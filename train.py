import copy
import functools
import os
import random
import warnings
from collections import OrderedDict
from pathlib import Path
from time import time

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from augment import AugmentPipe
from config import *
from dataset import get_dataset
from misc import *
from models import create_model
from torch_utils.misc import print_module_summary


OUTDIR_MAX_LEN = 1024


def parse_iter(filename):
    stem = filename.split('/')[-1].split('.')[0]
    try:
        # ckpt-10000.pt
        return int(Path(filename).stem.split('-')[-1])
    except ValueError:
        raise UserError("Failed to parse training step of checkpoint filename"
                        "Valid format is ckpt-<iter>.pt")


class Trainer():
    def __init__(self, cfg, local_rank=0, debug=False, use_wandb=False, **kwargs) -> None:
        t = time()
        self.cfg = cfg
        self.num_gpus = torch.cuda.device_count()
        self.local_rank = local_rank
        self.device = torch.device(f'cuda:{local_rank}')
        self.batch_gpu = cfg.TRAIN.batch_gpu
        self.start_iter = 0
        self.use_wandb = use_wandb
        self.stats_keys = []
        # ['g', 'd', 'real_score', 'fake_score', 'mean_path', 'r1', 'path', 'path_length']

        self.performance_and_reproducibility(**kwargs)
        self.wandb_id = None
        self.outdir = self.get_output_dir(cfg)
        self.log = setup_logger(self.outdir, local_rank, debug=debug)
        if local_rank == 0:
            (self.outdir / 'config.yml').write_text(cfg.dump())
            print(cfg)

        self.train_ds = get_dataset(cfg, split='all')
        self.g, self.d = create_model(cfg, deivce=self.device)
        self.g_ema = copy.deepcopy(self.g).eval()

        if cfg.ADA.enabled:
            self.augment_pipe = AugmentPipe(**cfg.ADA.KWARGS).train().requires_grad_(False).to(self.device)
            self.stats_keys.append('ada_p')
            self.augment_pipe.p.copy_(torch.as_tensor(cfg.ADA.p))

        # Define optimizers with Lazy regularizer
        g_reg_ratio = cfg.TRAIN.PPL.every / (cfg.TRAIN.PPL.every + 1)
        d_reg_ratio = cfg.TRAIN.R1.every / (cfg.TRAIN.R1.every + 1)
        self.g_optim = torch.optim.Adam(self.g.parameters(), lr=cfg.TRAIN.lrate * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))
        self.d_optim = torch.optim.Adam(self.d.parameters(), lr=cfg.TRAIN.lrate * d_reg_ratio, betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio))

        # Print network summary tables. TODO Using model infomation to create fake data
        if self.local_rank == 0:
            z = torch.empty([self.batch_gpu, self.g.z_dim], device=self.device)
            c = None
            heatmaps = torch.empty([self.batch_gpu, 17, 256, 256], device=self.device)
            imgs = print_module_summary(self.g, [z, c, heatmaps])
            print_module_summary(self.d, [imgs, c])

        # self.stats = OrderedDict((k, torch.tensor(0.0, dtype=torch.float, device=self.device)) for k in stats_keys)
        if cfg.TRAIN.ckpt:
            self.resume_from_checkpoint(cfg.TRAIN.ckpt)

        if self.num_gpus > 1:
            self.g = DDP(self.g, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
            self.d = DDP(self.d, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

        # if 'fid' in self.metrics:
        #     self.fid_tracker = FIDTracker(cfg, self.local_rank, self.num_gpus, self.out_dir)

        if use_wandb:
            self.run = self.launch_wandb()

        self.log.info(f"trainer initialized. ({time() - t :.2f} sec)")

    def performance_and_reproducibility(self, seed=0, cudnn_benchmark=False, amp=False):
        random.seed(seed * self.num_gpus + self.local_rank)
        np.random.seed(seed * self.num_gpus + self.local_rank)
        torch.manual_seed(seed * self.num_gpus + self.local_rank)

        torch.backends.cudnn.benchmark = cudnn_benchmark
        self.autocast = True if amp else False
        self.g_scaler = GradScaler(enabled=amp)
        self.d_scaler = GradScaler(enabled=amp)
        # TODO allow using tf32 for matmul and convolution

    def get_output_dir(self, cfg) -> Path:
        if self.num_gpus > 1:
            sync_data = torch.zeros([OUTDIR_MAX_LEN, ], dtype=torch.uint8, device=self.device)

        if self.local_rank == 0:
            root = Path(cfg.out_root)
            if not root.exists():
                print('populate {} as experiment root directory'.format(root))
                root.mkdir(parents=True)

            exp_name = ''
            existing_serial_num = [int(x.name[:5]) for x in root.glob("[0-9]" * 5 + "-*") if x.is_dir()]
            serial_num = max(existing_serial_num) + 1 if existing_serial_num else 0
            exp_name += str(serial_num).zfill(5)

            if self.use_wandb:
                if 'WANDB_RESUME' in os.environ and os.environ['WANDB_RESUME'] == 'must':
                    # resume training. reuse run id
                    self.wandb_id = os.environ['WANDB_RUN_ID']
                else:
                    self.wandb_id = wandb.util.generate_id()
                exp_name += f'-{self.wandb_id}'

            cfg_name = cfg.name if cfg.name else 'default'
            exp_name += f'-{self.num_gpus}gpu-{cfg_name}'
            outdir = root / exp_name
            outdir.mkdir(parents=True)

            if self.num_gpus > 1:
                ascii = [ord(c) for c in str(outdir)]
                assert len(ascii) <= OUTDIR_MAX_LEN
                for i in range(len(ascii)):
                    sync_data[i] = ascii[i]
                torch.distributed.broadcast(sync_data, src=0)
        elif self.num_gpus > 1:
            torch.distributed.broadcast(sync_data, src=0)
            ascii = list(sync_data[torch.nonzero(sync_data, as_tuple=True)].cpu())
            outdir = Path(''.join([chr(a) for a in ascii]))

        return outdir

    @master_only
    def launch_wandb(self):
        cfg = convert_to_dict(self.cfg)
        exp_name = cfg.pop('name')
        desc = cfg.pop('description')
        run = wandb.init(
            project=f'FaceHuman-KD_attention',
            name=exp_name,
            config=cfg,
            notes=desc,
            tags=['finetune'] if self.cfg.TRAIN.ckpt else None,
        )

        if run.resumed:
            # When WANDB_RESUME and WANDB_RUN_ID are set.
            assert self.cfg.TRAIN.ckpt
            start_iter = parse_iter(self.cfg.TRAIN.ckpt)
            if run.starting_step != start_iter:
                warnings.warn("non-increased step in log cal is not allowed in Wandb."
                              "It will cause wandb skip logging until last step in previous run")

        return run

    def get_trainloader(self):
        assert self.train_ds.targets, "Please call ds.update_targets([desired_targets])"
        sampler = (torch.utils.data.DistributedSampler(self.train_ds, shuffle=True)
                   if self.num_gpus > 1
                   else torch.utils.data.RandomSampler())

        loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_gpu,
            sampler=sampler,
            num_workers=self.cfg.DATASET.num_workers,
            pin_memory=self.cfg.DATASET.pin_memory,
            persistent_workers=True,
            worker_init_fn=self.train_ds.__class__.worker_init_fn
        )

        # Infinite loader
        while True:
            cur_epoch = self.cfg.TRAIN.iteration * self.batch_gpu / len(self.train_ds)
            if self.num_gpus > 1:
                assert isinstance(loader.sampler, torch.utils.data.distributed.DistributedSampler)
                loader.sampler.set_epoch(cur_epoch)

            for batch in loader:
                yield batch

            cur_epoch += 1

    def resume_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.log(f'resume from {ckpt_path}')

        # g, d, g_ema, g_optim, d_optim, g_scaler, d_scaler, mean_path_length, ada_p
        # TODO a mechanism to register which obj to checkpointing
        for key, state_dict in ckpt.items():
            obj = getattr(self, key, None)
            if obj is not None:
                # distinguish state and nn.Module
                obj.load_state_dict(state_dict)

        self.start_iter = parse_iter(ckpt_path)

    def train(self):
        if self.start_iter >= self.cfg.TRAIN.iteration:
            self.log.info("Training target has already achieved")

        self.train_ds.update_targets(['face', 'human', 'heatmap'])
    # default value for loss, metrics
    # sample data
    # train step (data + model -> loss & metric): D_main, D_Reg, G_main, G_Reg
    # backward + update
    # ema
    # synchronize metrics among devices
    # checkpointing, logging, progress bar
    # clean


if __name__ == "__main__":
    args = get_cmdline_args()
    cfg = get_cfg_defaults()
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    delattr(args, 'cfg')

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        assert torch.distributed.is_available()
        assert 0 <= args.local_rank < num_gpus
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://', rank=args.local_rank, world_size=num_gpus)

        assert torch.distributed.is_initialized()
        torch.distributed.barrier(device_ids=[args.local_rank])

    cfg.freeze()
    trainer = Trainer(cfg, **vars(args))
    # trainer.train()
