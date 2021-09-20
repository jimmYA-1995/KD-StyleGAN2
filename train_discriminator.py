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
from tqdm import tqdm
import wandb

from config import *
from dataset import get_dataset
from losses import *
from misc import *
from models import Discriminator
from torch_utils.misc import print_module_summary


OUTDIR_MAX_LEN = 1024
# warnings.filterwarnings("ignore")


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
        self.best = 0.0

        # stats
        self.stats_keys = []
        # ['g', 'd', 'real_score', 'fake_score', 'mean_path', 'r1', 'path', 'path_length']

        self.performance_and_reproducibility(**kwargs)
        self.wandb_id = None
        self.outdir = self.get_output_dir(cfg)
        self.log = setup_logger(self.outdir, local_rank, debug=debug)
        if local_rank == 0:
            (self.outdir / 'config.yml').write_text(cfg.dump())
            print(cfg)

        self.train_ds = get_dataset(cfg, split='train')
        self.val_ds = get_dataset(cfg, split='val', xflip=False)

        self.d = Discriminator(1, cfg.resolution, img_channels=6).to(self.device)

        # Define optimizers with Lazy regularizer
        d_reg_ratio = cfg.TRAIN.R1.every / (cfg.TRAIN.R1.every + 1)
        self.d_optim = torch.optim.Adam(self.d.parameters(), lr=cfg.TRAIN.lrate * d_reg_ratio, betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio))

        # Print network summary tables.
        if self.local_rank == 0:
            c = None
            imgs = torch.empty([self.batch_gpu, self.d.img_channels, cfg.resolution, cfg.resolution], device=self.device)
            print_module_summary(self.d, [imgs, c])

        # self.stats = OrderedDict((k, torch.tensor(0.0, dtype=torch.float, device=self.device)) for k in stats_keys)
        if cfg.TRAIN.ckpt:
            self.resume_from_checkpoint(cfg.TRAIN.ckpt)

        self.train_stats = torch.zeros([3], device=self.device)  # [total, real_pred, fake_pred]
        self.val_stats = torch.zeros([3], device=self.device)  # [total, real_pred, fake_pred]
        if self.num_gpus > 1:
            self.d = DDP(self.d, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

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
        self.wandb_stats = dict()
        cfg = convert_to_dict(self.cfg)
        exp_name = cfg.pop('name')
        desc = cfg.pop('description')
        run = wandb.init(
            project=f'train-D-only',
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

    def get_trainloader(self, ds):
        assert ds.targets, "Please call ds.update_targets([desired_targets])"
        sampler = (torch.utils.data.DistributedSampler(ds, shuffle=False)
                   if self.num_gpus > 1
                   else None)

        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_gpu,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=self.cfg.DATASET.num_workers,
            pin_memory=self.cfg.DATASET.pin_memory,
            persistent_workers=True,
            worker_init_fn=ds.__class__.worker_init_fn
        )

    def infinite_loader(self, loader, pbar=False):
        self.epoch = self.start_iter * self.batch_gpu // len(self.train_ds)
        while True:
            if self.num_gpus > 1:
                assert isinstance(loader.sampler, torch.utils.data.distributed.DistributedSampler)
                loader.sampler.set_epoch(self.epoch)

            if pbar:
                if getattr(self, 'pbar', None) is None:
                    self.pbar = tqdm(
                        total=self.cfg.TRAIN.iteration,
                        initial=self.start_iter,
                        dynamic_ncols=True,
                        smoothing=0, colour='yellow'
                    )

            for batch in loader:
                if pbar:
                    self.pbar.update(1)
                yield [x.to(self.device, non_blocking=self.cfg.DATASET.pin_memory) for x in batch]

            self.epoch += 1

    def evaluation(self, i):
        self.d.requires_grad_(False)
        self.d.eval()
        sampler = (torch.utils.data.DistributedSampler(self.val_ds, shuffle=True)
                   if self.num_gpus > 1
                   else None)
        loader = torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_gpu,
            sampler=sampler,
            num_workers=self.cfg.DATASET.num_workers,
            worker_init_fn=self.val_ds.__class__.worker_init_fn
        )
        real_preds, fake_preds = [], []
        self.val_ds.update_targets(['face', 'human'])
        for face, human in loader:
            face, human = face.to(self.device), human.to(self.device)

            with torch.no_grad():
                real = torch.cat([face, human], dim=1).detach()
                roll_face = torch.roll(face, 1, 0)
                fake = torch.cat([roll_face, human], dim=1).detach()

                real_pred = self.d(real)
                fake_pred = self.d(fake)

            if self.num_gpus > 1:
                _real_pred, _fake_pred = [], []
                for src in range(self.num_gpus):
                    x = real_pred.clone()
                    y = fake_pred.clone()
                    torch.distributed.broadcast(x, src=src)
                    torch.distributed.broadcast(y, src=src)
                    _real_pred.append(x)
                    _fake_pred.append(y)
                real_pred = torch.stack(_real_pred, dim=1).flatten(0, 1)
                fake_pred = torch.stack(_fake_pred, dim=1).flatten(0, 1)
            real_preds.append(real_pred)
            fake_preds.append(fake_pred)

        real_preds = torch.cat(real_preds, 0)[:len(self.val_ds)].cpu().numpy()
        fake_preds = torch.cat(fake_preds, 0)[:len(self.val_ds)].cpu().numpy()

        tp = (real_preds > 0).sum().item()
        tn = (fake_preds < 0).sum().item()
        fn = len(self.val_ds) - tp
        fp = len(self.val_ds) - tn
        sensitivity = tp / len(self.val_ds)
        specificity = tn / len(self.val_ds)
        if self.local_rank == 0:
            self.log.info(f"eval(epoch {self.epoch}): tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}, "
                          f"sensitivity: {sensitivity: .3f}, specificity: {specificity: .3f}")

        if self.use_wandb and self.local_rank == 0:
            self.wandb_stats.update({
                'Metric/sensitivity(val)': sensitivity,
                'Metric/specificity(val)': specificity
            })

        if (sensitivity + specificity) / 2 > self.best:
            self.best = (sensitivity + specificity) / 2
            d = self.d.module if self.num_gpus > 1 else self.d
            torch.save({'d': d.state_dict()}, f'ckpt-trainD_{sensitivity :.3f}_{specificity :.3f}-{str(i).zfill(6)}.pt')

    @master_only
    def update_metrics(self):
        tp = self.train_stats[1].item()
        fn = (self.train_stats[0] - self.train_stats[1]).item()
        tn = self.train_stats[2].item()
        fp = (self.train_stats[0] - self.train_stats[2]).item()
        sensitivity = tp / len(self.train_ds)
        specificity = tn / len(self.train_ds)

        if self.use_wandb:
            self.wandb_stats.update({
                'Metric/sensitivity(train)': sensitivity,
                'Metric/specificity(train)': specificity,
            })

        self.log.info(f"train(epoch {self.epoch}): tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}, "
                      f"sensitivity: {sensitivity: .3f}, specificity: {specificity: .3f}")

    def log_wandb(self, batch_stats, step):
        if not self.use_wandb:
            return

        self.wandb_stats['epoch'] = self.epoch
        self.wandb_stats.update(batch_stats)
        self.run.log(self.wandb_stats, step=step)
        self.wandb_stats = dict()

    def resume_from_checkpoint(self, ckpt_path):
        """ resume by mapping attribute name and ckpt key """
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.log.info(f'resume from {ckpt_path}')

        # g, d, g_ema, g_optim, d_optim, g_scaler, d_scaler, mean_path_length, ada_p
        # TODO a mechanism to register which obj to checkpointing
        missed_keys = set()
        for key, value in ckpt.items():
            obj = getattr(self, key, None)
            if obj is None:
                missed_keys.add(key)
                continue

            if isinstance(obj, torch.nn.Module):
                obj.load_state_dict(value)
            else:
                setattr(self, key, value)

        self.start_iter = parse_iter(ckpt_path)
        if self.local_rank == 0:
            if len(missed_keys):
                self.log.warning(f"Miss attr. to load: {missed_keys}")

    def train(self):
        t = time()
        if self.start_iter >= self.cfg.TRAIN.iteration:
            self.log.info("Training target has already achieved")
            return

        # default value for loss, metrics
        stats = OrderedDict()
        # TODO init value when resuming
        ema_beta = 0.5 ** (self.batch_gpu * self.num_gpus / (10 * 1000))
        self.train_ds.update_targets(['face', 'human'])
        train_loader = self.get_trainloader(self.train_ds)
        loader = self.infinite_loader(train_loader, pbar=self.local_rank == 0)

        for i in range(self.start_iter, self.cfg.TRAIN.iteration):
            face, human = next(loader)
            self.d.requires_grad_(True)

            do_Dr1 = (i % self.cfg.TRAIN.R1.every == 0)
            loss_Dmain = loss_Dr1 = 0
            real = torch.cat([face, human], dim=1).detach().requires_grad_(do_Dr1)
            roll_face = torch.roll(face, 1, 0)
            fake = torch.cat([roll_face, human], dim=1).detach().requires_grad_(do_Dr1)
            with autocast(enabled=self.autocast):
                real_pred = self.d(real)
                fake_pred = self.d(fake)
                loss_real = torch.nn.functional.softplus(-real_pred).mean()
                loss_fake = torch.nn.functional.softplus(fake_pred).mean()
                loss_Dmain = loss_real + loss_fake

                self.train_stats[0].add_(torch.ones_like(real_pred).sum())
                self.train_stats[1].add_((real_pred > 0).detach().flatten().sum())
                self.train_stats[2].add_((fake_pred < 0).detach().flatten().sum())

            stats['loss/real'] = loss_real.detach()
            stats['loss/fake'] = loss_fake.detach()
            if do_Dr1:
                r1 = r1_loss(real_pred, real)
                loss_Dr1 = self.cfg.TRAIN.R1.gamma / 2 * r1 * self.cfg.TRAIN.R1.every
                stats['loss/R1'] = loss_Dr1.detach()

            d_loss = real_pred[0] * 0 + loss_Dmain + loss_Dr1
            self.d.zero_grad(set_to_none=True)
            self.d_scaler.scale(d_loss).backward()
            self.d_scaler.step(self.d_optim)
            self.d_scaler.update()

            if self.num_gpus > 1:
                stat_list = [torch.stack(list(stats.values()), dim=0)]
                torch.distributed.reduce_multigpu(stat_list, dst=0)

                if self.local_rank == 0:
                    stat_keys = list(stats.keys())
                    stats = OrderedDict([(k, (v / self.num_gpus)) for k, v in zip(stat_keys, stat_list[0])])

            if i % (len(self.train_ds) // (self.batch_gpu * self.num_gpus)) == 0:
                # on epoch end
                if i > 0:
                    if self.num_gpus > 1:
                        torch.distributed.all_reduce(self.train_stats)

                    self.update_metrics()
                    self.train_stats.zero_()
                self.evaluation(i)

            if self.local_rank == 0:
                self.log_wandb(stats, step=i)

                desc = "epoch {}| real: {loss/real: .4f}; fake: {loss/fake: .4f}; r1: {loss/R1: .4f}"
                self.pbar.set_description(desc.format(self.epoch, **stats))

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
    trainer.train()
