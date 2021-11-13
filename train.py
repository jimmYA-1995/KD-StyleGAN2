import copy
import functools
import os
import random
import sys
from typing import Dict
import warnings
from collections import OrderedDict
from pathlib import Path
from time import time

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
from tqdm import tqdm
import wandb

from augment import AugmentPipe
from config import *
from dataset import get_dataset, get_sampler
from generate import image_generator
from losses import *
from misc import *
from models import *
from metrics.fid import FIDTracker
from torch_utils.misc import print_module_summary, constant


OUTDIR_MAX_LEN = 1024
warnings.filterwarnings('ignore')


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
        self.metrics = cfg.EVAL.metrics
        self.fid_tracker = None

        # stats
        stat_keys = ['mean_path_length']
        # ['g', 'd', 'real_score', 'fake_score', 'mean_path', 'r1', 'path', 'path_length']

        self.performance_and_reproducibility(**kwargs)
        self.wandb_id = None
        self.outdir = self.get_output_dir(cfg)
        self.log = setup_logger(self.outdir, local_rank, debug=debug)
        if local_rank == 0:
            (self.outdir / 'config.yml').write_text(cfg.dump())
            print(cfg)

        self.train_ds = get_dataset(cfg, split='all')
        self.num_classes = self.train_ds.num_classes
        self.val_ds = None  # get_dataset(cfg, split='val', xflip=True)
        self._samples = None

        self.g, self.d = create_model(cfg, self.train_ds.num_classes, device=self.device)
        self.g_ema = copy.deepcopy(self.g).eval().requires_grad_(False)

        # Define optimizers with Lazy regularizer
        g_reg_ratio = cfg.TRAIN.PPL.every / (cfg.TRAIN.PPL.every + 1) if cfg.TRAIN.PPL.every != -1 else 1
        d_reg_ratio = cfg.TRAIN.R1.every / (cfg.TRAIN.R1.every + 1) if cfg.TRAIN.R1.every != -1 else 1
        self.g_optim = torch.optim.Adam(self.g.parameters(), lr=cfg.TRAIN.lrate * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))
        self.d_optim = torch.optim.Adam(self.d.parameters(), lr=cfg.TRAIN.lrate * d_reg_ratio, betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio))

        # Print network summary tables.
        if self.local_rank == 0:
            z = torch.empty([self.batch_gpu, self.g.z_dim], device=self.device)
            gc = torch.empty([self.batch_gpu, self.num_classes + 1], device=self.device)
            imgs = print_module_summary(self.g, [z, gc])

            dc = torch.empty([self.batch_gpu, self.num_classes], device=self.device)
            print_module_summary(self.d, [*(imgs[k] for k in self.cfg.classes), dc])

        if 'fid' in self.metrics:
            self.fid_tracker = FIDTracker(cfg, self.local_rank, self.num_gpus, self.outdir)
            self.infer_fn = functools.partial(
                image_generator,
                self.g_ema,
                cfg.EVAL.batch_gpu,
                ds=self.val_ds,
                device=self.device,
                num_gpus=self.num_gpus
            )

        self.stats = OrderedDict([(k, torch.tensor(0.0, device=self.device)) for k in stat_keys])

        self.ckpt_required_keys = ["g", "d", "g_ema", "g_optim", "d_optim", "g_scaler", "d_scaler", "stats"]
        if cfg.TRAIN.CKPT.path:
            self.resume_from_checkpoint(cfg.TRAIN.CKPT.path)

        if cfg.ADA.enabled:
            self.log.info("build Augment Pipe")
            self.augment_pipe = AugmentPipe(**cfg.ADA.KWARGS).train().requires_grad_(False).to(self.device)
            if 'ada_p' not in self.stats:
                self.stats['ada_p'] = torch.as_tensor(cfg.ADA.p, device=self.device)
            self.augment_pipe.p.copy_(self.stats['ada_p'])
            if cfg.ADA.target > 0:
                self.ada_moments = torch.zeros([2], device=self.device)  # [num_scalars, sum_of_scalars]
                self.ada_sign = torch.tensor(0.0, dtype=torch.float, device=self.device)

        self.g_, self.d_ = self.g, self.d
        if self.num_gpus > 1:
            self.g = DDP(self.g, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
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
            (outdir / 'samples').mkdir()
            (outdir / 'checkpoints').mkdir()

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
    def launch_wandb(self) -> wandb.run:
        self.wandb_stats = dict()
        cfg = convert_to_dict(self.cfg)
        exp_name = cfg.pop('name')
        desc = cfg.pop('description')
        run = wandb.init(
            project=f'FaceHuman-KD_attention-fixed_face_pos',
            id=self.wandb_id,
            name=exp_name,
            config=cfg,
            notes=desc,
            tags=['resume'] if self.cfg.TRAIN.CKPT.path else None,
        )

        if run.resumed:
            # When WANDB_RESUME and WANDB_RUN_ID are set.
            assert self.cfg.TRAIN.CKPT.path
            start_iter = parse_iter(self.cfg.TRAIN.CKPT.path)
            if run.starting_step != start_iter:
                self.log.warning("non-increased step in log cal is not allowed in Wandb."
                                 "It will cause wandb skip logging until last step in previous run")

        return run

    @master_only
    def log_wandb(self, step):
        if not self.use_wandb:
            return

        self.wandb_stats['epoch'] = self.epoch
        self.wandb_stats.update(self.stats)
        self.run.log(self.wandb_stats, step=step)
        self.wandb_stats = dict()

    def sample_forever(self, loader, pbar=False):
        """ Inifinite loader with optional progress bar. """
        # epoch value may incorrect if we resume training with different num_gpus to previous run.
        self.epoch = self.start_iter * self.batch_gpu * self.num_gpus // len(loader.dataset)
        while True:
            if self.num_gpus > 1:
                loader.sampler.set_epoch(self.epoch)

            for batch, label in loader:
                yield ({k: v.to(self.device, non_blocking=self.cfg.DATASET.pin_memory) for k, v in batch.items()},
                       label.to(self.device, non_blocking=self.cfg.DATASET.pin_memory))

                # For aviod warmup(i.e. 1st iteration)
                if pbar and getattr(self, 'pbar', None) is None:
                    desc = "epoch {}|G: {loss/G-GAN: .4f}; D: {loss/D-Real: .4f}/{loss/D-Fake: .4f}"
                    self.pbar = tqdm(
                        total=self.cfg.TRAIN.iteration,
                        initial=self.start_iter + 1,
                        dynamic_ncols=True,
                        smoothing=0,
                        colour='yellow'
                    )

                if pbar:
                    self.pbar.update(1)
                    self.pbar.set_description(desc.format(self.epoch, **self.stats))
            self.epoch += 1

    def resume_from_checkpoint(self, ckpt_path):
        """ resume module and stats """
        self.start_iter = parse_iter(ckpt_path)
        if self.start_iter >= self.cfg.TRAIN.iteration:
            self.log.info(f"Training target has already achieved. training iter "
                          f"from ckpt {self.start_iter} v.s. target iteration {self.cfg.TRAIN.iteration}")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.log.info(f'resume from {ckpt_path}')

        for key, value in ckpt.items():
            if key not in self.ckpt_required_keys:
                self.log.warning(f"Missed key: {key}")
                continue

            obj = getattr(self, key, None)
            if obj is None:
                self.log.warning(f"Trainer has no attribute {key} to load")
                continue

            if key == 'stats':
                self.stats.update(value)
            else:
                obj.load_state_dict(value)

    @master_only
    def save_to_checkpoint(self, i):
        """ Save checktpoint. Default format: ckpt-000001.pt """
        cfg = self.cfg.TRAIN.CKPT
        ckpt_dir = self.outdir / 'checkpoints'

        snapshot = {
            'g': self.g_.state_dict(),
            'd': self.d_.state_dict(),
            'g_ema': self.g_ema.state_dict(),
            'g_optim': self.g_optim.state_dict(),
            'd_optim': self.d_optim.state_dict(),
            'g_scaler': self.g_scaler.state_dict(),
            'd_scaler': self.d_scaler.state_dict(),
            'stats': self.stats,
        }

        torch.save(snapshot, ckpt_dir / f'ckpt-{i :06d}.pt')
        ckpt_paths = list()
        if cfg.max_keep != -1 and len(ckpt_paths) > cfg.ckpt_max_keep:
            ckpts = sorted([p for p in ckpt_dir.glob('*.pt')], key=lambda p: p.name[5:11], reverse=True)
            for to_removed in ckpts[cfg.max_keep:]:
                os.remove(to_removed)

    def train(self):
        ema_beta = 0.5 ** (self.batch_gpu * self.num_gpus / (10 * 1000))
        train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_gpu,
            sampler=get_sampler(self.train_ds, num_gpus=self.num_gpus),
            num_workers=self.cfg.DATASET.num_workers,
            pin_memory=self.cfg.DATASET.pin_memory,
            persistent_workers=self.cfg.DATASET.num_workers > 0,
            worker_init_fn=self.train_ds.__class__.worker_init_fn if self.cfg.DATASET.num_workers > 0 else None
        )
        loader = self.sample_forever(train_loader, pbar=(self.local_rank == 0))

        for i in range(self.start_iter, self.cfg.TRAIN.iteration):
            data, label = next(loader)

            self.Dmain(data, label, r1_reg=(self.cfg.TRAIN.R1.every != -1 and i % self.cfg.TRAIN.R1.every == 0))
            self.Gmain(data, label, pl_reg=(self.cfg.TRAIN.PPL.every != -1 and i % self.cfg.TRAIN.PPL.every == 0))

            self.ema(ema_beta=ema_beta)
            self.reduce_stats()

            # FID
            if self.cfg.ADA.enabled and self.cfg.ADA.target > 0 and (i % self.cfg.ADA.interval == 0):
                self.update_ada()

            if self.fid_tracker is not None and (i == 0 or (i + 1) % self.cfg.EVAL.FID.every == 0):
                fids = self.fid_tracker(self.cfg.classes[:1], self.infer_fn, (i + 1), save=(self.local_rank == 0))
                self.stats.update({f'FID/{c}': torch.tensor(v, device=self.device) for c, v in fids.items()})

            if (i + 1) % self.cfg.TRAIN.CKPT.every == 0:
                self.save_to_checkpoint(i + 1)

            if i == 0 or (i + 1) % self.cfg.TRAIN.SAMPLE.every == 0:
                self.sampling(i + 1)

            if self.local_rank == 0:
                self.log_wandb(step=i)

        self.clear()

    def Dmain(self, data, label, r1_reg=False):
        """ GAN loss & (opt.)R1 regularization """
        self.g.requires_grad_(False)
        self.d.requires_grad_(True)

        loss_Dmain = loss_Dr1 = 0
        if self.cfg.ADA.enabled:
            aug_data = {k: self.augment_pipe(data[k]) for k in self.cfg.classes}
            data.update(aug_data)
        data = {k: v.detach().requires_grad_(r1_reg) for k, v in data.items()}
        one_hot = torch.nn.functional.one_hot(label, self.num_classes + 1).detach()

        z = torch.randn(label.shape[0], self.g_.z_dim, device=self.device)
        with autocast(enabled=self.autocast):
            fake_imgs = self.g(z, one_hot)
            aug_fake_imgs = {k: (self.augment_pipe(v) if self.cfg.ADA.enabled else v)
                             for k, v in fake_imgs.items()}

            real_pred = self.d(*(data[k] for k in self.cfg.classes), c=one_hot[:, 1:])
            fake_pred = self.d(*(aug_fake_imgs[k] for k in self.cfg.classes), c=one_hot[:, 1:])
            real_loss = torch.nn.functional.softplus(-real_pred).mean()
            fake_loss = torch.nn.functional.softplus(fake_pred).mean()
            self.stats[f"D-Real-Score"] = real_pred.mean().detach()
            self.stats[f"D-Fake-Score"] = fake_pred.mean().detach()
            self.stats[f"loss/D-Real"] = real_loss.detach()
            self.stats[f"loss/D-Fake"] = fake_loss.detach()

            loss_Dmain = loss_Dmain + real_loss + fake_loss

            if self.cfg.ADA.enabled and self.cfg.ADA.target > 0:
                self.ada_moments[0].add_(torch.ones_like(real_pred).sum())
                self.ada_moments[1].add_(real_pred.sign().detach().flatten().sum())

            if r1_reg:
                r1 = 0
                for cn in self.cfg.classes:
                    r1 = r1 + r1_loss(real_pred, data[cn])
                self.stats[f'loss/R1'] = r1.detach()
                loss_Dr1 = loss_Dr1 + self.cfg.TRAIN.R1.gamma / 2 * r1 * self.cfg.TRAIN.R1.every
                loss_Dr1 = loss_Dr1 + real_pred[0] * 0

        d_loss = loss_Dmain + loss_Dr1
        self.d.zero_grad(set_to_none=True)
        self.d_scaler.scale(d_loss).backward()
        self.d_scaler.step(self.d_optim)
        self.d_scaler.update()

    def Gmain(self, data, label, pl_reg=False):
        self.g.requires_grad_(True)
        self.d.requires_grad_(False)
        ref_c = self.cfg.classes[0]

        z = torch.randn(data[ref_c].shape[0], self.g_.z_dim, device=self.device)
        one_hot = torch.nn.functional.one_hot(label, self.num_classes + 1).detach()
        loss_Gmain = loss_Gpl = 0
        with autocast(enabled=self.autocast):
            # GAN loss
            fake_imgs = self.g(z, one_hot)

            aug_fake_imgs = {k: (self.augment_pipe(v) if self.cfg.ADA.enabled else v)
                             for k, v in fake_imgs.items()}

            fake_pred = self.d(*(aug_fake_imgs[k] for k in self.cfg.classes), one_hot[:, 1:])
            gan_loss = torch.nn.functional.softplus(-fake_pred).mean()
            self.stats['loss/G-GAN'] = gan_loss.detach()
            loss_Gmain = loss_Gmain + gan_loss

            # recontruction loss: patch guide
            loss = 0
            for r, t, l in zip(fake_imgs['patch'].detach(), fake_imgs['target'], label):
                loss = loss + torch.abs(r - t[(slice(None), *self.train_ds.plocation[l - 1])]).mean()
            rec_patch_target = loss / label.shape[0]
            loss_Gmain = loss_Gmain + rec_patch_target
            self.stats[f'loss/G-patch->target'] = rec_patch_target.detach()

        # if pl_reg:
        #     cfg = self.cfg.TRAIN.PPL
        #     pl_bs = max(1, self.batch_gpu // cfg.bs_shrink)
        #     with autocast(enabled=self.autocast):
        #         fake_imgs, ws = self.g(z[:pl_bs], one_hot[:pl_bs], return_dlatent=pl_reg)
        #         path_loss, self.stats['mean_path_length'], self.stats['path_length'] = path_regularize(
        #             fake_imgs[ref_c],
        #             ws[ref_c],
        #             self.stats['mean_path_length'].detach()
        #         )
        #     loss_Gpl = path_loss * cfg.gain * cfg.every + 0 * fake_imgs[ref_c][0, 0, 0, 0]
        #     self.stats['loss/PPL'] = path_loss.detach()

        g_loss = loss_Gmain + loss_Gpl
        self.g.zero_grad(set_to_none=True)
        self.g_scaler.scale(g_loss).backward()
        self.g_scaler.step(self.g_optim)
        self.g_scaler.update()

    def reduce_stats(self):
        """ Reduce all training stats to master for reporting. """
        if self.num_gpus == 1 or not self.stats:
            if not self.stats:
                self.log.warning("calling reduce Op. while there is no stats.")
            return

        stat_list = [torch.stack(list(self.stats.values()), dim=0)]
        torch.distributed.reduce_multigpu(stat_list, dst=0)

        if self.local_rank == 0:
            self.stats = OrderedDict([(k, (v / self.num_gpus)) for k, v in zip(list(self.stats.keys()), stat_list[0])])

    def ema(self, ema_beta=0.99):
        for p_ema, p in zip(self.g_ema.parameters(), self.g_.parameters()):
            p_ema.copy_(p.lerp(p_ema, ema_beta))

    def update_ada(self):
        cfg = self.cfg.ADA
        if self.num_gpus > 1:
            torch.distributed.all_reduce(self.ada_moments)

        ada_sign = (self.ada_moments[1] / self.ada_moments[0]).cpu().numpy()
        adjust = np.sign(ada_sign - cfg.target) * (self.batch_gpu * self.num_gpus * cfg.interval) / (cfg.kimg * 1000)
        self.augment_pipe.p.copy_((self.augment_pipe.p + adjust).max(constant(0, device=self.device)))
        self.stats['ada_p'] = self.augment_pipe.p
        self.ada_moments.zero_()

    def sampling(self, i):
        """ inference & save sample images """
        if self._samples is None:
            ds = get_dataset(self.cfg, split='test', xflip=False, num_items=self.cfg.n_sample)
            loader = torch.utils.data.DataLoader(
                ds,
                batch_size=self.cfg.n_sample // self.num_gpus,
                sampler=get_sampler(ds, eval=True, num_gpus=self.num_gpus)
            )
            data, _ = next(iter(loader))
            self._samples = {k: v.to(self.device) for k, v in data.items()}
            self.real_samples = {k: self.all_gather(v) for k, v in self._samples.items()}
            save_image(
                self.real_samples['target'],
                self.outdir / 'samples' / f'real.png',
                nrow=int(self.cfg.n_sample ** 0.5),
                normalize=True,
                value_range=(-1, 1),
            )
            save_image(
                self.real_samples['patch'],
                self.outdir / 'samples' / f'real-patch.png',
                nrow=int(self.cfg.n_sample ** 0.5),
                normalize=True,
                value_range=(-1, 1),
            )


            self.sample_z = torch.randn([self.cfg.n_sample // self.num_gpus, self.g_ema.z_dim], device=self.device)
            self.tgt_c = torch.nn.functional.one_hot(torch.zeros([self.sample_z.shape[0]], device=self.device, dtype=torch.int64), self.num_classes + 1)
            self.patch_c = torch.nn.functional.one_hot(torch.arange(self.num_classes, device=self.device, dtype=torch.int64) + 1, self.num_classes + 1).repeat(self.sample_z.shape[0], 1)
        with torch.no_grad():
            target = self.g_ema(self.sample_z, self.tgt_c)['target']
            patch = self.g_ema(self.sample_z.unsqueeze(1).repeat(1, self.num_classes, 1).flatten(0, 1), self.patch_c)['patch']

        target = self.all_gather(target)
        patch = self.all_gather(patch)
        if self.local_rank == 0:
            assert target.shape[0] == self.cfg.n_sample
            assert patch.shape[0] == self.cfg.n_sample * self.num_classes

        _, c, h, w = patch.shape
        # patch = patch.reshape(self.cfg.n_sample, self.num_classes, c, h, w)
        if self.local_rank == 0:
            save_image(
                target,
                self.outdir / 'samples' / f'fake-{i :06d}.png',
                nrow=int(self.cfg.n_sample ** 0.5),
                normalize=True,
                value_range=(-1, 1),
            )
            save_image(
                patch,
                self.outdir / 'samples' / f'fake-{i :06d}-patch.png',
                nrow=int(self.cfg.n_sample ** 0.5) * self.num_classes,
                normalize=True,
                value_range=(-1, 1),
            )

    def all_gather(self, tensor, cat_dim=0):
        """ All gather `tensor` and concatenate along `cat_dim`. When write this code,
            NCCL does not support `gather`
        """
        if self.num_gpus == 1:
            return tensor

        gather_list = [torch.zeros_like(tensor) for _ in range(self.num_gpus)]
        torch.distributed.all_gather(gather_list, tensor)

        return torch.cat(gather_list, dim=cat_dim)

    def clear(self):
        if getattr(self, 'pbar', None):
            self.pbar.close()


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
