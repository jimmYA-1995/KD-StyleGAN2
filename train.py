import copy
import functools
import os
os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1"
os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
os.environ["TPU_VISIBLE_DEVICES"] = "0,1,2,3"
import random
import sys
import warnings
from collections import OrderedDict
from pathlib import Path
from time import time

import numpy as np
import skimage.io as io
import torch
## from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid, save_image
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
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
from torch_xla.amp import autocast## , GradScaler
# try:
#   from torch_xla.amp import syncfree
# except ImportError:
#   assert False, "Missing package syncfree; the package is available in torch-xla>=1.11"


OUTDIR_MAX_LEN = 1024
warnings.filterwarnings('ignore')


def parse_iter(filename):
    try:
        # ckpt-10000.pt
        return int(Path(filename).stem.split('-')[-1])
    except ValueError:
        raise UserError("Failed to parse training step of checkpoint filename"
                        "Valid format is ckpt-<iter>.pt")


class Trainer():
    def __init__(self, cfg, local_rank=0, debug=False, use_wandb=False, outdir='test', **performance_opts):
        t = time()
        self.cfg = cfg
        self.num_gpus = xm.xrt_world_size()
        self.local_rank = local_rank
        self.device = xm.xla_device() # device in folked process always be xla:0
        self.batch_gpu = cfg.TRAIN.batch_gpu
        self.start_iter = 0
        self.outdir = outdir
        self.use_wandb = use_wandb
        self.metrics = cfg.EVAL.metrics
        self.fid_tracker = None

        # stats
        stat_keys = ['mean_path_length']
        # ['g', 'd', 'real_score', 'fake_score', 'mean_path', 'r1', 'path', 'path_length']

        conv2d_gradfix.enabled = True                         # Improves training speed.
        grid_sample_gradfix.enabled = True                    # Avoids errors with the augmentation pipe.
        self.performance_and_reproducibility(**performance_opts)
        self.wandb_id = None
        self.log = setup_logger(self.outdir, local_rank, debug=debug)
        if local_rank == 0:
            (self.outdir / 'config.yml').write_text(cfg.dump())
            print(cfg)

        self.train_ds = get_dataset(cfg, split='all')
        self.val_ds = None
        self._samples = None

        self.g, self.d = create_model(cfg, device=self.device)
        self.g_ema = copy.deepcopy(self.g).eval().requires_grad_(False)

        # Define optimizers with Lazy regularizer
        g_reg_ratio = cfg.TRAIN.PPL.every / (cfg.TRAIN.PPL.every + 1) if cfg.TRAIN.PPL.every != -1 else 1
        d_reg_ratio = cfg.TRAIN.R1.every / (cfg.TRAIN.R1.every + 1) if cfg.TRAIN.R1.every != -1 else 1
        self.g_optim = torch.optim.Adam(self.g.parameters(), lr=cfg.TRAIN.lrate * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))
        self.d_optim = torch.optim.Adam(self.d.parameters(), lr=cfg.TRAIN.lrate * d_reg_ratio, betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio))

        # Print network summary tables.
        if self.local_rank == 0:  ## if xm.is_master_ordinal():
            z = [torch.empty([self.batch_gpu, self.g.z_dim], device=self.device)]
            c = torch.empty([self.batch_gpu * len(cfg.classes), self.d.c_dim], device=self.device) if self.d.c_dim > 0 else None
            heatmaps = None
            imgs = print_module_summary(self.g, [z, heatmaps])
            print_module_summary(self.d, [torch.cat(list(imgs.values()), dim=0 if self.d.c_dim > 0 else 1), c])

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

        # Training stats will be aggregated accross all GPUs for monitoring
        self.stats = OrderedDict([(k, torch.tensor(0.0, device=self.device)) for k in stat_keys])

        self.ckpt_required_keys = ["g", "atten", "d", "g_ema", "g_optim", "d_optim", "stats"]
        if cfg.TRAIN.CKPT.path:
            self.resume_from_checkpoint(cfg.TRAIN.CKPT.path)

        if cfg.ADA.enabled:
            self.log.info("build Augment Pipe")
            self.augment_pipe = AugmentPipe(**cfg.ADA.KWARGS).train().requires_grad_(False).to(self.device)
            if 'ada_p' not in self.stats:
                self.stats['ada_p'] = torch.as_tensor(cfg.ADA.p, device=self.device)
            self.augment_pipe.p.copy_(self.stats['ada_p'])
            if cfg.ADA.target > 0:
                self.log.info("Adaptively adjust ADA probability")
                self.ada_moments = torch.zeros([2], device=self.device)  # [num_scalars, sum_of_scalars]
                self.ada_sign = torch.tensor(0.0, dtype=torch.float, device=self.device)

        # For compatibility of single/multi GPU training
        self.g_, self.d_ = self.g, self.d
        # if self.num_gpus > 1:
        #     self.g = DDP(self.g, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        #     self.d = DDP(self.d, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

        if use_wandb:
            self.run = self.launch_wandb()

        self.log.info(f"trainer initialized. ({time() - t :.2f} sec)")

    def performance_and_reproducibility(self, seed=0, cudnn_benchmark=False, amp=False):
        random.seed(seed * self.num_gpus + self.local_rank)
        np.random.seed(seed * self.num_gpus + self.local_rank)
        torch.manual_seed(seed * self.num_gpus + self.local_rank)

        torch.backends.cudnn.benchmark = cudnn_benchmark
        self.autocast = True if amp else False
        # TODO allow using tf32 for matmul and convolution
        # Decide which is better? gradscaler or `torch.nan_to_num`

    @master_only
    def launch_wandb(self) -> wandb.run:
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
                self.log.warning("non-increased step in log call is not allowed in Wandb."
                                 "It will cause wandb skip logging until last step in previous run")

        return run

    @master_only
    def log_wandb(self, step):
        if not self.use_wandb:
            return

        self.wandb_stats = {k: v.item() for k, v in self.stats.items()}
        self.wandb_stats['epoch'] = self.epoch
        self.run.log(self.wandb_stats, step=step)

    def sample_forever(self, loader, pbar=False):
        """ Inifinite loader with optional progress bar. """
        _loader = loader._loader._loader
        # _loader = loader._loader if isinstance(loader, pl.MpDeviceLoader) or isinstance(loader, pl.ParallelLoader) else loader
        # epoch value may incorrect if we resume training with different num_gpus.
        self.epoch = self.start_iter * self.batch_gpu * self.num_gpus // len(_loader.dataset)
        while True:
            if self.num_gpus > 1:
                _loader.sampler.set_epoch(self.epoch)

            for batch in loader:
                if isinstance(batch, dict):
                    yield {k: v.to(self.device, non_blocking=self.cfg.DATASET.pin_memory) for k, v in batch.items()}
                else:
                    yield [x.to(self.device, non_blocking=self.cfg.DATASET.pin_memory) for x in batch]

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

    # @master_only
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
            'stats': self.stats,
        }

        ## torch.save(snapshot, ckpt_dir / f'ckpt-{i :06d}.pt')
        xm.save(snapshot, ckpt_dir / f'ckpt-{i :06d}.pt')
        del snapshot  # Conserve memory
        if self.local_rank == 0:
            ckpt_paths = list()
            if cfg.max_keep != -1 and len(ckpt_paths) > cfg.ckpt_max_keep:
                ckpts = sorted([p for p in ckpt_dir.glob('*.pt')], key=lambda p: p.name[5:11], reverse=True)
                for to_removed in ckpts[cfg.max_keep:]:
                    os.remove(to_removed)

    def train(self):
        self.train_ds.update_targets(['ref', 'target'])
        ema_beta = 0.5 ** (self.batch_gpu * self.num_gpus / (self.cfg.TRAIN.ema * 1000))
        train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_gpu,
            sampler=get_sampler(self.train_ds, num_gpus=self.num_gpus, local_rank=self.local_rank),
            num_workers=self.cfg.DATASET.num_workers,
            pin_memory=self.cfg.DATASET.pin_memory,
            persistent_workers=self.cfg.DATASET.num_workers > 0,
            worker_init_fn=self.train_ds.__class__.worker_init_fn if self.cfg.DATASET.num_workers > 0 else None
        )
        # train_loader = pl.MpDeviceLoader(train_loader, self.device)
        train_loader = pl.ParallelLoader(train_loader, [self.device]).per_device_loader(self.device)
        loader = self.sample_forever(train_loader, pbar=(self.local_rank == 0))

        for i in range(self.start_iter, self.cfg.TRAIN.iteration):
            if self.local_rank == 0:
                t = time()
            data = next(loader)
            if self.local_rank == 0:
                print(f"fetch data: {time() - t: .2f} sec")
                t = time()

            self.Dmain(data, r1_reg=(self.cfg.TRAIN.R1.every != -1 and i % self.cfg.TRAIN.R1.every == 0))
            if self.local_rank == 0:
                print(f"Dmain: {time() - t: .2f} sec")
                t = time()
            self.Gmain(data, pl_reg=(self.cfg.TRAIN.PPL.every != -1 and i % self.cfg.TRAIN.PPL.every == 0))

            if self.local_rank == 0:
                print(f"Gmain: {time() - t: .2f} sec")
                t = time()
            # self.ema(ema_beta=ema_beta)

            # if self.local_rank == 0:
            #     print(f"EMA: {time() - t: .2f} sec")
                # t = time()
            self.reduce_stats()

            if self.local_rank == 0:
                print(f"reduce stats: {time() - t: .2f} sec")
                t = time()
            # FID
            if self.cfg.ADA.enabled and self.cfg.ADA.target > 0 and (i % self.cfg.ADA.interval == 0):
                self.update_ada()

            if self.local_rank == 0:
                print(f"update ADA: {time() - t: .2f} sec")
                t = time()
            if self.fid_tracker is not None and (i == 0 or (i + 1) % self.cfg.EVAL.FID.every == 0 or i == self.cfg.TRAIN.iteration - 1):
                fids = self.fid_tracker(self.g_ema.classes, self.infer_fn, (i + 1), save=(self.local_rank == 0))
                self.stats.update({f'FID/{c}': torch.tensor(v, device=self.device) for c, v in fids.items()})

            if (i + 1) % self.cfg.TRAIN.CKPT.every == 0 or i == self.cfg.TRAIN.iteration - 1:
                self.save_to_checkpoint(i + 1)

            if (i + 1) % self.cfg.TRAIN.SAMPLE.every == 0 or i == self.cfg.TRAIN.iteration - 1:
                self.sampling(i + 1)

            if self.local_rank == 0:
                self.log_wandb(step=i)

            if self.local_rank == 0:
                print(f"MISC: {time() - t: .2f} sec")
                t = time()
        self.clear()
        xm.rendezvous('finished')

    def Dmain(self, data, r1_reg=False):
        """ GAN loss & (opt.)R1 regularization """
        self.g.requires_grad_(False)
        self.d.requires_grad_(True)

        loss_Dmain = loss_Dr1 = 0
        if self.cfg.ADA.enabled:
            # augment data Separately
            data['ref'] = self.augment_pipe(data['ref'])
            data['target'] = self.augment_pipe(data['target'])

        real = torch.cat([data['ref'], data['target']], dim=0 if self.d_.c_dim > 0 else 1).detach().requires_grad_(r1_reg)

        # Style mixing regularization
        n = 2 if random.random() < self.cfg.TRAIN.style_mixing_prob else 1
        zs = torch.randn([n, data[self.cfg.classes[0]].shape[0], self.g_.z_dim], device=self.device).unbind(0)
        roi_row, roi_col = self.train_ds.ROI
        with autocast(enabled=self.autocast):
            fake_imgs = self.g(zs, None)
            if self.cfg.TRAIN.use_mix_loss:
                small_ref = torch.nn.functional.interpolate(fake_imgs['ref'], size=(roi_row.stop - roi_row.start, roi_col.stop - roi_col.start), mode='bicubic')
                fake_imgs['mix'] = fake_imgs['target'].clone()
                fake_imgs['mix'][:, :, roi_row, roi_col] = small_ref

            aug_fake_imgs = {k: (self.augment_pipe(v) if self.cfg.ADA.enabled else v)
                             for k, v in fake_imgs.items()}

            cat_dim, c = 1, None
            if self.d_.c_dim > 0:
                cat_dim = 0
                c = torch.eye(len(self.g_.classes), device=self.device).unsqueeze(1).repeat(1, zs[0].shape[0], 1).flatten(0, 1)

            fake = torch.cat([aug_fake_imgs['ref'], aug_fake_imgs['target']], dim=cat_dim)
            real_pred = self.d(real, c=c)
            fake_pred = self.d(fake, c=c)
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

            if self.cfg.TRAIN.use_mix_loss:
                fake_mix = torch.cat([aug_fake_imgs['ref'], aug_fake_imgs['mix']], dim=cat_dim)
                fake_pred_mix = self.d(fake_mix, c=c)
                fake_mix_loss = torch.nn.functional.softplus(fake_pred_mix).mean()
                self.stats[f"D-FakeMix-Score"] = fake_pred_mix.mean().detach()
                self.stats[f"loss/D-FakeMix"] = fake_mix_loss.detach()
                loss_Dmain = loss_Dmain + fake_mix_loss

            if r1_reg:
                r1 = r1_loss(real_pred, real)
                self.stats[f'loss/R1'] = r1.detach()
                loss_Dr1 = loss_Dr1 + self.cfg.TRAIN.R1.gamma / 2 * r1 * self.cfg.TRAIN.R1.every

                loss_Dr1 = loss_Dr1 + real_pred[0] * 0

        d_loss = loss_Dmain + loss_Dr1
        self.d.zero_grad(set_to_none=True)
        d_loss.backward()
        for param in self.d_.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        # self.d_optim.step()
        # xm.mark_step()
        xm.optimizer_step(self.d_optim, barrier=True)

    def Gmain(self, data, pl_reg=False):
        self.g.requires_grad_(True)
        self.d.requires_grad_(False)

        # Style mixing regularization
        n = 2 if random.random() < self.cfg.TRAIN.style_mixing_prob else 1
        zs = torch.randn([n, data['target'].shape[0], self.g_.z_dim], device=self.device).unbind(0)
        roi_row, roi_col = self.train_ds.ROI
        loss_Gmain = loss_Gpl = 0
        with autocast(enabled=self.autocast):
            fake_imgs = self.g(zs, None)
            small_ref = torch.nn.functional.interpolate(fake_imgs['ref'], size=(roi_row.stop - roi_row.start, roi_col.stop - roi_col.start), mode='bicubic')
            if self.cfg.TRAIN.use_mix_loss:
                fake_imgs['mix'] = fake_imgs['target'].clone()
                fake_imgs['mix'][:, :, roi_row, roi_col] = small_ref

            aug_fake_imgs = {k: (self.augment_pipe(v) if self.cfg.ADA.enabled else v)
                             for k, v in fake_imgs.items()}
            cat_dim, c = 1, None
            if self.d_.c_dim > 0:
                cat_dim = 0
                c = torch.eye(len(self.g_.classes), device=self.device).unsqueeze(1).repeat(1, zs[0].shape[0], 1).flatten(0, 1)

            fake = torch.cat([aug_fake_imgs['ref'], aug_fake_imgs['target']], dim=cat_dim)
            fake_pred = self.d(fake, c=c)
            gan_loss = torch.nn.functional.softplus(-fake_pred).mean()
            self.stats['loss/G-GAN'] = gan_loss.detach()
            loss_Gmain = loss_Gmain + gan_loss

            if self.cfg.TRAIN.use_mix_loss:
                fake_mix = torch.cat([aug_fake_imgs['ref'], aug_fake_imgs['mix']], dim=cat_dim)
                fake_pred_mix = self.d(fake_mix, c=c)
                gan_mix_loss = torch.nn.functional.softplus(-fake_pred_mix).mean()
                self.stats['loss/G-GANMix'] = gan_mix_loss.detach()
                loss_Gmain = loss_Gmain + gan_mix_loss

            loss_rec = torch.nn.functional.l1_loss(small_ref.detach(), fake_imgs['target'][:, :, roi_row, roi_col])
            self.stats['loss/G-reconstruction'] = loss_rec.detach()
            loss_Gmain = loss_Gmain + loss_rec

        if pl_reg:
            cfg = self.cfg.TRAIN.PPL
            pl_bs = max(1, self.batch_gpu // cfg.bs_shrink)
            n = 2 if random.random() < self.cfg.TRAIN.style_mixing_prob else 1
            zs = torch.randn([n, pl_bs, self.g_.z_dim], device=self.device).unbind(0)
            with autocast(enabled=self.autocast):
                fake_imgs, ws = self.g(zs, None, return_dlatent=True)
                path_loss, self.stats['mean_path_length'], self.stats['path_length'] = path_regularize(
                    fake_imgs['target'],
                    ws['target'],
                    self.stats['mean_path_length'].detach()
                )
            loss_Gpl = path_loss * cfg.gain * cfg.every + 0 * fake_imgs['target'][0, 0, 0, 0]
            self.stats['loss/PPL'] = path_loss.detach()

        g_loss = loss_Gmain + loss_Gpl
        self.g.zero_grad(set_to_none=True)
        g_loss.backward()
        for param in self.g_.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        # self.g_optim.step()
        # xm.mark_step()
        xm.optimizer_step(self.g_optim, barrier=True)

    def reduce_stats(self):
        """ Reduce all training stats to master for reporting. """
        if self.num_gpus == 1 or not self.stats:
            if not self.stats:
                self.log.warning("calling reduce Op. while there is no stats.")
            return

        stats = torch.stack(list(self.stats.values()), dim=0)
        xm.all_reduce('sum', stats, scale=1.0 / xm.xrt_world_size())
        # torch.distributed.reduce_multigpu(stats, dst=0)
        self.stats = OrderedDict([(k, v) for k, v in zip(list(self.stats.keys()), stats)])

    def ema(self, ema_beta=0.99):
        for p_ema, p in zip(self.g_ema.parameters(), self.g_.parameters()):
            p_ema.copy_(p.lerp(p_ema, ema_beta))

    def update_ada(self):
        cfg = self.cfg.ADA
        if self.num_gpus > 1:
            # torch.distributed.all_reduce(self.ada_moments)
            xm.all_reduce('sum', self.ada_moments)

        ada_sign = (self.ada_moments[1] / self.ada_moments[0]).cpu().numpy()
        adjust = np.sign(ada_sign - cfg.target) * (self.batch_gpu * self.num_gpus * cfg.interval) / (cfg.kimg * 1000)
        self.augment_pipe.p.copy_((self.augment_pipe.p + adjust).max(constant(0, device=self.device)))
        self.stats['ada_p'] = self.augment_pipe.p
        self.ada_moments.zero_()

    def sampling(self, i):
        """ inference & save sample images """
        if self._samples is None:
            bs_gpu = (self.cfg.n_sample // self.num_gpus) + int(self.cfg.n_sample % self.num_gpus != 0)
            loader = torch.utils.data.DataLoader(
                self.train_ds,
                batch_size=bs_gpu,
                sampler=get_sampler(self.train_ds, eval=True, num_gpus=self.num_gpus, local_rank=self.local_rank)
            )
            self._samples = {k: v.to(self.device) for k, v in next(iter(loader)).items()}
            self.real_samples = {k: xm.all_gather(v) for k, v in self._samples.items()}
            _samples = [self.real_samples[k] for k in self.cfg.classes]
            samples = torch.stack(_samples, dim=1).flatten(0, 1)
            grid = make_grid(
                samples,
                nrow=int(self.cfg.n_sample ** 0.5) * len(_samples),
                normalize=True,
                value_range=(-1, 1),
            ).cpu()

            if not xm.is_master_ordinal():
                xm.rendezvous('sampling-real')

            filepath = self.outdir / 'samples' / f'real.png'
            if not filepath.exists():
                grid = grid.numpy().transpose(1, 2, 0) * 255.
                grid = np.clip(grid, 0, 255).astype(np.uint8)
                io.imsave(filepath, grid)
                
            if xm.is_master_ordinal():
                xm.rendezvous('sampling-real')

        z = [torch.randn([self._samples['target'].shape[0], self.g_ema.z_dim], device=self.device)]
        with torch.no_grad():
            _fake_imgs = self.g_ema(z, None, noise_mode='const')

        fake_imgs = {k: xm.all_gather(v)[:self.cfg.n_sample] for k, v in _fake_imgs.items()}

        _samples = [fake_imgs[k] for k in self.cfg.classes]
        samples = torch.stack(_samples, dim=1).flatten(0, 1)
        grid = make_grid(
            samples,
            nrow=int(self.cfg.n_sample ** 0.5) * len(_samples),
            normalize=True,
            value_range=(-1, 1),
        ).cpu()
        if not xm.is_master_ordinal():
            xm.rendezvous('sampling-fake')

        filepath = self.outdir / 'samples' / f'fake-{i :06d}.png'
        if not filepath.exists():
            grid = grid.numpy().transpose(1, 2, 0) * 255.
            grid = np.clip(grid, 0, 255).astype(np.uint8)
            io.imsave(filepath, grid)

        if xm.is_master_ordinal():
            xm.rendezvous('sampling-fake')

    def clear(self):
        if getattr(self, 'pbar', None):
            self.pbar.close()


def get_output_dir(cfg, args, wandb_id=None) -> Path:
    """ Get output folder name
        1. Decide serial No. (the biggest + 1)
        2. concat unique ID produced by weight & bias
        3. concat name defined by user (in configuration file)
    """
    root = Path(cfg.out_root)
    if not root.exists():
        print('populate {} as experiment root directory'.format(root))
        root.mkdir(parents=True)

    exp_name = ''
    existing_serial_num = [int(x.name[:5]) for x in root.glob("[0-9]" * 5 + "-*") if x.is_dir()]
    serial_num = max(existing_serial_num) + 1 if existing_serial_num else 0
    exp_name += str(serial_num).zfill(5)

    if wandb_id is not None:
        exp_name += f'-{wandb_id}'

    cfg_name = cfg.name if cfg.name else 'default'
    exp_name += f'-{cfg_name}'
    outdir = root / exp_name
    outdir.mkdir(parents=True)
    (outdir / 'samples').mkdir()
    (outdir / 'checkpoints').mkdir()
    return outdir

def mp_fn(index, cfg, args):
    args.local_rank = index
    trainer = Trainer(cfg, **vars(args))
    trainer.train()


if __name__ == "__main__":
    args = get_cmdline_args()
    cfg = get_cfg_defaults()
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    delattr(args, 'cfg')
    cfg.freeze()

    wandb_id = None
    if args.use_wandb:
        if 'WANDB_RESUME' in os.environ and os.environ['WANDB_RESUME'] == 'must':
            # resume training. reuse run id
            wandb_id = os.environ['WANDB_RUN_ID']
        else:
            wandb_id = wandb.util.generate_id()
    args.outdir = get_output_dir(cfg, args, wandb_id=wandb_id)

    if 'XRT_TPU_CONFIG' in os.environ:  # TPU
        num_cores = 8# len(xm.get_xla_supported_devices(devkind='TPU'))
        if num_cores > 1:
            xmp.spawn(mp_fn, args=(cfg, args,), nprocs=num_cores, start_method='fork')
            sys.exit(0)
    # elif num_gpus > 1:
    #     assert torch.distributed.is_available()
    #     assert 0 <= args.local_rank < num_gpus
    #     torch.cuda.set_device(args.local_rank)
    #     torch.distributed.init_process_group(
    #         backend='nccl', init_method='env://', rank=args.local_rank, world_size=num_gpus)

    #     assert torch.distributed.is_initialized()
    #     torch.distributed.barrier(device_ids=[args.local_rank])

    trainer = Trainer(cfg, **vars(args))
    trainer.train()
