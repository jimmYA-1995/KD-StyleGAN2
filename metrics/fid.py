import os
import sys
import time
import json
import pickle
import logging
from functools import partial
from pathlib import Path

import torch
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from tqdm import trange

import dnnlib
from config import get_cfg_defaults
from dataset import get_dataset
from generate import image_generator
from misc import setup_logger, master_only


def pbar(rank, force=False):
    def range_wrapper(*args, **kwargs):
        return trange(*args, **kwargs) if rank == 0 or force else range(*args, **kwargs)
    return range_wrapper


class FIDTracker():
    def __init__(self, cfg, rank, num_gpus, out_dir):
        detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        print(f"[{os.getpid()}] rank: {rank}({num_gpus} gpus)")

        self.rank = rank
        self.num_gpus = num_gpus
        self.device = torch.device('cuda', rank) if num_gpus > 1 else 'cuda'
        self.classes = cfg.classes
        self.cfg = cfg.EVAL.FID
        self.log = logging.getLogger(f'GPU{rank}')
        self.pbar = pbar(rank)
        self.out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir

        # metrics
        self.n_sample = cfg.EVAL.FID.n_sample
        self.real_means = None  # [ndarray(2048) float32] * num_class
        self.real_covs = None  # [ndarray(2048, 2048) float64] * num_class
        self.iterations = []
        self.fids = []

        start = time.time()
        self.log.info("load inceptionV3 model...")
        if num_gpus == 1:
            with dnnlib.util.open_url(detector_url, verbose=True) as f:
                self.inceptionV3 = pickle.load(f).to(self.device)
        else:
            if self.rank != 0:
                torch.distributed.barrier(device_ids=[rank])
            with dnnlib.util.open_url(detector_url, verbose=(rank == 0)) as f:
                self.inceptionV3 = pickle.load(f).to(self.device)
            # self.inceptionV3 = load_patched_inception_v3().eval().to(self.device)
            if self.rank == 0:
                torch.distributed.barrier(device_ids=[rank])
        self.log.info("load inceptionV3 model complete ({:.2f} sec)".format(time.time() - start))

        # get features for real images
        if cfg.EVAL.FID.inception_cache:
            self.log.info(f"load inception cache from {cfg.EVAL.FID.inception_cache}")
            embeds = pickle.load(open(cfg.EVAL.FID.inception_cache, 'rb'))
            self.real_means = embeds['mean']
            self.real_covs = embeds['cov']
            self.classes = list(self.real_means.keys())
        else:
            real_means, real_covs = {}, {}
            ds = get_dataset(cfg, split='all')

            def generator(ds, target_class):
                ds.update_targets([target_class])
                num_items = min(len(ds), 50000)
                item_subset = [(i * num_gpus + rank) % num_items
                               for i in range((num_items - 1) // num_gpus + 1)]
                loader = torch.utils.data.DataLoader(
                    ds,
                    sampler=item_subset,
                    batch_size=self.cfg.batch_gpu,
                    num_workers=3,
                )
                for batch in loader:
                    yield batch[target_class].to(self.device)

            for c in self.classes:
                self.log.info(f"Extract real features from '{c}'")
                t = time.time()
                real_means[c], real_covs[c] = self.extract_features(generator(ds, c))

                self.log.info(f"cost {time.time() - t :.2f} sec")

            if self.rank == 0:
                self.log.info(f"save inception cache to {self.out_dir / 'inception_cache.pkl'}")
                with open(self.out_dir / 'inception_cache.pkl', 'wb') as f:
                    pickle.dump(dict(mean=real_means, cov=real_covs), f)

            self.real_means, self.real_covs = real_means, real_covs

    @classmethod
    def calc_fid(cls, real_mean, real_cov, sample_mean, sample_cov, eps=1e-6):
        cov_sqrt, _ = scipy.linalg.sqrtm(sample_cov @ real_cov, disp=False)

        if not np.isfinite(cov_sqrt).all():
            print('product of cov matrices is singular')
            offset = np.eye(sample_cov.shape[0]) * eps
            cov_sqrt = scipy.linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

        if np.iscomplexobj(cov_sqrt):
            print("cov_sqrt is complex number")
            if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
                m = np.max(np.abs(cov_sqrt.imag))

                raise ValueError(f'Imaginary component {m}')

            cov_sqrt = cov_sqrt.real

        mean_diff = sample_mean - real_mean
        mean_norm = mean_diff @ mean_diff

        trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)
        return mean_norm + trace

    def __call__(self, classes, generator_fn, iteration, save=False, eps=1e-6):
        assert self.real_means is not None and self.real_covs is not None
        assert all(c in self.classes for c in classes)
        start = time.time()
        self.iterations.append(iteration)
        fid = {}

        for c in classes:
            self.log.info(f"Extract feature from {c} on {iteration} iteration")
            sample_mean, sample_cov = self.extract_features(generator_fn(c))
            fid[c] = FIDTracker.calc_fid(self.real_means[c], self.real_covs[c], sample_mean, sample_cov, eps=eps)

        self.fids.append(fid)
        total_time = time.time() - start
        self.log.info(f'FID on {iteration} iterations: "{fid}". [costs {total_time: .2f} sec(s)]')

        if self.rank == 0 and save:
            with open(self.out_dir / 'fid.txt', 'a+') as f:
                f.write(f'{iteration}: {json.dumps(fid)}\n')

            # compatible with NVLab
            result_dict = {"results": {"fid50k_full": fid},
                           "metric": "fid50k_full",
                           "total_time": total_time,
                           "total_time_str": f"{int(total_time // 60)}m {int(total_time % 60)}s",
                           "num_gpus": self.num_gpus,
                           "snapshot_pkl": "none",
                           "timestamp": time.time()}

            with open(self.out_dir / 'metric-fid50k_full.jsonl', 'at') as f:
                f.write(f"{json.dumps(result_dict)}\n")

        return fid

    @torch.no_grad()
    def extract_features(self, img_generator):
        cnt = 0
        features = []
        while True:
            try:
                imgs = next(img_generator)
                imgs = (imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            except StopIteration:
                self.log.warn(f"Only get {cnt} images")
                break

            feature = self.inceptionV3(imgs, return_features=True)
            if self.num_gpus > 1:
                _features = []
                for src in range(self.num_gpus):
                    y = feature.clone()
                    torch.distributed.broadcast(y, src=src)
                    _features.append(y)
                feature = torch.stack(_features, dim=1).flatten(0, 1)
            features.append(feature)
            cnt += feature.shape[0]
            if cnt >= self.n_sample:
                break

        features = torch.cat(features, dim=0)[:self.n_sample].cpu().numpy()
        mean = np.mean(features, 0)
        cov = np.cov(features, rowvar=False)
        return mean, cov

    @torch.no_grad()
    def extract_feature_dict(self, img_generator):
        cnt = 0
        slicing_map = {
            'human': (slice(None),) * 4,
            'human-TopHalf': (slice(None), slice(None), slice(None, 128), slice(None)),
            'human-BtmHalf': (slice(None), slice(None), slice(128, None), slice(None)),
            'human-TopCentral': (slice(None), slice(None), slice(None, 128), slice(64, 192))
        }
        features_dict = {k: [] for k in slicing_map.keys()}
        features = []
        while True:
            try:
                imgs = next(img_generator)
                imgs = (imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            except StopIteration:
                self.log.warn(f"Only get {cnt} images")

            for k, features in features_dict.items():
                feature = self.inceptionV3(imgs[slicing_map[k]], return_features=True)
                if self.num_gpus > 1:
                    _features = []
                    for src in range(self.num_gpus):
                        y = feature.clone()
                        torch.distributed.broadcast(y, src=src)
                        _features.append(y)
                    feature = torch.stack(_features, dim=1).flatten(0, 1)
                features.append(feature)
            cnt += feature.shape[0]
            if cnt >= self.n_sample:
                break

        mean_dict, cov_dict = dict(), dict()
        for k, features in features_dict.items():
            features = torch.cat(features, dim=0)[:self.n_sample].cpu().numpy()
            mean_dict[k] = np.mean(features, 0)
            cov_dict[k] = np.cov(features, rowvar=False)
        return mean_dict, cov_dict

    def plot_figure(self):
        self.log.info(f"save FID figure in {self.out_dir / 'fid.png'}")
        kiter = np.array(self.iterations) / 1000.
        for c in self.fids[0].keys():
            plt.plot(kiter, np.array([x[c] for x in self.fids]))

        plt.xlabel('k iterations')
        plt.ylabel('FID')
        plt.legend(self.classes, loc='upper right')
        plt.savefig(self.out_dir / 'fid.png')


def subprocess_fn(rank, args, cfg, temp_dir):
    from models import create_model
    if args.num_gpus > 1:
        torch.cuda.set_device(rank)
        init_method = f"file://{os.path.join(temp_dir, '.torch_distributed_init')}"
        torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    setup_logger(args.out_dir, rank, debug=args.debug)
    device = torch.device('cuda', rank) if args.num_gpus > 1 else 'cuda'
    fid_tracker = FIDTracker(cfg, rank, args.num_gpus, args.out_dir)

    if args.ckpt is None:
        if rank == 0:
            print("only get statistics on real data. return...")
        return

    val_ds = None  # get_dataset(cfg, split='val', xflip=True)
    args.ckpt = Path(args.ckpt)
    args.ckpt = sorted(p for p in args.ckpt.glob('*.pt')) if args.ckpt.is_dir() else [args.ckpt]
    for ckpt in args.ckpt:
        print(f"load {str(ckpt)}")
        iteration = int(str(ckpt.name)[5:11])
        ckpt = torch.load(ckpt)

        g = create_model(cfg, device=device, eval_only=True)
        g.load_state_dict(ckpt['g_ema'])
        infer_fn = partial(
            image_generator, g, cfg.EVAL.batch_gpu, ds=val_ds, device=device, num_gpus=args.num_gpus)

        if args.num_gpus > 1:
            torch.distributed.barrier(device_ids=[rank])

        fid_tracker(g.classes, infer_fn, iteration, save=args.save)

    if rank == 0:
        fid_tracker.plot_figure()


if __name__ == '__main__':
    import argparse
    import tempfile

    parser = argparse.ArgumentParser()
    # parser.add_argument('--truncation', type=float, default=1)
    # parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument("--cfg", type=str, help="path to the configuration file")
    parser.add_argument("--gpus", type=int, default=1, dest='num_gpus')
    parser.add_argument('--out_dir', type=str, default='/tmp/fid_result')
    parser.add_argument('--ckpt', type=str, default=None, metavar='CHECKPOINT', help='model ckpt or dir')
    parser.add_argument('--save', action='store_true', default=False, help='save fid.txt')
    parser.add_argument("--debug", action='store_true', default=False, help="whether to use debug mode")

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    if args.cfg:
        cfg.merge_from_file(args.cfg)

    assert args.num_gpus >= 1
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, cfg=cfg, temp_dir=temp_dir)
        else:
            os.environ['OMP_NUM_THREADS'] = '1'  # for scipy performance issue
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, cfg, temp_dir), nprocs=args.num_gpus)
