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


def calc_KID(real_features, gen_features, num_subsets=100, max_subset_size=1000):
    n = real_features.shape[1]
    m = min(min(real_features.shape[0], gen_features.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = gen_features[np.random.choice(gen_features.shape[0], m, replace=False)]
        y = real_features[np.random.choice(real_features.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_subsets / m
    return float(kid)


class KIDTracker():
    def __init__(self, cfg, rank, num_gpus, out_dir):
        detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        print(f"[{os.getpid()}] rank: {rank}({num_gpus} gpus)")

        self.rank = rank
        self.num_gpus = num_gpus
        self.device = torch.device('cuda', rank) if num_gpus > 1 else 'cuda'
        self.classes = cfg.classes
        self.cfg = cfg.EVAL.KID
        self.log = logging.getLogger(f'GPU{rank}')
        self.pbar = pbar(rank)
        self.out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir

        # metrics
        self.n_sample = cfg.EVAL.KID.n_sample
        self.real_features = None
        self.iterations = []
        self.kids = []

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

            if self.rank == 0:
                torch.distributed.barrier(device_ids=[rank])
        self.log.info("load inceptionV3 model complete ({:.2f} sec)".format(time.time() - start))

        # get features for real images
        if cfg.EVAL.KID.inception_cache:
            self.log.info(f"load inception cache from {cfg.EVAL.KID.inception_cache}")
            embeds = pickle.load(open(cfg.EVAL.KID.inception_cache, 'rb'))
            self.real_features = embeds['feature']
            self.classes = list(self.real_features.keys())
        else:
            real_features = {}
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
                real_features[c] = self.extract_features(generator(ds, c))
                self.log.info(f"cost {time.time() - t :.2f} sec")

            if self.rank == 0:
                self.log.info(f"save inception cache to {self.out_dir / 'inception_cache.pkl'}")
                with open(self.out_dir / 'inception_cache_kid.pkl', 'wb') as f:
                    pickle.dump(dict(feature=real_features), f)

            self.real_features = real_features

    def __call__(self, classes, generator_fn, iteration, save=False, eps=1e-6):
        assert all(c in self.classes for c in classes)
        start = time.time()
        self.iterations.append(iteration)
        kid = {}

        for c in classes:
            self.log.info(f"Extract feature from {c} on {iteration} iteration")
            sample_feature = self.extract_features(generator_fn(c))
            kid[c] = calc_KID(self.real_features[c], sample_feature)

        self.kids.append(kid)
        total_time = time.time() - start
        self.log.info(f'KID on {iteration} iterations: "{kid}". [costs {total_time: .2f} sec(s)]')

        if self.rank == 0 and save:
            with open(self.out_dir / 'kid.txt', 'a+') as f:
                f.write(f'{iteration}: {json.dumps(kid)}\n')

            # compatible with NVLab
            result_dict = {"results": {"kid50k_full": kid},
                           "metric": "kid50k_full",
                           "total_time": total_time,
                           "total_time_str": f"{int(total_time // 60)}m {int(total_time % 60)}s",
                           "num_gpus": self.num_gpus,
                           "snapshot_pkl": f"ckpt-{iteration :06d}.pt",
                           "timestamp": time.time()}

            with open(self.out_dir / 'metric-kid50k_full.jsonl', 'at') as f:
                f.write(f"{json.dumps(result_dict)}\n")

        return kid

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
        return features

    def plot_figure(self):
        self.log.info(f"save KID figure in {self.out_dir / 'kid.png'}")
        kiter = np.array(self.iterations) / 1000.
        for c in self.kids[0].keys():
            plt.plot(kiter, np.array([x[c] for x in self.kids]))

        plt.xlabel('k iterations')
        plt.ylabel('KID')
        plt.legend(self.classes, loc='upper right')
        plt.savefig(self.out_dir / 'kid.png')


def subprocess_fn(rank, args, cfg, temp_dir):
    from models import create_model
    if args.num_gpus > 1:
        torch.cuda.set_device(rank)
        init_method = f"file://{os.path.join(temp_dir, '.torch_distributed_init')}"
        torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    setup_logger(args.out_dir, rank, debug=args.debug)
    device = torch.device('cuda', rank) if args.num_gpus > 1 else 'cuda'
    kid_tracker = KIDTracker(cfg, rank, args.num_gpus, args.out_dir)

    if args.ckpt is None:
        if rank == 0:
            print("only get statistics on real data. return...")
        return

    val_ds = None  # get_dataset(cfg, split='val', xflip=True)
    args.ckpt = Path(args.ckpt)
    args.ckpt = sorted(p for p in args.ckpt.glob('*.pt')) if args.ckpt.is_dir() else [args.ckpt]
    for ckpt in args.ckpt:
        print(f"load {str(ckpt)}")
        iteration = int(ckpt.stem.split('-')[-1])
        ckpt = torch.load(ckpt)

        g = create_model(cfg, device=device, eval_only=True)
        g.load_state_dict(ckpt['g_ema'])
        infer_fn = partial(
            image_generator, g, cfg.EVAL.batch_gpu, ds=val_ds, device=device, num_gpus=args.num_gpus)

        if args.num_gpus > 1:
            torch.distributed.barrier(device_ids=[rank])

        kid_tracker(g.classes, infer_fn, iteration, save=args.save)

    if rank == 0:
        kid_tracker.plot_figure()


if __name__ == '__main__':
    import argparse
    import tempfile

    parser = argparse.ArgumentParser()
    # parser.add_argument('--truncation', type=float, default=1)
    # parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument("--cfg", type=str, help="path to the configuration file")
    parser.add_argument("--gpus", type=int, default=1, dest='num_gpus')
    parser.add_argument('--out_dir', type=str, default='/tmp/kid_result')
    parser.add_argument('--ckpt', type=str, default=None, metavar='CHECKPOINT', help='model ckpt or dir')
    parser.add_argument('--save', action='store_true', default=False, help='save kid.txt')
    parser.add_argument("--debug", action='store_true', default=False, help="whether to use debug mode")

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    if args.cfg:
        cfg.merge_from_file(args.cfg)

    if not Path(args.out_dir).exists():
        Path(args.out_dir).mkdir(parents=True)
    assert args.num_gpus >= 1
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, cfg=cfg, temp_dir=temp_dir)
        else:
            os.environ['OMP_NUM_THREADS'] = '1'  # for scipy performance issue
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, cfg, temp_dir), nprocs=args.num_gpus)
