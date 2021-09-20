import random
import pickle
from pathlib import Path
from typing import List, Set

import numpy as np
import torch
import skimage.io as io
from torch.utils import data
from torchvision import transforms
from PIL import Image

from config import configurable
from misc import cords_to_map, draw_pose_from_cords, UserError


DATASET = {}


def register(cls):
    DATASET[cls.__name__] = cls
    return cls


def get_dataset(cfg, **override_kwargs):
    ds_cls = DATASET.get(cfg.DATASET.name, None)
    if ds_cls is None:
        raise ValueError(f"{cfg.DATASET.name} is not available")

    return ds_cls(cfg, **override_kwargs)


class DefaultDataset(data.Dataset):
    def __init__(self, cfg, split='train'):
        assert len(cfg.roots) == len(cfg.source) == 1
        assert split in ['train', 'val', 'test', 'all']
        self.cfg = cfg
        self.root = Path(cfg.roots[0]).expanduser()
        self.face_dir = None
        self.fileIDs = None
        self.idx = None
        self.resolution = cfg.resolution
        self.split = split
        self.xflip = cfg.xflip
        self._img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std, inplace=True),
        ])
        self._mask_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.fileIDs) * 2 if self.xflip else len(self.fileIDs)

    def maybe_xflip(self, img):
        """ xflip if xflip enabled and index > len(ds) / 2,
            no op. otherwise
        """
        assert isinstance(img, Image.Image) and self.idx is not None
        if not self.xflip or self.idx < len(self.fileIDs):
            return img

        return img.transpose(method=Image.FLIP_LEFT_RIGHT)

    def img_transform(self, img):
        img = self.maybe_xflip(img)
        return self._img_transform(img)

    def mask_transform(self, img):
        img = self.maybe_xflip(img)
        return self._mask_transform(img)

    @classmethod
    def worker_init_fn(cls, worker_id):
        """ For reproducibility & randomness in multi-worker mode """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        if hasattr(dataset, 'rng'):
            dataset.rng = np.random.default_rng(worker_seed)


@register
class DeepFashion(data.Dataset):
    @configurable()
    def __init__(
        self,
        resolution: int = 256,
        roots: List[str] = None,
        sources: List[str] = None,
        split: str = 'all',
        xflip: bool = False,
        num_items: int = float('inf')
    ):
        assert roots is not None and sources is not None
        self.classes = ["DF_face", "DF_human"]
        self.res = resolution
        self.xflip = xflip
        self.available_targets = ['face', 'human', 'heatmap', 'vis_kp']
        self.targets = []

        root = Path(roots[0]).expanduser()
        split_map = pickle.load(open(root / 'new_split.pkl', 'rb'))
        self.fileIDs = [ID for IDs in split_map.values() for ID in IDs] if split == 'all' else split_map[split]
        self.fileIDs.sort()

        # self.size = {"DF_face": self.__len__(), "DF_human": self.__len__()}
        # self.labels = np.zeros((len(self),), dtype=int)

        src = sources[0]
        self.face_dir = root / src / 'face'
        self.human_dir = root / f'r{self.res}' / 'images'
        self.kp_dir = root / 'kp_heatmaps/keypoints'
        assert self.face_dir.exists() and self.human_dir.exists() and self.kp_dir.exists()
        assert set(self.fileIDs) <= set(p.stem for p in self.face_dir.glob('*.png'))
        assert set(self.fileIDs) <= set(p.stem for p in self.human_dir.glob('*.png'))
        assert set(self.fileIDs) <= set(p.stem for p in self.kp_dir.glob('*.pkl'))

        total = len(self.fileIDs) * 2 if xflip else len(self.fileIDs)
        self.num_items = min(total, num_items)

    @classmethod
    def from_config(cls, cfg):
        return {
            'resolution': cfg.resolution,
            'roots': cfg.DATASET.roots,
            'sources': cfg.DATASET.sources,
            'xflip': cfg.DATASET.xflip
        }

    @classmethod
    def worker_init_fn(cls, worker_id):
        """ For reproducibility & randomness in multi-worker mode """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        # worker_info = torch.utils.data.get_worker_info()

    def __len__(self):
        return self.num_items

    def update_targets(self, targets: List[str]) -> None:
        if not all([t in self.available_targets for t in targets]):
            raise ValueError(f"Some of desire targets is not available. "
                             f"Available targets for {self.__class__.__name__} dataset: {self.available_targets}")
        self.targets = targets

    def transform(self, img: np.ndarray, normalize=True, channel_first=True) -> torch.Tensor:
        """ normalize, channel first, maybe xflip, to Tensor """
        img = (img.astype(np.float32) - 127.5) / 127.5
        if self.xflip and self.idx > len(self.fileIDs):
            img = img[:, ::-1, :]
        if channel_first:
            img = img.transpose(2, 0, 1)

        return torch.from_numpy(img.copy())

    def __getitem__(self, idx) -> List[torch.Tensor]:
        data = {}
        self.idx = idx
        try:
            fileID = self.fileIDs[idx % len(self.fileIDs)] if self.xflip else self.fileIDs[idx]
        except IndexError as e:
            print(self.xflip, idx)
            raise RuntimeError(e)

        if 'face' in self.targets:
            img = io.imread(self.face_dir / f'{fileID}.png')
            data['face'] = self.transform(img)

        if 'human' in self.targets:
            img = io.imread(self.human_dir / f'{fileID}.png')
            data['human'] = self.transform(img)

        if 'heatmap' in self.targets or 'vis_kp' in self.targets:
            kp = pickle.load(open(self.kp_dir / f'{fileID}.pkl', 'rb'))[0][:, (1, 0, 2)]  # [K, (y, x, score)]
            cords = np.where(kp[:, 2:3] > 0.1, kp[:, :2], -np.ones_like(kp[:, :2]))

        if 'heatmap' in self.targets:
            heatmap = cords_to_map(cords, (self.res, self.res), sigma=8)
            data['heatmap'] = self.transform(heatmap, normalize=False)

        if 'vis_kp' in self.targets:
            vis_kp, _ = draw_pose_from_cords(cords.astype(int), (self.res, self.res))
            data['vis_kp'] = self.transform(vis_kp)

        return data


if __name__ == "__main__":
    from config import get_cfg_defaults
    cfg = get_cfg_defaults()
    ds = get_dataset(cfg, split='train', xflip=False)
    print(len(ds))
    ds.update_targets(["face", "human", "heatmap"])
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=2
    )
    data = next(iter(loader))
    print("finish")

