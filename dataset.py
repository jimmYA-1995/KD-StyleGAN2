import random
import json
import pickle
from collections import namedtuple
from pathlib import Path
from typing import List, Set

import numpy as np
import torch
import skimage.io as io
from torch.utils import data
from torchvision import transforms
from PIL import Image

from config import configurable
from misc import cords_to_map, draw_pose_from_cords, ffhq_alignment


DATASET = {}


def register(cls):
    DATASET[cls.__name__] = cls
    return cls


def get_dataset(cfg, **override_kwargs):
    ds_cls = DATASET.get(cfg.DATASET.name, None)
    if ds_cls is None:
        raise ValueError(f"{cfg.DATASET.name} is not available")

    return ds_cls(cfg, **override_kwargs)


def get_sampler(ds, eval=False, num_gpus=1):
    """ Using state to decide common dataloader kwargs. """
    # TODO workers kwargs
    assert ds.targets, "Please call ds.update_targets([desired_targets])"
    if num_gpus > 1:
        sampler = torch.utils.data.DistributedSampler(
            ds, shuffle=(not eval), drop_last=(not eval))
    elif not eval:
        sampler = torch.utils.data.RandomSampler(ds)
    else:
        sampler = torch.utils.data.SequentialSampler(ds)

    return sampler


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
    FacePosition = namedtuple('FacePosition', 'X_mean X_std cx_mean cx_std cy_mean cy_std')
    FacePosition.__qualname__ = "DeepFashion.FacePosition"

    @configurable()
    def __init__(
        self,
        resolution: int = 256,
        roots: List[str] = None,
        sources: List[str] = None,
        split: str = 'all',
        xflip: bool = False,
        num_items: int = float('inf'),
        resampling: bool = False,  # Whether to resample face position when create masked face
    ):
        assert roots is not None and sources is not None
        self.classes = ["DF_face", "DF_human"]
        self.res = resolution
        self.xflip = xflip
        # all targets: ['face', 'human', 'heatmap', 'masked_face', 'vis_kp', 'lm', 'face_lm', 'quad_mask']
        self.available_targets = ['face', 'human', 'face_lm']
        self.mask_slice = (slice(16, 48), slice(112, 144))
        self.mask_size = (32, 32)
        self.targets = []

        root = Path(roots[0]).expanduser()
        split_map = pickle.load(open(root / 'new_split.pkl', 'rb'))
        self.fileIDs = [ID for IDs in split_map.values() for ID in IDs] if split == 'all' else split_map[split]
        self.fileIDs.sort()

        # statistics
        self.big = DeepFashion.FacePosition(78.08, 11.18, 517.075, 26.655, 131.60, 24.41)
        self.small = DeepFashion.FacePosition(44.56, 3.32, 517.075, 26.655, 100.36, 17.22)
        self.rng = np.random.default_rng()

        self.src = sources[0]
        self.face_dir = root / self.src / 'face'
        self.human_dir = root / f'r{self.res}' / 'fixedface_unalign1.0_0.125'
        self.kp_dir = root / 'kp_heatmaps/keypoints'
        self.dlib_ann = json.load(open(root / 'df_landmarks.json', 'r'))
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

        if 'masked_face' in self.targets:
            assert 'face' in data, "require face targets to make masked_face"
            # 4 channel masked face with face
            dist = self.big if np.random.random() > 0.7 else self.small
            x = self.rng.normal(loc=dist.X_mean, scale=dist.X_std, size=()) * 0.45
            cx = self.rng.normal(loc=dist.cx_mean, scale=dist.cx_std, size=()) * (self.res / 1024)
            cx = np.clip(cx, int(0.125 * self.res), int(0.875 * self.res))
            cy = self.rng.normal(loc=dist.cy_mean, scale=dist.cy_std, size=()) * (self.res / 1024)
            cx = np.clip(cx, 0, int(0.5 * self.res))
            w = h = int(np.abs(x) * 2)
            masked_face = torch.zeros_like(data['face'])
            face = torch.nn.functional.interpolate(data['face'][None, ...], (w, h), mode='bicubic', align_corners=True)[0]
            x1, y1 = int(max(cx - w / 2, 0)), int(max(cy - h / 2, 0))
            x2, y2 = int(min(cx + w / 2, self.res)), int(min(cy + h / 2, self.res))
            crop_face = face[:, (y1 - y2):, :(x2 - x1)] if y1 == 0 else face[:, :(y2 - y1), :(x2 - x1)]
            masked_face[:, y1:y2, x1:x2] = crop_face
            mask = torch.any(masked_face != 0, dim=0)

            data['masked_face'] = torch.cat([masked_face, mask[None, ...]], dim=0)

        if 'heatmap' in self.targets or 'vis_kp' in self.targets:
            kp = pickle.load(open(self.kp_dir / f'{fileID}.pkl', 'rb'))[0][:, (1, 0, 2)]  # [K, (y, x, score)]
            cords = np.where(kp[:, 2:3] > 0.1, kp[:, :2], -np.ones_like(kp[:, :2]))

        if 'heatmap' in self.targets:
            heatmap = cords_to_map(cords, (self.res, self.res), sigma=8)
            data['heatmap'] = self.transform(heatmap, normalize=False)

        if 'vis_kp' in self.targets:
            vis_kp, _ = draw_pose_from_cords(cords.astype(int), (self.res, self.res))
            data['vis_kp'] = self.transform(vis_kp)

        if any(x in self.targets for x in ['lm', 'quad_mask']):
            landmarks = np.array(self.dlib_ann[fileID]['face_landmarks'])
            if self.xflip and self.idx > len(self.fileIDs):
                landmarks[:, 0] = 1. - landmarks[:, 0]

            if 'lm' in self.targets:
                data['lm'] = torch.from_numpy(landmarks.copy())

            landmarks = (landmarks * self.res).astype(int)
            if 'quad_mask' in self.targets:
                quad_mask = ffhq_alignment(landmarks, output_size=self.res, ratio=float(self.src.split('_')[-1]))
                data['quad_mask'] = torch.from_numpy(quad_mask.copy())[None, ...]  # (c, h, w)

        if 'face_lm' in self.targets:
            face_lm = np.array(self.dlib_ann[fileID]['landmarks_unalign1.0'])
            eye_left = np.mean(face_lm[36:42], axis=0, keepdims=True)
            eye_right = np.mean(face_lm[42:48], axis=0, keepdims=True)
            nose = face_lm[30:31]
            mouth_avg = (face_lm[48:49] + face_lm[54:55]) * 0.5
            profile = face_lm[(0, 3, 6, 8, 10, 13, 16), :]
            face_lm = np.concatenate([eye_left, eye_right, nose, mouth_avg, profile], axis=0)
            if self.xflip and self.idx > len(self.fileIDs):
                face_lm[:, 0] = 1. - face_lm[:, 0]

            data['face_lm'] = torch.from_numpy(face_lm.copy())

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
