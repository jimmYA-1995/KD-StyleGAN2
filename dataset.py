import random
import json
import pickle
from collections import namedtuple
from pathlib import Path
from typing import List, Dict

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


class BaseDataset(data.Dataset):
    def __init__(
        self,
        root: str,
        resolution: int = 256,
        sources: List[str] = None,
        xflip: bool = False,
        split: str = None,
    ):
        root = Path(root).expanduser()
        assert root.exists(), f"Data root does not exists: {root}"
        if split is not None:
            assert split in ['all', 'train', 'val', 'test']

        self.root = root
        self.res = resolution
        self.src = sources
        self.xflip = xflip
        self.num_items = None  # must be defined in inherited class

    @classmethod
    def from_config(cls, cfg):
        return {
            'resolution': cfg.resolution,
            'root': cfg.DATASET.root,
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

    def transform(
        self,
        img: np.ndarray,
        normalize: bool = True,      # Whether to normalize image from [0,255] to [-1, 1]
        channel_first: bool = True,  # transpose color channel to first channel (for pytorch tensor)
        xflip: bool = False,         # horizontal flip
    ) -> torch.Tensor:
        """ normalize, xflip, transform, to Tensor """
        img = img.astype(np.float32)
        if normalize:
            img = (img - 127.5) / 127.5

        if xflip:
            img = img[:, ::-1, :]

        if channel_first:
            img = img.transpose(2, 0, 1)

        return torch.from_numpy(img.copy())

    def __getitem__(self, idx):
        raise NotImplementedError


@register
class DeepFashion(BaseDataset):
    FacePosition = namedtuple('FacePosition', 'X_mean X_std cx_mean cx_std cy_mean cy_std')
    FacePosition.__qualname__ = "DeepFashion.FacePosition"

    @configurable()
    def __init__(
        self,
        root: str,
        resolution: int = 256,
        sources: List[str] = None,
        xflip: bool = False,
        split: str = 'all',
        num_items: int = float('inf'),
        resampling: bool = False,  # Whether to resample face position when create masked face
    ):
        super().__init__(root, resolution, sources, xflip, split)
        assert len(self.src) == 2, "Assume source1 for Face & source2 for target(human)"
        self.classes = ["DF_face", "DF_human"]
        # all targets: ['face', 'human', 'heatmap', 'masked_face', 'vis_kp', 'lm', 'face_lm', 'quad_mask']
        self.available_targets = ['face', 'human', 'face_lm']
        face_ratio = float(sources[1].split('_')[-1])
        face_size = int(resolution * face_ratio)
        self.mask_slice = (slice(16, 16 + face_size), slice((resolution - face_size) // 2, (resolution + face_size) // 2))
        self.mask_size = (face_size, face_size)
        self.targets = []

        split_map = pickle.load(open(self.root / 'split.pkl', 'rb'))
        self.fileIDs = [ID for IDs in split_map.values() for ID in IDs] if split == 'all' else split_map[split]
        self.fileIDs.sort()

        # statistics
        self.big = DeepFashion.FacePosition(78.08, 11.18, 517.075, 26.655, 131.60, 24.41)
        self.small = DeepFashion.FacePosition(44.56, 3.32, 517.075, 26.655, 100.36, 17.22)
        self.rng = np.random.default_rng()

        self.face_dir = self.root / self.src[0] / 'face'
        self.human_dir = self.root / f'r{self.res}' / sources[1]
        self.kp_dir = self.root / 'kp_heatmaps/keypoints'
        self.dlib_ann = json.load(open(self.root / 'df_landmarks.json', 'r'))
        assert self.face_dir.exists() and self.human_dir.exists() and self.kp_dir.exists()
        assert set(self.fileIDs) <= set(p.stem for p in self.face_dir.glob('*.png'))
        assert set(self.fileIDs) <= set(p.stem for p in self.human_dir.glob('*.png'))
        assert set(self.fileIDs) <= set(p.stem for p in self.kp_dir.glob('*.pkl'))

        total = len(self.fileIDs) * 2 if xflip else len(self.fileIDs)
        self.num_items = min(total, num_items)

    def update_targets(self, targets: List[str]) -> None:
        if not all([t in self.available_targets for t in targets]):
            raise ValueError(f"Some of desire targets is not available. "
                             f"Available targets for {self.__class__.__name__} dataset: {self.available_targets}")
        self.targets = targets

    def __getitem__(self, idx) -> Dict[torch.Tensor]:
        data = {}

        try:
            xflip = self.xflip and idx >= len(self.fileIDs)
            fileID = self.fileIDs[idx % len(self.fileIDs)]
        except IndexError as e:
            raise RuntimeError("Invalid dataset index: {idx} (xflip={xflip})") from e

        if 'face' in self.targets:
            img = io.imread(self.face_dir / f'{fileID}.png')
            data['face'] = self.transform(img, xflip=xflip)

        if 'human' in self.targets:
            img = io.imread(self.human_dir / f'{fileID}.png')
            data['human'] = self.transform(img, xflip=xflip)

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
            data['heatmap'] = self.transform(heatmap, normalize=False, xflip=xflip)

        if 'vis_kp' in self.targets:
            vis_kp, _ = draw_pose_from_cords(cords.astype(int), (self.res, self.res))
            data['vis_kp'] = self.transform(vis_kp, xflip=xflip)

        if any(x in self.targets for x in ['lm', 'quad_mask']):
            landmarks = np.array(self.dlib_ann[fileID]['face_landmarks'])
            if xflip:
                landmarks[:, 0] = 1. - landmarks[:, 0]

            if 'lm' in self.targets:
                data['lm'] = torch.from_numpy(landmarks.copy())

            landmarks = (landmarks * self.res).astype(int)
            if 'quad_mask' in self.targets:
                quad_mask = ffhq_alignment(landmarks, output_size=self.res, ratio=float(self.src[0].split('_')[-1]))
                data['quad_mask'] = torch.from_numpy(quad_mask.copy())[None, ...]  # (c, h, w)

        if 'face_lm' in self.targets:
            face_lm = np.array(self.dlib_ann[fileID]['landmarks_unalign1.0'])
            eye_left = np.mean(face_lm[36:42], axis=0, keepdims=True)
            eye_right = np.mean(face_lm[42:48], axis=0, keepdims=True)
            nose = face_lm[30:31]
            mouth_avg = (face_lm[48:49] + face_lm[54:55]) * 0.5
            profile = face_lm[(0, 3, 6, 8, 10, 13, 16), :]
            face_lm = np.concatenate([eye_left, eye_right, nose, mouth_avg, profile], axis=0)
            if xflip:
                face_lm[:, 0] = 1. - face_lm[:, 0]

            data['face_lm'] = torch.from_numpy(face_lm.copy())

        return data


@register
class FFHQ256(BaseDataset):
    @configurable
    def __init__(
        self,
        root: str,
        resolution: int = 256,
        sources: List[str] = None,
        xflip: bool = False,
        split: str = None,
        num_items: int = None
    ):
        super().__init__(root, resolution=resolution, sources=sources, xflip=xflip, split=split)
        assert split is None
        self.ref_dir = self.root / self.src[0]
        self.target_dir = self.root / self.src[1]
        self.fileIDs = sorted(p.relative_to(self.ref_dir) for p in self.ref_dir.glob('**/*.png'))
        assert set(self.fileIDs) <= set([p.relative_to(self.target_dir) for p in self.target_dir.glob('**/*.png')])

        self.num_items = len(self.fileIDs)
        if num_items is not None:
            assert num_items > 0
            if num_items > len(self.fileIDs):
                print(f"required #items is bigger than all dataset. Using whole dataset({len(self.fileIDs)} items)")

            self.num_items = min(len(self.fileIDs), num_items)

        if self.xflip:
            self.num_items *= 2

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        assert idx < len(self), f"{idx} v.s {len(self)}"
        try:
            xflip = self.xflip and idx > len(self.fileIDs)
            fileID = self.fileIDs[idx % len(self.fileIDs)]
        except IndexError as e:
            raise RuntimeError("Invalid dataset index: {idx} (xflip={xflip})") from e

        data = {}
        data['ref'] = self.transform(io.imread(self.ref_dir / fileID), xflip=xflip)
        data['target'] = self.transform(io.imread(self.target_dir / fileID), xflip=xflip)

        return data


if __name__ == "__main__":
    # Test Code
    import torch
    import numpy as np
    from PIL import Image
    from config import get_cfg_defaults

    def display(t):
        if t.is_cuda:
            t = t.cpu()
        t = np.clip(t.numpy() * 127.5 + 128, 0, 255).astype(np.uint8)
        if t.ndim == 4:
            b, c, h, w = t.shape
            t = t.transpose(2, 0, 3, 1).reshape(h, b * w, c)
        elif t.ndim == 3:
            t = t.transpose(1, 2, 0)
        else:
            raise RuntimeError(f"Unknown shape: {t.shape}")
        Image.fromarray(t).show()

    cfg = get_cfg_defaults()
    cfg.merge_from_file('exp.yml')
    ds = get_dataset(cfg, split='train', xflip=False)

    print(cfg)
    print(len(ds))
    # print(ds.available_targets)
    # ds.update_targets(ds.available_targets)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=2
    )
    batch = next(iter(loader))
    print("Test Done")
