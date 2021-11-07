import random
import json
import pickle
from collections import namedtuple
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import torch
import skimage.io as io
from PIL import Image
from torch.utils import data
from torchvision import transforms

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
    if num_gpus > 1:
        sampler = torch.utils.data.DistributedSampler(
            ds, shuffle=(not eval), drop_last=(not eval))
    elif not eval:
        sampler = torch.utils.data.RandomSampler(ds)
    else:
        sampler = torch.utils.data.SequentialSampler(ds)

    return sampler


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
        self.res = resolution
        self.xflip = xflip
        # all targets: ['face', 'human', 'heatmap', 'masked_face', 'vis_kp', 'lm', 'face_lm', 'quad_mask']
        self.available_targets = ['face', 'human', 'face_lm']
        face_ratio = float(sources[1].split('_')[-1])
        face_size = int(resolution * face_ratio)
        self.mask_slice = (slice(16, 16 + face_size), slice((resolution - face_size) // 2, (resolution + face_size) // 2))
        self.mask_size = (face_size, face_size)
        self.targets = []

        root = Path(roots[0]).expanduser()
        split_map = pickle.load(open(root / 'split.pkl', 'rb'))
        self.fileIDs = [ID for IDs in split_map.values() for ID in IDs] if split == 'all' else split_map[split]
        self.fileIDs.sort()

        # statistics
        self.big = DeepFashion.FacePosition(78.08, 11.18, 517.075, 26.655, 131.60, 24.41)
        self.small = DeepFashion.FacePosition(44.56, 3.32, 517.075, 26.655, 100.36, 17.22)
        self.rng = np.random.default_rng()

        self.src = sources[0]
        self.face_dir = root / self.src / 'face'
        self.human_dir = root / f'r{self.res}' / sources[1]
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


@register
class DeepFashion_TS(data.Dataset):
    """ DeepFashion dataset with discrete condition for translation & scaling
        Target is 1/8 fixed face position:
        [x1, y1, x2, y2] = [(0.5 - 0.125) / 2, 0.125 * (1 - 0.5), (0.5 + 0.125) / 2, 0.125 * (1 + 0.5)]
        the label is: 0(face), 1(target), >1, >2, <1, <2, ^1, v1, v2, x1.5, x2.0
        total 10 classes.
        class2 to class9 will translate (unit: 32 pixels) & scale
    """
    @configurable()
    def __init__(
        self,
        resolution: int = 256,
        roots: List[str] = None,
        sources: List[str] = None,
        split: str = 'all',
        xflip: bool = False,
        num_items: int = float('inf'),
        target_ratio: float = 0.125
    ):
        assert roots is not None and sources is not None
        assert len(roots) == 1, "Only support 1 data root directory. List is just for compatibility"
        assert len(sources) == 2, "assume source 1 for ref(face), 2 for target(human)"
        try:
            face_ratio = float(sources[1].split('_')[-1])
            assert target_ratio == face_ratio
        except ValueError:
            print("Warning: Cannot infer target ratio via pasing folder name."
                  "Please make sure the target ratio is correct")

        self.res = resolution
        self.xflip = xflip
        self.target_ratio = target_ratio
        self.affineMs = None
        self.define_transformation()
        self.num_classes = 2 + len(self.affineMs)
        self.available_targets = ["ref", "target", "trans"]
        self.targets = []

        root = Path(roots[0]).expanduser()
        split_map = pickle.load(open(root / 'split.pkl', 'rb'))
        self.fileIDs = [ID for IDs in split_map.values() for ID in IDs] if split == 'all' else split_map[split]
        self.fileIDs.sort()

        self.ref_dir = root / sources[0] / 'face'
        self.target_dir = root / f'r{self.res}' / sources[1]
        assert all(dir for dir in [self.ref_dir, self.target_dir])
        assert all(set(self.fileIDs) <= set(p.stem for p in dir.glob('*.png'))
                   for dir in [self.ref_dir, self.target_dir])

        total = len(self.fileIDs) * 2 if xflip else len(self.fileIDs)
        self._num_items = min(total, num_items)

    @classmethod
    def from_config(cls, cfg):
        return {
            'resolution': cfg.resolution,
            'roots': cfg.DATASET.roots,
            'sources': cfg.DATASET.sources,
            'xflip': cfg.DATASET.xflip
        }

    def __len__(self):
        return self._num_items

    def update_targets(self, targets: List[str]) -> None:
        if not all([t in self.available_targets for t in targets]):
            raise ValueError(f"Some of desire targets is not available. "
                             f"Available targets for {self.__class__.__name__} dataset: {self.available_targets}")
        self.targets = targets

    @classmethod
    def worker_init_fn(cls, worker_id):
        """ For reproducibility & randomness in multi-worker mode """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)

    def define_transformation(self):
        """ Define mapping from label index to transformation. Offset by face and target class.
            0(face), 1(target), >1, >2, <1, <2, ^1, v1, v2, x1.5, x2.0
        """
        face_size = int(self.res * self.target_ratio)
        self.target_box = [(self.res - face_size) // 2, int(face_size * (1 - 0.5)), (self.res + face_size) // 2, int(face_size * (1 + 0.5))]
        u = face_size // 2  # translation unit
        self.affineMs = [
            np.array([[1, 0, u], [0, 1, 0]], dtype=np.float32),
            np.array([[1, 0, u * 2], [0, 1, 0]], dtype=np.float32),
            np.array([[1, 0, -u], [0, 1, 0]], dtype=np.float32),
            np.array([[1, 0, -(u * 2)], [0, 1, 0]], dtype=np.float32),
            np.array([[1, 0, 0], [0, 1, -u]], dtype=np.float32),
            np.array([[1, 0, 0], [0, 1, u]], dtype=np.float32),
            np.array([[1, 0, 0], [0, 1, u * 2]], dtype=np.float32),
            cv2.getRotationMatrix2D((self.res // 2, face_size), 0, 1.5),
            cv2.getRotationMatrix2D((self.res // 2, face_size), 0, 2.0)
        ]

        # pre-calculate face box after translation or scaling
        x1, y1, x2, y2 = self.target_box
        query_points = np.array([[x1, y1, 1], [x2, y2, 1], [0, 0, 1], [self.res - 1, self.res - 1, 1]], dtype=np.float32).T[None, ...]
        homogenous_vector = np.array([[0, 0, 1]], dtype=np.float32)
        square_affineMs = [np.concatenate([M, homogenous_vector.copy()]) for M in self.affineMs]
        batch_affineMs = np.stack(square_affineMs, axis=0)
        affined_points = (batch_affineMs @ query_points).astype(int)[:, :2, :]
        self.facebox = affined_points[..., :2].transpose(0, 2, 1).reshape(-1, 4)  # [N, 4] for (x1', y1', x2', y2')

        boundary = np.array([[0, 0],
                             [self.res, self.res]], dtype=np.int32)[None, ...]
        transbox = affined_points[..., 2:].transpose(0, 2, 1)
        transbox[:, 0] = np.maximum(boundary[:, 0], transbox[:, 0])
        transbox[:, 1] = np.minimum(boundary[:, 1], transbox[:, 1])
        self.transbox = transbox.reshape(-1, 4)

        # inverse query
        query_points = np.array([[0, 0, 1], [self.res - 1, self.res - 1, 1]], dtype=np.float32).T[None, ...]
        batch_affineMs_inv = np.stack([np.linalg.inv(M) for M in square_affineMs], axis=0)
        targetbox = (batch_affineMs_inv @ query_points).astype(int)[:, :2, :].transpose(0, 2, 1)
        targetbox[:, 0] = np.maximum(boundary[:, 0], targetbox[:, 0])
        targetbox[:, 1] = np.minimum(boundary[:, 1], targetbox[:, 1])
        self.targetbox = targetbox.reshape(-1, 4)

    def transform(self, img: np.ndarray, xflip: bool, affine=None, channel_first=True) -> torch.Tensor:
        """ normalize, xflip, transform, to Tensor """
        img = (img.copy().astype(np.float32) - 127.5) / 127.5
        if xflip:
            img = img[:, ::-1, :]

        if affine is not None:
            img = cv2.warpAffine(img, affine, (self.res, self.res), borderValue=(1, 1, 1))

        if channel_first:
            img = img.transpose(2, 0, 1)

        return torch.from_numpy(img.copy())

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        data = {}
        fileID = self.fileIDs[idx % len(self.fileIDs)]
        xflip = self.xflip and idx > len(self.fileIDs)
        if 'ref' in self.targets:
            data['ref'] = self.transform(io.imread(self.ref_dir / f'{fileID}.png'), xflip=xflip)

        trans_idx = np.random.randint(len(self.affineMs))
        label = trans_idx + 2  # offset by ref_class and target class

        if any(x in self.targets for x in ['target', 'trans']):
            target = io.imread(self.target_dir / f'{fileID}.png')
            if 'target' in self.targets:
                data['target'] = self.transform(target, xflip=xflip)

            if 'trans' in self.targets:
                data['trans'] = self.transform(target, xflip=xflip, affine=self.affineMs[trans_idx])

        return data, label
