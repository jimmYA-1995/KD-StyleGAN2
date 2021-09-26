import atexit
import argparse
import functools
import inspect
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy.ndimage
from PIL import Image
from skimage.draw import circle, line_aa


LIMB_SEQ = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [5, 11], [6, 12], [11, 13], [12, 14], [13, 15], [14, 16]]
COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
MISSING_VALUE = -1


def get_cmdline_args():
    parser = argparse.ArgumentParser(prog='torch.distributed.launch')
    parser.add_argument('--cfg', default=None, metavar='FILE', help="path to the config file")
    parser.add_argument('--local_rank', type=int, default=0, metavar='INT', help="Automatically given by %(prog)s in multi-gpu training")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--nobench', default=True, action='store_false', dest='cudnn_benchmark', help="disable cuDNN benchmarking")
    parser.add_argument('--amp', default=False, action='store_true', help="whether to use auto mixed precision training")
    parser.add_argument('--no-wandb', default=True, action='store_false', dest='use_wandb', help="disable wandb logging")
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def setup_logger(outdir, local_rank=0, debug=False):
    """
    Initialize loggers. Only master can log info. to stdout and the messages
    from other workers will displayed in logfile & debug mode.
    """
    if not isinstance(outdir, Path):
        outdir = Path(outdir)
    assert outdir.exists() and outdir.is_dir()

    logger_name = f"GPU{local_rank}"
    loglevel = 'DEBUG' if debug else ('INFO' if local_rank == 0 else 'WARN')

    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logger.setLevel(getattr(logging, loglevel))
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, loglevel))
    ch.setFormatter(ColorfulFormatter(logger_name))
    logger.addHandler(ch)

    # disable PIL dubug mode
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)

    # https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes
    if outdir:
        filename = outdir / f'log_{logger_name}.txt'
        plain_formatter = "%(levelname)-8s - %(asctime)-15s - %(message)s (%(filename)s:%(lineno)d)"
        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(getattr(logging, loglevel))
        fh.setFormatter(logging.Formatter(plain_formatter))
        logger.addHandler(fh)

    return logger


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@lru_cache(maxsize=None)
def _cached_log_stream(filename):
    io = open(filename, "a", buffering=-1)
    atexit.register(io.close)
    return io


def master_only(func_or_method):
    """
    As using decorator on class method, it will be recognized as function
    instead of method by inspect module. Thus, I just use parameter name to
    decide whether it is class method or not.
    """
    def wrapper(*args, **kwargs):
        signature = inspect.signature(func_or_method)
        if list(signature.parameters.keys())[0] == 'self':
            obj = args[0]
            assert hasattr(obj, 'local_rank')
            local_rank = obj.local_rank
        else:
            assert list(signature.parameters.keys())[0] == 'local_rank', \
                "assume local_rank is the first argumnet of function"
            local_rank = args[0]

        if local_rank != 0:
            return

        return func_or_method(*args, **kwargs)
    return wrapper


# https://github.com/RenYurui/Global-Flow-Local-Attention/blob/3afa8fe9e0c1ed148eff6720a1345c4a428ec76b/util/pose_utils.py#L52
def cords_to_map(cords, img_size, old_size=None, affine_matrix=None, sigma=6):
    old_size = img_size if old_size is None else old_size
    cords = cords.astype(float)
    result = np.zeros(img_size + cords.shape[0:1], dtype='float32')
    for i, point in enumerate(cords):
        if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
            continue
        point[0] = point[0] / old_size[0] * img_size[0]
        point[1] = point[1] / old_size[1] * img_size[1]
        if affine_matrix is not None:
            point_ = np.dot(affine_matrix, np.matrix([point[1], point[0], 1]).reshape(3, 1))
            point_0 = int(point_[1])
            point_1 = int(point_[0])
        else:
            point_0 = int(point[0])
            point_1 = int(point[1])
        xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
        result[..., i] = np.exp(-((yy - point_0) ** 2 + (xx - point_1) ** 2) / (2 * sigma ** 2))
    return result


def draw_pose_from_cords(pose_joints, img_size, radius=2, draw_joints=True):
    colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)
    mask = np.zeros(shape=img_size, dtype=bool)

    if draw_joints:
        for f, t in LIMB_SEQ:
            from_missing = pose_joints[f][0] == MISSING_VALUE or pose_joints[f][1] == MISSING_VALUE
            to_missing = pose_joints[t][0] == MISSING_VALUE or pose_joints[t][1] == MISSING_VALUE
            if from_missing or to_missing:
                continue
            yy, xx, val = line_aa(pose_joints[f][0], pose_joints[f][1], pose_joints[t][0], pose_joints[t][1])
            colors[yy, xx] = np.expand_dims(val, 1) * 255
            mask[yy, xx] = True

    for i, joint in enumerate(pose_joints):
        if pose_joints[i][0] == MISSING_VALUE or pose_joints[i][1] == MISSING_VALUE:
            continue
        yy, xx = circle(joint[0], joint[1], radius=radius, shape=img_size)
        colors[yy, xx] = COLORS[i]
        mask[yy, xx] = True

    return colors, mask


def ffhq_alignment(
    lm,
    output_size=1024,
    ratio=2.0,
):

    # Parse landmarks.
    # pylint: disable=unused-variable
    lm_chin = lm[0:17]  # left-right
    lm_eyebrow_left = lm[17:22]  # left-right
    lm_eyebrow_right = lm[22:27]  # left-right
    lm_nose = lm[27:31]  # top-down
    lm_nostrils = lm[31:36]  # top-down
    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise
    lm_mouth_outer = lm[48:60]  # left-clockwise
    lm_mouth_inner = lm[60:68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)

    x *= max(np.hypot(*eye_to_eye) * ratio, np.hypot(*eye_to_mouth) * ratio * 0.9)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y]).astype(int)

    # 
    mask = np.zeros((output_size, output_size), np.float32)
    cv2.fillConvexPoly(mask, quad, 1)

    return mask


class ColorfulFormatter(logging.Formatter):
    """ Logging Formatter  to add colors and count warning / errors"""
    def __init__(self, name):
        gray = "\x1b[38;21m"
        green = "\x1b[32;21m"
        yellow = "\x1b[33;21m"
        blink_red = "\x1b[5m\x1b[31;21m"
        blink_bold_red = "\x1b[5m\x1b[31;1m"
        reset = "\x1b[0m"
        format = f"<{name}> %(message)s (%(filename)s:%(lineno)d)"

        self.FORMATS = {
            'DEBUG': gray + format + reset,
            'INFO': green + format + reset,
            'WARNING': yellow + format + reset,
            'ERROR': blink_red + format + reset,
            'CRITICAL': blink_bold_red + format + reset,
        }
        super(ColorfulFormatter, self).__init__()

    def format(self, record):
        log_fmt = self.FORMATS[record.levelname]
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class UserError(Exception):
    pass


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]
