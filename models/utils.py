import torch
import torch.nn as nn


REGISTRY = {}


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


def build_pose_encoder(bottom_res: int, out_channels: int, **pose_encoder_kwargs) -> nn.Module:

    pose_encoder_name = pose_encoder_kwargs.pop('name')
    return REGISTRY['pose'].get(pose_encoder_name)(bottom_res, out_channels, **pose_encoder_kwargs)


def register(key):
    def wrapper(cls):
        registry = REGISTRY.setdefault(key, {})
        registry[cls.__name__] = cls

        return cls
    return wrapper


def ema(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
