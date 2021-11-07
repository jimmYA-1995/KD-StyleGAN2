from .networks import *


__all__ = ['Generator', 'Discriminator', 'create_model']


def create_model(cfg, num_clases, device=None, eval_only=False):
    g = Generator(
        cfg.MODEL.z_dim,
        cfg.MODEL.w_dim,
        num_clases,
        cfg.resolution,
        mapping_kwargs=cfg.MODEL.MAPPING,
        synthesis_kwargs=dict(cfg.MODEL.SYNTHESIS),
    ).to(device)

    if eval_only:
        return g.eval()

    d = Discriminator(cfg.resolution, **cfg.MODEL.DISCRIMINATOR).to(device)
    return g, d


def map_keys(k):
    ks = k.split('.')
    if k.startswith('mapping'):
        return f"mapping.face.fc.{ks[1][-1]}.{ks[-1]}"

    # synthesis
    new_key = "synthesis.face."
    if ks[-1] == 'const':
        return new_key + 'const'

    conv_map = ['Up', '']
    new_key += f"{ks[1]}_"
    if ks[2] == 'resample_filter':
        new_key += "trgb.resample_filter"
        return new_key

    if ks[2] == 'torgb':
        new_key += "trgb."
    else:
        try:
            new_key += f"{ks[2][:-1]}{conv_map[int(ks[2][-1])]}."
        except ValueError:
            print(ks)
            print(k)

    # 我的多 resample_filter & noise_const
    new_key += '.'.join(ks[3:])
    return new_key
