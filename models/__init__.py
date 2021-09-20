from .networks import Generator, Discriminator


__all__ = ['Generator', 'Discriminator', 'create_model', 'resume_teachNet_from_NV_weights']


def create_model(cfg, device=None, eval_only=False):
    g = Generator(
        cfg.MODEL.z_dim,
        cfg.MODEL.w_dim,
        cfg.classes,
        cfg.resolution,
        mode=cfg.MODEL.mode,
        freeze_teacher=cfg.MODEL.freeze_teacher,
        attn_res=cfg.MODEL.attn_res,
        mapping_kwargs=cfg.MODEL.MAPPING,
        synthesis_kwargs=dict(cfg.MODEL.SYNTHESIS),
    ).to(device).requires_grad_(False)

    if eval_only:
        return g.eval()

    d = Discriminator(1, cfg.resolution, img_channels=6).to(device).requires_grad_(False)
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


def resume_teacherNet_from_NV_weights(g, c, verbose=False):
    cnt = 0
    g_keys = set([x[0] for x in g.named_parameters()] + [x[0] for x in g.named_buffers()])
    for old_k, value in c.items():
        if old_k in ['mapping.w_avg']:
            continue

        new_k = map_keys(old_k)
        obj = g
        for attr in new_k.split('.'):
            obj = getattr(obj, attr)
            assert obj is not None, f"attr not found: {obj} has no {attr}"

        assert obj.shape == value.shape, f"shape not match: {new_k}({obj.shape}) v.s {old_k}({value.shape})"
        obj.copy_(value)
        cnt += 1
        g_keys.remove(new_k)
        if verbose:
            print(f"{old_k} -> {new_k}")

    if verbose:
        print(f"load: {cnt}/{len(c.items())};")
        print("missed:")
        print([x for x in g_keys if 'human' not in x])
