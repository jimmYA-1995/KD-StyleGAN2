from .networks import Generator, Discriminator


__all__ = ['Generator', 'Discriminator', 'create_model']


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
    ).to(device)

    if eval_only:
        return g.eval()

    d = Discriminator(1, cfg.resolution, img_channels=6).to(device)
    return g, d
