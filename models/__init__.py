from .networks import Generator, Discriminator


__all__ = ['Generator', 'Discriminator', 'create_model']


def create_model(cfg, device=None, eval_only=False):
    g = Generator(
        cfg.MODEL.z_dim,
        cfg.MODEL.w_dim,
        cfg.classes,
        cfg.resolution,
        mapping_kwargs=cfg.MODEL.MAPPING,
        synthesis_kwargs=cfg.MODEL.SYNTHESIS,
    ).to(device)

    if eval_only:
        return g.eval()

    d = Discriminator(1, cfg.resolution, img_channels=6).to(device)  # fix #class to 1
    return g, d
