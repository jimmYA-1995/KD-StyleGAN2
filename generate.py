from pathlib import Path
import torch
import numpy as np
from PIL import Image
# from dataset import get_sampler


def image_generator(model, batch_size, target_class, ds=None, device='cuda', num_gpus=1):
    assert ds is None
    # ds.update_targets(['face', 'masked_face'])
    # loader = torch.utils.data.DataLoader(
    #     ds,
    #     batch_size=batch_size,
    #     sampler=get_sampler(ds, eval=True, num_gpus=num_gpus),
    #     pin_memory=False,
    #     num_workers=3,
    #     worker_init_fn=ds.__class__.worker_init_fn
    # )

    # Infinite sampling
    epoch = 0
    while True:
        # if num_gpus > 1:
        #     loader.sampler.set_epoch(epoch)

        # for batch in loader:
        #     data = {k: v.to(device) for k, v in batch.items()}
        z = torch.randn([batch_size, model.z_dim], device=device)
        yield model.inference(z, None, target_class, noise_mode='const')
        # epoch += 1


if __name__ == '__main__':
    import argparse
    from functools import partial
    from config import get_cfg_defaults
    from models import create_model
    from dataset import get_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="path to the configuration file")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for inference")
    parser.add_argument("--outdir", type=str, help="output directory")
    parser.add_argument("--total", type=int, help="Num of image to generate")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    g = create_model(cfg, device='cuda', eval_only=True)
    if cfg.TRAIN.CKPT.path:
        g.load_state_dict(torch.load(cfg.TRAIN.CKPT.path)['g_ema'])

    ds = None  # get_dataset(cfg, split='test')
    generate_fn = partial(
        image_generator, g, args.batch_size, ds=ds, device='cuda', num_gpus=1)

    outdir = Path(args.outdir)
    if outdir.exists():
        pass
    else:
        outdir.mkdir(parents=True)
    serial_no = 0

    for c in g.classes[1:]:
        count = 0
        for fake_imgs in generate_fn(c):
            if fake_imgs.is_cuda:
                fake_imgs = fake_imgs.cpu()

            imgs = fake_imgs.numpy().transpose(0, 2, 3, 1)
            imgs = np.clip(imgs * 127.5 + 127.5, 0, 255).astype(np.uint8)
            for img in imgs:
                Image.fromarray(img).save(outdir / f'fake-{c}-{serial_no :06d}.png')
                serial_no += 1
                count += 1
                if count == args.total:
                    break
            else:
                if count >= args.total:
                    break

        