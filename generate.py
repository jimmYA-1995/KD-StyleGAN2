import torch
from dataset import get_sampler


def image_generator(model, batch_size, target_class, ds=None, device='cuda', num_gpus=1):
    assert ds is not None, "current methods needs ds for conditional input"
    ds.update_targets(['face', 'masked_face'])
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        sampler=get_sampler(ds, eval=True, num_gpus=num_gpus),
        pin_memory=False,
        num_workers=3,
        worker_init_fn=ds.__class__.worker_init_fn
    )

    # Infinite sampling
    epoch = 0
    while True:
        if num_gpus > 1:
            loader.sampler.set_epoch(epoch)

        for batch in loader:
            data = {k: v.to(device) for k, v in batch.items()}
            z = torch.randn([data['face'].shape[0], model.z_dim], device=device)
            fake_imgs = model(z, face=data['face'], content=data['masked_face'])
            yield fake_imgs[target_class]
        epoch += 1


if __name__ == '__main__':
    import argparse
    from functools import partial
    from config import get_cfg_defaults
    from models import create_model
    from dataset import get_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="path to the configuration file")
    parser.add_argument("--batch_size", type=str, help="batch size for inference")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    g = create_model(cfg, device='cuda', eval_only=True)
    ds = get_dataset(cfg, split='test')
    generate_fn = partial(
        image_generator, g, args.batch_size, ds=ds, device='cuda', num_gpus=1)

    outs = {}
    for c in g.classes:
        img_list = []
        for fake_imgs in generate_fn(c):
            img_list.append(img_list)
        outs[c] = torch.cat(img_list, dim=0)
