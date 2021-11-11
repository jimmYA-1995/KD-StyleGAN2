import torch
# from dataset import get_sampler


def image_generator(model, batch_size, target_class, ds=None, device='cuda', num_gpus=1):
    assert ds is None

    # Infinite sampling
    while True:
        z = torch.randn([batch_size, model.z_dim], device=device)
        c = torch.nn.functional.one_hot(torch.zeros([z.shape[0]], device=device, dtype=torch.int64), model.num_classes)
        yield model.inference(z, c)


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
    ds = None  # get_dataset(cfg, split='test')
    generate_fn = partial(
        image_generator, g, args.batch_size, ds=ds, device='cuda', num_gpus=1)

    outs = {}
    for c in g.classes:
        img_list = []
        for fake_imgs in generate_fn(c):
            img_list.append(img_list)
        outs[c] = torch.cat(img_list, dim=0)
