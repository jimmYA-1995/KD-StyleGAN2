import math
from numpy.core.fromnumeric import nonzero
import torch
from torch.nn import functional as F
from torch import autograd


__all__ = ['MaskedRecLoss', 'logistic_loss', 'r1_loss', 'nonsaturating_loss', 'path_regularize']


def gaussian_kernel(size, sigma=2., dim=2, channels=3):
    kernel_size = 2 * size + 1
    kernel_size = [kernel_size] * dim
    sigma = [sigma] * dim
    kernel = 1
    meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])

    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
    return kernel


class MaskedRecLoss:
    def __init__(self, dist='L1', mask=None, ksize=3, sigma=2., dim=2, num_channels=3, device='cuda'):
        assert dist in ['L1', 'L2']
        assert mask in [None, 'binary', 'gaussian']
        self.dist = dist
        self.mask = mask
        self.gaussian_weight = None
        if mask == 'gaussian':
            self.gaussian_weight = gaussian_kernel(ksize, sigma, dim, num_channels).to(device)
            self.pad = ksize

    def __call__(self, real_img, fake_img, mask=None):
        assert (mask is None) == (self.mask is None)
        diff = real_img - fake_img
        if self.dist == 'L1':
            loss = torch.abs(diff)
        elif self.dist == 'L2':
            loss = diff ** 2

        if mask is None:
            return loss.mean()

        if self.mask == 'gaussian':
            with torch.no_grad():
                blur_mask = F.conv2d(mask, self.gaussian_weight, padding=self.pad)
                final_mask = blur_mask / blur_mask.max() * mask
        else:
            final_mask = mask

        loss = (loss * final_mask).sum(dim=[1, 2, 3]) / mask.sum(dim=[1, 2, 3])
        return loss.mean()


def r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
    grad, = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths.mean().detach()
