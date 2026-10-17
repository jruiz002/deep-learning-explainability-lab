"""Deletion and insertion metrics for a saliency map (Task 4.1), written with PyTorch and SciPy.

Pixels of the 224x224 crop are ordered by the map, from highest to lowest. At step k = 1..P the first k/P of them are
copied from an ``end`` image into a ``start`` image and the class probability c_k is recorded; AUC = (1/P) sum_k c_k.

- deletion:  start = original image, end = reference (mid gray, or a blurred copy)  -> a good map makes c_k fall fast;
- insertion: start = heavily blurred image,  end = original image                    -> a good map makes c_k rise fast.

All images are built in [0, 1] pixel space and normalised with the ImageNet statistics right before the model.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch import nn
from torchvision.transforms.functional import to_tensor

from . import config

_MEAN = torch.tensor(config.IMAGENET_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(config.IMAGENET_STD).view(1, 3, 1, 1)


def to_unit(crop: Image.Image) -> torch.Tensor:
    """(3, H, W) float tensor in [0, 1] of the 224x224 crop (before the ImageNet normalisation)."""
    return to_tensor(crop)


def normalize(x01: torch.Tensor) -> torch.Tensor:
    """ImageNet normalisation of a (B, 3, H, W) batch in [0, 1]."""
    return (x01 - _MEAN) / _STD


def gaussian_blur(x01: torch.Tensor, sigma: float = config.BLUR_SIGMA) -> torch.Tensor:
    """Per-channel Gaussian blur with standard deviation ``sigma`` pixels (reflected borders)."""
    arr = x01.numpy()
    out = np.stack([gaussian_filter(c, sigma=sigma, mode="reflect") for c in arr])
    return torch.from_numpy(out).to(x01.dtype)


def gray_like(x01: torch.Tensor, level: float = config.GRAY_LEVEL) -> torch.Tensor:
    return torch.full_like(x01, level)


def order_from_map(saliency: torch.Tensor) -> torch.Tensor:
    """Flat pixel indices sorted by decreasing map value (stable: ties keep raster order)."""
    return torch.argsort(saliency.flatten(), descending=True, stable=True)


def random_order(n_pixels: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n_pixels, generator=g)


def pixels_at_step(k: int, steps: int, n_pixels: int) -> int:
    return int(round(k / steps * n_pixels))


def perturbation_images(start: torch.Tensor, end: torch.Tensor, order: torch.Tensor, steps: int) -> torch.Tensor:
    """(steps + 1, 3, H, W): image k has the first k/steps of ``order`` copied from ``end`` into ``start``."""
    _, h, w = start.shape
    n = h * w
    s, e = start.reshape(3, n), end.reshape(3, n)
    frames = []
    for k in range(steps + 1):
        img = s.clone()
        idx = order[: pixels_at_step(k, steps, n)]
        img[:, idx] = e[:, idx]
        frames.append(img.view(3, h, w))
    return torch.stack(frames)


@torch.no_grad()
def class_curve(model: nn.Module, frames: torch.Tensor, class_idx: int) -> np.ndarray:
    """Softmax probability of ``class_idx`` for every frame (k = 0..P)."""
    return model(normalize(frames)).softmax(dim=1)[:, class_idx].numpy().astype(float)


def curve(model: nn.Module, start, end, order, class_idx: int, steps: int = config.PERTURB_STEPS) -> np.ndarray:
    return class_curve(model, perturbation_images(start, end, order, steps), class_idx)


def auc(c: np.ndarray) -> float:
    """AUC = (1/P) sum_{k=1}^{P} c_k (c[0] is the unperturbed start image and is not counted)."""
    return float(np.mean(c[1:]))


def cells_touched(order: torch.Tensor, k: int, steps: int, size: int = config.CROP_SIZE,
                  cell: int = config.CROP_SIZE // config.FEATURE_SIZE) -> int:
    """How many of the 7x7 cells of the original map (32x32 pixels each) contain a pixel changed by step k."""
    idx = order[: pixels_at_step(k, steps, size * size)]
    rows, cols = idx // size // cell, idx % size // cell
    return int(torch.unique(rows * config.FEATURE_SIZE + cols).numel())
