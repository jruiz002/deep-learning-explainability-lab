"""Saliency-map utilities: normalisation, upsampling, masks, IoU, box mass."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import config


def relu_minmax(m: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """ReLU followed by min-max normalisation to [0, 1]. A constant map becomes all zeros."""
    m = F.relu(m)
    lo, hi = m.min(), m.max()
    if (hi - lo) < eps:
        return torch.zeros_like(m)
    return (m - lo) / (hi - lo)


def minmax(m: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    lo, hi = m.min(), m.max()
    if (hi - lo) < eps:
        return torch.zeros_like(m)
    return (m - lo) / (hi - lo)


def upsample(m: torch.Tensor, size: int = config.CROP_SIZE) -> torch.Tensor:
    """Bilinear upsampling of an (H, W) map to (size, size)."""
    out = F.interpolate(m[None, None], size=(size, size), mode="bilinear", align_corners=False)
    return out[0, 0]


def top_fraction_mask(m: torch.Tensor, fraction: float = config.IOU_TOP_FRACTION) -> torch.Tensor:
    """Boolean mask selecting exactly the ``fraction`` of pixels with the highest value."""
    flat = m.flatten()
    k = max(1, int(round(fraction * flat.numel())))
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[flat.topk(k).indices] = True
    return mask.view_as(m)


def iou(a: torch.Tensor, b: torch.Tensor) -> float:
    """Intersection over union of two boolean masks."""
    union = (a | b).sum().item()
    return float((a & b).sum().item() / union) if union else 0.0


def box_mask(box: tuple[float, float, float, float], size: int = config.CROP_SIZE) -> torch.Tensor:
    """Boolean (size, size) mask of the pixels inside ``box`` = (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    c0, r0 = int(math.floor(x0)), int(math.floor(y0))
    c1, r1 = int(math.ceil(x1)), int(math.ceil(y1))
    mask = torch.zeros(size, size, dtype=torch.bool)
    mask[r0:r1, c0:c1] = True
    return mask


def mass_in_box(m: torch.Tensor, box: tuple[float, float, float, float]) -> float:
    """Fraction of the (non-negative) map mass that lies inside ``box``."""
    total = m.sum().item()
    if total == 0:
        return 0.0
    return float(m[box_mask(box, m.shape[-1])].sum().item() / total)
