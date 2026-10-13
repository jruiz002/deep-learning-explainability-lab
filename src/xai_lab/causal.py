"""Causal (occlusion) test: replace a region of the input by gray and re-score the class."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from . import config
from .data import to_normalized_tensor
from .maps import box_mask


def gray_out(
    crop: Image.Image,
    box: tuple[float, float, float, float],
    keep: str,
    gray_level: float = config.GRAY_LEVEL,
) -> torch.Tensor:
    """Gray out everything outside (``keep="inside"``) or inside (``keep="outside"``) the box.

    Returns the normalised (1, 3, 224, 224) tensor ready for the model.
    """
    if keep not in {"inside", "outside"}:
        raise ValueError("keep must be 'inside' or 'outside'")
    arr = np.asarray(crop).copy()
    inside = box_mask(box, arr.shape[0]).numpy()
    to_gray = ~inside if keep == "inside" else inside
    arr[to_gray] = int(round(gray_level * 255))
    return to_normalized_tensor(Image.fromarray(arr))


def gray_image(crop: Image.Image, box, keep: str) -> Image.Image:
    """Same as :func:`gray_out` but returns a PIL image (for figures)."""
    arr = np.asarray(crop).copy()
    inside = box_mask(box, arr.shape[0]).numpy()
    arr[(~inside) if keep == "inside" else inside] = int(round(config.GRAY_LEVEL * 255))
    return Image.fromarray(arr)
