"""Cascading parameter randomisation (Task 4.2): re-initialise ResNet-18 from the top layer down.

Each stage re-initialises one more block with PyTorch's own initialisers (``reset_parameters`` of every Conv2d, Linear
and BatchNorm2d inside it; for BatchNorm this also resets the running statistics), keeping the blocks randomised in
earlier stages. The Grad-CAM map of the class that was originally the most probable is recomputed after every stage.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from torch import nn


def reinitialize(module: nn.Module) -> int:
    """Call ``reset_parameters`` on every sub-module that has one; returns how many were reset."""
    n = 0
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
            n += 1
    return n


def spearman_maps(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation of two maps over their positions; NaN when one of them is constant (e.g. all zero)."""
    a, b = np.ravel(a), np.ravel(b)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)
