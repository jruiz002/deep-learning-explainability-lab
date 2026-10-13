"""Class Activation Mapping (CAM) computed directly from the final linear layer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from . import config


@dataclass(frozen=True)
class CAMResult:
    class_idx: int
    map: torch.Tensor  # M^c_ij = sum_k w^c_k A^k_ij: (7, 7)
    weights: torch.Tensor  # w^c: (512,)
    bias: float  # b_c
    logit_from_cam: float  # (1/Z) sum_ij M^c_ij + b_c


@torch.no_grad()
def feature_maps(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Activations A of the last conv stage (before global pooling): (512, 7, 7)."""
    captured: list[torch.Tensor] = []
    handle = model.layer4.register_forward_hook(lambda _m, _i, out: captured.append(out))
    try:
        model(x)
    finally:
        handle.remove()
    return captured[0][0]


@torch.no_grad()
def compute_cam(model: nn.Module, activations: torch.Tensor, class_idx: int) -> CAMResult:
    """CAM from the ``fc`` weights of class ``class_idx``; no gradients involved."""
    w = model.fc.weight[class_idx]  # (512,)
    b = float(model.fc.bias[class_idx])
    m = torch.einsum("k,kij->ij", w, activations)
    return CAMResult(class_idx, m, w.clone(), b, float(m.sum() / config.Z + b))
