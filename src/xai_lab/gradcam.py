"""Grad-CAM implemented from scratch with a forward and a backward hook on ``layer4``."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from . import config
from .maps import minmax, upsample


@dataclass(frozen=True)
class GradCAMResult:
    class_idx: int
    logit: float
    activations: torch.Tensor  # A:           (512, 7, 7)
    gradients: torch.Tensor  # dy^c/dA:     (512, 7, 7)
    alpha: torch.Tensor  # alpha_k^c:   (512,)
    raw_map: torch.Tensor  # sum_k alpha_k A^k, no ReLU / no normalisation: (7, 7)
    cam_7x7: torch.Tensor  # ReLU(raw_map): (7, 7)
    cam_224: torch.Tensor  # ReLU -> bilinear x32 -> min-max: (224, 224)


class GradCAM:
    """Hook-based Grad-CAM on a ResNet's last convolutional stage.

    Use as a context manager so the hooks are always removed::

        with GradCAM(model, model.layer4) as cam:
            result = cam(x, class_idx)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, _module, _inputs, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def __call__(self, x: torch.Tensor, class_idx: int) -> GradCAMResult:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        # Back-propagate from the raw logit y^c (before softmax), not from a probability or a loss.
        logits[0, class_idx].backward()

        assert self._activations is not None and self._gradients is not None
        a = self._activations[0]
        g = self._gradients[0]
        alpha = g.mean(dim=(1, 2))  # spatial average of each channel's gradient
        raw_map = (alpha[:, None, None] * a).sum(dim=0)  # (7, 7)
        cam_7x7 = F.relu(raw_map)
        cam_224 = minmax(upsample(cam_7x7, config.CROP_SIZE))
        return GradCAMResult(
            class_idx=class_idx,
            logit=float(logits[0, class_idx].detach()),
            activations=a.clone(),
            gradients=g.clone(),
            alpha=alpha.clone(),
            raw_map=raw_map.clone(),
            cam_7x7=cam_7x7.clone(),
            cam_224=cam_224.clone(),
        )
