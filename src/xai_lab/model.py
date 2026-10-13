"""ResNet-18 loading and prediction helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
CATEGORIES: list[str] = list(WEIGHTS.meta["categories"])


@dataclass(frozen=True)
class Prediction:
    index: int
    name: str
    probability: float
    logit: float


def load_model(device: str | torch.device = "cpu") -> nn.Module:
    """ImageNet-pretrained ResNet-18 in evaluation mode."""
    model = resnet18(weights=WEIGHTS).to(device)
    return model.eval()


@torch.no_grad()
def predict_topk(model: nn.Module, x: torch.Tensor, k: int = 5) -> list[Prediction]:
    logits = model(x)[0]
    probs = logits.softmax(dim=0)
    top = probs.topk(k)
    return [
        Prediction(int(i), CATEGORIES[int(i)], float(probs[i]), float(logits[i]))
        for i in top.indices
    ]


@torch.no_grad()
def class_probability(model: nn.Module, x: torch.Tensor, class_idx: int) -> float:
    return float(model(x)[0].softmax(dim=0)[class_idx])
