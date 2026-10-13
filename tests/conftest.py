import pytest
import torch
from torchvision.models import resnet18

from xai_lab.reproducibility import set_seed


@pytest.fixture(scope="module")
def model():
    """Randomly initialised ResNet-18: the identities under test hold for any weights."""
    set_seed(0)
    return resnet18(weights=None).eval()


@pytest.fixture()
def x():
    g = torch.Generator().manual_seed(1)
    return torch.randn(1, 3, 224, 224, generator=g)
