"""Image manifest loading, ImageNet preprocessing and bounding-box mapping."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

from . import config


@dataclass(frozen=True)
class ImageSpec:
    """One lab image as described in ``data/images.json``."""

    id: str
    role: str  # "a", "b" or "c"
    filename: str
    description: str
    source_url: str
    page_url: str
    author: str
    license: str
    sha256: str = ""
    # Box enclosing the object, in ORIGINAL image pixel coordinates (x0, y0, x1, y1).
    bbox_original: tuple[float, float, float, float] | None = None
    # Groups of ImageNet class indices used to pick "the class of each object" in image (b).
    class_groups: dict[str, list[int]] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return config.IMAGES_DIR / self.filename


@dataclass(frozen=True)
class PreparedImage:
    spec: ImageSpec
    crop: Image.Image  # 224x224 RGB crop shown in figures
    tensor: torch.Tensor  # (1, 3, 224, 224), normalised
    bbox_224: tuple[float, float, float, float] | None


def load_manifest(path: Path = config.MANIFEST_PATH) -> list[ImageSpec]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    specs = []
    for e in entries:
        bbox = e.get("bbox_original")
        specs.append(
            ImageSpec(
                **{k: v for k, v in e.items() if k not in {"bbox_original", "class_groups"}},
                bbox_original=tuple(bbox) if bbox else None,
                class_groups=e.get("class_groups", {}),
            )
        )
    return specs


_normalize = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ]
)


def resize_and_crop(img: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Resize the short side to 256 and centre-crop 224. Returns (resized, crop)."""
    resized = T.Resize(config.RESIZE_SHORT_SIDE)(img)
    crop = T.CenterCrop(config.CROP_SIZE)(resized)
    return resized, crop


def map_bbox_to_crop(
    bbox: tuple[float, float, float, float],
    original_size: tuple[int, int],
    resized_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Transform a box from original pixels to the 224x224 crop, clipped to the crop."""
    (w, h), (rw, rh) = original_size, resized_size
    sx, sy = rw / w, rh / h
    left = int(round((rw - config.CROP_SIZE) / 2.0))  # same rounding as CenterCrop
    top = int(round((rh - config.CROP_SIZE) / 2.0))
    x0, y0, x1, y1 = bbox
    mapped = (x0 * sx - left, y0 * sy - top, x1 * sx - left, y1 * sy - top)
    return tuple(min(max(v, 0.0), float(config.CROP_SIZE)) for v in mapped)  # type: ignore[return-value]


def prepare_image(spec: ImageSpec) -> PreparedImage:
    img = Image.open(spec.path).convert("RGB")
    resized, crop = resize_and_crop(img)
    bbox_224 = (
        map_bbox_to_crop(spec.bbox_original, img.size, resized.size)
        if spec.bbox_original
        else None
    )
    return PreparedImage(spec, crop, _normalize(crop).unsqueeze(0), bbox_224)


def to_normalized_tensor(crop: Image.Image) -> torch.Tensor:
    """Normalise an already cropped 224x224 PIL image into a (1, 3, 224, 224) tensor."""
    return _normalize(crop).unsqueeze(0)
