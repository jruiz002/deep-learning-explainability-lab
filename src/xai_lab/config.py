"""Project-wide constants and paths."""

from __future__ import annotations

from pathlib import Path

SEED = 42

# ImageNet preprocessing (as required by the lab statement).
RESIZE_SHORT_SIDE = 256
CROP_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Geometry of the last convolutional stage of ResNet-18 (layer4).
FEATURE_CHANNELS = 512
FEATURE_SIZE = 7
Z = FEATURE_SIZE * FEATURE_SIZE  # 49 spatial positions

TOP_K = 5
IOU_TOP_FRACTION = 0.20
GRAY_LEVEL = 0.5  # pixel value in [0, 1] used by the causal (occlusion) test

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
MANIFEST_PATH = DATA_DIR / "images.json"
OUTPUT_DIR = ROOT / "outputs" / "task1"
FIGURES_DIR = OUTPUT_DIR / "figures"
