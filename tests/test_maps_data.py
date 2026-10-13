import numpy as np
import torch
from PIL import Image

from xai_lab import maps
from xai_lab.causal import gray_out
from xai_lab.data import map_bbox_to_crop


def test_relu_minmax_range_and_constant():
    m = maps.relu_minmax(torch.tensor([[-1.0, 0.0], [2.0, 4.0]]))
    assert m.min() == 0 and m.max() == 1
    assert torch.all(maps.relu_minmax(torch.full((3, 3), 5.0)) == 0)


def test_iou_identical_and_disjoint():
    a = torch.zeros(10, 10, dtype=torch.bool)
    a[:5] = True
    assert maps.iou(a, a) == 1.0
    assert maps.iou(a, ~a) == 0.0


def test_top_fraction_size():
    m = torch.rand(224, 224)
    assert int(maps.top_fraction_mask(m, 0.2).sum()) == round(0.2 * 224 * 224)


def test_mass_in_box():
    m = torch.ones(224, 224)
    assert maps.mass_in_box(m, (0, 0, 224, 224)) == 1.0
    assert abs(maps.mass_in_box(m, (0, 0, 112, 224)) - 0.5) < 1e-9


def test_bbox_mapping_square_image():
    # 512x512 -> resized to 256 (x0.5) -> crop offset 16
    assert map_bbox_to_crop((100, 100, 300, 300), (512, 512), (256, 256)) == (34.0, 34.0, 134.0, 134.0)


def test_bbox_is_clipped():
    assert map_bbox_to_crop((0, 0, 512, 512), (512, 512), (256, 256)) == (0.0, 0.0, 224.0, 224.0)


def test_gray_out_keeps_box_only():
    crop = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    x = gray_out(crop, (0, 0, 112, 224), keep="inside")
    gray = (0.5 - 0.485) / 0.229  # channel 0 normalised gray vs black
    black = (0.0 - 0.485) / 0.229
    assert abs(float(x[0, 0, 0, 0]) - black) < 1e-2  # inside the box: untouched
    assert abs(float(x[0, 0, 0, 200]) - gray) < 2e-2  # outside: gray
