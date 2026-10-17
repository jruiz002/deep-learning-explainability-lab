"""Deletion / insertion frames and AUC, and cascading randomisation helpers (Task 4), on small tensors or a random ResNet."""

import copy

import numpy as np
import torch

from xai_lab import config, faithfulness as fa, sanity
from xai_lab.gradcam import GradCAM


def test_frames_follow_the_order_and_end_on_the_target_image():
    start, end = torch.zeros(3, 4, 4), torch.ones(3, 4, 4)
    order = torch.arange(16).flip(0)  # last pixel first
    F = fa.perturbation_images(start, end, order, steps=4)
    assert torch.equal(F[0], start) and torch.equal(F[-1], end)
    assert F[1].flatten(1)[:, 12:].eq(1).all() and F[1].flatten(1)[:, :12].eq(0).all()  # first 4 of the order


def test_order_from_map_is_descending_and_stable():
    m = torch.tensor([[0.1, 0.9], [0.9, 0.0]])
    assert fa.order_from_map(m).tolist() == [1, 2, 0, 3]


def test_auc_ignores_the_start_frame():
    assert np.isclose(fa.auc(np.array([1.0, 0.5, 0.25, 0.0, 0.25])), 0.25)


def test_blur_preserves_a_constant_image_and_smooths_an_impulse():
    flat = torch.full((3, 32, 32), 0.3)
    assert torch.allclose(fa.gaussian_blur(flat, 3.0), flat, atol=1e-6)
    imp = torch.zeros(3, 33, 33)
    imp[:, 16, 16] = 1.0
    b = fa.gaussian_blur(imp, 3.0)
    assert b[0, 16, 16] < 0.1 and np.isclose(float(b[0].sum()), 1.0, atol=1e-3)


def test_cells_touched_counts_7x7_cells():
    order = torch.arange(config.CROP_SIZE ** 2)  # raster order: first 1/7 of the rows = first row of cells
    assert fa.cells_touched(order, 1, 7) == config.FEATURE_SIZE


def test_reinitialize_only_touches_the_given_block(model):
    m = copy.deepcopy(model)
    before = {n: p.clone() for n, p in m.named_parameters()}
    torch.manual_seed(123)
    assert sanity.reinitialize(m.fc) == 1
    changed = {n for n, p in m.named_parameters() if not torch.equal(p, before[n])}
    assert changed == {"fc.weight", "fc.bias"}


def test_random_fc_map_is_the_new_weights_times_the_same_activations(model, x):
    """alpha = w / Z: after re-initialising fc only, the raw map is sum_k w_rand_k A^k / Z with unchanged activations."""
    m = copy.deepcopy(model)
    with GradCAM(m, m.layer4) as g:
        r0 = g(x, 3)
    torch.manual_seed(7)
    sanity.reinitialize(m.fc)
    with GradCAM(m, m.layer4) as g:
        r1 = g(x, 3)
    pred = (m.fc.weight[3].detach()[:, None, None] * r0.activations).sum(0) / config.Z
    assert torch.equal(r0.activations, r1.activations)
    assert torch.allclose(pred, r1.raw_map, atol=1e-5)


def test_spearman_maps_handles_constant_maps():
    a = np.arange(49.0)
    assert np.isclose(sanity.spearman_maps(a, 2 * a + 1), 1.0)
    assert np.isnan(sanity.spearman_maps(a, np.zeros(49)))
