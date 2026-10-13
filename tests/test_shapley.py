"""Exact Shapley: linear closed form, efficiency, symmetry and dummy axioms (offline, instant)."""

import numpy as np

from xai_lab.shapley import coalition_masks, exact_shapley, value_function


def _data(n=5, k=12, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n), rng.normal(size=(k, n))


def test_masks_enumerate_all_coalitions():
    m = coalition_masks(4)
    assert m.shape == (16, 4) and len({tuple(r) for r in m}) == 16
    assert not m[0].any() and m[-1].all()


def test_value_function_single_call():
    x, bg = _data()
    calls = []
    f = lambda X: (calls.append(X.shape), X.sum(axis=1))[1]
    value_function(f, x, bg, coalition_masks(5))
    assert calls == [(32 * 12, 5)]


def test_linear_model_matches_closed_form():
    x, bg = _data()
    w, b = np.array([1.5, -2.0, 0.3, 0.0, 4.0]), 0.7
    phi, v0, vn = exact_shapley(lambda X: X @ w + b, x, bg)
    np.testing.assert_allclose(phi, w * (x - bg.mean(axis=0)), atol=1e-12)


def test_efficiency_nonlinear():
    x, bg = _data()
    f = lambda X: np.tanh(X[:, 0] * X[:, 1]) + X[:, 2] ** 2 - X[:, 3] * X[:, 4]
    phi, v0, vn = exact_shapley(f, x, bg)
    assert abs(phi.sum() - (vn - v0)) < 1e-12
    assert abs(v0 - f(bg).mean()) < 1e-12 and abs(vn - f(x[None])[0]) < 1e-12


def test_symmetry_duplicated_features():
    x, bg = _data(4)
    x[1], bg[:, 1] = x[0], bg[:, 0]  # features 0 and 1 identical in x and in every background row
    f = lambda X: np.sin(X[:, 0] + X[:, 1]) + X[:, 2] * X[:, 3]
    phi, _, _ = exact_shapley(f, x, bg)
    assert abs(phi[0] - phi[1]) < 1e-12


def test_dummy_feature_gets_zero():
    x, bg = _data()
    phi, _, _ = exact_shapley(lambda X: X[:, 0] * X[:, 1] + X[:, 2], x, bg)
    assert abs(phi[3]) < 1e-12 and abs(phi[4]) < 1e-12
