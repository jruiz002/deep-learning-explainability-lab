"""LIME from scratch: kernel, effective sample size, weighted R^2, closed-form ridge and sampling (offline, instant)."""

import numpy as np

from xai_lab.lime_tabular import (effective_sample_size, explain, kernel_weights, sample_perturbations,
                                  weighted_r2, weighted_ridge)


def test_kernel_weights_identity_and_monotone():
    d = 20
    z = np.array([np.r_[np.ones(d - m), np.zeros(m)] for m in range(d + 1)])
    pi = kernel_weights(z, 0.75)
    assert pi[0] == 1.0
    assert np.all(np.diff(pi) < 0)
    assert np.allclose(pi, np.exp(-np.arange(d + 1) / (0.75 * np.sqrt(20)) ** 2))


def test_effective_sample_size_uniform_and_degenerate():
    assert np.isclose(effective_sample_size(np.full(300, 0.4)), 300)
    assert np.isclose(effective_sample_size(np.array([1.0, 1e-12, 1e-12])), 1.0, atol=1e-6)


def test_weighted_r2_exact_linear_and_formula():
    rng = np.random.default_rng(0)
    y, pi = rng.normal(size=50), rng.uniform(0.1, 1, 50)
    assert np.isclose(weighted_r2(y, y, pi), 1.0)
    g = y + rng.normal(scale=0.3, size=50)
    yb = (pi * y).sum() / pi.sum()
    assert np.isclose(weighted_r2(y, g, pi), 1 - (pi * (y - g) ** 2).sum() / (pi * (y - yb) ** 2).sum())


def _linear_problem(n=400, d=6, seed=1):
    rng = np.random.default_rng(seed)
    Zp = (rng.random((n, d)) < 0.5).astype(float)
    w = rng.normal(size=d)
    return Zp, 0.7 + Zp @ w, w, rng.uniform(0.2, 1, n)


def test_closed_form_recovers_exact_linear_coefficients():
    Zp, y, w, pi = _linear_problem()
    est = weighted_ridge(Zp, y, pi, lam=1e-10)
    assert np.allclose(est[0], 0.7, atol=1e-6) and np.allclose(est[1:], w, atol=1e-6)


def test_closed_form_matches_lstsq_on_augmented_system():
    Zp, y, _, pi = _linear_problem()
    y = y + np.random.default_rng(2).normal(scale=0.2, size=len(y))
    lam = 1.0
    A = np.column_stack([np.ones(len(Zp)), Zp])
    s = np.sqrt(pi)
    pen = np.sqrt(lam) * np.hstack([np.zeros((Zp.shape[1], 1)), np.eye(Zp.shape[1])])
    ref = np.linalg.lstsq(np.vstack([s[:, None] * A, pen]), np.r_[s * y, np.zeros(Zp.shape[1])], rcond=None)[0]
    assert np.allclose(weighted_ridge(Zp, y, pi, lam), ref, atol=1e-10)


def test_intercept_is_not_penalised():
    Zp, y, _, pi = _linear_problem()
    est = weighted_ridge(Zp, y, pi, lam=1e9)
    assert np.allclose(est[1:], 0, atol=1e-6)
    assert np.isclose(est[0], (pi * y).sum() / pi.sum(), atol=1e-6)
    assert not np.isclose(weighted_ridge(Zp, y, pi, 1e9, penalize_intercept=True)[0], est[0], atol=1e-3)


def test_sampling_keeps_present_and_draws_absent_from_training_column():
    rng = np.random.default_rng(3)
    d, n = 5, 2000
    x = np.arange(d, dtype=float) + 100
    X_train = np.column_stack([np.arange(10) + 1000 * j for j in range(d)]).astype(float)
    zp, Z = sample_perturbations(x, X_train, n, rng)
    assert zp.shape == Z.shape == (n, d) and set(np.unique(zp)) == {0.0, 1.0}
    assert abs(zp.mean() - 0.5) < 0.02
    for j in range(d):
        assert np.all(Z[zp[:, j] == 1, j] == x[j])
        assert np.all(np.isin(Z[zp[:, j] == 0, j], X_train[:, j]))
        assert len(np.unique(Z[zp[:, j] == 0, j])) > 5  # values vary across samples


def test_explain_finds_the_active_variables_of_a_linear_model():
    rng = np.random.default_rng(4)
    X_train = rng.normal(size=(200, 8))
    x = rng.normal(size=8)
    # f depends only on whether v1 / v5 keep the client value, so it is exactly linear in z'
    f = lambda Z: 2.0 * (Z[:, 1] == x[1]) - 1.0 * (Z[:, 5] == x[5])
    names = [f"v{i}" for i in range(8)]
    r = explain(x, f, X_train, names, n=3000, k=2, lam=1e-6, sigma_rel=100.0, seed=0)
    assert set(r.features) == {"v1", "v5"} and r.r2w > 0.999
    assert 0.0 <= r.r2w <= 1.0 and r.n_eff <= 3000
