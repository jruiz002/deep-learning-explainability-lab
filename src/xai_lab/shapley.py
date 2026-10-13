"""Exact Shapley values by brute-force coalition enumeration (pure NumPy, no ``shap``)."""

from __future__ import annotations

from math import factorial

import numpy as np


def coalition_masks(n: int) -> np.ndarray:
    """(2^n, n) boolean matrix; row m has feature j in the coalition iff bit j of m is set."""
    return ((np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1).astype(bool)


def value_function(f, x: np.ndarray, bg: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """v(S) = mean_b f(x_S, b_{not S}) for every mask, with ONE call to ``f``.

    Builds a (len(masks) * len(bg), n) matrix: features in S take x's value, the rest the background row's.
    """
    x, bg = np.asarray(x, dtype=float), np.asarray(bg, dtype=float)
    m, k = len(masks), len(bg)
    data = np.where(masks[:, None, :], x[None, None, :], bg[None, :, :])  # (m, k, n)
    return np.asarray(f(data.reshape(m * k, -1))).reshape(m, k).mean(axis=1)


def exact_shapley(f, x: np.ndarray, bg: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(phi, v(empty), v(N)) with weights |S|!(n-|S|-1)!/n! over S subset of N \ {i}."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    masks = coalition_masks(n)
    v = value_function(f, x, bg, masks)
    size = masks.sum(axis=1)
    weight = np.array([factorial(s) * factorial(n - s - 1) / factorial(n) if s < n else 0.0 for s in range(n + 1)])
    ids = np.arange(2**n)
    phi = np.zeros(n)
    for i in range(n):
        without = ids[~masks[:, i]]  # S not containing i; S u {i} is ids + 2^i
        phi[i] = np.sum(weight[size[without]] * (v[without + (1 << i)] - v[without]))
    return phi, float(v[0]), float(v[-1])
