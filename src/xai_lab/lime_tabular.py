"""LIME for tabular data from scratch (NumPy): sampling, kernel, weighted ridge in closed form, weighted R^2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import config


@dataclass
class LimeResult:
    """One LIME explanation: top-K variables with the refit coefficients plus the diagnostics of the fit."""

    features: list[str]  # top-K names ordered by |w| of the first fit
    coefs: np.ndarray  # refit coefficients of those K variables
    intercept: float
    r2w: float
    n_eff: float
    full_coefs: np.ndarray  # 20 coefficients of the first fit (all variables)
    selected: np.ndarray  # column indices of the K variables
    # samples kept for the numerical checks (not serialised)
    zp: np.ndarray = field(repr=False, default=None)
    y: np.ndarray = field(repr=False, default=None)
    pi: np.ndarray = field(repr=False, default=None)


def sample_perturbations(x: np.ndarray, X_train: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Z' (n x d, Bernoulli(0.5)) and Z (n x d): z'_j = 1 keeps x_j, z'_j = 0 draws a random training value of column j."""
    x = np.asarray(x, dtype=float)
    X_train = np.asarray(X_train, dtype=float)
    d = x.shape[0]
    z_prime = (rng.random((n, d)) < 0.5).astype(float)
    rows = rng.integers(0, X_train.shape[0], size=(n, d))  # independent per sample and feature
    drawn = X_train[rows, np.arange(d)]
    return z_prime, np.where(z_prime == 1.0, x, drawn)


def kernel_weights(z_prime: np.ndarray, sigma_rel: float, d_ref: int = config.LIME_D_PRIME) -> np.ndarray:
    """pi = exp(-D^2 / sigma^2), D^2 = number of absent variables, sigma = sigma_rel * sqrt(d_ref)."""
    d2 = z_prime.shape[1] - z_prime.sum(axis=1)
    sigma = sigma_rel * np.sqrt(d_ref)
    return np.exp(-d2 / sigma**2)


def weighted_ridge(Zp: np.ndarray, y: np.ndarray, pi: np.ndarray, lam: float, penalize_intercept: bool = False) -> np.ndarray:
    """w = (Z'^T Pi Z' + lam I)^-1 Z'^T Pi y with a ones column prepended; returns [intercept, slopes]."""
    A = np.column_stack([np.ones(len(Zp)), Zp])
    reg = lam * np.eye(A.shape[1])
    if not penalize_intercept:
        reg[0, 0] = 0.0
    return np.linalg.solve(A.T @ (pi[:, None] * A) + reg, A.T @ (pi * y))


def weighted_r2(y: np.ndarray, g: np.ndarray, pi: np.ndarray) -> float:
    """R^2_w = 1 - sum pi (y - g)^2 / sum pi (y - ybar_w)^2, with ybar_w = sum pi y / sum pi."""
    ybar = np.sum(pi * y) / np.sum(pi)
    return float(1.0 - np.sum(pi * (y - g) ** 2) / np.sum(pi * (y - ybar) ** 2))


def effective_sample_size(pi: np.ndarray) -> float:
    return float(np.sum(pi) ** 2 / np.sum(pi**2))


def explain(
    x: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_train: np.ndarray,
    feature_names: list[str],
    n: int = config.LIME_N,
    k: int = config.LIME_K,
    lam: float = config.LIME_LAMBDA,
    sigma_rel: float = config.LIME_SIGMA_REL,
    seed: int = config.SEED,
) -> LimeResult:
    """Sample, weight, fit on all variables, keep the K largest |w|, refit with only those and score R^2_w."""
    rng = np.random.default_rng(seed)
    z_prime, Z = sample_perturbations(x, X_train, n, rng)
    y = np.asarray(predict_fn(Z), dtype=float)
    pi = kernel_weights(z_prime, sigma_rel)
    full = weighted_ridge(z_prime, y, pi, lam)[1:]
    sel = np.argsort(-np.abs(full), kind="stable")[:k]
    w = weighted_ridge(z_prime[:, sel], y, pi, lam)
    g = w[0] + z_prime[:, sel] @ w[1:]
    return LimeResult([feature_names[i] for i in sel], w[1:], float(w[0]), weighted_r2(y, g, pi),
                      effective_sample_size(pi), full, sel, z_prime, y, pi)
