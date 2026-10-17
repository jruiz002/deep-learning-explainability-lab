"""Counterfactual explanations by random sampling (Task 3.4), written with NumPy.

A candidate x' copies the client x and changes a random subset of 1..max_changes actionable variables; every new value
is drawn from the training column (observed values different from x_j, so a "changed" variable really changes). It is
valid when P(good | x') >= theta, and its distance to x is

    d(x, x') = sum_{j numeric} |x_j - x'_j| / MAD_j + sum_{j categorical} 1[x_j != x'_j]

with MAD_j the median absolute deviation of column j in training (mean absolute deviation when the MAD is zero).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class CandidateSet:
    candidates: np.ndarray  # (n, d) encoded rows
    changed: np.ndarray  # (n, d) bool: variables changed in each candidate
    p_good: np.ndarray  # (n,)
    distance: np.ndarray  # (n,)
    valid: np.ndarray  # (n,) bool, p_good >= theta
    max_changes: int


def robust_scale(X_train: np.ndarray, numeric_idx: list[int]) -> dict[int, tuple[float, str]]:
    """{column: (scale, kind)} with kind 'MAD' or 'media' (mean absolute deviation, used when the MAD is 0)."""
    out = {}
    for j in numeric_idx:
        c = np.asarray(X_train[:, j], dtype=float)
        mad = float(np.median(np.abs(c - np.median(c))))
        out[j] = (mad, "MAD") if mad > 0 else (float(np.mean(np.abs(c - c.mean()))), "media")
    return out


def distance(x: np.ndarray, C: np.ndarray, scale: dict[int, tuple[float, str]], categorical_idx: list[int]) -> np.ndarray:
    """d(x, x') for every row of C: scaled L1 over numeric columns plus a 0/1 mismatch per categorical column."""
    C = np.atleast_2d(C)
    d = np.zeros(len(C))
    for j, (s, _) in scale.items():
        d += np.abs(C[:, j] - x[j]) / s
    for j in categorical_idx:
        d += C[:, j] != x[j]
    return d


def sample_candidates(
    x: np.ndarray,
    X_train: np.ndarray,
    actionable_idx: list[int],
    n: int,
    max_changes: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Candidates and their change masks: each row changes a uniform number 1..max_changes of actionable columns."""
    d = x.shape[0]
    m = min(max_changes, len(actionable_idx))
    C = np.tile(np.asarray(x, dtype=float), (n, 1))
    changed = np.zeros((n, d), dtype=bool)
    # training values of each actionable column that differ from the client (keeps their empirical frequency)
    pools = {j: X_train[:, j][X_train[:, j] != x[j]] for j in actionable_idx}
    for i in range(n):
        k = int(rng.integers(1, m + 1))
        for j in rng.choice(actionable_idx, size=k, replace=False):
            C[i, j] = pools[j][rng.integers(len(pools[j]))]
            changed[i, j] = True
    return C, changed


def search(
    x: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_train: np.ndarray,
    actionable_idx: list[int],
    numeric_idx: list[int],
    categorical_idx: list[int],
    n: int,
    max_changes: int,
    theta: float,
    seed: int,
) -> CandidateSet:
    rng = np.random.default_rng(seed)
    C, changed = sample_candidates(x, X_train, actionable_idx, n, max_changes, rng)
    p = np.asarray(predict_fn(C), dtype=float)
    dist = distance(x, C, robust_scale(X_train, numeric_idx), categorical_idx)
    return CandidateSet(C, changed, p, dist, p >= theta, max_changes)


def select_diverse(cs: CandidateSet, k: int) -> list[int]:
    """Closest valid candidate, then (greedily) the closest whose set of changed variables differs from those chosen."""
    chosen: list[int] = []
    seen: set[frozenset] = set()
    for i in np.argsort(cs.distance, kind="stable"):
        if not cs.valid[i]:
            continue
        key = frozenset(np.flatnonzero(cs.changed[i]).tolist())
        if key in seen:
            continue
        chosen.append(int(i))
        seen.add(key)
        if len(chosen) == k:
            break
    return chosen
