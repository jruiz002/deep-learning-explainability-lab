"""Counterfactual search (Task 3.4): MAD scale and its fallback, distance, sampling rules and diverse selection (offline)."""

import numpy as np

from xai_lab import counterfactual as cf


def test_robust_scale_uses_mad_and_falls_back_to_mean_absolute_deviation():
    X = np.column_stack([[1.0, 2, 3, 4, 100], [1.0, 1, 1, 1, 6]])
    s = cf.robust_scale(X, [0, 1])
    assert s[0] == (1.0, "MAD")  # median 3, |x - 3| = 2 1 0 1 97 -> median 1
    assert s[1][1] == "media" and np.isclose(s[1][0], np.mean(np.abs(X[:, 1] - X[:, 1].mean())))


def test_distance_formula():
    x = np.array([10.0, 2.0, 0.0])
    C = np.array([[10.0, 2.0, 0.0], [16.0, 2.0, 1.0], [7.0, 2.0, 0.0]])
    scale = {0: (3.0, "MAD")}
    assert np.allclose(cf.distance(x, C, scale, [2]), [0.0, 2.0 + 1.0, 1.0])


def test_candidates_change_only_actionable_columns_with_training_values():
    rng = np.random.default_rng(0)
    X_train = rng.integers(0, 5, size=(200, 6)).astype(float)
    x = np.zeros(6)
    act = [1, 3, 4]
    C, changed = cf.sample_candidates(x, X_train, act, 3000, 2, rng)
    k = changed.sum(axis=1)
    assert k.min() == 1 and k.max() == 2  # 1..max_changes
    assert not changed[:, [0, 2, 5]].any()  # non-actionable columns never move
    assert np.array_equal(C != x, changed)  # a changed column always gets a different value
    for j in act:
        assert np.all(np.isin(C[changed[:, j], j], X_train[:, j]))


def test_select_diverse_skips_invalid_and_repeated_change_sets():
    changed = np.array([[1, 0], [1, 0], [0, 1], [1, 1], [0, 1]], dtype=bool)
    cs = cf.CandidateSet(np.zeros((5, 2)), changed, np.array([0.9, 0.8, 0.2, 0.7, 0.6]),
                         np.array([1.0, 2.0, 0.5, 3.0, 4.0]), np.array([True, True, False, True, True]), 2)
    # 2 is invalid, 1 repeats the set of 0 -> 0 (d=1), 3 (d=3), 4 (d=4)
    assert cf.select_diverse(cs, 3) == [0, 3, 4]


def test_search_marks_validity_with_threshold():
    X_train = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    x = np.array([0.0, 0.0])
    cs = cf.search(x, lambda C: C[:, 0] / 2.0, X_train, [0, 1], [0, 1], [], 500, 2, 0.5, seed=1)
    assert np.array_equal(cs.valid, cs.p_good >= 0.5) and cs.valid.any() and (~cs.valid).any()
