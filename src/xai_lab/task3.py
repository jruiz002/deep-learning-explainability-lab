"""Experiments of Task 3 (LIME from scratch): 3.1 explanations of three clients and 3.2 stability."""

from __future__ import annotations

import platform
import time
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge

from . import config, credit, lime_tabular as lt


@dataclass
class Task3Run:
    """JSON-serialisable ``results`` plus the arrays the figures need."""

    results: dict
    extra: dict = field(default_factory=dict)


def _closed_form_checks(res: lt.LimeResult, lam: float) -> dict:
    """Compare the closed-form fit on all 20 variables with lstsq (sqrt(pi)-weighted augmented system) and sklearn Ridge."""
    Zp, y, pi = res.zp, res.y, res.pi
    w = lt.weighted_ridge(Zp, y, pi, lam)
    A = np.column_stack([np.ones(len(Zp)), Zp])
    s = np.sqrt(pi)[:, None]
    pen = np.sqrt(lam) * np.hstack([np.zeros((Zp.shape[1], 1)), np.eye(Zp.shape[1])])
    w_ls = np.linalg.lstsq(np.vstack([s * A, pen]), np.concatenate([np.sqrt(pi) * y, np.zeros(Zp.shape[1])]), rcond=None)[0]
    rd = Ridge(alpha=lam, fit_intercept=True).fit(Zp, y, sample_weight=pi)
    return {"max_abs_diff_lstsq": float(np.max(np.abs(w - w_ls))),
            "max_abs_diff_sklearn_ridge": float(max(np.max(np.abs(w[1:] - rd.coef_)), abs(w[0] - rd.intercept_)))}


def run_task3(seed: int = config.SEED) -> Task3Run:
    t0 = time.perf_counter()
    base = credit.build_base(seed)
    d, sp, model = base.data, base.split, base.model
    names = d.feature_names
    X_train = sp.X_train.to_numpy()
    predict = lambda Z: credit.proba_good(model, pd.DataFrame(Z, columns=names))
    p_test = credit.proba_good(model, sp.X_test)
    clients = credit.select_clients(p_test)
    cfg = {"seed": seed, "n": config.LIME_N, "k": config.LIME_K, "lambda": config.LIME_LAMBDA,
           "sigma_rel": config.LIME_SIGMA_REL, "d_prime": config.LIME_D_PRIME,
           "sigma": config.LIME_SIGMA_REL * float(np.sqrt(config.LIME_D_PRIME)), "intercept_penalized": False}
    res: dict = {"environment": {"seed": seed, "python": platform.python_version(), "sklearn": sklearn.__version__,
                                 "numpy": np.__version__, "pandas": pd.__version__}, "config": cfg}

    # ---- 3.1 ------------------------------------------------------------------------------
    t31: dict = {}
    for key, pos in clients.items():
        row = sp.X_test.iloc[pos]
        r = lt.explain(row.to_numpy(), predict, X_train, names, seed=seed)
        t31[key] = {
            "test_position": pos, "p_good": float(p_test[pos]), "intercept": r.intercept, "r2w": r.r2w, "n_eff": r.n_eff,
            "top": [{"feature": f, "value": d.decode(f, row[f]), "coef": float(c)} for f, c in zip(r.features, r.coefs)],
            "full_coefs": {n: float(c) for n, c in zip(names, r.full_coefs)},
            **_closed_form_checks(r, config.LIME_LAMBDA),
        }
    res["t3_1"] = t31

    # ---- 3.2 ------------------------------------------------------------------------------
    pos = clients["rechazado"]
    x = sp.X_test.iloc[pos].to_numpy()
    seeds = list(range(config.LIME_STAB_RUNS))
    jac, neff, cells = [], [], []
    for n in config.LIME_STAB_N:
        jrow, nrow = [], []
        for sr in config.LIME_STAB_SIGMA_REL:
            runs = [lt.explain(x, predict, X_train, names, n=n, sigma_rel=sr, seed=s) for s in seeds]
            sets = [set(r.features) for r in runs]
            j = [len(a & b) / len(a | b) for a, b in combinations(sets, 2)]
            freq = Counter(f for s_ in sets for f in s_)
            jrow.append(float(np.mean(j)))
            nrow.append(float(np.mean([r.n_eff for r in runs])))
            cells.append({"n": n, "sigma_rel": sr, "n_pairs": len(j), "jaccard_min": float(min(j)), "jaccard_max": float(max(j)),
                          "most_frequent": [{"feature": f, "frequency": c / len(seeds)}
                                            for f, c in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:config.LIME_STAB_TOP]]})
        jac.append(jrow)
        neff.append(nrow)
    res["t3_2"] = {"client": "rechazado", "test_position": pos, "n_values": list(config.LIME_STAB_N),
                   "sigma_rel_values": list(config.LIME_STAB_SIGMA_REL), "seeds": seeds, "k": config.LIME_K,
                   "lambda": config.LIME_LAMBDA, "jaccard": jac, "n_eff": neff, "cells": cells}
    res["time_s"] = time.perf_counter() - t0
    return Task3Run(res)
