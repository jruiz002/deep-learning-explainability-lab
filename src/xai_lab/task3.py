"""Experiments of Task 3: LIME from scratch (3.1 - 3.3) and counterfactuals (3.4)."""

from __future__ import annotations

import platform
import time
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
import shap
import sklearn
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from . import config, counterfactual as cf, credit, lime_tabular as lt


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
           "sigma": config.LIME_SIGMA_REL * float(np.sqrt(config.LIME_D_PRIME)), "intercept_penalized": False,
           "cf_max": config.CF_MAX_CHANGES}
    res: dict = {"environment": {"seed": seed, "python": platform.python_version(), "sklearn": sklearn.__version__,
                                 "shap": shap.__version__, "numpy": np.__version__, "pandas": pd.__version__}, "config": cfg}

    # ---- 3.1 ------------------------------------------------------------------------------
    t31: dict = {}
    lime_runs: dict[str, lt.LimeResult] = {}
    for key, pos in clients.items():
        row = sp.X_test.iloc[pos]
        r = lime_runs[key] = lt.explain(row.to_numpy(), predict, X_train, names, seed=seed)
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

    # ---- 3.3 ------------------------------------------------------------------------------
    # SHAP exactly as in Task 2.2 (interventional TreeSHAP, 100-row background, log-odds); the probability-scale
    # version is only used in 3.5(b) to separate the effect of the output scale from that of the method.
    tree = shap.TreeExplainer(model, data=base.background, feature_perturbation="interventional", model_output="raw")
    tree_p = shap.TreeExplainer(model, data=base.background, feature_perturbation="interventional", model_output="probability")
    rows = sp.X_test.iloc[list(clients.values())]
    phi = dict(zip(clients, np.asarray(tree.shap_values(rows, check_additivity=False))))
    phi_p = dict(zip(clients, np.asarray(tree_p.shap_values(rows, check_additivity=False))))
    res["t3_3"] = _t3_3(clients, lime_runs, phi, phi_p, names, x, predict, X_train, seed)

    # ---- 3.4 ------------------------------------------------------------------------------
    res["t3_4"] = _t3_4(d, sp, model, clients["rechazado"], seed)

    res["time_s"] = time.perf_counter() - t0
    return Task3Run(res)


def _ranks(v: np.ndarray) -> np.ndarray:
    """Rank 1 = largest |v| (ties broken by column order)."""
    r = np.empty(len(v), dtype=int)
    r[np.argsort(-np.abs(v), kind="stable")] = np.arange(1, len(v) + 1)
    return r


def _t3_3(clients, lime_runs, phi, phi_p, names, x_rej, predict, X_train, seed) -> dict:
    """(a) Spearman between |w| (LIME on all 20 variables) and |phi| (TreeSHAP 2.2) and top-3 overlap; (b) R^2_w vs K."""
    top = config.TOP_COMPARE
    a = {}
    for key in clients:
        w, f, fp = lime_runs[key].full_coefs, phi[key], phi_p[key]
        top_w = [names[j] for j in np.argsort(-np.abs(w), kind="stable")[:top]]
        top_f = [names[j] for j in np.argsort(-np.abs(f), kind="stable")[:top]]
        a[key] = {
            "spearman": float(spearmanr(np.abs(w), np.abs(f)).statistic),
            "spearman_vs_shap_probability": float(spearmanr(np.abs(w), np.abs(fp)).statistic),
            "top_lime": top_w, "top_shap": top_f, "top_overlap": len(set(top_w) & set(top_f)),
            "lime_w": {n: float(v) for n, v in zip(names, w)},
            "shap_phi": {n: float(v) for n, v in zip(names, f)},
            "shap_phi_probability": {n: float(v) for n, v in zip(names, fp)},
            "rank_lime": {n: int(r) for n, r in zip(names, _ranks(w))},
            "rank_shap": {n: int(r) for n, r in zip(names, _ranks(f))},
        }
    b = []
    for k in config.LIME_K_VALUES:
        r = lt.explain(x_rej, predict, X_train, names, k=k, seed=seed)  # same seed -> same 5000 samples for every K
        b.append({"k": k, "r2w": r.r2w, "features": r.features})
    return {"a": a, "b": {"client": "rechazado", "by_k": b}}


def _t3_4(d: credit.CreditData, sp: credit.Split, model, pos: int, seed: int) -> dict:
    """Sampling search of counterfactuals for the rejected client (max. 3 actionable changes, 5 if none is valid)."""
    names = d.feature_names
    X_train = sp.X_train.to_numpy()
    x = sp.X_test.iloc[pos].to_numpy()
    predict = lambda Z: credit.proba_good(model, pd.DataFrame(Z, columns=names))
    idx = lambda cols: [names.index(c) for c in cols]
    act, num, cat = idx(config.CF_ACTIONABLE), idx(d.numeric), idx(d.categorical)
    assert not set(config.CF_ACTIONABLE) & set(config.CF_IMMUTABLE)
    t0 = time.perf_counter()
    cs = cf.search(x, predict, X_train, act, num, cat, config.CF_N_CANDIDATES, config.CF_MAX_CHANGES, config.CF_THRESHOLD, seed)
    fallback = not cs.valid.any()
    if fallback:
        cs = cf.search(x, predict, X_train, act, num, cat, config.CF_N_CANDIDATES, config.CF_MAX_CHANGES_FALLBACK,
                       config.CF_THRESHOLD, seed)
    elapsed = time.perf_counter() - t0
    chosen = cf.select_diverse(cs, config.CF_N_SELECTED)
    scale = cf.robust_scale(X_train, num)
    n_changed = cs.changed.sum(axis=1)
    sel = []
    for i in chosen:
        cols = np.flatnonzero(cs.changed[i])
        sel.append({
            "changes": [{"feature": names[j], "original": d.decode(names[j], x[j]), "new": d.decode(names[j], cs.candidates[i, j]),
                         "cost": float(abs(cs.candidates[i, j] - x[j]) / scale[j][0]) if j in scale else 1.0} for j in cols],
            "distance": float(cs.distance[i]), "p_good": float(cs.p_good[i]), "n_changed": int(len(cols)),
        })
    by_m = {int(m): {"candidates": int((n_changed == m).sum()), "valid": int((cs.valid & (n_changed == m)).sum())}
            for m in range(1, cs.max_changes + 1)}
    by_f = {names[j]: {"candidates": int(cs.changed[:, j].sum()), "valid": int((cs.valid & cs.changed[:, j]).sum())}
            for j in act}
    return {
        "client": "rechazado", "test_position": pos, "p_good_original": float(predict(x[None])[0]),
        "n_candidates": config.CF_N_CANDIDATES, "max_changes": cs.max_changes, "fallback_to_5": fallback,
        "threshold": config.CF_THRESHOLD, "actionable": list(config.CF_ACTIONABLE), "immutable": list(config.CF_IMMUTABLE),
        "scales": {names[j]: {"value": v, "kind": k} for j, (v, k) in scale.items() if names[j] in config.CF_ACTIONABLE},
        "n_valid": int(cs.valid.sum()), "validity": float(cs.valid.mean()), "by_n_changed": by_m, "by_feature": by_f,
        "selected": sel,
        "proximity_selected": float(np.mean([s["distance"] for s in sel])) if sel else None,
        "sparsity_selected": float(np.mean([s["n_changed"] for s in sel])) if sel else None,
        "proximity_valid": float(cs.distance[cs.valid].mean()) if cs.valid.any() else None,
        "sparsity_valid": float(n_changed[cs.valid].mean()) if cs.valid.any() else None,
        "max_p_good_candidates": float(cs.p_good.max()), "time_s": elapsed,
    }
