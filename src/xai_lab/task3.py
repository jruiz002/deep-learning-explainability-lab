"""Experiments of Task 3: LIME from scratch (3.1 - 3.3), counterfactuals (3.4) and the analysis runs of 3.5."""

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
           "cf_max": config.CF_MAX_CHANGES, "disagree_seeds": config.LIME_DISAGREE_SEEDS}
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

    # ---- 3.5 ------------------------------------------------------------------------------
    res["t3_5"] = {
        "a": _t3_5a(res["t3_2"], x, predict, X_train, names),
        "b": _t3_5b(res["t3_3"], clients, sp, d, predict, model, base.background.to_numpy(), X_train, names),
        "d": _t3_5d(res["t3_3"], res["t3_4"], d, sp, model, clients["rechazado"]),
    }
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
    if len(chosen) < config.CF_N_SELECTED:  # the report, the checks and Task 4 assume all of them exist
        raise RuntimeError(f"Sólo {len(chosen)} contrafactuales válidos con conjuntos de variables distintos "
                           f"(se piden {config.CF_N_SELECTED}; máximo de cambios = {cs.max_changes}).")
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


def _lime_jaccard(x, predict, X_train, names, n: int, sigma_rel: float) -> tuple[float, float]:
    """Mean pairwise Jaccard of the top-K sets and mean N_ef over the 20 seeds used in 3.2."""
    runs = [lt.explain(x, predict, X_train, names, n=n, sigma_rel=sigma_rel, seed=s) for s in range(config.LIME_STAB_RUNS)]
    sets = [set(r.features) for r in runs]
    return (float(np.mean([len(a & b) / len(a | b) for a, b in combinations(sets, 2)])),
            float(np.mean([r.n_eff for r in runs])))


def _t3_5a(t2: dict, x, predict, X_train, names) -> dict:
    """N needed for a mean Jaccard of 0.9 at sigma_rel = 0.75, estimated from the 3.2 table and checked with new runs.

    Model: 1 - J = a * N_ef^(-b), fitted on the two grid points that bracket the target (log-log interpolation).
    It encodes that the noise of the coefficients, and hence the chance of swapping the K-th and (K+1)-th variable,
    shrinks with the effective sample size; N_ef / N is constant for a given sigma_rel, so N = N_ef / (N_ef / N).
    """
    target, sr = config.LIME_TARGET_JACCARD, config.LIME_SIGMA_REL
    col = t2["sigma_rel_values"].index(sr)
    ns = np.array(t2["n_values"], dtype=float)
    J = np.array([row[col] for row in t2["jaccard"]])
    E = np.array([row[col] for row in t2["n_eff"]])
    hi = int(np.argmax(J >= target)) if (J >= target).any() else len(J) - 1
    lo = max(hi - 1, 0)
    # the power law needs two different points with 0 < 1 - J and J increasing; otherwise b is NaN or meaningless
    if hi == lo or J[hi] >= 1 or J[hi] <= J[lo]:
        raise ValueError(f"No se puede ajustar 1 - J = a·N_ef^(-b): Jaccard {np.round(J, 4).tolist()} con N = "
                         f"{ns.astype(int).tolist()} (el menor N ya alcanza {target}, o J no crece). Agregue un N menor a la tabla del 3.2.")
    b = float(np.log((1 - J[lo]) / (1 - J[hi])) / np.log(E[hi] / E[lo]))
    a = float((1 - J[hi]) * E[hi] ** b)
    n_eff_needed = float((a / (1 - target)) ** (1 / b))
    ratio = float(np.mean(E / ns))
    n_est = int(np.ceil(n_eff_needed / ratio / 100.0) * 100)  # rounded up to a multiple of 100
    j_check, e_check = _lime_jaccard(x, predict, X_train, names, n_est, sr)
    # pooled relation between N_ef and Jaccard over the 9 cells of 3.2
    allJ = [v for row in t2["jaccard"] for v in row]
    allE = [v for row in t2["n_eff"] for v in row]
    rho = spearmanr(allE, allJ)
    return {"target": target, "sigma_rel": sr, "bracket_n": [int(ns[lo]), int(ns[hi])], "bracket_n_eff": [float(E[lo]), float(E[hi])], "fit_a": a, "fit_b": b,
            "n_eff_over_n": ratio, "n_eff_needed": n_eff_needed, "n_estimate": n_est,
            "check_jaccard": j_check, "check_n_eff": e_check, "check_runs": config.LIME_STAB_RUNS,
            "pooled_spearman_neff_jaccard": float(rho.statistic), "pooled_spearman_p": float(rho.pvalue)}


def _t3_5b(t33: dict, clients, sp, d, predict, model, bg: np.ndarray, X_train, names) -> dict:
    """Client and variable where LIME and SHAP disagree most (rank difference) and three tests of the cause.

    locality:    LIME coefficient of the variable for several kernel widths (10 seeds each);
    nonlinearity: P(good) and log-odds of the client when only that variable takes each value seen in training;
    interaction: mean effect of keeping the client's value, in log-odds and probability, as a function of how many of the
                 other 19 variables keep the client's value (the rest come from the SHAP background). Shapley averages
                 these sizes uniformly, so the mean over sizes must reproduce phi (a check of the test itself).
    """
    a = t33["a"]
    key, var = max(((k, n) for k in a for n in names),
                   key=lambda kn: (abs(a[kn[0]]["rank_lime"][kn[1]] - a[kn[0]]["rank_shap"][kn[1]]),
                                   abs(a[kn[0]]["lime_w"][kn[1]])))
    pos = clients[key]
    x = sp.X_test.iloc[pos].to_numpy()
    j = names.index(var)
    f_lo = lambda Z: credit.log_odds(model, pd.DataFrame(Z, columns=names))

    loc = []
    for sr in config.LIME_DISAGREE_SIGMA_REL:
        runs = [lt.explain(x, predict, X_train, names, sigma_rel=sr, seed=s) for s in range(config.LIME_DISAGREE_SEEDS)]
        w = np.array([r.full_coefs[j] for r in runs])
        loc.append({"sigma_rel": sr, "w_mean": float(w.mean()), "w_std": float(w.std()),
                    "rank_mean": float(np.mean([_ranks(r.full_coefs)[j] for r in runs])),
                    "n_eff_mean": float(np.mean([r.n_eff for r in runs])), "positive_runs": int((w > 0).sum())})

    vals = np.unique(X_train[:, j])
    grid = np.tile(x, (len(vals), 1))
    grid[:, j] = vals
    p_grid, f_grid = predict(grid), f_lo(grid)
    col = X_train[:, j]
    rep = np.tile(x, (len(col), 1))
    rep[:, j] = col  # client with this variable replaced by every training value (its empirical distribution)
    ice = {"values": [d.decode(var, v) for v in vals], "codes": vals.tolist(), "p_good": p_grid.tolist(),
           "log_odds": f_grid.tolist(), "client_value": d.decode(var, x[j]), "client_p": float(predict(x[None])[0]),
           "mean_p_replaced": float(predict(rep).mean()),
           "share_train_values_with_lower_p": float((predict(rep) < predict(x[None])[0]).mean()),
           "monotone": bool(np.all(np.diff(p_grid) >= -1e-12) or np.all(np.diff(p_grid) <= 1e-12))}

    rng = np.random.default_rng(config.SEED)
    others = np.array([i for i in range(len(names)) if i != j])
    eff_f, eff_p = [], []
    for k in range(len(others) + 1):
        Z0 = bg[rng.integers(len(bg), size=config.COALITION_SAMPLES)].copy()
        for r in range(len(Z0)):
            S = rng.choice(others, size=k, replace=False)
            Z0[r, S] = x[S]
        Z1 = Z0.copy()
        Z1[:, j] = x[j]
        eff_f.append(float((f_lo(Z1) - f_lo(Z0)).mean()))
        eff_p.append(float((predict(Z1) - predict(Z0)).mean()))
    return {"client": key, "test_position": pos, "feature": var, "rank_lime": a[key]["rank_lime"][var],
            "rank_shap": a[key]["rank_shap"][var], "lime_w": a[key]["lime_w"][var], "shap_phi": a[key]["shap_phi"][var],
            "shap_phi_probability": a[key]["shap_phi_probability"][var],
            "rank_shap_probability": int(_ranks(np.array([a[key]["shap_phi_probability"][n] for n in names]))[j]),
            "locality": loc, "ice": ice,
            "interaction": {"sizes": list(range(len(others) + 1)), "effect_log_odds": eff_f, "effect_probability": eff_p,
                            "samples_per_size": config.COALITION_SAMPLES,
                            "shapley_from_sizes_log_odds": float(np.mean(eff_f)),
                            "shapley_from_sizes_probability": float(np.mean(eff_p))}}


def _t3_5d(t33: dict, t34: dict, d, sp, model, pos: int) -> dict:
    """Top-3 |phi| of the rejected client against the variables changed by the counterfactuals, with the reason why a
    high-|phi| variable does not appear: best P(good) reachable by changing only that variable, and its cost."""
    names = d.feature_names
    phi = t33["a"]["rechazado"]["shap_phi"]
    top3 = sorted(names, key=lambda n: -abs(phi[n]))[:config.TOP_COMPARE]
    in_cf = sorted({c["feature"] for s in t34["selected"] for c in s["changes"]})
    X_train = sp.X_train.to_numpy()
    x = sp.X_test.iloc[pos].to_numpy()
    scale = cf.robust_scale(X_train, [names.index(n) for n in d.numeric])
    rows = []
    for n in top3:
        j = names.index(n)
        vals = np.unique(X_train[:, j][X_train[:, j] != x[j]])
        Z = np.tile(x, (len(vals), 1))
        Z[:, j] = vals
        p = credit.proba_good(model, pd.DataFrame(Z, columns=names))
        b = int(np.argmax(p))
        cost = abs(vals[b] - x[j]) / scale[j][0] if j in scale else 1.0
        rows.append({"feature": n, "phi": phi[n], "value": d.decode(n, x[j]), "in_counterfactuals": n in in_cf,
                     "role": "accionable" if n in config.CF_ACTIONABLE else "inmutable" if n in config.CF_IMMUTABLE
                     else "no accionable",
                     "best_single_value": d.decode(n, vals[b]), "best_single_p": float(p[b]), "best_single_cost": float(cost),
                     "mean_single_p": float(p.mean())})
    return {"top3_shap": rows, "cf_features": in_cf}
