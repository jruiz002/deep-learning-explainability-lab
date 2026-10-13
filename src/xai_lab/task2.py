"""Task 2 experiments (2.1 - 2.6): SHAP on German Credit with a gradient-boosting classifier."""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from . import config, credit
from .shapley import exact_shapley

TOP_N = 5


@dataclass
class Task2Run:
    """JSON-serialisable ``results`` plus the arrays / objects the figures need."""

    results: dict
    extra: dict = field(default_factory=dict)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def _rank_desc(v: np.ndarray) -> np.ndarray:
    """Rank 1 = most important (ties broken by position)."""
    order = np.argsort(-v, kind="stable")
    r = np.empty(len(v), dtype=int)
    r[order] = np.arange(1, len(v) + 1)
    return r


def _frame_f(model, names):
    """decision_function on a NumPy matrix, keeping the feature names the model was fitted with."""
    return lambda A: model.decision_function(pd.DataFrame(np.asarray(A, dtype=float), columns=names))


def _explain_client(base: credit.Base, phi_row: np.ndarray, row: pd.Series, p: float, f: float, pos: int) -> dict:
    d = base.data
    order = np.argsort(-np.abs(phi_row))
    return {
        "test_position": pos,
        "p_good": p,
        "log_odds": f,
        "phi": {n: float(v) for n, v in zip(d.feature_names, phi_row)},
        "values": {n: d.decode(n, row[n]) for n in d.feature_names},
        "top3": [{
            "feature": d.feature_names[j],
            "code": float(row[d.feature_names[j]]),
            "value": d.decode(d.feature_names[j], row[d.feature_names[j]]),
            "phi": float(phi_row[j]),
            "direction": "aumenta" if phi_row[j] > 0 else "reduce",
        } for j in order[:3]],
    }


def run_task2(seed: int = config.SEED) -> Task2Run:
    t_start = time.perf_counter()
    base = credit.build_base(seed)
    d, sp, model, bg = base.data, base.split, base.model, base.background
    names = d.feature_names
    n_feat = len(names)
    X_test, y_test = sp.X_test, sp.y_test
    res: dict = {"environment": {
        "seed": seed, "python": platform.python_version(), "sklearn": sklearn.__version__,
        "shap": shap.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__}}
    extra: dict = {"base": base}

    # ---- 2.1 ------------------------------------------------------------------------------
    f_test = credit.log_odds(model, X_test)
    p_test = credit.proba_good(model, X_test)
    res["t2_1"] = {
        "n_total": len(d.y), "n_train": len(sp.y_train), "n_test": len(y_test),
        "n_features": n_feat, "n_categorical": len(d.categorical), "n_numeric": len(d.numeric),
        "categorical": d.categorical, "numeric": d.numeric,
        "good_rate_total": float(d.y.mean()), "good_rate_train": float(sp.y_train.mean()),
        "good_rate_test": float(y_test.mean()),
        "n_estimators": config.GBC_N_ESTIMATORS, "max_depth": config.GBC_MAX_DEPTH,
        "test_accuracy": float(model.score(X_test, y_test)),
        "test_auc": float(roc_auc_score(y_test, p_test)),
    }

    # ---- 2.2 ------------------------------------------------------------------------------
    expl = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional", model_output="raw")
    phi = np.asarray(expl.shap_values(X_test, check_additivity=False))
    base_val = float(np.ravel(expl.expected_value)[0])
    eff_err = np.abs(base_val + phi.sum(axis=1) - f_test)
    mean_bg = float(credit.log_odds(model, bg).mean())
    I = np.abs(phi).mean(axis=0)
    perm = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                                  n_repeats=config.PERM_REPEATS, random_state=seed)
    rho = spearmanr(I, perm.importances_mean)
    rk_I, rk_perm = _rank_desc(I), _rank_desc(perm.importances_mean)
    top = lambda v: [{"feature": names[j], "value": float(v[j])} for j in np.argsort(-v, kind="stable")[:TOP_N]]
    sel = credit.select_clients(p_test)
    clients = {}
    for key in ("aprobado", "rechazado"):
        pos = sel[key]
        clients[key] = _explain_client(base, phi[pos], X_test.iloc[pos], float(p_test[pos]), float(f_test[pos]), pos)
    res["t2_2"] = {
        "a_max_efficiency_error": float(eff_err.max()),
        "b_base_value": base_val, "b_mean_bg_log_odds": mean_bg, "b_abs_diff": abs(base_val - mean_bg),
        "c_importance_shap": {n: float(v) for n, v in zip(names, I)},
        "c_importance_perm_mean": {n: float(v) for n, v in zip(names, perm.importances_mean)},
        "c_importance_perm_std": {n: float(v) for n, v in zip(names, perm.importances_std)},
        "c_top5_shap": top(I), "c_top5_perm": top(perm.importances_mean),
        "c_spearman": float(rho.statistic), "c_spearman_p": float(rho.pvalue),
        "c_top5_overlap": len({r["feature"] for r in top(I)} & {r["feature"] for r in top(perm.importances_mean)}),
        "c_rank_shap": {n: int(r) for n, r in zip(names, rk_I)},
        "c_rank_perm": {n: int(r) for n, r in zip(names, rk_perm)},
        "d_selected": sel, "d_clients": clients,
        # direction of each numeric feature: Spearman between its value and its contribution (test set)
        "d_value_phi_spearman": {n: float(spearmanr(X_test[n], phi[:, names.index(n)]).statistic) for n in d.numeric},
    }
    extra.update(phi=phi, expl=expl, I=I, perm=perm, base_val=base_val)

    # ---- 2.3 ------------------------------------------------------------------------------
    num_idx = [names.index(n) for n in d.numeric]
    chosen = names[max(num_idx, key=lambda j: I[j])]
    copy_name = f"{chosen}_copy"
    rng = np.random.default_rng(seed)
    sigma = float(d.X[chosen].std())
    dup = d.X[chosen].to_numpy() + config.DUP_NOISE_SCALE * sigma * rng.standard_normal(len(d.y))
    X_aug = d.X.copy()
    X_aug[copy_name] = dup
    Xtr2, Xte2 = X_aug.iloc[sp.idx_train], X_aug.iloc[sp.idx_test]
    model2 = credit.train_gbc(Xtr2, sp.y_train, seed)
    bg2 = Xtr2.loc[bg.index]
    names2 = list(X_aug.columns)
    jo, jc = names2.index(chosen), names2.index(copy_name)
    cfg = {}
    e_int = shap.TreeExplainer(model2, data=bg2, feature_perturbation="interventional", model_output="raw")
    e_path = shap.TreeExplainer(model2, feature_perturbation="tree_path_dependent", model_output="raw")
    for key, e in (("interventional", e_int), ("tree_path_dependent", e_path)):
        ph = np.asarray(e.shap_values(Xte2, check_additivity=False))
        a_o, a_c = np.abs(ph[:, jo]).mean(), np.abs(ph[:, jc]).mean()
        cfg[key] = {
            "mean_abs_orig": float(a_o),
            "mean_abs_copy": float(a_c),
            "mean_abs_sum": float(np.abs(ph[:, jo] + ph[:, jc]).mean()),
            "share_orig": float(a_o / (a_o + a_c)),
            "corr_phi_orig_copy": float(np.corrcoef(ph[:, jo], ph[:, jc])[0, 1]),
        }
        extra[f"phi_dup_{key}"] = ph
    imp2 = np.abs(extra["phi_dup_interventional"]).mean(axis=0)
    res["t2_3"] = {
        "feature": chosen, "copy": copy_name, "sigma": sigma,
        "corr_orig_copy": float(np.corrcoef(d.X[chosen], dup)[0, 1]),
        "test_accuracy_21": float(model2.score(Xte2, y_test)),
        "test_auc_21": float(roc_auc_score(y_test, model2.predict_proba(Xte2)[:, 1])),
        "I_orig_without_copy": float(I[names.index(chosen)]),
        "mean_abs_other_features_21": float(np.delete(imp2, [jo, jc]).mean()),
        "configs": cfg,
    }

    # ---- 2.4 ------------------------------------------------------------------------------
    Xk = X_test.iloc[: config.KERNEL_N_INSTANCES]
    kern = []
    for k in config.BACKGROUND_SIZES_KERNEL:
        bgk = bg.iloc[:k]
        counter = {"n": 0}
        base_f = _frame_f(model, names)

        def f_counted(A, _c=counter, _f=base_f):
            _c["n"] += len(A)
            return _f(A)

        ke = shap.KernelExplainer(f_counted, bgk.to_numpy())
        counter["n"] = 0
        np.random.seed(seed)  # KernelExplainer draws its coalitions from the global NumPy state
        t0 = time.perf_counter()
        phi_k = np.asarray(ke.shap_values(Xk.to_numpy(), nsamples=config.KERNEL_NSAMPLES, silent=True))
        t_kernel = time.perf_counter() - t0
        te = shap.TreeExplainer(model, data=bgk, feature_perturbation="interventional", model_output="raw")
        t0 = time.perf_counter()
        phi_t = np.asarray(te.shap_values(Xk, check_additivity=False))
        t_tree = time.perf_counter() - t0
        n_inst = len(Xk)
        kern.append({
            "k": k, "mae": float(np.abs(phi_k - phi_t).mean()), "max_abs_error": float(np.abs(phi_k - phi_t).max()),
            "time_kernel_s": t_kernel, "time_tree_s": t_tree,
            "time_kernel_per_instance_s": t_kernel / n_inst,
            "model_evaluations": int(counter["n"]), "evaluations_nominal": config.KERNEL_NSAMPLES * k * n_inst,
            "evaluations_per_instance": counter["n"] / n_inst,
            "time_per_evaluation_us": 1e6 * t_kernel / counter["n"],
            "speedup_tree_vs_kernel": t_kernel / t_tree,
        })
    res["t2_4"] = {"n_instances": len(Xk), "nsamples": config.KERNEL_NSAMPLES, "by_background": kern}

    # ---- 2.5 ------------------------------------------------------------------------------
    cols = list(config.SMALL_MODEL_FEATURES)
    Xtr_s, Xte_s = sp.X_train[cols], sp.X_test[cols]
    model_s = credit.train_gbc(Xtr_s, sp.y_train, seed)
    bg_s = credit.background(Xtr_s, config.SMALL_MODEL_BACKGROUND, seed)
    f_s = _frame_f(model_s, cols)
    te_s = shap.TreeExplainer(model_s, data=bg_s, feature_perturbation="interventional", model_output="raw")
    n_s = config.SMALL_MODEL_INSTANCES
    own, v0s, vns = [], [], []
    t0 = time.perf_counter()
    for i in range(n_s):
        ph, v0, vn = exact_shapley(f_s, Xte_s.iloc[i].to_numpy(), bg_s.to_numpy())
        own.append(ph)
        v0s.append(v0)
        vns.append(vn)
    t_own = time.perf_counter() - t0
    own = np.array(own)
    ref = np.asarray(te_s.shap_values(Xte_s.iloc[:n_s], check_additivity=False))
    fx = model_s.decision_function(Xte_s.iloc[:n_s])
    res["t2_5"] = {
        "features": cols, "n_coalitions": 2 ** len(cols), "n_instances": n_s, "n_background": len(bg_s),
        "test_accuracy": float(model_s.score(Xte_s, y_test)),
        "test_auc": float(roc_auc_score(y_test, credit.proba_good(model_s, Xte_s))),
        "max_abs_error_vs_treeshap": float(np.abs(own - ref).max()),
        "max_efficiency_error": float(np.abs(own.sum(axis=1) - (np.array(vns) - np.array(v0s))).max()),
        "max_vN_vs_f_error": float(np.abs(np.array(vns) - fx).max()),
        "max_v0_vs_expected_error": float(np.abs(np.array(v0s) - float(np.ravel(te_s.expected_value)[0])).max()),
        "time_own_s": t_own,
        "phi_own": own.tolist(), "phi_treeshap": ref.tolist(), "v_empty": v0s, "v_full": vns,
        "f_x": fx.tolist(),
    }
    extra.update(own=own, ref=ref)

    # ---- 2.6 ------------------------------------------------------------------------------
    # (a) rejected client: additivity holds in log-odds, not in probability
    rc = clients["rechazado"]
    pos = rc["test_position"]
    phi_r = phi[pos]
    s0 = float(_sigmoid(base_val))
    sj = _sigmoid(base_val + phi_r) - s0
    p_x = rc["p_good"]
    res["t2_6"] = {"a": {
        "test_position": pos, "p": p_x, "log_odds": rc["log_odds"], "phi0": base_val, "sigma_phi0": s0,
        "sum_phi": float(phi_r.sum()), "sigma_effects": {n: float(v) for n, v in zip(names, sj)},
        "sum_sigma_effects": float(sj.sum()), "p_minus_sigma_phi0": p_x - s0,
        "gap": float(p_x - s0 - sj.sum()),
    }}
    # (b) largest rank disagreement between the two importance measures
    diff = np.abs(rk_I - rk_perm)
    jb = int(np.argmax(diff))
    fb = names[jb]
    corr = sp.X_train.corr(method="spearman")[fb].drop(fb)
    top_corr = corr.reindex(corr.abs().sort_values(ascending=False).index)[:3]
    abs_phi = np.abs(phi[:, jb])
    srt = np.sort(abs_phi)[::-1]
    n50 = int(np.searchsorted(np.cumsum(srt), 0.5 * srt.sum()) + 1)
    res["t2_6"]["b"] = {
        "feature": fb, "rank_shap": int(rk_I[jb]), "rank_perm": int(rk_perm[jb]), "rank_gap": int(diff[jb]),
        "I": float(I[jb]), "perm_mean": float(perm.importances_mean[jb]), "perm_std": float(perm.importances_std[jb]),
        "top_corr": [{"feature": n, "spearman": float(v)} for n, v in top_corr.items()],
        "max_abs_corr": float(top_corr.abs().max()),
        "n_clients_50pct": n50, "frac_clients_50pct": n50 / len(abs_phi),
        "is_categorical": fb in d.categorical,
        "n_categories": int(d.X[fb].nunique()),
        "ties_in_rank_gap": int((diff == diff.max()).sum()),
    }
    # (d) cost model
    xb = bg.to_numpy()
    f_all = _frame_f(model, names)
    f_all(xb)  # warm-up
    reps = []
    for _ in range(30):
        t0 = time.perf_counter()
        f_all(xb)
        reps.append(time.perf_counter() - t0)
    t100 = float(np.median(reps))
    k100 = next(r for r in kern if r["k"] == 100)
    exact20 = 2 ** 20 * t100
    res["t2_6"]["d"] = {
        "t_decision_function_100_s": t100, "n_repeats": len(reps),
        "exact_20_vars_s": exact20, "exact_20_vars_h": exact20 / 3600,
        "kernel_per_instance_s_k100": k100["time_kernel_per_instance_s"],
        "exact_over_kernel": exact20 / k100["time_kernel_per_instance_s"],
        "years_60_vars_1us": 2 ** 60 * 1e-6 / (365.25 * 24 * 3600),
        "by_background": [{"k": r["k"], "mae": r["mae"], "time_per_evaluation_us": r["time_per_evaluation_us"],
                           "evaluations": r["model_evaluations"], "time_kernel_s": r["time_kernel_s"]} for r in kern],
    }
    res["elapsed_s"] = time.perf_counter() - t_start
    return Task2Run(res, extra)
