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

    res["elapsed_s"] = time.perf_counter() - t_start
    return Task2Run(res, extra)
