"""Figures for Task 2 (SHAP on German Credit). One function per figure; each saves a PNG and returns its path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import shap

from . import viz  # noqa: F401  (sets the Agg backend and the shared matplotlib style)
from .viz import ACCENT, INK, MUTED, _save, plt

ORANGE = "#d9822b"


def _explanation(run, pos: int) -> shap.Explanation:
    """Explanation of one test client; data shows decoded category labels instead of ordinal codes."""
    base = run.extra["base"]
    d = base.data
    row = base.split.X_test.iloc[pos]
    shown = np.array([d.decode(n, row[n]) for n in d.feature_names], dtype=object)
    return shap.Explanation(values=run.extra["phi"][pos], base_values=run.extra["base_val"],
                            data=shown, feature_names=d.feature_names)


def fig_waterfall(run, key: str, out: Path) -> Path:
    c = run.results["t2_2"]["d_clients"][key]
    shap.plots.waterfall(_explanation(run, c["test_position"]), max_display=10, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 4.8)
    fig.suptitle(f"Cliente {key} (posición {c['test_position']} del test): p(good) = {c['p_good']:.3f}", fontsize=10, y=1.0)
    return _save(fig, out)


def fig_beeswarm(run, out: Path) -> Path:
    base = run.extra["base"]
    ex = shap.Explanation(values=run.extra["phi"], base_values=np.full(len(run.extra["phi"]), run.extra["base_val"]),
                          data=base.split.X_test.to_numpy(), feature_names=base.data.feature_names)
    shap.plots.beeswarm(ex, max_display=20, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 6.5)
    return _save(fig, out)


def fig_importance(run, out: Path) -> Path:
    t = run.results["t2_2"]
    names = list(t["c_importance_shap"])
    order = sorted(names, key=lambda n: -t["c_importance_shap"][n])
    y = np.arange(len(order))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    axes[0].barh(y, [t["c_importance_shap"][n] for n in order], color=ACCENT, height=0.7)
    axes[0].set_title("$I_j$ = media de |$\\phi_j$| (log-odds)", fontsize=9)
    axes[1].barh(y, [t["c_importance_perm_mean"][n] for n in order], xerr=[t["c_importance_perm_std"][n] for n in order],
                 color=ORANGE, height=0.7, ecolor=INK, error_kw={"lw": 0.8})
    axes[1].axvline(0, color=MUTED, lw=0.8)
    axes[1].set_title("Permutación: caída del AUC (media ± desv.)", fontsize=9)
    axes[0].set_yticks(y, [f"{n} (#{t['c_rank_perm'][n]})" for n in order])
    axes[0].set_ylabel("variable (entre paréntesis, el rango por permutación)")
    fig.suptitle(f"Spearman entre ambos vectores = {t['c_spearman']:.3f}", fontsize=10)
    fig.tight_layout()
    return _save(fig, out)


def fig_duplicate(run, out: Path) -> Path:
    t = run.results["t2_3"]
    labels = ["interventional", "tree_path_dependent"]
    series = [("mean|$\\phi$| original", "mean_abs_orig", ACCENT), ("mean|$\\phi$| copia", "mean_abs_copy", ORANGE),
              ("mean|$\\phi_{orig}+\\phi_{copia}$|", "mean_abs_sum", INK)]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    w = 0.2
    for i, (lab, key, col) in enumerate(series):
        vals = [t["configs"][c][key] for c in labels]
        bars = ax.bar(np.arange(2) + (i - 1.5) * w, vals, w, label=lab, color=col)
        ax.bar_label(bars, fmt="%.3f", fontsize=7)
    ax.bar(np.arange(2) + 1.5 * w, [t["I_orig_without_copy"]] * 2, w, label="$I_j$ original, modelo sin copia", color="none",
           edgecolor=MUTED, hatch="//")
    ax.set_xticks(np.arange(2), labels)
    ax.set_ylabel("log-odds")
    ax.set_title(f"Variable duplicada: {t['feature']} (copia con 5 % de ruido)", fontsize=9)
    ax.set_ylim(0, 1.35 * max(t["configs"][c]["mean_abs_sum"] for c in labels))
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center")
    return _save(fig, out)


def fig_kernel(run, out: Path) -> Path:
    rows = run.results["t2_4"]["by_background"]
    ks = [r["k"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(ks, [r["mae"] for r in rows], "o-", color=ACCENT)
    axes[0].set_xlabel("tamaño del fondo k")
    axes[0].set_ylabel("MAE contra TreeSHAP (log-odds)")
    axes[0].set_xscale("log")
    axes[0].set_xticks(ks, [str(k) for k in ks])
    axes[1].plot(ks, [r["time_kernel_s"] for r in rows], "o-", color=ORANGE, label="KernelSHAP")
    axes[1].plot(ks, [r["time_tree_s"] for r in rows], "s-", color=ACCENT, label="TreeSHAP")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xticks(ks, [str(k) for k in ks])
    axes[1].set_xlabel("tamaño del fondo k")
    axes[1].set_ylabel(f"tiempo para {run.results['t2_4']['n_instances']} instancias (s)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    return _save(fig, out)
