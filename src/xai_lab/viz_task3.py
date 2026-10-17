"""Figures for Task 3 (LIME). One function per figure; each saves a PNG and returns its path."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .viz import ACCENT, CMAP, INK, MUTED, _save, plt
from .viz_tabular import ORANGE


def fig_lime_coefs(results: dict, out: Path) -> Path:
    clients = results["t3_1"]
    fig, axes = plt.subplots(1, len(clients), figsize=(4.4 * len(clients), 3.6))
    for ax, (key, c) in zip(np.atleast_1d(axes), clients.items()):
        top = c["top"][::-1]  # largest |w| at the top
        coef = [t["coef"] for t in top]
        ax.barh(range(len(top)), coef, color=[ACCENT if v >= 0 else ORANGE for v in coef], height=0.65)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(range(len(top)), [f"{t['feature']} = {t['value']}" for t in top], fontsize=8)
        ax.set_xlabel("coeficiente (cambio en P(good))")
        ax.set_title(f"{key} (p = {c['p_good']:.3f}, $R^2_w$ = {c['r2w']:.3f})", fontsize=9)
    fig.tight_layout()
    return _save(fig, out)


def _heatmap(ax, M, rows, cols, title: str, fmt: str) -> None:
    M = np.asarray(M)
    im = ax.imshow(M, cmap=CMAP, aspect="auto")
    ax.set_xticks(range(len(cols)), [f"{c:g}" for c in cols])
    ax.set_yticks(range(len(rows)), [f"{r:,}" for r in rows])
    ax.set_xlabel("$\\sigma_{rel}$")
    ax.set_ylabel("N")
    ax.set_title(title, fontsize=9)
    mid = (M.min() + M.max()) / 2
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, format(M[i, j], fmt), ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] < mid else INK)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def fig_stability(results: dict, out: Path) -> Path:
    t = results["t3_2"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    _heatmap(axes[0], t["jaccard"], t["n_values"], t["sigma_rel_values"], "Jaccard medio (190 pares)", ".3f")
    _heatmap(axes[1], t["n_eff"], t["n_values"], t["sigma_rel_values"], "$N_{ef}$ medio (20 corridas)", ".0f")
    fig.tight_layout()
    return _save(fig, out)


def fig_lime_vs_shap(results: dict, out: Path) -> Path:
    """3.3: |w| of LIME (20 variables) against |phi| of TreeSHAP per client, plus R^2_w as a function of K."""
    a, b = results["t3_3"]["a"], results["t3_3"]["b"]
    fig, axes = plt.subplots(1, len(a) + 1, figsize=(4.0 * (len(a) + 1), 3.7))
    for ax, (key, c) in zip(axes, a.items()):
        names = list(c["lime_w"])
        xs = np.array([abs(c["shap_phi"][n]) for n in names])
        ys = np.array([abs(c["lime_w"][n]) for n in names])
        ax.scatter(xs, ys, s=16, color=ACCENT, alpha=0.75)
        for n in set(c["top_lime"]) | set(c["top_shap"]):
            i = names.index(n)
            ax.annotate(n, (xs[i], ys[i]), fontsize=7, xytext=(3, 2), textcoords="offset points", color=INK)
        ax.set_xlabel("|φ| TreeSHAP (log-odds)")
        ax.set_ylabel("|w| LIME (P(good))")
        ax.set_title(f"{key}: Spearman = {c['spearman']:.3f}, top-3 común = {c['top_overlap']}", fontsize=8.5)
    ks = [r["k"] for r in b["by_k"]]
    r2 = [r["r2w"] for r in b["by_k"]]
    ax = axes[-1]
    ax.plot(ks, r2, marker="o", color=ACCENT)
    for k, v in zip(ks, r2):
        ax.annotate(f"{v:.3f}", (k, v), fontsize=8, xytext=(0, 6), textcoords="offset points", ha="center")
    ax.set_xticks(ks)
    ax.set_ylim(0, max(r2) * 1.3)
    ax.set_xlabel("K (variables del modelo lineal)")
    ax.set_ylabel("$R^2_w$")
    ax.set_title("Cliente rechazado: fidelidad según K", fontsize=8.5)
    fig.tight_layout()
    return _save(fig, out)
