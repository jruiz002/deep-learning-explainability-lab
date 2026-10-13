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
