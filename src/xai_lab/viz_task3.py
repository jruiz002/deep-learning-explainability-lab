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
