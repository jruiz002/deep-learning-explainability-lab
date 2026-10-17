"""Figures for Task 4 (deletion / insertion and cascading randomisation). One function per figure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import config, faithfulness as fa
from .viz import ACCENT, CMAP, INK, MUTED, _img, _save, plt
from .viz_tabular import ORANGE


def _curve(ax, mean_map, r_mean, r_std, title: str, auc_map: float, r: dict) -> None:
    ks = np.arange(len(mean_map)) / config.PERTURB_STEPS
    r_mean, r_std = np.asarray(r_mean), np.asarray(r_std)
    ax.fill_between(ks, r_mean - r_std, r_mean + r_std, color=MUTED, alpha=0.25, lw=0)
    ax.plot(ks, r_mean, color=MUTED, lw=1.2, label=f"azar: AUC {r['random_mean']:.3f} ± {r['random_std']:.3f}")
    ax.plot(ks, mean_map, color=ACCENT, marker="o", ms=2.5, lw=1.4, label=f"Grad-CAM: AUC {auc_map:.3f}")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("fracción de píxeles (k / P)")
    ax.set_ylabel("probabilidad de la clase")
    ax.set_title(title, fontsize=8.5)
    ax.legend(frameon=False, fontsize=7)


def fig_curves(results: dict, out: Path) -> Path:
    t = results["t4_1"]
    fig, axes = plt.subplots(len(t), 2, figsize=(9.5, 3.0 * len(t)))
    for row, (rid, r) in zip(axes, t.items()):
        c = r["curves"]
        _curve(row[0], c["deletion_map"], c["deletion_random_mean"], c["deletion_random_std"],
               f"({rid}) {r['class_name']}: deletion (gris)", r["deletion"]["map"], r["deletion"])
        _curve(row[1], c["insertion_map"], c["insertion_random_mean"], c["insertion_random_std"],
               f"({rid}) {r['class_name']}: insertion (desde desenfoque σ = {config.BLUR_SIGMA:g})", r["insertion"]["map"], r["insertion"])
    fig.tight_layout()
    return _save(fig, out)


def fig_frames(prep, saliency: torch.Tensor, out: Path, steps_shown=(0, 2, 5, 10, 15)) -> Path:
    """Deletion and insertion frames of one image with the Grad-CAM order and with one random order."""
    x01 = fa.to_unit(prep.crop)
    blurred = fa.gaussian_blur(x01, config.BLUR_SIGMA)
    order = fa.order_from_map(saliency)
    rnd = fa.random_order(order.numel(), config.SEED)
    rows = [("deletion, Grad-CAM", x01, fa.gray_like(x01), order), ("deletion, azar", x01, fa.gray_like(x01), rnd),
            ("insertion, Grad-CAM", blurred, x01, order), ("insertion, azar", blurred, x01, rnd)]
    fig, axes = plt.subplots(len(rows), len(steps_shown), figsize=(2.0 * len(steps_shown), 2.1 * len(rows)))
    for ax_row, (label, start, end, o) in zip(axes, rows):
        frames = fa.perturbation_images(start, end, o, config.PERTURB_STEPS)
        for ax, k in zip(ax_row, steps_shown):
            _img(ax, frames[k].permute(1, 2, 0).clamp(0, 1).numpy(), f"{label}\nk = {k} ({k / config.PERTURB_STEPS:.0%})")
            ax.title.set_fontsize(7)
    fig.tight_layout()
    return _save(fig, out)


def fig_blur_reference(results: dict, out: Path) -> Path:
    t = results["t4_3a_blur_reference"]
    g = results["t4_1"]["a"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    _curve(axes[0], g["curves"]["deletion_map"], g["curves"]["deletion_random_mean"], g["curves"]["deletion_random_std"],
           "(a) deletion con gris", g["deletion"]["map"], g["deletion"])
    c = t["curves"]
    _curve(axes[1], c["deletion_map"], c["deletion_random_mean"], c["deletion_random_std"],
           f"(a) deletion con referencia desenfocada (σ = {config.BLUR_SIGMA:g})", t["deletion"]["map"], t["deletion"])
    fig.tight_layout()
    return _save(fig, out)


def fig_cascade_maps(results: dict, extra: dict, out: Path) -> Path:
    stages = results["t4_2"]["stages"]
    cols = ["original"] + [f"+ {s['randomized'][-1]}" for s in stages]
    maps = extra["cascade_maps"]
    fig, axes = plt.subplots(len(maps), len(cols), figsize=(1.75 * len(cols), 1.95 * len(maps)))
    for ax_row, (rid, ms) in zip(axes, maps.items()):
        for j, (ax, m) in enumerate(zip(ax_row, ms)):
            m = m.numpy()
            ax.imshow(m / m.max() if m.max() > 0 else m, cmap=CMAP, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"({rid}) {results['t4_1'][rid]['class_name']}", fontsize=7.5)
                ax.set_title(cols[0], fontsize=7.5)
            else:
                rho = stages[j - 1]["images"][rid]["spearman"]
                ax.set_title(f"{cols[j]}\nρ = {rho:.2f}" if rho == rho else f"{cols[j]}\nρ indefinido", fontsize=7.5)
    fig.tight_layout()
    return _save(fig, out)


def fig_cascade_spearman(results: dict, out: Path) -> Path:
    stages = results["t4_2"]["stages"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    xs = [s["stage"] for s in stages]
    for rid, c in zip(stages[0]["images"], [ACCENT, ORANGE, INK]):
        ax.plot(xs, [s["images"][rid]["spearman"] for s in stages], marker="o", color=c,
                label=f"({rid}) {results['t4_1'][rid]['class_name']}")
    ax.axhline(config.CASCADE_SPEARMAN_LOW, color=MUTED, ls="--", lw=1, label=f"ρ = {config.CASCADE_SPEARMAN_LOW:g}")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks(xs, [f"{s['stage']}\n+{s['randomized'][-1]}" for s in stages], fontsize=7.5)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("etapa de la cascada (capa reinicializada que se agrega)")
    ax.set_ylabel("Spearman con el mapa original (7×7)")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    return _save(fig, out)
