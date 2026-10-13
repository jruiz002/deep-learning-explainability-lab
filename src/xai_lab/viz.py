"""Figures for Task 1. One function per figure; each saves a PNG and returns its path."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from . import config  # noqa: E402
from .model import CATEGORIES  # noqa: E402

CMAP = "inferno"  # perceptually uniform, single sequential scale for all heat maps
INK, MUTED, ACCENT = "#1f2933", "#7b8794", "#2f6fb0"
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _img(ax, crop, title: str | None = None) -> None:
    ax.imshow(crop)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)


def _box(ax, box, color="#39d353") -> None:
    x0, y0, x1, y1 = box
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=2))


def fig_predictions(runs, out: Path) -> Path:
    fig, axes = plt.subplots(len(runs), 2, figsize=(9, 3.2 * len(runs)), gridspec_kw={"width_ratios": [1, 1.6]})
    for row, (rid, r) in zip(axes, runs.items()):
        _img(row[0], r.prepared.crop, f"Image ({rid}) - 224x224 input")
        if r.prepared.bbox_224:
            _box(row[0], r.prepared.bbox_224)
        names = [p.name for p in r.top5][::-1]
        probs = [p.probability for p in r.top5][::-1]
        row[1].barh(names, probs, color=ACCENT, height=0.55)
        for y, p in enumerate(probs):
            row[1].text(p + 0.01, y, f"{p:.3f}", va="center", fontsize=8)
        row[1].set_xlim(0, 1.1)
        row[1].set_xlabel("probability")
        row[1].set_title("Top-5 ImageNet classes", fontsize=9)
    fig.tight_layout()
    return _save(fig, out)


def fig_gradcam(runs, out: Path) -> Path:
    rows = []
    for rid, r in runs.items():
        for c, gc in r.gradcams.items():
            rows.append((rid, r, gc))
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
    for ax_row, (rid, r, gc) in zip(axes, rows):
        label = f"({rid}) {CATEGORIES[gc.class_idx]}"
        _img(ax_row[0], r.prepared.crop, f"{label}: input")
        ax_row[1].imshow(gc.cam_224, cmap=CMAP, vmin=0, vmax=1)
        ax_row[1].axis("off")
        ax_row[1].set_title("Grad-CAM (normalised)", fontsize=9)
        _img(ax_row[2], r.prepared.crop, "overlay")
        ax_row[2].imshow(gc.cam_224, cmap=CMAP, alpha=0.5, vmin=0, vmax=1)
    fig.tight_layout()
    return _save(fig, out)


def fig_cam_vs_gradcam(runs, out: Path) -> Path:
    fig, axes = plt.subplots(len(runs), 4, figsize=(12, 3 * len(runs)))
    for ax_row, (rid, r) in zip(axes, runs.items()):
        c = r.cam
        panels = [
            (c["cam"].map, "CAM  $M^c$"),
            (c["z_times_L"], "Grad-CAM  $Z \\cdot L^c$ (no ReLU / norm.)"),
            (c["c_abs_diff_map"], f"|difference|  (max {c['c_max_abs_error']:.1e})"),
            (c["d_norm_cam"] - c["d_norm_gradcam"], f"normalised diff (max {c['d_max_abs_error']:.1e})"),
        ]
        for ax, (m, t) in zip(ax_row, panels):
            im = ax.imshow(m.numpy(), cmap=CMAP)
            ax.set_title(f"({rid}) {t}", fontsize=8)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return _save(fig, out)


def fig_alpha_vs_w(run_a, out: Path) -> Path:
    gc = run_a.gradcams[run_a.extra["classes"][0]]
    from_w = (run_a.cam["cam"].weights / config.Z).numpy()
    alpha = gc.alpha.numpy()
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(from_w, alpha, s=8, color=ACCENT, alpha=0.6, label="512 channels")
    lim = [min(from_w.min(), alpha.min()), max(from_w.max(), alpha.max())]
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--", label="y = x")
    top = [r["channel"] for r in run_a.extra["t14a"]]
    ax.scatter(from_w[top], alpha[top], s=28, facecolor="none", edgecolor=INK, lw=1.2, label="top-5 weights")
    ax.set_xlabel("$w^c_k / Z$")
    ax.set_ylabel("$\\alpha^c_k$ (Grad-CAM)")
    ax.set_title("Image (a): $\\alpha^c_k$ vs $w^c_k/Z$", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, out)


def fig_iou(run_b, out: Path) -> Path:
    t = run_b.extra["t14b"]
    m1, m2 = t["mask1"].numpy(), t["mask2"].numpy()
    c1, c2 = t["class1"], t["class2"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    _img(axes[0], run_b.prepared.crop, "Input")
    for ax, m, c in ((axes[1], m1, c1), (axes[2], m2, c2)):
        _img(ax, run_b.prepared.crop, f"Top 20 % of Grad-CAM: {CATEGORIES[c]}")
        ov = np.zeros((*m.shape, 4))
        ov[m] = (0.97, 0.45, 0.09, 0.55)
        ax.imshow(ov)
    fig.suptitle(f"IoU between the two top-20 % sets = {t['iou']:.4f}", fontsize=10)
    fig.tight_layout()
    return _save(fig, out)


def fig_causal(run_c, out: Path) -> Path:
    from .causal import gray_image

    t = run_c.extra["t14c"]
    gc = run_c.gradcams[t["class_idx"]]
    crop, box = run_c.prepared.crop, t["bbox_224"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    _img(axes[0], crop, f"Original  p({t['class_name']}) = {t['p_original']:.3f}")
    _box(axes[0], box)
    _img(axes[1], crop, f"Grad-CAM; {t['mass_in_box']:.1%} of mass inside box")
    axes[1].imshow(gc.cam_224, cmap=CMAP, alpha=0.55, vmin=0, vmax=1)
    _box(axes[1], box)
    _img(axes[2], gray_image(crop, box, "inside"), f"Background gray  p = {t['p_background_gray']:.3f}")
    _img(axes[3], gray_image(crop, box, "outside"), f"Object gray  p = {t['p_object_gray']:.4f}")
    fig.tight_layout()
    return _save(fig, out)
