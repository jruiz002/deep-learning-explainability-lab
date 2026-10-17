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


def fig_counterfactuals(results: dict, out: Path) -> Path:
    """3.4: P(good) of the rejected client and of the three counterfactuals, with the changes of each one."""
    t = results["t3_4"]
    labels = ["original"] + [
        "\n".join(f"{c['feature']}: {c['original']} → {c['new']}" for c in s["changes"]) + f"\n(d = {s['distance']:.2f})"
        for s in t["selected"]]
    ps = [t["p_good_original"]] + [s["p_good"] for s in t["selected"]]
    fig, ax = plt.subplots(figsize=(8.5, 1.15 * len(ps) + 0.8))
    y = np.arange(len(ps))[::-1]
    ax.barh(y, ps, color=[MUTED] + [ACCENT] * (len(ps) - 1), height=0.6)
    ax.axvline(t["threshold"], color=ORANGE, ls="--", lw=1.2, label=f"θ = {t['threshold']:g}")
    for yi, p in zip(y, ps):
        ax.text(p + 0.01, yi, f"{p:.3f}", va="center", fontsize=8)
    ax.set_yticks(y, labels, fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(good)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title(f"Contrafactuales del cliente rechazado ({t['n_valid']} válidos de {t['n_candidates']:,})", fontsize=9)
    fig.tight_layout()
    return _save(fig, out)


def fig_jaccard_neff(results: dict, out: Path) -> Path:
    """3.5(a): mean Jaccard against mean N_ef for the 9 cells of 3.2, the fitted curve and the verification run."""
    t2, a = results["t3_2"], results["t3_5"]["a"]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    colors = [MUTED, ACCENT, ORANGE]
    for j, (sr, c) in enumerate(zip(t2["sigma_rel_values"], colors)):
        E = [row[j] for row in t2["n_eff"]]
        J = [row[j] for row in t2["jaccard"]]
        ax.plot(E, J, marker="o", color=c, lw=1, label=f"σ_rel = {sr:g} (3.2)")
    e = np.linspace(*a["bracket_n_eff"], 100)  # the fit is only used between the two bracketing cells
    ax.plot(e, 1 - a["fit_a"] * e ** (-a["fit_b"]), color=ACCENT, ls=":", lw=1.2,
            label=f"ajuste 1 − J = a·N_ef^(−{a['fit_b']:.2f}) (entre N = {a['bracket_n'][0]} y {a['bracket_n'][1]})")
    ax.axhline(a["target"], color=INK, lw=0.8, ls="--")
    ax.scatter([a["check_n_eff"]], [a["check_jaccard"]], marker="*", s=160, color=INK, zorder=5,
               label=f"verificación N = {a['n_estimate']}: J = {a['check_jaccard']:.3f}")
    ax.set_ylim(0.6, 1.0)
    ax.set_xlabel("$N_{ef}$ medio")
    ax.set_ylabel("Jaccard medio (top-5)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_title("Estabilidad de LIME frente al tamaño efectivo de muestra", fontsize=9)
    fig.tight_layout()
    return _save(fig, out)


def fig_disagreement(results: dict, out: Path) -> Path:
    """3.5(b): the three tests of the largest LIME-SHAP disagreement (non-linearity, interaction, locality)."""
    b = results["t3_5"]["b"]
    var, ice, it, loc = b["feature"], b["ice"], b["interaction"], b["locality"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    ax = axes[0]
    xs = np.arange(len(ice["values"]))
    ax.step(xs, ice["p_good"], where="mid", color=ACCENT)
    ci = ice["values"].index(ice["client_value"])
    ax.scatter([ci], [ice["p_good"][ci]], color=ORANGE, zorder=5, label=f"valor del cliente ({ice['client_value']})")
    ax.axhline(ice["mean_p_replaced"], color=MUTED, ls="--", lw=1, label=f"media con valores de entrenamiento ({ice['mean_p_replaced']:.3f})")
    step = max(1, len(xs) // 12)
    ax.set_xticks(xs[::step], ice["values"][::step], fontsize=7, rotation=45)
    ax.set_xlabel(f"{var} (sólo cambia esta variable)")
    ax.set_ylabel("P(good)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(f"Cliente {b['client']}: P(good) al variar {var}", fontsize=8.5)

    ax = axes[1]
    ax.plot(it["sizes"], it["effect_log_odds"], marker="o", ms=3, color=ACCENT, label="log-odds")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axhline(b["shap_phi"], color=ACCENT, ls=":", lw=1, label=f"φ TreeSHAP = {b['shap_phi']:+.3f}")
    ax.set_xlabel("otras variables con el valor del cliente (de 19)")
    ax.set_ylabel("efecto medio de conservar la variable (log-odds)")
    ax2 = ax.twinx()
    ax2.plot(it["sizes"], it["effect_probability"], marker="s", ms=3, color=ORANGE, label="probabilidad")
    ax2.set_ylabel("efecto medio (probabilidad)", color=ORANGE)
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="upper left")
    ax.set_title("Prueba de interacción (efecto según el contexto)", fontsize=8.5)

    ax = axes[2]
    srs = [r["sigma_rel"] for r in loc]
    ax.errorbar(range(len(srs)), [r["w_mean"] for r in loc], yerr=[r["w_std"] for r in loc], marker="o", color=ACCENT,
                capsize=3, label="w LIME (media ± desv., 10 semillas)")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axhline(b["shap_phi_probability"], color=ORANGE, ls=":", lw=1, label=f"φ TreeSHAP en probabilidad = {b['shap_phi_probability']:+.3f}")
    ax.set_xticks(range(len(srs)), [f"{s:g}" for s in srs])
    ax.set_xlabel("$\\sigma_{rel}$")
    ax.set_ylabel(f"coeficiente de {var}")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Prueba de localidad (ancho del kernel)", fontsize=8.5)
    fig.tight_layout()
    return _save(fig, out)
