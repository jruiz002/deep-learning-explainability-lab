"""Plain-text execution report of Task 4 (ASCII tables, numerical self-checks)."""

from __future__ import annotations

import numpy as np

from . import config
from .console import WIDTH, _check, _table, _title


def collect_checks(results: dict) -> list[tuple[str, float, float, bool]]:
    """(description, worst value, tolerance, passed) for every numerical self-check of Task 4."""
    t41, t42 = results["t4_1"], results["t4_2"]
    curves = [r["curves"] for r in t41.values()]
    s1 = t42["stages"][0]["images"].values()
    out = [
        # deletion ends on the all-gray image and insertion on the original, whatever the order
        ("4.1 |c_P deletion (mapa) - c_P deletion (azar)|", max(abs(c["deletion_map"][-1] - c["deletion_random_mean"][-1]) for c in curves), 1e-6),
        ("4.1 |c_P insertion - p(top-1) de la imagen original|",
         max(abs(r["curves"]["insertion_map"][-1] - r["p_top1"]) for r in t41.values()), 1e-4),
        ("4.1 |c_0 deletion - p(top-1)|", max(abs(r["curves"]["deletion_map"][0] - r["p_top1"]) for r in t41.values()), 1e-4),
        ("4.1 AUC fuera de [0, 1]", max(max(0.0, -v, v - 1.0) for r in t41.values() for k in ("deletion", "insertion")
                                       for v in [r[k]["map"], *r[k]["random_all"]]), 1e-12),
        ("4.2 etapa fc: activaciones de layer4 cambiadas", max(i["activations_unchanged"] for i in s1), 1e-12),
        ("4.2 etapa fc: max |mapa - sum_k w_rand A^k / Z|", t42["fc_prediction_max_abs_error"], config.TOL_CASCADE_FC),
    ]
    return [(d, v, t, v < t) for d, v, t in out]


def render_console(results: dict, figures: list[str], elapsed: float) -> tuple[str, bool]:
    env, cfg, t41, t42 = results["environment"], results["config"], results["t4_1"], results["t4_2"]
    L: list[str] = ["=" * WIDTH, " LABORATORIO DE EXPLICABILIDAD - TASK 4", " Validación de Grad-CAM: deletion / insertion y aleatorización en cascada",
                    "=" * WIDTH]
    L += [f" Entorno     semilla {env['seed']}, torch {env['torch']}, torchvision {env['torchvision']}, Python {env['python']}",
          f" Parámetros  P = {cfg['steps']}, {cfg['random_orders']} órdenes aleatorios, gris = {cfg['gray_level']}, desenfoque sigma = {cfg['blur_sigma']}"]

    L += _title("Task 4.1  Deletion e insertion (clase top-1)")
    for rid, r in t41.items():
        L += ["", f" Imagen ({rid}) {r['class_name']}  p = {r['p_top1']:.4f}"]
        L += _table(["Métrica", "Grad-CAM", "Azar media", "Azar desv.", "Diferencia", "Desv. a favor"],
                    [[k, f"{r[k]['map']:.4f}", f"{r[k]['random_mean']:.4f}", f"{r[k]['random_std']:.4f}",
                      f"{r[k]['diff_vs_random']:+.4f}", f"{r[k]['sds_better_than_random']:+.1f}"] for k in ("deletion", "insertion")])

    b = results["t4_3a_blur_reference"]
    L += _title("Task 4.3(a)  Deletion de (a) con referencia desenfocada")
    L += [""] + _table(["Referencia", "Grad-CAM", "Azar media", "Azar desv.", "Desv. a favor"],
                       [[name, f"{d['map']:.4f}", f"{d['random_mean']:.4f}", f"{d['random_std']:.4f}", f"{d['sds_better_than_random']:+.1f}"]
                        for name, d in (("gris", b["deletion_gray"]), ("desenfocada", b["deletion"]))])

    L += _title("Task 4.2  Aleatorización en cascada (Spearman con el mapa original 7x7)")
    ids = list(t41)
    fmt = lambda v: "indef." if v != v else f"{v:+.3f}"
    L += [""] + _table(["Etapa", "Última capa", *[f"({i})" for i in ids]],
                       [[str(s["stage"]), s["randomized"][-1], *[fmt(s["images"][i]["spearman"]) for i in ids]] for s in t42["stages"]])
    L += ["", " Primera etapa con Spearman < " + f"{config.CASCADE_SPEARMAN_LOW}: "
          + ", ".join(f"({i}) {t42['first_stage_below'][i]}" for i in ids)]

    checks = collect_checks(results)
    L += _title("Verificaciones numéricas")
    L += [""] + _table(["Verificación", "Peor caso", "Tolerancia", "Estado"],
                       [[desc, f"{v:.2e}", f"< {tol:.0e}", _check(ok)] for desc, v, tol, ok in checks], "lrrr")
    all_ok = all(ok for *_, ok in checks)
    L += ["", f" Resultado: {sum(ok for *_, ok in checks)}/{len(checks)} verificaciones correctas"]
    L += _title("Archivos generados")
    L += [f"  figura   outputs/task4/figures/{f}" for f in figures]
    L += ["  datos    outputs/task4/results.json", "  informe  README.md (sección Task 4)"]
    L += ["", f" Tiempo total: {elapsed:.1f} s", "=" * WIDTH]
    return "\n".join(L), all_ok
