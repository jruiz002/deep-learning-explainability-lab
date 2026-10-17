"""Plain-text execution report of Task 3 (ASCII tables, numerical self-checks)."""

from __future__ import annotations

from . import config
from .console import WIDTH, _check, _table, _title

_P_RANGE = {"aprobado": (config.PROB_APPROVED, 1.0), "rechazado": (0.0, config.PROB_REJECTED),
            "dudoso": config.PROB_DOUBTFUL_BAND}


def _outside(v: float, lo: float, hi: float) -> float:
    """Distance by which ``v`` falls outside [lo, hi] (0 when inside)."""
    return max(0.0, lo - v, v - hi)


def collect_checks(results: dict) -> list[tuple[str, float, float, bool]]:
    """(description, worst value, tolerance, passed) for every numerical self-check of Task 3."""
    t1, t2, cfg = results["t3_1"], results["t3_2"], results["config"]
    cl = list(t1.values())
    n_eff_cells = [(n, v) for n, row in zip(t2["n_values"], t2["n_eff"]) for v in row]
    out = [
        ("3.1 max |w cerrada - lstsq ponderado| (20 coef., 3 clientes)", max(c["max_abs_diff_lstsq"] for c in cl), config.TOL_LIME),
        ("3.1 max |w cerrada - sklearn Ridge| (20 coef., 3 clientes)", max(c["max_abs_diff_sklearn_ridge"] for c in cl), config.TOL_LIME),
        ("3.1 R2_w fuera de [0, 1]", max(_outside(c["r2w"], 0.0, 1.0) for c in cl), 1e-12),
        ("3.1 N_ef por encima de N (o no positivo)", max(max(0.0, c["n_eff"] - cfg["n"], -c["n_eff"]) for c in cl), 1e-9),
        ("3.1 p(good) fuera del rango pedido de cada cliente", max(_outside(c["p_good"], *_P_RANGE[k]) for k, c in t1.items()), 1e-12),
        ("3.2 Jaccard medio fuera de [0, 1]", max(_outside(v, 0.0, 1.0) for row in t2["jaccard"] for v in row), 1e-12),
        ("3.2 N_ef medio por encima de N (o no positivo)", max(max(0.0, v - n, -v) for n, v in n_eff_cells), 1e-9),
        ("3.2 pares por celda distintos de C(20, 2) = 190", float(max(abs(c["n_pairs"] - 190) for c in t2["cells"])), 1e-12),
    ]
    return [(d, v, t, v < t) for d, v, t in out]


def render_console(results: dict, figures: list[str], elapsed: float) -> tuple[str, bool]:
    env, cfg, t1, t2 = results["environment"], results["config"], results["t3_1"], results["t3_2"]
    L: list[str] = ["=" * WIDTH, " LABORATORIO DE EXPLICABILIDAD - TASK 3", " LIME tabular desde cero (NumPy) sobre German Credit", "=" * WIDTH]
    L += [f" Entorno     semilla {env['seed']}, Python {env['python']}",
          f" Versiones   scikit-learn {env['sklearn']} | numpy {env['numpy']} | pandas {env['pandas']}",
          f" Parámetros  N = {cfg['n']}, K = {cfg['k']}, lambda = {cfg['lambda']}, sigma_rel = {cfg['sigma_rel']}, "
          f"d' = {cfg['d_prime']}, sigma = {cfg['sigma']:.4f} (intercepto sin penalizar)"]

    L += _title("Task 3.1  LIME sobre tres clientes (salida explicada: P(good))")
    for key, c in t1.items():
        L += ["", f" {key}: posición {c['test_position']} del test, p(good) = {c['p_good']:.4f}, intercepto = {c['intercept']:+.4f}, "
                  f"R2_w = {c['r2w']:.4f}, N_ef = {c['n_eff']:.1f}"]
        L += _table(["Variable", "Valor", "Coeficiente"], [[x["feature"], x["value"], f"{x['coef']:+.4f}"] for x in c["top"]], "llr", indent=5)

    L += _title("Task 3.2  Estabilidad (cliente rechazado)")
    ns, srs = t2["n_values"], t2["sigma_rel_values"]
    L += ["", f" Cliente rechazado (posición {t2['test_position']}); {len(t2['seeds'])} corridas por celda "
              f"(semillas {t2['seeds'][0]}..{t2['seeds'][-1]}), top-{t2['k']}, 190 pares", "", " Jaccard medio (filas N, columnas sigma_rel):"]
    L += _table(["N", *[f"{s:g}" for s in srs]], [[str(n), *[f"{v:.4f}" for v in row]] for n, row in zip(ns, t2["jaccard"])])
    L += ["", " N_ef medio (filas N, columnas sigma_rel):"]
    L += _table(["N", *[f"{s:g}" for s in srs]], [[str(n), *[f"{v:.1f}" for v in row]] for n, row in zip(ns, t2["n_eff"])])

    if "t3_3" in results:
        a, b = results["t3_3"]["a"], results["t3_3"]["b"]
        L += _title("Task 3.3  LIME frente a SHAP")
        L += [""] + _table(["Cliente", "Spearman", "Top-3 LIME", "Top-3 SHAP", "Comunes"],
                           [[k, f"{c['spearman']:.3f}", ", ".join(c["top_lime"]), ", ".join(c["top_shap"]), f"{c['top_overlap']}/3"]
                            for k, c in a.items()], "lrllr")
        L += ["", " R2_w del cliente rechazado según K:"] + _table(["K", "R2_w"], [[str(r["k"]), f"{r['r2w']:.4f}"] for r in b["by_k"]])

    checks = collect_checks(results)
    L += _title("Verificaciones numéricas")
    L += [""] + _table(["Verificación", "Peor caso", "Tolerancia", "Estado"],
                       [[desc, f"{v:.2e}", f"< {tol:.0e}", _check(ok)] for desc, v, tol, ok in checks], "lrrr")
    all_ok = all(ok for *_, ok in checks)
    L += ["", f" Resultado: {sum(ok for *_, ok in checks)}/{len(checks)} verificaciones correctas"]
    L += _title("Archivos generados")
    L += [f"  figura   outputs/task3/figures/{f}" for f in figures]
    L += ["  datos    outputs/task3/results.json", "  informe  README.md (sección Task 3)"]
    L += ["", f" Tiempo total: {elapsed:.1f} s", "=" * WIDTH]
    return "\n".join(L), all_ok
