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
    if "t3_4" in results:
        t4, t5 = results["t3_4"], results["t3_5"]
        sel = t4["selected"]
        feats = [{c["feature"] for c in s["changes"]} for s in sel]
        out += [
            ("3.4 contrafactuales seleccionados con P(good) < theta", max(_outside(s["p_good"], t4["threshold"], 1.0) for s in sel), 1e-12),
            ("3.4 seleccionados que cambian una variable no accionable", float(sum(len(f - set(t4["actionable"])) for f in feats)), 0.5),
            ("3.4 seleccionados con el mismo conjunto de variables", float(len(feats) - len({frozenset(f) for f in feats})), 0.5),
            ("3.4 |distancia - suma de costos por variable|", max(abs(s["distance"] - sum(c["cost"] for c in s["changes"])) for s in sel), 1e-9),
            ("3.5(a) Jaccard de la corrida de verificación bajo 0.9", max(0.0, t5["a"]["target"] - t5["a"]["check_jaccard"]), 1e-12),
            ("3.5(b) |promedio por tamaños - phi TreeSHAP| (log-odds)",
             abs(t5["b"]["interaction"]["shapley_from_sizes_log_odds"] - t5["b"]["shap_phi"]), 0.05),
        ]
    return [(d, v, t, v < t) for d, v, t in out]


def render_console(results: dict, figures: list[str], elapsed: float) -> tuple[str, bool]:
    env, cfg, t1, t2 = results["environment"], results["config"], results["t3_1"], results["t3_2"]
    L: list[str] = ["=" * WIDTH, " LABORATORIO DE EXPLICABILIDAD - TASK 3", " LIME desde cero, comparación con SHAP y contrafactuales (German Credit)", "=" * WIDTH]
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
        t4 = results["t3_4"]
        L += _title("Task 3.4  Contrafactuales (cliente rechazado)")
        L += ["", f" p(good) original = {t4['p_good_original']:.4f}; {t4['n_valid']} válidos de {t4['n_candidates']} "
                  f"(validez {t4['validity']:.4%}); máx. {t4['max_changes']} cambios" + (" (se subió a 5)" if t4["fallback_to_5"] else "")]
        L += _table(["#", "Cambios", "Distancia", "P(good)"],
                    [[str(i), "; ".join(f"{c['feature']}: {c['original']} -> {c['new']}" for c in s["changes"]), f"{s['distance']:.3f}",
                      f"{s['p_good']:.4f}"] for i, s in enumerate(t4["selected"], 1)], "rlrr")
        L += ["", f" Proximidad promedio {t4['proximity_selected']:.3f}; dispersión promedio {t4['sparsity_selected']:.1f} variables"]
        t5 = results["t3_5"]
        L += _title("Task 3.5  Análisis")
        a5, b5 = t5["a"], t5["b"]
        L += ["", f" (a) N para Jaccard {a5['target']} con sigma_rel = {a5['sigma_rel']}: N = {a5['n_estimate']} "
                  f"(1 - J = a N_ef^-{a5['fit_b']:.2f}); verificación: J = {a5['check_jaccard']:.4f}, N_ef = {a5['check_n_eff']:.0f}",
              f" (b) mayor desacuerdo: cliente {b5['client']}, {b5['feature']} (puesto LIME {b5['rank_lime']}, SHAP {b5['rank_shap']}); "
              f"w = {b5['lime_w']:+.4f}, phi = {b5['shap_phi']:+.4f}",
              f"     efecto por tamaño (log-odds): {b5['interaction']['effect_log_odds'][0]:+.3f} (0 presentes) -> "
              f"{b5['interaction']['effect_log_odds'][-1]:+.3f} (19 presentes); promedio = {b5['interaction']['shapley_from_sizes_log_odds']:+.4f}"]
        L += ["", " (d) Top-3 |phi| del cliente rechazado frente a los contrafactuales:"]
        L += _table(["Variable", "phi", "Tipo", "En CF", "Mejor P(good) cambiándola sola"],
                    [[r["feature"], f"{r['phi']:+.3f}", r["role"], "sí" if r["in_counterfactuals"] else "no", f"{r['best_single_p']:.3f}"]
                     for r in t5["d"]["top3_shap"]], "lrllr")

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
