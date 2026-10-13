"""Plain-text execution report of Task 2 (ASCII tables, numerical self-checks)."""

from __future__ import annotations

from . import config
from .console import WIDTH, _check, _table, _title


def collect_checks(results: dict) -> list[tuple[str, float, float, bool]]:
    """(description, worst value, tolerance, passed) for every numerical self-check of Task 2."""
    t22, t25 = results["t2_2"], results["t2_5"]
    out = [
        ("2.2(a) max |base + sum(phi) - f(x)| (200 clientes)", t22["a_max_efficiency_error"], config.TOL_SHAP),
        ("2.2(b) |valor base - media f(fondo)|", t22["b_abs_diff"], config.TOL_SHAP),
        ("2.5 max |phi propio - TreeSHAP| (5 x 6)", t25["max_abs_error_vs_treeshap"], config.TOL_SHAP),
        ("2.5 max |sum(phi) - (v(N) - v(vacio))|", t25["max_efficiency_error"], config.TOL_SHAP_EXACT),
        ("2.5 max |v(N) - f(x)|", t25["max_vN_vs_f_error"], config.TOL_SHAP_EXACT),
        ("2.5 max |v(vacio) - valor base TreeSHAP|", t25["max_v0_vs_expected_error"], config.TOL_SHAP),
    ]
    return [(d, v, t, v < t) for d, v, t in out]


def render_console(results: dict, figures: list[str], elapsed: float) -> tuple[str, bool]:
    env, t1, t2, t3 = results["environment"], results["t2_1"], results["t2_2"], results["t2_3"]
    t4, t5, t6 = results["t2_4"], results["t2_5"], results["t2_6"]
    L: list[str] = ["=" * WIDTH, " LABORATORIO DE EXPLICABILIDAD - TASK 2", " SHAP sobre German Credit (GradientBoostingClassifier)", "=" * WIDTH]
    L += [f" Entorno     semilla {env['seed']}, Python {env['python']}",
          f" Versiones   scikit-learn {env['sklearn']} | shap {env['shap']} | numpy {env['numpy']} | pandas {env['pandas']}"]

    L += _title("Task 2.1  Datos y modelo")
    L += [f" Clientes {t1['n_total']} (train {t1['n_train']} / test {t1['n_test']}), {t1['n_features']} variables "
          f"({t1['n_categorical']} categóricas codificadas ordinalmente, {t1['n_numeric']} numéricas)",
          f" GBC {t1['n_estimators']} árboles, profundidad {t1['max_depth']}",
          f" Exactitud (test) {t1['test_accuracy']:.4f}   AUC (test) {t1['test_auc']:.4f}   (proporción 'good' en test {t1['good_rate_test']:.2f})"]

    L += _title("Task 2.2  TreeSHAP interventional (log-odds) y permutación")
    L += ["", f" (a) max |base + sum(phi) - f(x)| = {t2['a_max_efficiency_error']:.2e}",
          f" (b) valor base = {t2['b_base_value']:.6f}   media f(fondo 100) = {t2['b_mean_bg_log_odds']:.6f}", "",
          " (c) Top-5 por importancia:"]
    L += _table(["#", "SHAP", "I_j", "Permutación", "Caída AUC"],
                [[str(i + 1), a["feature"], f"{a['value']:.4f}", b["feature"], f"{b['value']:.4f}"]
                 for i, (a, b) in enumerate(zip(t2["c_top5_shap"], t2["c_top5_perm"]))], "lrrlr")
    L += [f"     Spearman (20 variables) = {t2['c_spearman']:.4f}  (p = {t2['c_spearman_p']:.4f});  coincidencias en el top-5: {t2['c_top5_overlap']}/5", "",
          " (d) Clientes seleccionados y tres mayores contribuciones (log-odds):"]
    for key, c in t2["d_clients"].items():
        L += ["", f"   {key}: posición {c['test_position']} del test, p(good) = {c['p_good']:.4f}, f(x) = {c['log_odds']:.4f}"]
        L += _table(["Variable", "Valor", "phi"], [[x["feature"], x["value"], f"{x['phi']:+.4f}"] for x in c["top3"]], "llr", indent=5)

    L += _title("Task 2.3  Variable duplicada con ruido")
    L += ["", f" Variable: {t3['feature']} (sigma = {t3['sigma']:.1f}); corr(original, copia) = {t3['corr_orig_copy']:.5f}",
          f" Modelo con 21 variables: exactitud {t3['test_accuracy_21']:.4f}, AUC {t3['test_auc_21']:.4f}",
          f" I_j del original en el modelo sin copia = {t3['I_orig_without_copy']:.4f}", ""]
    L += _table(["Configuración", "mean|phi_orig|", "mean|phi_copia|", "mean|suma|", "% orig."],
                [[k, f"{v['mean_abs_orig']:.4f}", f"{v['mean_abs_copy']:.4f}", f"{v['mean_abs_sum']:.4f}", f"{v['share_orig']:.1%}"]
                 for k, v in t3["configs"].items()])

    L += _title("Task 2.4  KernelSHAP vs TreeSHAP (mismo fondo)")
    L += ["", f" {t4['n_instances']} instancias de test, nsamples = {t4['nsamples']}", ""]
    L += _table(["k", "MAE", "max error", "t Kernel (s)", "t Tree (s)", "evaluaciones", "s / inst."],
                [[str(r["k"]), f"{r['mae']:.4f}", f"{r['max_abs_error']:.4f}", f"{r['time_kernel_s']:.2f}",
                  f"{r['time_tree_s']:.4f}", str(r["model_evaluations"]), f"{r['time_kernel_per_instance_s']:.3f}"]
                 for r in t4["by_background"]])

    L += _title("Task 2.5  Shapley exacto propio (6 variables, 64 coaliciones)")
    L += ["", f" Modelo reducido: exactitud {t5['test_accuracy']:.4f}, AUC {t5['test_auc']:.4f}; fondo de {t5['n_background']} filas", ""]
    L += _table(["Cliente", *t5["features"]], [[str(i), *[f"{v:+.4f}" for v in row]] for i, row in enumerate(t5["phi_own"])])
    L += ["", f" max |propio - TreeSHAP| = {t5['max_abs_error_vs_treeshap']:.2e}",
          f" max |sum(phi) - (v(N) - v(vacio))| = {t5['max_efficiency_error']:.2e}   (tiempo propio: {t5['time_own_s']:.3f} s)"]

    a, b, d = t6["a"], t6["b"], t6["d"]
    L += _title("Task 2.6  Análisis")
    L += ["", f" (a) Cliente rechazado (posición {a['test_position']}): p(x) = {a['p']:.4f}, sigma(phi0) = {a['sigma_phi0']:.4f}",
          f"     p(x) - sigma(phi0) = {a['p_minus_sigma_phi0']:+.4f}   suma de efectos sigma(phi0+phi_j)-sigma(phi0) = {a['sum_sigma_effects']:+.4f}",
          f"     sum(phi) = {a['sum_phi']:+.4f}  vs  f(x) - phi0 = {a['log_odds'] - a['phi0']:+.4f}  (aditivo en log-odds)", "",
          f" (b) Mayor desacuerdo de rangos: {b['feature']}  (rango I_j #{b['rank_shap']}, rango permutación #{b['rank_perm']})",
          f"     I_j = {b['I']:.4f}; permutación = {b['perm_mean']:+.4f} +/- {b['perm_std']:.4f}",
          "     Mayores correlaciones de Spearman (train): " + ", ".join(f"{c['feature']} {c['spearman']:+.3f}" for c in b["top_corr"]),
          f"     Clientes que acumulan el 50 % de su suma de |phi|: {b['n_clients_50pct']} de {t1['n_test']} ({b['frac_clients_50pct']:.1%})", "",
          " (d) Costo:"]
    L += _table(["k", "evaluaciones", "t Kernel (s)", "us / evaluación", "MAE"],
                [[str(r["k"]), str(r["evaluations"]), f"{r['time_kernel_s']:.2f}", f"{r['time_per_evaluation_us']:.2f}", f"{r['mae']:.4f}"]
                 for r in d["by_background"]], indent=5)
    L += [f"     t(decision_function, 100 filas) = {d['t_decision_function_100_s'] * 1e3:.3f} ms (mediana de {d['n_repeats']})",
          f"     Shapley exacto con 20 variables ~ 2^20 * t(100) = {d['exact_20_vars_s']:.0f} s ({d['exact_20_vars_h']:.2f} h) por instancia; "
          f"KernelSHAP (k=100) {d['kernel_per_instance_s_k100']:.3f} s por instancia ({d['exact_over_kernel']:.0f} veces menos)",
          f"     60 variables a 1 us por evaluación: {d['years_60_vars_1us']:.3e} años"]

    checks = collect_checks(results)
    L += _title("Verificaciones numéricas")
    L += [""] + _table(["Verificación", "Peor caso", "Tolerancia", "Estado"],
                       [[desc, f"{v:.2e}", f"< {tol:.0e}", _check(ok)] for desc, v, tol, ok in checks], "lrrr")
    all_ok = all(ok for *_, ok in checks)
    L += ["", f" Resultado: {sum(ok for *_, ok in checks)}/{len(checks)} verificaciones correctas"]
    L += _title("Archivos generados")
    L += [f"  figura   outputs/task2/figures/{f}" for f in figures]
    L += ["  datos    outputs/task2/results.json", "  informe  README.md (sección Task 2)"]
    L += ["", f" Tiempo total: {elapsed:.1f} s", "=" * WIDTH]
    return "\n".join(L), all_ok
