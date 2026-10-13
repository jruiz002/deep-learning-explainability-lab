"""Written report of Task 2 (Spanish, impersonal voice), built only from the numbers in results.json."""

from __future__ import annotations

from .report import _table

TASK2_BEGIN = "<!-- INFORME_TASK2:INICIO (generado por scripts/run_task2.py; no editar a mano) -->"
TASK2_END = "<!-- INFORME_TASK2:FIN -->"
FIRST_FIGURE = 7  # Task 1 uses Figura 1 - Figura 6

PROMPT = (
    "Usa German Credit con `fetch_openml(\"credit-g\", version=1)`, "
    "codifica `good`=1 y `bad`=0, pasa las 13 categóricas por un `OrdinalEncoder` (tratando los códigos como numéricos) "
    "y divide 80/20 estratificado con semilla 42. Entrena un `GradientBoostingClassifier` con 200 árboles y profundidad 3. "
    "Para SHAP usa `TreeExplainer` con `feature_perturbation=\"interventional\"` y `model_output=\"raw\"`, de modo que la "
    "salida explicada sea el log-odds de `decision_function`, con un fondo de 100 filas del entrenamiento elegidas con la "
    "semilla. Verifica la eficiencia y que el valor base sea el promedio del log-odds en el fondo, compara la importancia "
    "media |φ| con `permutation_importance` (AUC, 10 repeticiones) usando Spearman, y en KernelSHAP usa exactamente el "
    "mismo fondo que en TreeSHAP para que el error compare lo mismo. La fórmula de Shapley del 2.5 la quiero escrita a "
    "mano: enumera las 64 coaliciones con máscaras de bits, evalúalas todas en una sola llamada al modelo y usa los pesos "
    "|S|!(n−|S|−1)!/n!. Guarda los números en `outputs/task2/results.json` e imprime una tabla de verificaciones OK/FALLO "
    "con tolerancias."
)

WHY = (
    "Fijamos los detalles que cambian los números en lugar de pedir simplemente \"calcula SHAP\". Con "
    "`model_output=\"raw\"` SHAP explica el log-odds; sin eso explicaría otra salida y la eficiencia en log-odds no "
    "cerraría. Nombrar `interventional` frente a `tree_path_dependent` evitó que la librería escogiera una de las dos por "
    "defecto sin que lo supiéramos. Usar el mismo fondo en KernelSHAP y en TreeSHAP hizo que el MAE midiera sólo la "
    "aproximación y no un cambio de referencia. Pedir las 64 coaliciones vectorizadas en una sola llamada permitió que el "
    "cálculo exacto corriera en segundos, y las verificaciones automáticas con tolerancias detectaron errores antes de "
    "escribir el informe."
)


def _f(v: float, nd: int = 4) -> str:
    return f"{v:.{nd}f}"


def _sg(v: float, nd: int = 3) -> str:
    return f"{v:+.{nd}f}"


def render_informe_task2(results: dict) -> str:
    env, t1, t2, t3 = results["environment"], results["t2_1"], results["t2_2"], results["t2_3"]
    t4, t5, t6 = results["t2_4"], results["t2_5"], results["t2_6"]
    a6, b6, d6 = t6["a"], t6["b"], t6["d"]
    clients = t2["d_clients"]
    n_fig = iter(range(FIRST_FIGURE, FIRST_FIGURE + 20))

    def fig(name: str, cap: str) -> str:
        return f"![Figura {(n := next(n_fig))}. {cap}](outputs/task2/figures/{name})\n\n*Figura {n}. {cap}*\n"

    # ---- 2.1 ----
    gain = t1["test_accuracy"] - t1["good_rate_test"]
    acc_txt = (f"supera en {gain * 100:.1f} puntos porcentuales a predecir siempre `good` ({t1['good_rate_test']:.3f})"
               if gain > 0 else f"no supera a predecir siempre `good` ({t1['good_rate_test']:.3f})")

    # ---- 2.2 ----
    eff_ok = t2["a_max_efficiency_error"] < 1e-6
    top_s, top_p = t2["c_top5_shap"], t2["c_top5_perm"]
    rho = t2["c_spearman"]
    rho_word = "alta" if rho >= 0.7 else "moderada" if rho >= 0.4 else "baja"
    only_s = [r["feature"] for r in top_s if r["feature"] not in {q["feature"] for q in top_p}]
    only_p = [r["feature"] for r in top_p if r["feature"] not in {q["feature"] for q in top_s}]
    t_top = _table(["#", "mean\\|φ\\| (variable)", "I_j", "Permutación (variable)", "Caída de AUC"],
                   [[str(i + 1), f"`{x['feature']}`", _f(x["value"]), f"`{y['feature']}`", _f(y["value"])]
                    for i, (x, y) in enumerate(zip(top_s, top_p))])

    def client_block(key: str) -> list[str]:
        c = clients[key]
        rows = [[f"`{x['feature']}` = {x['value']}", _sg(x["phi"]), x["direction"]] for x in c["top3"]]
        return [f"**Cliente {key}** (posición {c['test_position']} del conjunto de prueba; p(good) = {_f(c['p_good'], 3)}, "
                f"f(x) = {_f(c['log_odds'], 3)}):", ""] + _table(["Variable y valor", "φ (log-odds)", "Efecto sobre el log-odds"], rows)

    def client_sentence(key: str) -> str:
        c = clients[key]
        t = c["top3"]
        parts = [f"`{x['feature']}` = {x['value']} ({x['direction']} el log-odds en {abs(x['phi']):.3f})" for x in t]
        side = "por encima" if c["log_odds"] > t2["b_base_value"] else "por debajo"
        return (f"En el cliente {key}, las tres mayores contribuciones son {parts[0]}, {parts[1]} y {parts[2]}; el log-odds final "
                f"({_f(c['log_odds'], 3)}) queda {side} del valor base ({_f(t2['b_base_value'], 3)}).")

    sp_num = t2["d_value_phi_spearman"]
    top_num = sorted(sp_num, key=lambda n: -t2["c_importance_shap"][n])[:3]
    dir_txt = "; ".join(
        f"en `{n}`, " + (f"un valor mayor se asocia con {'mayor' if sp_num[n] > 0 else 'menor'} contribución (ρ = {sp_num[n]:+.2f})"
                         if abs(sp_num[n]) >= 0.3 else f"no hay una relación monótona clara entre el valor y la contribución (ρ = {sp_num[n]:+.2f})")
        for n in top_num)

    # ---- 2.3 ----
    cf_i, cf_p = t3["configs"]["interventional"], t3["configs"]["tree_path_dependent"]
    t_dup = _table(["Configuración", "mean\\|φ_orig\\|", "mean\\|φ_copia\\|", "mean\\|φ_orig + φ_copia\\|", "% del original", "corr(φ_orig, φ_copia)"],
                   [[k, _f(v["mean_abs_orig"]), _f(v["mean_abs_copy"]), _f(v["mean_abs_sum"]), f"{v['share_orig']:.1%}",
                     _sg(v["corr_phi_orig_copy"], 2)] for k, v in t3["configs"].items()])
    gap_i, gap_p = abs(cf_i["share_orig"] - 0.5), abs(cf_p["share_orig"] - 0.5)
    closer = "tree_path_dependent" if gap_p < gap_i else "interventional"

    # ---- 2.4 ----
    kr = t4["by_background"]
    mae_mono = all(x["mae"] >= y["mae"] for x, y in zip(kr, kr[1:]))
    mae_drop = 1 - kr[-1]["mae"] / kr[0]["mae"]
    big = [r for r in kr if r["k"] >= 50]
    rate_spread = max(r["time_per_evaluation_us"] for r in big) / min(r["time_per_evaluation_us"] for r in big) - 1

    # ---- 2.5 ----
    cols = t5["features"]
    t_own = _table(["Cliente (test)", *[f"`{c}`" for c in cols], "Σφ", "f(x) − v(∅)"],
                   [[str(i), *[_sg(v, 4) for v in row], _sg(sum(row), 4), _sg(fx - t5["v_empty"][i], 4)]
                    for i, (row, fx) in enumerate(zip(t5["phi_own"], t5["f_x"]))])

    # ---- 2.6 ----
    sum_phi_check = a6["log_odds"] - a6["phi0"]
    top_sig = sorted(a6["sigma_effects"].items(), key=lambda kv: -abs(kv[1]))[:5]
    t_sig = _table(["Variable", "σ(φ0 + φ_j) − σ(φ0)"], [[f"`{k}`", _sg(v, 4)] for k, v in top_sig])
    over = "sobreestima la magnitud del" if abs(a6["sum_sigma_effects"]) > abs(a6["p_minus_sigma_phi0"]) else "subestima el"
    corr_txt = ", ".join(f"`{c['feature']}` ({c['spearman']:+.2f})" for c in b6["top_corr"])
    corr_low = b6["max_abs_corr"] < 0.4
    noisy = b6["perm_std"] >= abs(b6["perm_mean"])
    spread = b6["frac_clients_50pct"] >= 0.2
    t_cost = _table(["k", "Evaluaciones del modelo", "Tiempo KernelSHAP (s)", "Tiempo / evaluación (µs)", "MAE"],
                    [[str(r["k"]), f"{r['evaluations']:,}", _f(r["time_kernel_s"], 2), _f(r["time_per_evaluation_us"], 2), _f(r["mae"])]
                     for r in d6["by_background"]])
    t_kern = _table(["k", "MAE", "Error máximo", "KernelSHAP (s)", "TreeSHAP (s)", "KernelSHAP por instancia (s)", "Evaluaciones"],
                    [[str(r["k"]), _f(r["mae"]), _f(r["max_abs_error"]), _f(r["time_kernel_s"], 2), _f(r["time_tree_s"], 4),
                      _f(r["time_kernel_per_instance_s"], 3), f"{r['model_evaluations']:,}"] for r in kr])

    return f"""# Task 2

## Configuración

- Datos: German Credit (`fetch_openml("credit-g", version=1)`), {t1['n_total']} clientes y {t1['n_features']} variables
  ({t1['n_categorical']} categóricas y {t1['n_numeric']} numéricas). Objetivo: `good` = 1, `bad` = 0.
- Semilla aleatoria: **{env['seed']}** (partición, modelo, fondo, permutaciones y ruido de 2.3).
- Versiones: scikit-learn {env['sklearn']}, shap {env['shap']}, NumPy {env['numpy']}, pandas {env['pandas']}, Python {env['python']}.
- Limitación: las 13 categóricas se codificaron con `OrdinalEncoder`, cuyos códigos siguen el orden alfabético de las categorías y no un
  orden con significado, pero el modelo los trata como numéricos. Por eso un umbral de un árbol sobre una categórica agrupa categorías por su
  código, y los φ de esas variables resumen el efecto de ese código y no el de una categoría por separado.

## Task 2.1 - Datos y modelo

Partición estratificada 80/20: {t1['n_train']} clientes de entrenamiento y {t1['n_test']} de prueba, con una proporción de `good` de
{t1['good_rate_train']:.2f} y {t1['good_rate_test']:.2f}. Modelo: `GradientBoostingClassifier` con {t1['n_estimators']} árboles de profundidad
{t1['max_depth']}.

| Métrica (prueba) | Valor |
|---|---|
| Exactitud | **{_f(t1['test_accuracy'], 3)}** |
| AUC | **{_f(t1['test_auc'], 3)}** |

La exactitud {acc_txt}.

## Task 2.2 - TreeSHAP interventional

Se usó `TreeExplainer(model, data=fondo, feature_perturbation="interventional", model_output="raw")`, con un fondo de 100 filas de entrenamiento
(semilla {env['seed']}), sobre las {t1['n_test']} filas de prueba. La salida explicada es el log-odds, es decir, `decision_function`.

### (a) Eficiencia

max<sub>n</sub> |φ0 + Σ<sub>j</sub> φ<sub>j</sub>(x<sub>n</sub>) − f(x<sub>n</sub>)| = **{t2['a_max_efficiency_error']:.2e}**
{"(dentro de la tolerancia de 1e-6; el error no es cero y es del orden de la precisión simple, lo que sugiere que la extensión en C++ de TreeSHAP no calcula todo en float64)." if eff_ok else "(fuera de la tolerancia de 1e-6)."}

### (b) Valor base

| φ0 (`expected_value`) | Media de f en el fondo | Diferencia |
|---|---|---|
| {_f(t2['b_base_value'], 6)} | {_f(t2['b_mean_bg_log_odds'], 6)} | {t2['b_abs_diff']:.2e} |

{"Coinciden: con `interventional` el valor base es el promedio de la salida del modelo sobre el fondo." if t2['b_abs_diff'] < 1e-6 else "No coinciden dentro de la tolerancia."}

### (c) Importancia SHAP frente a permutación

I<sub>j</sub> = mean|φ<sub>j</sub>| sobre la prueba; permutación con AUC, 10 repeticiones.

{chr(10).join(t_top)}
Correlación de Spearman entre los vectores completos de las 20 variables: **ρ = {_f(rho, 3)}** (p = {t2['c_spearman_p']:.4f}), una concordancia {rho_word}.
Los dos top-5 comparten {t2['c_top5_overlap']} de 5 variables; sólo aparece en el top-5 de I<sub>j</sub> {', '.join('`'+v+'`' for v in only_s) or 'ninguna'} y sólo en
el de permutación {', '.join('`'+v+'`' for v in only_p) or 'ninguna'}.

{fig('fig_2_2_importance.png', 'Importancia por variable: media de |φ| (izquierda) y caída del AUC por permutación, media ± desviación (derecha). Entre paréntesis, el rango por permutación.')}
### (d) Dos clientes

{chr(10).join(client_block('aprobado'))}
{chr(10).join(client_block('rechazado'))}
{client_sentence('aprobado')} {client_sentence('rechazado')} Los valores categóricos se muestran con su etiqueta original.

{fig('fig_2_2_waterfall_aprobado.png', 'Cascada SHAP del cliente aprobado (p > 0.8).')}
{fig('fig_2_2_waterfall_rechazado.png', 'Cascada SHAP del cliente rechazado (p < 0.2).')}
{fig('fig_2_2_beeswarm.png', 'Gráfico de enjambre SHAP global sobre el conjunto de prueba (el color de las categóricas corresponde a su código ordinal).')}
En el enjambre, las variables ordenadas por I<sub>j</sub> son `{top_s[0]['feature']}`, `{top_s[1]['feature']}` y `{top_s[2]['feature']}`. Para las numéricas de mayor
importancia: {dir_txt}.

## Task 2.3 - Variable duplicada con ruido

La numérica con mayor I<sub>j</sub> es `{t3['feature']}` (I<sub>j</sub> = {_f(t3['I_orig_without_copy'])}). Se creó `{t3['copy']}` = x + 0.05·σ·ε, con σ = {t3['sigma']:.1f} (calculada en el conjunto
completo) y ε ~ N(0, 1) con semilla {env['seed']}; la correlación entre la variable y su copia es {t3['corr_orig_copy']:.5f}. El modelo se reentrenó con 21 variables
(exactitud {_f(t3['test_accuracy_21'], 3)}, AUC {_f(t3['test_auc_21'], 3)}) y se usaron las mismas filas de fondo.

{chr(10).join(t_dup)}
I<sub>j</sub> de `{t3['feature']}` en el modelo **sin** copia: {_f(t3['I_orig_without_copy'])} (es otro modelo, por lo que no es exactamente comparable).

{fig('fig_2_3_duplicate.png', 'Crédito repartido entre la variable y su copia, con ambos métodos de TreeSHAP.')}
## Task 2.4 - KernelSHAP frente a TreeSHAP

KernelSHAP (`nsamples` = {t4['nsamples']}) sobre las primeras {t4['n_instances']} filas de prueba, con fondos anidados de k = 10, 50 y 100 filas; TreeSHAP interventional usa
**el mismo** fondo en cada caso. El MAE se promedia sobre {t4['n_instances']} × {t1['n_features']} valores φ.

{chr(10).join(t_kern)}
El conteo de evaluaciones supera en {kr[0]['model_evaluations'] - kr[0]['evaluations_nominal']} (una por instancia, para f(x)) el producto nsamples·k·instancias. Con el mismo fondo, TreeSHAP es entre
{min(r['speedup_tree_vs_kernel'] for r in kr):.0f} y {max(r['speedup_tree_vs_kernel'] for r in kr):.0f} veces más rápido que KernelSHAP (los tiempos dependen de la máquina y de la corrida).

{fig('fig_2_4_kernel_vs_tree.png', 'MAE de KernelSHAP contra TreeSHAP y tiempos de cómputo según el tamaño del fondo.')}
## Task 2.5 - Shapley exacto escrito a mano

Modelo con seis variables (`{'`, `'.join(cols)}`), misma partición, semilla e hiperparámetros (exactitud {_f(t5['test_accuracy'], 3)}, AUC {_f(t5['test_auc'], 3)}). Con un fondo de
{t5['n_background']} filas se enumeraron las {t5['n_coalitions']} coaliciones con máscaras de bits y se evaluó v(S) = media<sub>b</sub> f(x<sub>S</sub>, b<sub>¬S</sub>) en una sola llamada al modelo; los
pesos son |S|!(n−|S|−1)!/n!. Resultados de las primeras {t5['n_instances']} filas de prueba (log-odds):

{chr(10).join(t_own)}
- Máximo error absoluto contra `TreeExplainer` interventional con el mismo fondo: **{t5['max_abs_error_vs_treeshap']:.2e}**.
- Eficiencia, máx. |Σφ − (v(N) − v(∅))|: **{t5['max_efficiency_error']:.2e}**; además v(N) coincide con f(x) (error {t5['max_vN_vs_f_error']:.1e}).
- Tiempo del cálculo propio para las {t5['n_instances']} instancias: {t5['time_own_s']:.3f} s.

{fig('fig_2_5_own_vs_shap.png', 'Shapley exacto propio contra TreeSHAP interventional (30 valores).')}
El error contra TreeSHAP ({t5['max_abs_error_vs_treeshap']:.1e}) es {"mucho mayor" if t5['max_abs_error_vs_treeshap'] > 1e3 * t5['max_efficiency_error'] else "similar"} que el de la eficiencia del cálculo propio
({t5['max_efficiency_error']:.1e}), lo que es coherente con que TreeSHAP acumule en precisión simple y el cálculo propio en float64.

## Task 2.6 - Análisis

### (a) Aditividad en log-odds frente a probabilidad (cliente rechazado)

Cliente de la posición {a6['test_position']}: p(x) = {_f(a6['p'])}, f(x) = {_f(a6['log_odds'])}, φ0 = {_f(a6['phi0'])}, σ(φ0) = {_f(a6['sigma_phi0'])}.

| Cantidad | Valor |
|---|---|
| Σ<sub>j</sub> φ<sub>j</sub> | {_sg(a6['sum_phi'], 4)} |
| f(x) − φ0 | {_sg(sum_phi_check, 4)} |
| p(x) − σ(φ0) | {_sg(a6['p_minus_sigma_phi0'], 4)} |
| Σ<sub>j</sub> [σ(φ0 + φ<sub>j</sub>) − σ(φ0)] | {_sg(a6['sum_sigma_effects'], 4)} |
| [p(x) − σ(φ0)] − Σ<sub>j</sub> [σ(φ0 + φ<sub>j</sub>) − σ(φ0)] | {_sg(a6['gap'], 4)} |

Mayores efectos individuales en probabilidad:

{chr(10).join(t_sig)}
La suma de los φ coincide con f(x) − φ0 porque la función explicada es f (el log-odds) y los valores de Shapley son aditivos en esa escala. La probabilidad
es σ(f), y σ no es lineal: σ(a + b) − σ(a) ≠ [σ(φ0 + φ_1) − σ(φ0)] + [σ(φ0 + φ_2) − σ(φ0)] porque cada término evalúa la pendiente de σ en
un punto distinto (φ0 + φ_j) y no en el punto final (f(x) = {_f(a6['log_odds'], 3)}). Con estos números, la suma de efectos individuales ({_sg(a6['sum_sigma_effects'], 4)}) {over}
cambio real de probabilidad ({_sg(a6['p_minus_sigma_phi0'], 4)}) en {abs(a6['gap']):.4f}.

### (b) Mayor desacuerdo entre las dos importancias: `{b6['feature']}`

`{b6['feature']}` ocupa el puesto {b6['rank_shap']} por I<sub>j</sub> ({_f(b6['I'])}) y el {b6['rank_perm']} por permutación ({_sg(b6['perm_mean'], 4)} ± {b6['perm_std']:.4f}; diferencia de {b6['rank_gap']} puestos).
I<sub>j</sub> mide cuánto mueve la variable el log-odds de las predicciones (contribución media absoluta, sin mirar si acierta); la permutación mide cuánto
cae el AUC en el conjunto de prueba al romper la relación entre la variable y el resultado.

Evidencia calculada: sus mayores correlaciones de Spearman (entrenamiento) son {corr_txt}; {"el máximo en valor absoluto es " + _f(b6['max_abs_corr'], 2) + ", bajo, por lo que la correlación con otras variables no explica el desacuerdo" if corr_low else "el máximo en valor absoluto es " + _f(b6['max_abs_corr'], 2) + ", por lo que la correlación con otras variables podría explicar parte del desacuerdo"}.
{"La desviación de la permutación (" + f"{b6['perm_std']:.4f}" + ") es al menos tan grande como el valor absoluto de su media (" + f"{abs(b6['perm_mean']):.4f}" + "), así que su efecto sobre el AUC no se distingue de cero" if noisy else "La permutación tiene una media de " + _sg(b6['perm_mean'], 4) + " con desviación " + f"{b6['perm_std']:.4f}" + ", distinguible de cero"}{" y la media es negativa (mezclar la variable no empeora el AUC)" if b6['perm_mean'] < 0 else ""}.
Además, hacen falta {b6['n_clients_50pct']} de {t1['n_test']} clientes ({b6['frac_clients_50pct']:.1%}) para acumular la mitad de su Σ|φ|, {"de modo que su efecto está repartido y no concentrado en pocos clientes" if spread else "de modo que su efecto está concentrado en pocos clientes"}.
**Hipótesis** (no se probó con un experimento adicional): {"el modelo usa `" + b6['feature'] + "` (" + str(b6['n_categories']) + " valores distintos, con lo que los árboles pueden ajustar con ella detalles del entrenamiento) sin que ese uso mejore el AUC en prueba" if noisy or b6['perm_mean'] <= 0 else "la variable aporta al AUC, pero menos de lo que su contribución al log-odds sugiere"}. Con un
conjunto de prueba de {t1['n_test']} clientes y una sola semilla, esta explicación es tentativa.

### (c) Crédito entre la variable y su copia (tabla de 2.3)

En el modelo interventional, `{t3['feature']}` recibe {_f(cf_i['mean_abs_orig'])} y la copia {_f(cf_i['mean_abs_copy'])} ({cf_i['share_orig']:.1%} para el original); con `tree_path_dependent`,
{_f(cf_p['mean_abs_orig'])} y {_f(cf_p['mean_abs_copy'])} ({cf_p['share_orig']:.1%}). El axioma de simetría de Shapley pediría un reparto igual (50 %) si el modelo tratara las dos columnas de forma
simétrica, y el reparto observado {"se acerca más a ese 50 %" if closer else ""} con {closer} (desviación de {min(gap_i, gap_p) * 100:.1f} puntos frente a {max(gap_i, gap_p) * 100:.1f}). No es exacto porque la simetría es una propiedad de la función
explicada y no del método: los árboles eligen en cada división una de las dos columnas casi idénticas, el ruido del 5 % rompe los empates y el f ajustado no es simétrico en ellas.
La correlación entre φ_orig y φ_copia es {_sg(cf_i['corr_phi_orig_copy'], 2)} (interventional) y {_sg(cf_p['corr_phi_orig_copy'], 2)} (`tree_path_dependent`).

La suma de ambas contribuciones ({_f(cf_i['mean_abs_sum'])} y {_f(cf_p['mean_abs_sum'])}) se compara con I<sub>j</sub> = {_f(t3['I_orig_without_copy'])} del modelo sin copia
({(cf_i['mean_abs_sum'] / t3['I_orig_without_copy'] - 1) * 100:+.1f} % y {(cf_p['mean_abs_sum'] / t3['I_orig_without_copy'] - 1) * 100:+.1f} %), mientras que cada columna por separado queda por debajo de esa suma. Con `interventional`, el
fondo rompe la dependencia entre columnas (evalúa combinaciones del original de un cliente con la copia de otro, que no ocurren en los datos); con `tree_path_dependent` sólo se usan las
proporciones de entrenamiento (`cover`) a lo largo de cada camino del árbol. Esta última **no** equivale a la esperanza condicional verdadera: sólo captura la dependencia que los
propios cortes del árbol reflejan y la ignora donde las divisiones no la expresan. Para una importancia real, esto significa que variables casi duplicadas se reparten el crédito (aquí, ninguna de las dos alcanza por sí sola la suma) y que la importancia
individual de cada una subestima la del concepto compartido; conviene sumar los grupos correlacionados, como la columna "suma" de la tabla.

### (d) Costo computacional

{chr(10).join(t_cost)}
Tiempo por evaluación entre k = 50 y k = 100: {_f(big[0]['time_per_evaluation_us'], 2)} µs y {_f(big[-1]['time_per_evaluation_us'], 2)} µs (diferencia de {rate_spread:.1%}); con k = 10 es {_f(kr[0]['time_kernel_s'] / kr[0]['model_evaluations'] * 1e6, 2)} µs. El tiempo {"crece de forma aproximadamente lineal con nsamples·k (para k ≥ 50)" if rate_spread < 0.15 else "no crece de forma lineal con nsamples·k"}; el costo por evaluación más alto
con k = 10 {"sugiere un costo fijo por instancia que se amortiza al subir k" if kr[0]['time_per_evaluation_us'] > big[0]['time_per_evaluation_us'] else "no se distingue de los demás"}. El MAE {"disminuye de forma monótona" if mae_mono else "no disminuye de forma monótona"} con k ({_f(kr[0]['mae'])} → {_f(kr[1]['mae'])} → {_f(kr[-1]['mae'])}); {"con k = 100 es " + f"{mae_drop:.1%}" + " menor que con k = 10" if mae_drop > 0 else "con k = 100 no es menor que con k = 10"}.
Con {t4['n_instances']} instancias y una sola corrida (el muestreo de KernelSHAP es aleatorio) este experimento no basta para afirmar que más fondo reduzca el error de forma sistemática.

Con un tiempo de `decision_function` sobre 100 filas de {d6['t_decision_function_100_s'] * 1e3:.3f} ms (mediana de {d6['n_repeats']} repeticiones), el Shapley exacto con 20 variables requiere
2<sup>20</sup> coaliciones × t(100) ≈ **{d6['exact_20_vars_s']:.0f} s ({d6['exact_20_vars_h']:.2f} h) por instancia**, frente a {_f(d6['kernel_per_instance_s_k100'], 3)} s de KernelSHAP con k = 100
({d6['exact_over_kernel']:.0f} veces menos). Con 60 variables y 1 µs por evaluación, 2<sup>60</sup> × 10<sup>−6</sup> s ≈ **{d6['years_60_vars_1us']:.3e} años** por instancia, por lo que el cálculo exacto es inviable
y se necesitan aproximaciones (KernelSHAP) o estructura del modelo (TreeSHAP).

**Prompt utilizado**

{PROMPT}

**Por qué funcionó este prompt**

{WHY}
"""
