"""Written report of Task 3 (Spanish, impersonal voice), built only from the numbers in results.json."""

from __future__ import annotations

from .report import _table

TASK3_BEGIN = "<!-- INFORME_TASK3:INICIO (generado por scripts/run_task3.py; no editar a mano) -->"
TASK3_END = "<!-- INFORME_TASK3:FIN -->"
FIRST_FIGURE = 14  # Task 1 uses Figura 1 - 6 and Task 2 uses Figura 7 - 13

PROMPT = """*3.1 y 3.2:* Pendiente (lo completa quien implementó esos incisos).

*3.3 a 3.5:* Con el enunciado del Laboratorio 10 y el repositorio, que ya tiene implementados los Tasks 1, 2, 3.1 y 3.2, implementa lo que falta del
Task 3 siguiendo la misma estructura (`task3.py` para los cálculos, `viz_task3.py`, `console_task3.py` y `report_task3.py`). En 3.3 compara |w| del
primer ajuste de LIME (20 variables, misma semilla) con |φ| de TreeSHAP interventional del Task 2.2 y calcula R²_w con K ∈ {3, 5, 10, 20} sobre las
mismas muestras. En 3.4 escribe la búsqueda de contrafactuales con NumPy: 20 000 candidatos que cambian de 1 a 3 variables accionables con
valores observados en el entrenamiento, distancia con MAD (o desviación absoluta media si la MAD es 0) y selección codiciosa de tres con
conjuntos de variables distintos. En 3.5 no escribas conclusiones sin datos: estima el N para un Jaccard de 0.9 a partir de la tabla del 3.2
y verifícalo con una corrida nueva, y prueba la causa del mayor desacuerdo LIME-SHAP con experimentos (ancho del kernel, P(good) al variar
una sola variable y efecto según el tamaño de la coalición). Agrega verificaciones numéricas con tolerancia a la consola."""

WHY = """Dar el enunciado completo y pedir que se siguiera la estructura existente fijó las fórmulas, los parámetros y los nombres, y permitió
reutilizar el modelo, los clientes y el LIME del 3.1 sin reimplementarlos. Pedir que cada conclusión del 3.5 saliera de un experimento evitó
respuestas genéricas: la prueba de interacción se validó a sí misma porque su promedio sobre tamaños reproduce el φ de TreeSHAP, y la
estimación de N se comprobó con una corrida que no se usó para ajustarla."""


def _f(v: float, nd: int = 3) -> str:
    return f"{v:.{nd}f}"


def _sg(v: float, nd: int = 3) -> str:
    return f"{v:+.{nd}f}"


def _join(items: list[str]) -> str:
    items = [f"`{i}`" for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " y " + items[-1]


def _non_decreasing(v: list[float]) -> bool:
    return all(b >= a for a, b in zip(v, v[1:]))


def _non_increasing(v: list[float]) -> bool:
    return all(b <= a for a, b in zip(v, v[1:]))


def render_informe_task3(results: dict) -> str:
    env, cfg, t1, t2 = results["environment"], results["config"], results["t3_1"], results["t3_2"]
    n_fig = iter(range(FIRST_FIGURE, FIRST_FIGURE + 10))

    def fig(name: str, cap: str) -> str:
        return f"![Figura {(n := next(n_fig))}. {cap}](outputs/task3/figures/{name})\n\n*Figura {n}. {cap}*\n"

    # ---- 3.1 ----
    blocks = []
    for key, c in t1.items():
        rows = [[f"`{x['feature']}`", x["value"], _sg(x["coef"])] for x in c["top"]]
        blocks.append(f"**Cliente {key}** (posición {c['test_position']} del conjunto de prueba; p(good) = {_f(c['p_good'])}; "
                      f"intercepto = {_f(c['intercept'])}; R²_w = {_f(c['r2w'])}; N_ef = {c['n_eff']:.1f}):\n\n"
                      + "\n".join(_table(["Variable", "Valor del cliente", "Coeficiente"], rows)))
    rej = t1["rechazado"]
    neg = [x["feature"] for x in rej["top"] if x["coef"] < 0]
    pos = [x["feature"] for x in rej["top"] if x["coef"] > 0]
    big = max((x for x in rej["top"] if x["coef"] < 0), key=lambda x: abs(x["coef"]), default=None)
    rej_txt = (f"En el cliente rechazado (p = {_f(rej['p_good'])}), {_join(neg)} {'tiene' if len(neg) == 1 else 'tienen'} coeficiente negativo, "
               f"es decir, conservar su valor en lugar de sustituirlo por uno de entrenamiento baja P(good) en la aproximación local; "
               f"el de mayor magnitud es `{big['feature']}` = {big['value']} ({_sg(big['coef'])})"
               if neg else "En el cliente rechazado ninguna de las cinco variables tiene coeficiente negativo")
    pos_items = [f"`{x['feature']}` = {x['value']} ({_sg(x['coef'])})" for x in rej["top"] if x["coef"] > 0]
    rej_txt += (f". En sentido contrario, {', '.join(pos_items)} {'tiene' if len(pos_items) == 1 else 'tienen'} coeficiente positivo."
                if pos_items else ".")
    r2 = {k: c["r2w"] for k, c in t1.items()}
    lo_k, hi_k = min(r2, key=r2.get), max(r2, key=r2.get)
    r2_txt = (f"R²_w va de {_f(r2[lo_k])} (cliente {lo_k}) a {_f(r2[hi_k])} (cliente {hi_k})"
              + (": en los tres clientes el modelo lineal con 5 variables explica menos de la mitad de la varianza ponderada de P(good), "
                 "de modo que los coeficientes describen sólo una parte del comportamiento local del modelo." if r2[hi_k] < 0.5
                 else "."))
    keys = list(t1)
    shared = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            s = sorted({x["feature"] for x in t1[a]["top"]} & {x["feature"] for x in t1[b]["top"]})
            if s:
                shared.append(f"{a} y {b}: {_join(s)}")
    shared_txt = ("Variables presentes en el top-5 de dos clientes: " + "; ".join(shared) + "." if shared
                  else "Ninguna variable aparece en el top-5 de dos clientes.")

    # ---- 3.2 ----
    ns, srs = t2["n_values"], t2["sigma_rel_values"]
    J, E = t2["jaccard"], t2["n_eff"]
    t_j = _table(["N \\ σ_rel", *[f"{s:g}" for s in srs]], [[str(n), *[_f(v, 4) for v in row]] for n, row in zip(ns, J)])
    t_e = _table(["N \\ σ_rel", *[f"{s:g}" for s in srs]], [[str(n), *[f"{v:.1f}" for v in row]] for n, row in zip(ns, E)])
    jcols = [[row[j] for row in J] for j in range(len(srs))]
    jrows = J
    grows_n = all(_non_decreasing(c) for c in jcols)
    grows_s = all(_non_decreasing(r) for r in jrows)
    falls_s = all(_non_increasing(r) for r in jrows)
    best = max(((J[i][j], ns[i], srs[j]) for i in range(len(ns)) for j in range(len(srs))))
    worst = min(((J[i][j], ns[i], srs[j]) for i in range(len(ns)) for j in range(len(srs))))
    n_txt = (f"El Jaccard medio crece con N en las tres anchuras (por ejemplo, con σ_rel = {srs[1]:g} pasa de {_f(jcols[1][0])} con N = {ns[0]} "
             f"a {_f(jcols[1][-1])} con N = {ns[-1]})" if grows_n
             else "El Jaccard medio no crece de forma monótona con N en todas las anchuras")
    s_txt = ("; con N fijo no disminuye al aumentar σ_rel" if grows_s else "; con N fijo disminuye al aumentar σ_rel" if falls_s
             else "; con N fijo no sigue un patrón monótono en σ_rel")
    ratio = [[E[i][j] / ns[i] for j in range(len(srs))] for i in range(len(ns))]
    mean_ratio = [sum(ratio[i][j] for i in range(len(ns))) / len(ns) for j in range(len(srs))]
    ratio_inc = _non_decreasing(mean_ratio)
    ratio_cols = [[ratio[i][j] for i in range(len(ns))] for j in range(len(srs))]
    spread = max(max(c) - min(c) for c in ratio_cols)
    e_txt = (f"N_ef equivale a una fracción de N que "
             + (f"casi no cambia con N (variación máxima de {spread:.1%} de N dentro de cada σ_rel)" if spread < 0.02 else
                f"cambia con N (variación máxima de {spread:.1%} de N dentro de una misma σ_rel)")
             + f"; promedia {mean_ratio[0]:.1%} de N con σ_rel = {srs[0]:g}, "
             f"{mean_ratio[1]:.1%} con {srs[1]:g} y {mean_ratio[2]:.1%} con {srs[2]:g}"
             + (", de modo que un kernel más ancho reparte el peso entre más muestras." if ratio_inc else "."))

    return f"""# Task 3

## Configuración

- Salida explicada: P(good) del mismo `GradientBoostingClassifier` y los mismos tres clientes del Task 2 (`credit.build_base` y `select_clients`).
- Semilla aleatoria: **{env['seed']}**. Muestras por explicación: N = {cfg['n']}; variables conservadas: K = {cfg['k']}; λ = {cfg['lambda']:g};
  σ_rel = {cfg['sigma_rel']:g} con d' = {cfg['d_prime']}, es decir σ = σ_rel·√d' = {cfg['sigma']:.4f}.
- Variable ausente: z'_j ~ Bernoulli(0.5) de forma independiente por muestra y variable; con z'_j = 1 se conserva el valor del cliente y
  con z'_j = 0 se toma un valor al azar de la columna j del conjunto de entrenamiento. Peso π = exp(−D²/σ²), con D² igual al número de variables ausentes.
- Ajuste: regresión ridge ponderada en forma cerrada con `np.linalg.solve`; el intercepto **no se penaliza**. Se ajustan las 20 variables, se
  conservan las K de mayor |w| y se reajusta sólo con ellas; R²_w se calcula sobre ese reajuste con las mismas muestras y pesos.
- Versiones: scikit-learn {env['sklearn']}, NumPy {env['numpy']}, pandas {env['pandas']}, Python {env['python']}.

## Task 3.1 - LIME desde cero

{chr(10).join(b + chr(10) for b in blocks)}
{fig("fig_3_1_coefs.png", "Coeficientes LIME (top-5) de los tres clientes; R²_w del reajuste en el título de cada panel.")}
{rej_txt} {r2_txt} {shared_txt}

## Task 3.2 - Estabilidad

Cliente rechazado (posición {t2['test_position']}); para cada combinación de N y σ_rel se corrieron {len(t2['seeds'])} explicaciones con las semillas
{t2['seeds'][0]} a {t2['seeds'][-1]} y se promedió el Jaccard del top-{t2['k']} sobre los 190 pares.

**Jaccard medio** (filas N, columnas σ_rel):

{chr(10).join(t_j)}
**N_ef medio** (filas N, columnas σ_rel):

{chr(10).join(t_e)}
{fig("fig_3_2_stability.png", "Jaccard medio y tamaño efectivo de muestra N_ef según N y σ_rel (cliente rechazado).")}
{n_txt}{s_txt}; el valor más alto es {_f(best[0])} (N = {best[1]}, σ_rel = {best[2]:g}) y el más bajo {_f(worst[0])} (N = {worst[1]}, σ_rel = {worst[2]:g}). {e_txt}

{_render_3_3(results, fig)}
{_render_3_4(results, fig)}
{_render_3_5(results, fig)}
**Prompt utilizado**

{PROMPT}

**Por qué funcionó este prompt**

{WHY}
"""


# Markdown headers with literal "|" (escaped), kept outside f-string expressions for Python < 3.12
H_3_3A = ["Cliente", "p(good)", "Spearman \\|w\\| vs \\|φ\\|", "Spearman vs \\|φ\\| (probabilidad)", "Top-3 LIME", "Top-3 SHAP", "Coincidencia"]
H_TOP3_PHI = "Variable (top-3 \\|φ\\|)"

MIN_FIDELITY = 0.8  # 3.5(c): minimum R^2_w accepted for an explanation that justifies a credit decision
K_SHARE = 0.9  # 3.5(c): K = smallest value that reaches this share of the R^2_w obtained with all 20 variables


def _render_3_3(results: dict, fig) -> str:
    a, b = results["t3_3"]["a"], results["t3_3"]["b"]
    t1 = results["t3_1"]
    rows = [[k, _f(t1[k]["p_good"]), _f(c["spearman"]), _f(c["spearman_vs_shap_probability"]), _join(c["top_lime"]),
             _join(c["top_shap"]), f"{c['top_overlap']}/3"] for k, c in a.items()]
    rho = {k: c["spearman"] for k, c in a.items()}
    lo, hi = min(rho, key=rho.get), max(rho, key=rho.get)
    full = [k for k, c in a.items() if c["top_overlap"] == 3]
    part = [f"{k} ({c['top_overlap']}/3: LIME pone {_join(sorted(set(c['top_lime']) - set(c['top_shap'])))} y SHAP "
            f"{_join(sorted(set(c['top_shap']) - set(c['top_lime'])))})" for k, c in a.items() if c["top_overlap"] < 3]
    by_k = b["by_k"]
    t_k = _table(["K", "R²_w", "Variables añadidas respecto del K anterior"],
                 [[str(r["k"]), _f(r["r2w"]), _join(r["features"][len(prev["features"]) if prev else 0:])]
                  for prev, r in zip([None, *by_k[:-1]], by_k)])
    r2 = [r["r2w"] for r in by_k]
    gains = [f"{_sg(r2[i + 1] - r2[i])} de K = {by_k[i]['k']} a {by_k[i + 1]['k']}" for i in range(len(r2) - 1)]
    return f"""## Task 3.3 - Comparación de LIME con SHAP

### (a) Ranking y coincidencia de variables

Spearman entre |w| de LIME (primer ajuste, 20 variables, misma semilla que en 3.1) y |φ| de TreeSHAP interventional del Task 2.2
(mismo modelo y mismo fondo de 100 filas), y coincidencia entre las tres variables principales de cada método. Como LIME explica P(good)
y SHAP el log-odds, se agrega como control la correlación con |φ| calculado por TreeSHAP sobre la probabilidad.

{chr(10).join(_table(H_3_3A, rows))}
La correlación va de {_f(rho[lo])} (cliente {lo}) a {_f(rho[hi])} (cliente {hi}): ambos métodos ordenan las variables de forma parecida.
{("El top-3 coincide por completo en " + " y ".join(full) + ". ") if full else ""}{("Difiere en " + "; ".join(part) + ".") if part else ""}

### (b) Fidelidad según el número de variables (cliente rechazado)

{chr(10).join(t_k)}
{fig("fig_3_3_lime_vs_shap.png", "|w| de LIME contra |φ| de TreeSHAP por cliente y R²_w del cliente rechazado según K.")}
R²_w crece con K ({"; ".join(gains)}), pero incluso con las 20 variables el modelo lineal en z' explica sólo {r2[-1]:.1%} de la varianza
ponderada de P(good). El límite no lo pone K sino la forma del modelo sustituto: un modelo aditivo en presencia/ausencia no puede
representar las interacciones del modelo de árboles (ver 3.5(b)).
"""


def _render_3_4(results: dict, fig) -> str:
    t = results["t3_4"]
    sc = t["scales"]
    sc_txt = ", ".join(f"`{n}` = {v['value']:g} ({v['kind']})" for n, v in sc.items())
    rows = []
    for i, s in enumerate(t["selected"], 1):
        rows.append([str(i), _join([c["feature"] for c in s["changes"]]),
                     "; ".join(f"{c['original']} → {c['new']}" for c in s["changes"]), _f(s["distance"]), _f(s["p_good"])])
    by_m = t["by_n_changed"]
    m_txt = "; ".join(f"{m} variable{'s' if int(m) > 1 else ''}: {v['valid']} de {v['candidates']:,}" for m, v in by_m.items())
    fb = (f"Ningún candidato fue válido con {results['config'].get('cf_max', 3)} cambios, así que se aumentó el máximo a "
          f"{t['max_changes']} variables." if t["fallback_to_5"] else
          f"Hubo candidatos válidos con el máximo de {t['max_changes']} variables, por lo que no fue necesario subirlo a 5.")
    return f"""## Task 3.4 - Explicaciones contrafactuales

Cliente rechazado (posición {t['test_position']}, P(good) = {_f(t['p_good_original'])}). Se generaron {t['n_candidates']:,} candidatos; cada uno
cambia un número uniforme de 1 a {t['max_changes']} variables accionables ({_join(t['actionable'])}) elegidas al azar, con valores tomados
al azar de los valores observados en el entrenamiento distintos del original (con su frecuencia). Las inmutables ({_join(t['immutable'])})
y el resto de variables no se modifican. Un candidato es válido si P(good) ≥ θ = {t['threshold']:g}. Escalas de la distancia: {sc_txt};
las categóricas cuentan 1 si cambian.

{chr(10).join(_table(["#", "Variables modificadas", "Original → nuevo", "Distancia", "P(good)"], rows))}
{fig("fig_3_4_counterfactuals.png", "P(good) del cliente rechazado y de los tres contrafactuales seleccionados (línea: umbral θ).")}
| Métrica | Valor |
|---|---|
| Validez (candidatos válidos / total) | **{t['validity']:.3%}** ({t['n_valid']} de {t['n_candidates']:,}) |
| Proximidad promedio (3 seleccionados) | **{_f(t['proximity_selected'])}** (todos los válidos: {_f(t['proximity_valid'])}) |
| Dispersión: variables cambiadas en promedio (3 seleccionados) | **{t['sparsity_selected']:.1f}** (todos los válidos: {t['sparsity_valid']:.1f}) |
| Máximo P(good) entre todos los candidatos | {_f(t['max_p_good_candidates'])} |

Válidos según el número de variables cambiadas: {m_txt}. {fb} Ningún cambio de una o dos variables basta para pasar el umbral, de modo que
el rechazo no depende de una sola variable accionable.
"""


def _render_3_5(results: dict, fig) -> str:
    t2, a, b, d = results["t3_2"], results["t3_5"]["a"], results["t3_5"]["b"], results["t3_5"]["d"]
    by_k = results["t3_3"]["b"]["by_k"]
    t4 = results["t3_4"]

    # ---- (a) ----
    ok = a["check_jaccard"] >= a["target"]
    a_txt = f"""### (a) Relación entre estabilidad y tamaño efectivo

En las nueve celdas del Task 3.2 el Jaccard medio aumenta con N_ef (Spearman = {_f(a['pooled_spearman_neff_jaccard'])}, p = {a['pooled_spearman_p']:.1e}):
la estabilidad depende de cuántas muestras pesan de verdad en el ajuste, más que de N o de σ_rel por separado. Para σ_rel = {a['sigma_rel']:g}, N_ef es
{a['n_eff_over_n']:.1%} de N, así que N_ef y N son casi proporcionales.

**Estimación.** El error de los coeficientes de la regresión ponderada disminuye como 1/√N_ef, y el top-{t2['k']} cambia cuando ese error intercambia
la variable K-ésima con la siguiente; por eso se modeló 1 − J = a·N_ef^(−b) y se ajustó con los dos puntos de la tabla que encierran el objetivo
(N = {a['bracket_n'][0]} y N = {a['bracket_n'][1]}), es decir, una interpolación en escala log-log y no una extrapolación fuera del rango medido. Resultado:
b = {a['fit_b']:.2f}, N_ef necesario = {a['n_eff_needed']:.0f} y **N ≈ {a['n_estimate']}** (N_ef / {a['n_eff_over_n']:.3f}, redondeado hacia arriba a un múltiplo de 100).

**Verificación.** Una corrida adicional con N = {a['n_estimate']}, σ_rel = {a['sigma_rel']:g} y las mismas {a['check_runs']} semillas dio un Jaccard medio de
**{_f(a['check_jaccard'])}** (N_ef = {a['check_n_eff']:.0f}), {"que alcanza" if ok else "que no alcanza"} el objetivo de {a['target']:g}.
{"La estimación se confirma." if ok else "La estimación se queda corta; hace falta un N mayor."} El valor tiene su propio ruido: con 20 corridas el Jaccard medio
de una celda varía de una ejecución a otra, por lo que la cifra debe leerse como orden de magnitud (unas {a['n_estimate'] // 100 * 100} muestras) y no como un umbral exacto.

{fig("fig_3_5a_jaccard_neff.png", "Jaccard medio contra N_ef en las nueve celdas del 3.2, curva ajustada y corrida de verificación.")}"""

    # ---- (b) ----
    loc, ice, it = b["locality"], b["ice"], b["interaction"]
    eff = it["effect_log_odds"]
    flip = next((k for k, v in zip(it["sizes"], eff) if v > 0), None)
    pos_runs = sum(r["positive_runs"] for r in loc)
    tot_runs = len(loc) * results["config"].get("disagree_seeds", 10)
    same_sign_prob = (b["lime_w"] > 0) == (b["shap_phi_probability"] > 0)
    mid = [k for k in it["sizes"] if 8 <= k <= 11]
    b_txt = f"""### (b) Desacuerdo entre LIME y SHAP

Se buscó la mayor diferencia de puesto entre el ranking de |w| (LIME) y el de |φ| (SHAP) en los tres clientes: **cliente {b['client']},
variable `{b['feature']}`** (puesto {b['rank_lime']} en LIME y {b['rank_shap']} en SHAP). Además, los signos son opuestos: w = {_sg(b['lime_w'], 4)}
(P(good)) y φ = {_sg(b['shap_phi'], 4)} (log-odds).

**Hipótesis:** el efecto de `{b['feature']}` = {ice['client_value']} depende de los valores de las demás variables (interacción). SHAP promedia su
contribución sobre coaliciones de todos los tamaños con el mismo peso, mientras que LIME concentra sus muestras en coaliciones con la mitad de las
variables presentes y explica la probabilidad, no el log-odds. Se probaron las tres explicaciones posibles:

1. **Localidad.** Con σ_rel ∈ {{{", ".join(f"{r['sigma_rel']:g}" for r in loc)}}} y 10 semillas por valor, el coeficiente medio de LIME fue
   {", ".join(_sg(r['w_mean'], 4) for r in loc)} (positivo en {pos_runs} de {tot_runs} corridas) y su puesto medio
   {", ".join(f"{r['rank_mean']:.1f}" for r in loc)}. Reducir el kernel **no** acerca LIME a SHAP: el signo no cambia, de modo que la
   discrepancia no se debe a la localidad.
2. **Relación no lineal.** Al variar sólo `{b['feature']}` en el cliente (panel izquierdo de la figura siguiente), P(good) es una función escalonada y
   {"monótona" if ice['monotone'] else "no monótona"}, que va de {_f(max(ice['p_good']))} a {_f(min(ice['p_good']))}. Con su valor actual ({ice['client_value']}) P(good) = {_f(ice['client_p'])},
   por encima del promedio al sustituirlo por los valores de entrenamiento ({_f(ice['mean_p_replaced'])}); {ice['share_train_values_with_lower_p']:.1%} de esos valores dan una
   probabilidad menor. La no linealidad existe, pero con las demás variables del cliente fijas el valor actual favorece la aprobación, lo
   que coincide con el signo de LIME y no con el de SHAP.
3. **Interacción.** Se midió el efecto medio de conservar `{b['feature']}` = {ice['client_value']} según cuántas de las otras 19 variables conservan
   el valor del cliente ({it['samples_per_size']} coaliciones por tamaño, el resto tomado del fondo de SHAP). En log-odds el efecto pasa de
   {_sg(eff[0])} (ninguna otra variable del cliente) a {_sg(eff[-1])} (las 19){f" y cambia de signo a partir de {flip} variables" if flip is not None else ""}.
   El promedio uniforme sobre los tamaños, que es lo que hace Shapley, da {_sg(it['shapley_from_sizes_log_odds'], 4)}, prácticamente igual a
   φ = {_sg(b['shap_phi'], 4)}, lo que valida la prueba. En los tamaños donde se concentra el muestreo de LIME ({mid[0]} a {mid[-1]} variables) el efecto en
   log-odds es {", ".join(_sg(eff[k]) for k in mid)}, y en probabilidad el promedio sobre tamaños es {_sg(it['shapley_from_sizes_probability'], 4)}.
   Con TreeSHAP sobre la probabilidad, φ = {_sg(b['shap_phi_probability'], 4)} (puesto {b['rank_shap_probability']}){", del mismo signo que LIME" if same_sign_prob else ""}.

{fig("fig_3_5b_disagreement.png", f"Cliente {b['client']}, variable {b['feature']}: P(good) al variar sólo esa variable, efecto según el número de variables del cliente presentes y coeficiente de LIME según σ_rel.")}
**Conclusión:** la discrepancia se debe a una **interacción**: el efecto de `{b['feature']}` cambia de signo según el contexto, y cada método
promedia contextos distintos (SHAP pondera por igual las coaliciones pequeñas, donde el efecto es negativo). La escala de salida la acentúa,
porque la misma interacción promediada en probabilidad da un φ positivo. No se debe a la localidad, y la no linealidad de `{b['feature']}`
por sí sola no explica el cambio de signo."""

    # ---- (c) ----
    r2 = {r["k"]: r["r2w"] for r in by_k}
    kmax = max(r2)
    chosen = next(k for k in sorted(r2) if r2[k] >= K_SHARE * r2[kmax])
    ks = sorted(r2)
    prev = ks[ks.index(chosen) - 1] if ks.index(chosen) else None
    below = (f" por debajo, con K = {prev}, se pierde {r2[chosen] - r2[prev]:.3f} de R²_w y sólo queda el "
             f"{r2[prev] / r2[kmax]:.1%} de la fidelidad máxima." if prev is not None else "")
    # comprehensibility: a person follows about 5 to 7 reasons; say explicitly when the chosen K exceeds it
    if chosen > 7:
        trade = (f"La comprensibilidad favorece pocos términos (una persona sigue con comodidad unas 5 a 7 razones) y K = {chosen} "
                 f"las supera: se acepta ese costo porque la fidelidad ya es baja y recortarla{f' a K = {prev}' if prev else ''} dejaría una "
                 f"explicación que describe todavía menos el comportamiento local del modelo.")
    else:
        trade = (f"K = {chosen} cabe dentro de las 5 a 7 razones que una persona sigue con comodidad, así que cumple a la vez "
                 f"fidelidad y comprensibilidad.")
    c_txt = f"""### (c) Fidelidad y comprensibilidad

Con la tabla de R²_w del 3.3(b) ({", ".join(f"K = {k}: {_f(r2[k])}" for k in ks)}) se eligió **K = {chosen}**, el menor K que alcanza el
{K_SHARE:.0%} de la fidelidad obtenida con las 20 variables ({r2[chosen] / r2[kmax]:.1%}). A partir de ahí, duplicar las variables sólo agrega
{_sg(r2[kmax] - r2[chosen])} de R²_w y vuelve la explicación más difícil de leer;{below}
{trade}

Para una decisión de crédito se exigiría **R²_w ≥ {MIN_FIDELITY:g}**: la explicación se usa para justificar un rechazo ante el cliente y el regulador,
así que debe reproducir la mayor parte del comportamiento local del modelo; con una fidelidad menor, una razón dada podría no ser la que
realmente movió la decisión. **Ningún K la alcanza** (el máximo es {_f(r2[kmax])}). En ese caso no se entregaría la explicación de LIME como justificación.
Se usaría SHAP, cuya eficiencia se cumple con error de 1e-7 (Task 2.2), o el contrafactual del 3.4, cuya P(good) la calcula el propio modelo.
Para LIME, se podría intentar un sustituto que incluya interacciones (3.5(b) muestra que son la causa de la baja fidelidad) y reportar siempre R²_w junto a los coeficientes."""

    # ---- (d) ----
    top = d["top3_shap"]
    t_d = _table([H_TOP3_PHI, "Valor", "φ", "Tipo", "¿En contrafactuales?", "Mejor cambio único", "P(good) con ese cambio", "Costo"],
                 [[f"`{r['feature']}`", r["value"], _sg(r["phi"]), r["role"], "sí" if r["in_counterfactuals"] else "no",
                   r["best_single_value"], _f(r["best_single_p"]), _f(r["best_single_cost"], 2)] for r in top])
    missing = [r for r in top if not r["in_counterfactuals"]]
    reasons = []
    for r in missing:
        f = r["feature"]
        if r["role"] != "accionable":
            reasons.append(f"`{f}` (φ = {_sg(r['phi'])}) {'es inmutable' if r['role'] == 'inmutable' else 'no es accionable'} en este ejercicio, así que la búsqueda nunca la modifica. Aunque se pudiera, "
                           f"el mejor valor ({r['best_single_value']}) sólo lleva P(good) de {_f(t4['p_good_original'])} a {_f(r['best_single_p'])} con costo {r['best_single_cost']:g}, lejos de θ = {t4['threshold']:g}")
        else:
            bf = t4["by_feature"][f]
            reasons.append(f"`{f}` (φ = {_sg(r['phi'])}) es accionable, pero su φ es {'positivo: su valor actual ya empuja hacia la aprobación' if r['phi'] > 0 else 'negativo'}; "
                           f"cambiarla nunca ayuda lo suficiente: el mejor valor ({r['best_single_value']}) da P(good) = {_f(r['best_single_p'])} (el promedio de sus valores, {_f(r['mean_single_p'])}), "
                           f"y de {bf['candidates']:,} candidatos que la cambian, {bf['valid']} son válidos. Su efecto por cambio es pequeño o contrario")
    d_txt = f"""### (d) SHAP frente a contrafactuales

{chr(10).join(t_d)}
Los contrafactuales cambian {_join(d['cf_features'])}. Del top-3 de |φ| sólo aparece {_join([r['feature'] for r in top if r['in_counterfactuals']]) if any(r['in_counterfactuals'] for r in top) else 'ninguna'}.
{". ".join(reasons)}. A la vez, los contrafactuales usan variables con |φ| bajo en este cliente ({_join(sorted(set(d['cf_features']) - {r['feature'] for r in top}))}) porque son baratas en la distancia y su
combinación cruza el umbral.

SHAP responde **"¿cuánto aportó cada variable a llevar el puntaje desde el promedio del fondo hasta el de este cliente?"**: reparte la
diferencia f(x) − φ0 entre las variables, sean o no modificables. El contrafactual responde **"¿qué cambio mínimo y factible haría que el
modelo aprobara la solicitud?"**: busca la frontera de decisión más cercana respetando las restricciones de accionabilidad. Una variable puede
ser la principal razón del puntaje y no aparecer en ningún contrafactual, si es inmutable o si cambiarla no alcanza el umbral."""

    return f"""## Task 3.5 - Análisis

{a_txt}
{b_txt}

{c_txt}

{d_txt}
"""
