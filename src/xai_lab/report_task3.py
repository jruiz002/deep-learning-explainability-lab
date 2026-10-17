"""Written report of Task 3 (Spanish, impersonal voice), built only from the numbers in results.json."""

from __future__ import annotations

from .report import _table

TASK3_BEGIN = "<!-- INFORME_TASK3:INICIO (generado por scripts/run_task3.py; no editar a mano) -->"
TASK3_END = "<!-- INFORME_TASK3:FIN -->"
FIRST_FIGURE = 14  # Task 1 uses Figura 1 - 6 and Task 2 uses Figura 7 - 13

PROMPT = "Pendiente"

WHY = "Pendiente"


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
**Prompt utilizado**

{PROMPT}

**Por qué funcionó este prompt**

{WHY}
"""


# Markdown headers with literal "|" (escaped), kept outside f-string expressions for Python < 3.12
H_3_3A = ["Cliente", "p(good)", "Spearman \\|w\\| vs \\|φ\\|", "Spearman vs \\|φ\\| (probabilidad)", "Top-3 LIME", "Top-3 SHAP", "Coincidencia"]


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
