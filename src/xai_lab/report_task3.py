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

**Prompt utilizado**

{PROMPT}

**Por qué funcionó este prompt**

{WHY}
"""
