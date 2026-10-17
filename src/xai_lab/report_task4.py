"""Written report of Task 4 (Spanish, impersonal voice), built from results.json of Task 4 and, for 4.3(c), of Tasks 2 and 3."""

from __future__ import annotations

from .report import _table

TASK4_BEGIN = "<!-- INFORME_TASK4:INICIO (generado por scripts/run_task4.py; no editar a mano) -->"
TASK4_END = "<!-- INFORME_TASK4:FIN -->"
FIRST_FIGURE = 20  # Task 1: 1 - 6, Task 2: 7 - 13, Task 3: 14 - 19

PROMPT = """Con el enunciado del Laboratorio 10 y el repositorio (Grad-CAM propio del Task 1 en `gradcam.py`), implementa el Task 4 con la
misma estructura que los demás (`task4.py`, `viz_task4.py`, `console_task4.py`, `report_task4.py` y `scripts/run_task4.py`), sin
`pytorch-grad-cam` ni `captum`. En 4.1 ordena los 224×224 píxeles por el mapa normalizado de la clase top-1; en P = 20 pasos reemplaza por gris
medio (deletion) o inserta sobre la imagen desenfocada con σ = 10 (insertion) la fracción k/P, trabajando en [0, 1] y normalizando justo antes
del modelo, y compara con 10 órdenes aleatorios con semilla. En 4.2 reinicializa con `reset_parameters` de PyTorch, de forma acumulada, `fc`,
`layer4`, `layer3`, `layer2`, `layer1` y `conv1`, y calcula el Spearman entre mapas de 7×7 de la clase original. Para 4.3 corre los
experimentos que piden los incisos (deletion de (a) con referencia desenfocada y la comprobación de que, con sólo `fc` aleatorio, el mapa es
Σ_k w_k A^k / Z con las activaciones originales) y usa los números de los Tasks 2 y 3 para la parte de audiencias. Imprime verificaciones
numéricas con tolerancia."""

WHY = """Nombrar el espacio de píxeles ([0, 1] antes de normalizar), el gris, el desenfoque y las semillas evitó ambigüedades que cambian las
curvas, y exigir comparación contra órdenes aleatorios convirtió cada AUC en una cifra interpretable. Pedir la comprobación de α = w/Z en la
primera etapa de la cascada transformó la "predicción" del 4.3(b) en una prueba numérica (error del orden de 1e-8), y obligar a usar los
números de los Tasks 2 y 3 en 4.3(c) evitó recomendaciones genéricas para cada audiencia."""


def _f(v: float, nd: int = 3) -> str:
    return f"{v:.{nd}f}"


def _sg(v: float, nd: int = 3) -> str:
    return f"{v:+.{nd}f}"


def _rho(v: float) -> str:
    return "indef." if v != v else f"{v:+.3f}"


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " y " + items[-1]


def _sd(v: float) -> str:
    return f"{abs(v):.1f} desviaciones estándar {'a favor del mapa' if v > 0 else 'en contra del mapa'}"


def render_informe_task4(results: dict, prior: dict) -> str:
    env, cfg, t41, t42 = results["environment"], results["config"], results["t4_1"], results["t4_2"]
    blur = results["t4_3a_blur_reference"]
    n_fig = iter(range(FIRST_FIGURE, FIRST_FIGURE + 10))

    def fig(name: str, cap: str) -> str:
        return f"![Figura {(n := next(n_fig))}. {cap}](outputs/task4/figures/{name})\n\n*Figura {n}. {cap}*\n"

    # ---- 4.1 ----
    tables = []
    for rid, r in t41.items():
        rows = []
        for key, label in (("deletion", "AUC_del (menor es mejor)"), ("insertion", "AUC_ins (mayor es mejor)")):
            s = r[key]
            rows.append([label, _f(s["map"], 4), f"{s['random_mean']:.4f} ± {s['random_std']:.4f}", _sg(s["diff_vs_random"], 4),
                         f"{s['sds_better_than_random']:+.1f}"])
        tables.append(f"**Imagen ({rid})**, clase `{r['class_name']}` (p = {_f(r['p_top1'], 4)}; con la imagen toda en gris p = {r['p_reference']:.4f}; "
                      f"desenfocada p = {r['p_blurred']:.4f}):\n\n"
                      + "\n".join(_table(["Métrica", "Grad-CAM", "Azar (media ± desv., 10 órdenes)", "Diferencia", "Desviaciones a favor del mapa"], rows)))

    # ---- 4.2 ----
    ids = list(t41)
    st = t42["stages"]
    rows = [[str(s["stage"]), ", ".join(f"`{n}`" for n in s["randomized"]),
             *[f"{_rho(s['images'][i]['spearman'])} ({_rho(s['images'][i]['spearman_raw'])})" for i in ids]] for s in st]
    t_cas = _table(["Etapa", "Capas reinicializadas", *[f"({i}) {t41[i]['class_name']}" for i in ids]], rows)
    below = t42["first_stage_below"]

    # figures of 4.1 and 4.2 are numbered before the one inside 4.3(a), in the order they appear in the document
    fig_curves = fig("fig_4_1_curves.png", "Curvas de deletion (gris) e insertion (desde la imagen desenfocada) con el orden de Grad-CAM y con 10 órdenes aleatorios (media ± desviación).")
    fig_frames = fig("fig_4_1_frames.png", "Imagen (a): pasos de deletion e insertion con el orden de Grad-CAM y con un orden aleatorio.")
    fig_maps = fig("fig_4_2_cascade_maps.png", "Mapas de 7×7 de la clase original en cada etapa de la cascada (cada mapa escalado a su máximo) y su Spearman con el original.")
    fig_rho = fig("fig_4_2_cascade_spearman.png", "Spearman con el mapa original en cada etapa de la cascada.")

    # ---- 4.3(a) ----
    ins_sd = {i: t41[i]["insertion"]["sds_better_than_random"] for i in ids}
    del_sd = {i: t41[i]["deletion"]["sds_better_than_random"] for i in ids}
    del_win = [i for i in ids if del_sd[i] > 2]
    del_tie = [i for i in ids if -2 <= del_sd[i] <= 2]
    del_lose = [i for i in ids if del_sd[i] < -2]
    a = t41["a"]
    ca = a["curves"]["deletion_map"]
    ra = a["curves"]["deletion_random_mean"]
    plateau = next((k for k in range(len(ca)) if ca[k] < 0.9 * ca[0]), len(ca)) - 1  # last step before the first drop
    half_r = [h for h in a["half_drop_step_random"] if h is not None]
    bd, bg = blur["deletion"], blur["deletion_gray"]
    low_p = [i for i in ids if t41[i]["p_top1"] < 0.5]
    a_txt = f"""### (a) Validación contra el azar

Con el criterio de 2 desviaciones estándar: en **insertion** Grad-CAM supera al azar en las tres imágenes
({", ".join(f"({i}) {ins_sd[i]:.1f}" for i in ids)} desviaciones estándar). En **deletion** con gris
{"; ".join(x for x in [("lo supera en " + _and([f"({i})" for i in del_win])) if del_win else "", ("no se distingue del azar en " + _and([f"({i}) ({del_sd[i]:+.1f})" for i in del_tie])) if del_tie else "", ("queda **por debajo** del azar en " + _and([f"({i}) ({del_sd[i]:+.1f})" for i in del_lose])) if del_lose else ""] if x)}.

**Forma de las curvas.** El mapa de 224×224 es la interpolación bilineal de un mapa de 7×7, así que tiene sólo 49 grados de libertad: cada
paso cambia {cfg['pixels_per_step']:.0f} píxeles (unas {cfg['pixels_per_step'] / 1024:.1f} celdas de 32×32) y el orden de Grad-CAM recorre bloques contiguos celda por celda.
En (a), el primer paso con el orden del mapa toca {a['cells_touched_map'][0]} de las 49 celdas, mientras que un orden aleatorio toca las {a['cells_touched_random_step1']} desde el primer paso. Por eso la
curva del mapa tiene mesetas y saltos: la probabilidad se mantiene por encima del 90 % de la inicial hasta el paso {plateau} ({plateau / cfg['steps']:.0%} de los píxeles),
cae a la mitad en el paso {a['half_drop_step_map']} y el salto más grande ({_f(a['largest_step_drop_map'])}) ocurre en el paso {a['largest_step_drop_map_at']}, cuando se borran
las celdas que cubren el animal. Con órdenes aleatorios la probabilidad decae de forma suave desde el primer paso ({_f(ra[0])} → {_f(ra[1])}) y cae a la mitad en el paso
{min(half_r)} a {max(half_r)}: los píxeles grises dispersos funcionan como ruido sobre toda la imagen y destruyen la textura que usa la red, aunque no borren ninguna región completa.
Por eso el azar resulta más "eficaz" en deletion que el mapa. En insertion ocurre lo contrario: unos pocos píxeles nítidos dispersos sobre la imagen desenfocada no
reconstruyen nada (la curva aleatoria de (a) sigue en {_f(a['curves']['insertion_random_mean'][10])} a la mitad de los pasos), mientras que el mapa inserta regiones enteras del objeto
({_f(a['curves']['insertion_map'][2])} en el paso 2).{(" En " + _and([f"({i})" for i in low_p]) + " la probabilidad inicial ya es baja (" + _and([_f(t41[i]['p_top1']) for i in low_p]) + ") y cualquier orden la lleva casi a cero en el primer paso, de modo que AUC_del queda dominada por ese piso y discrimina poco.") if low_p else ""}

**Efecto de los parches grises (imagen (a)).** Se repitió deletion reemplazando los píxeles por los de la imagen desenfocada en lugar de gris:

| Referencia | AUC_del Grad-CAM | AUC_del azar (media ± desv.) | Desviaciones a favor del mapa |
|---|---|---|---|
| Gris | {_f(bg['map'], 4)} | {bg['random_mean']:.4f} ± {bg['random_std']:.4f} | {bg['sds_better_than_random']:+.1f} |
| Desenfocada (σ = {cfg['blur_sigma']:g}) | {_f(bd['map'], 4)} | {bd['random_mean']:.4f} ± {bd['random_std']:.4f} | {bd['sds_better_than_random']:+.1f} |

{fig("fig_4_3a_blur_reference.png", "Deletion de la imagen (a) con referencia gris y con referencia desenfocada, frente al azar.")}
Con la referencia desenfocada, el AUC_del del mapa cambia {_sg(bd['map'] - bg['map'], 4)} y el del azar {_sg(bd['random_mean'] - bg['random_mean'], 4)}: los parches grises castigan
sobre todo al orden aleatorio, porque introducen bordes artificiales que la red no vio en el entrenamiento. La desventaja del mapa se reduce de {abs(bg['sds_better_than_random']):.1f} a
{abs(bd['sds_better_than_random']):.1f} desviaciones estándar{", pero no desaparece" if bd['sds_better_than_random'] < -2 else ""}: aun con desenfoque, cambiar píxeles dispersos degrada la textura de toda la imagen. Por eso
deletion con órdenes aleatorios a nivel de píxel no es una referencia justa para un mapa de 7×7; insertion, o un azar que permute celdas de 32×32,
mide mejor la calidad del mapa."""

    # ---- 4.3(b) ----
    s1 = st[0]["images"]
    pos = [i for i in ids if s1[i]["spearman"] > 0 and s1[i]["fc_weight_sum"] > 0]
    neg = [i for i in ids if s1[i]["spearman"] < 0 and s1[i]["fc_weight_sum"] < 0]
    resid_txt = " ".join(x for x in [
        (f"En {_and([f'({i})' for i in pos])} la suma de pesos aleatorios es positiva y el mapa original se parece a Σ_k A^k, así que el mapa "
         "aleatorio conserva una correlación positiva con el original.") if pos else "",
        (f"En {_and([f'({i})' for i in neg])} la suma es negativa y la correlación también lo es.") if neg else ""] if x)

    b_txt = f"""### (b) Sensibilidad a los parámetros del modelo

Etapa en la que el Spearman cae por primera vez por debajo de {cfg['spearman_low']:g}: {", ".join(f"({i}) etapa {below[i]} (`{st[below[i] - 1]['randomized'][-1]}`)" if below[i] else f"({i}) nunca" for i in ids)}.

**Predicción.** En ResNet-18, α_k^c = w_k^c / Z, de modo que L^c = ReLU(Σ_k w_k^c A^k / Z). Si sólo se aleatoriza `fc`, las activaciones A de `layer4`
no cambian (diferencia máxima medida: {max(s1[i]['activations_unchanged'] for i in ids):.1e}) y el nuevo mapa es exactamente Σ_k w_k^rand A^k / Z: una combinación con pesos al azar
de los mismos 512 mapas de activación. Debería perder la relación con la clase y conservar sólo la estructura espacial común a todos los canales
(dónde hay activación en general), así que la correlación con el original debería caer muy por debajo de 1 y quedar con un signo y una magnitud
que dependen del sorteo de los pesos.

**Comparación con la etapa 1.** La identidad se cumple con error máximo de {t42['fc_prediction_max_abs_error']:.1e} entre el mapa calculado y Σ_k w_k^rand A^k / Z. Los Spearman
de la etapa 1 son {", ".join(f"({i}) {_rho(s1[i]['spearman'])}" for i in ids)}, todos lejos de 1. Lo que queda de correlación sale de la estructura común: el
Spearman entre el mapa original y la suma sin pesos Σ_k A^k es {", ".join(f"({i}) {_rho(s1[i]['spearman_original_vs_sum_A'])}" for i in ids)}, y la suma de los pesos
aleatorios de la clase es {", ".join(f"({i}) {_sg(s1[i]['fc_weight_sum'])}" for i in ids)}. {resid_txt} Grad-CAM **pasa** la prueba de aleatorización: el mapa depende
de los parámetros que la red aprendió, y desde la etapa 2 la correlación queda por debajo de {cfg['spearman_low']:g} en las tres imágenes. Aun así, la etapa 1 muestra
un riesgo práctico: con una capa final al azar el mapa puede seguir marcando el objeto, sólo porque ahí se concentra la activación, y verse
"plausible" sin decir nada sobre la clase."""

    # ---- 4.3(c) ----
    c_txt = _audiences(prior)

    return f"""# Task 4

## Configuración

- Modelo y entrada: ResNet-18 (`IMAGENET1K_V1`) y las mismas tres imágenes del Task 1; clase de mayor probabilidad de cada imagen.
- Mapa: Grad-CAM del Task 1.2 (224×224, normalizado). Los píxeles se ordenan de mayor a menor valor del mapa; los empates (celdas con valor 0)
  conservan el orden de barrido.
- Deletion y insertion: P = {cfg['steps']} pasos, {cfg['random_orders']} órdenes aleatorios (semillas {env['seed']} a {env['seed'] + cfg['random_orders'] - 1}); gris medio = {cfg['gray_level']:g} (RGB 128);
  desenfoque gaussiano con σ = {cfg['blur_sigma']:g} píxeles por canal. Las imágenes se construyen en [0, 1] y se normalizan con la media y la desviación de ImageNet
  justo antes del modelo. AUC = (1/P) Σ_(k=1..P) c_k.
- Cascada: `reset_parameters` de PyTorch sobre cada Conv2d, Linear y BatchNorm2d de la capa (en BatchNorm también reinicia las estadísticas),
  acumulando {", ".join(f"`{s}`" for s in cfg['stages'])}; semilla {env['seed']}. Mapa de 7×7 (Grad-CAM con ReLU, sin interpolar) de la clase original.
- Semilla aleatoria: **{env['seed']}**. Versiones: torch {env['torch']}, torchvision {env['torchvision']}, Python {env['python']}.

## Task 4.1 - Deletion e insertion

{chr(10).join(t + chr(10) for t in tables)}
{fig_curves}
{fig_frames}
## Task 4.2 - Aleatorización de parámetros en cascada

Spearman entre el mapa de 7×7 original y el de cada etapa (entre paréntesis, el mismo cálculo sobre el mapa antes de ReLU, que no tiene empates en cero):

{chr(10).join(t_cas)}
{fig_maps}
{fig_rho}
## Task 4.3 - Análisis

{a_txt}

{b_txt}

{c_txt}

**Prompt utilizado**

{PROMPT}

**Por qué funcionó este prompt**

{WHY}
"""


def _audiences(prior: dict) -> str:
    t2, t3 = prior["t2"], prior["t3"]
    eff = t2["t2_2"]["a_max_efficiency_error"]
    base = t2["t2_2"]["b_abs_diff"]
    rho = t2["t2_2"]["c_spearman"]
    auc, acc = t2["t2_1"]["test_auc"], t2["t2_1"]["test_accuracy"]
    ks = t2["t2_4"]["by_background"][-1]
    exact = t2["t2_5"]["max_abs_error_vs_treeshap"]
    dup = t2["t2_3"]
    share = dup["configs"]["interventional"]["share_orig"]
    rej = t2["t2_2"]["d_clients"]["rechazado"]
    top3 = ", ".join(f"`{x['feature']}` = {x['value']} ({x['phi']:+.2f})" for x in rej["top3"])
    J = t3["t3_2"]["jaccard"]
    r2max = max(r["r2w"] for r in t3["t3_3"]["b"]["by_k"])
    cf = t3["t3_4"]
    best = cf["selected"][0]
    best_txt = "; ".join(f"{c['feature']} de {c['original']} a {c['new']}" for c in best["changes"])
    n09 = t3["t3_5"]["a"]["n_estimate"]
    return f"""### (c) Explicaciones para distintas audiencias

| Audiencia | Qué se entrega | Validaciones con números propios | Qué **no** se entrega |
|---|---|---|---|
| Persona solicitante | Las razones principales del rechazo en lenguaje llano a partir de SHAP (en el cliente rechazado: {top3}) y un contrafactual accionable del 3.4 ({best_txt}; P(good) pasa de {cf['p_good_original']:.3f} a {best['p_good']:.3f}), sin cambiar variables inmutables. | El contrafactual se comprueba con el propio modelo (P(good) ≥ 0.5) y sólo cambia variables accionables; las razones SHAP suman exactamente el puntaje (error de eficiencia {eff:.1e}). | Coeficientes de LIME (R²_w máximo {r2max:.3f} y top-5 inestable), códigos ordinales, el modelo o umbrales internos, ni contrafactuales que toquen `age`, `personal_status` o `foreign_worker`. |
| Oficial de cumplimiento interno | SHAP local de cada decisión y SHAP global (importancia media \\|φ\\|) junto con la importancia por permutación; contrafactuales para revisar si las acciones sugeridas son razonables. | Eficiencia {eff:.1e}; valor base igual al promedio del fondo (diferencia {base:.1e}); Spearman entre importancias {rho:.3f}; Shapley exacto propio contra TreeSHAP con error {exact:.1e}; KernelSHAP con MAE {ks['mae']:.3f} respecto de TreeSHAP; LIME con Jaccard de {min(min(r) for r in J):.2f} a {max(max(r) for r in J):.2f} según N y σ_rel (hacen falta unas {n09} muestras para 0.9). | LIME como justificación de una decisión (no alcanza R²_w ≥ 0.8) ni importancias individuales de variables correlacionadas sin agruparlas (con una copia ruidosa, la original se queda con {share:.0%} del crédito). |
| Regulador | Documentación del modelo y del método: datos, codificación (ordinal, con su limitación), desempeño (AUC {auc:.3f}, exactitud {acc:.3f}), importancia global SHAP y por permutación, y el protocolo de explicaciones individuales con sus pruebas. | Todas las verificaciones numéricas anteriores, el reparto del crédito entre variables duplicadas (2.3), la estabilidad de LIME (3.2) y la validación de Grad-CAM del Task 4 como ejemplo de que una explicación debe compararse con el azar y pasar la prueba de aleatorización. | Explicaciones aisladas sin su validación, ni una sola corrida de LIME o una sola semilla presentadas como si fueran estables. |

**Contrafactuales en un sistema de detección de lavado (Proyecto 2).** Un contrafactual como el del 3.4 le dice a quien es evaluado qué cambio mínimo
haría que el modelo cambiara de decisión. En crédito eso es deseable porque el cambio (pedir menos monto o plazo) es legítimo. En detección
de lavado, la persona evaluada puede ser el adversario: "si la transferencia fuera menor que X o se repartiera en N operaciones no se
marcaría" es una receta para evadir el modelo (fraccionamiento, cambio de contrapartes), y repetir consultas permite reconstruir la frontera de
decisión. Además, avisar al cliente de que está siendo investigado está prohibido en la normativa antilavado (*tipping off*). En su lugar se
entregaría al cliente sólo una notificación genérica del resultado y el canal de revisión, y las explicaciones SHAP locales y globales se
reservarían para los analistas y el oficial de cumplimiento, con registro de accesos y sin publicar umbrales ni contrafactuales."""
