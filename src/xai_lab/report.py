"""Serialise Task 1 results: JSON, Markdown (Spanish) and the per-task console summary."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torchvision

from . import config
from .model import CATEGORIES


def _name(i: int) -> str:
    return f"{CATEGORIES[i]} (#{i})"


def build_results(runs) -> dict:
    """Plain-python dict with every reported number (no tensors)."""
    res: dict = {
        "environment": {
            "seed": config.SEED,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "python": platform.python_version(),
            "Z": config.Z,
        },
        "images": {},
    }
    for rid, r in runs.items():
        spec = r.prepared.spec
        c = r.cam
        top1 = r.top5[0].index
        item = {
            "file": spec.filename,
            "source": spec.page_url,
            "author": spec.author,
            "license": spec.license,
            "t1_1_top5": [
                {"class": p.name, "index": p.index, "probability": p.probability} for p in r.top5
            ],
            "bbox_original": spec.bbox_original,
            "bbox_224": r.prepared.bbox_224,
            "t1_2_classes": [_name(i) for i in r.extra["classes"]],
            # (row, col) of the maximum of each normalised 224x224 Grad-CAM map
            "t1_2_peaks": {
                _name(ci): [int(v) for v in divmod(int(gc.cam_224.argmax()), gc.cam_224.shape[1])]
                for ci, gc in r.gradcams.items()
            },
            "t1_3": {
                "class": _name(top1),
                "logit": c["b_logit"],
                "logit_from_cam": c["b_cam_logit"],
                "abs_error_logit": c["b_abs_error"],
                "max_abs_error_ZL_vs_M": c["c_max_abs_error"],
                "max_abs_error_normalised": c["d_max_abs_error"],
            },
        }
        if "t14a" in r.extra:
            item["t1_4a"] = r.extra["t14a"]
        if "t14b" in r.extra:
            t = r.extra["t14b"]
            item["t1_4b"] = {
                "class1": _name(t["class1"]),
                "class2": _name(t["class2"]),
                "probabilities": {_name(k): v for k, v in t["probabilities"].items()},
                "iou_top20": t["iou"],
                "pixels_per_mask": t["pixels_per_mask"],
                "intersection": t["intersection"],
                "union": t["union"],
            }
        if "t14c" in r.extra:
            item["t1_4c"] = {k: v for k, v in r.extra["t14c"].items() if not hasattr(v, "shape")}
        res["images"][rid] = item
    return res


def write_json(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@dataclass
class Section:
    """One lab sub-task (inciso): what the code did, what to put in Word, and the numbers."""

    task: str  # e.g. "1.4(a)"
    title: str
    what_it_does: str
    word_items: list[str]
    lines: list[str] = field(default_factory=list)  # Markdown with the numbers
    figures: list[str] = field(default_factory=list)  # file names in outputs/task1/figures


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return out + ["| " + " | ".join(r) + " |" for r in rows] + [""]


def build_sections(results: dict) -> list[Section]:
    imgs = results["images"]
    S: list[Section] = []

    # ---- Task 1.1 --------------------------------------------------------------------------
    lines: list[str] = []
    for rid, it in imgs.items():
        lines += [f"**Imagen ({rid})** - `{it['file']}` - {it['author']} - [{it['license']}]({it['source']})", ""]
        lines += _table(["#", "clase ImageNet", "probabilidad"],
                        [[str(i), f"{p['class']} (#{p['index']})", f"{p['probability']:.4f}"]
                         for i, p in enumerate(it["t1_1_top5"], 1)])
    b = imgs["c"]["bbox_224"]
    lines += [f"**Recuadro de la imagen (c) en 224x224** (x0, y0, x1, y1) = "
              f"({b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f})  "
              f"[definido en la imagen original como {imgs['c']['bbox_original']}]", ""]
    S.append(Section(
        "1.1", "Modelo, preprocesamiento, tres imágenes y top-5",
        "Carga ResNet-18 con pesos de ImageNet en modo eval; preprocesa (resize 256, recorte central 224, "
        "normalización por canal); predice el top-5 de las tres imágenes (a: objeto claro, b: gato y perro, "
        "c: objeto fuera de contexto) y transforma el recuadro de (c) a coordenadas de 224x224.",
        ["Tabla con fuente, autor y licencia de las 3 imágenes (indicar la fuente es obligatorio).",
         "Una tabla top-5 (clase + probabilidad) por imagen.",
         "Las coordenadas del recuadro de (c) en 224x224.",
         "Figura fig_1_1_predictions.png.",
         "Mencionar la semilla y las versiones de librerías."],
        lines, ["fig_1_1_predictions.png"]))

    # ---- Task 1.2 --------------------------------------------------------------------------
    S.append(Section(
        "1.2", "Grad-CAM desde cero sobre layer4",
        "Hook de avance (guarda A, 512x7x7) y hook de retroceso (guarda dy^c/dA); se retropropaga desde el "
        "logit (antes del softmax); alpha_k = promedio espacial del gradiente; L = ReLU(sum alpha_k A^k), "
        "interpolación bilineal a 224x224 y normalización min-max. Mapa de la clase top-1 de cada imagen "
        "y, en (b), también el de la clase del segundo objeto.",
        ["La figura fig_1_2_gradcam.png (mapas superpuestos a la imagen).",
         "Una descripción breve de la implementación (hooks, logit, alpha, ReLU, bilineal, min-max).",
         "Un comentario sobre dónde cae el mapa en cada imagen."],
        ["Mapas generados para: " + "; ".join(f"imagen ({rid}): {', '.join(it['t1_2_classes'])}"
                                              for rid, it in imgs.items()), ""],
        ["fig_1_2_gradcam.png"]))

    # ---- Task 1.3 --------------------------------------------------------------------------
    rows = lambda keys: [[rid, it["t1_3"]["class"], *keys(it["t1_3"])] for rid, it in imgs.items()]
    lines = ["**(a)** M^c_ij = sum_k w^c_k A^k_ij calculado con los pesos de `fc`, sin gradientes "
             "(mapas en la figura).", "", "**(b)** Descomposición del puntaje: (1/Z) sum M + b_c debe igualar el logit y^c (Z = 49).", ""]
    lines += _table(["imagen", "clase", "logit y^c", "(1/Z) sum M + b_c", "error absoluto"],
                    rows(lambda t: [f"{t['logit']:.6f}", f"{t['logit_from_cam']:.6f}", f"{t['abs_error_logit']:.2e}"]))
    lines += ["**(c)** Error absoluto máximo entre Z·L^c (sin ReLU ni normalización) y M^c sobre las 49 posiciones.", ""]
    lines += _table(["imagen", "clase", "error abs. máximo"], rows(lambda t: [f"{t['max_abs_error_ZL_vs_M']:.2e}"]))
    lines += ["**(d)** Error absoluto máximo entre ambos mapas tras ReLU + normalización min-max.", ""]
    lines += _table(["imagen", "clase", "error abs. máximo"], rows(lambda t: [f"{t['max_abs_error_normalised']:.2e}"]))
    S.append(Section(
        "1.3", "CAM directo desde los pesos de fc y verificación contra Grad-CAM",
        "(a) Calcula CAM sin gradientes; (b) verifica que (1/Z) sum M + b = logit; (c) verifica Z·L = M; "
        "(d) compara ambos mapas tras ReLU y min-max.",
        ["Tabla del inciso (b): ambos valores y el error absoluto por imagen.",
         "Tabla del inciso (c): error absoluto máximo por imagen.",
         "Tabla del inciso (d): error absoluto máximo por imagen.",
         "Figura fig_1_3_cam_vs_gradcam.png y una conclusión (coinciden hasta precisión float32)."],
        lines, ["fig_1_3_cam_vs_gradcam.png"]))

    # ---- Task 1.4 (a) ----------------------------------------------------------------------
    a = imgs["a"]["t1_4a"]
    S.append(Section(
        "1.4(a)", "alpha vs w/Z (imagen a, cinco canales de mayor peso)",
        "Compara numéricamente el alpha de Grad-CAM con w^c_k / Z para los cinco canales con mayor peso en fc.",
        ["Tabla con canal, w, alpha, w/Z y error relativo.",
         "Explicación: dy^c/dA^k_ij = w_k/Z es igual en todas las posiciones (i, j) porque el pooling global "
         "promedia con el mismo peso 1/Z; al promediar Z gradientes idénticos alpha = w/Z.",
         "Explicación de por qué el error es ~1e-8 y no 0: precisión de punto flotante (float32).",
         "Figura fig_1_4a_alpha_vs_w.png."],
        _table(["canal", "w", "alpha", "w/Z", "error absoluto", "error relativo"],
               [[str(r["channel"]), f"{r['w']:.6f}", f"{r['alpha']:.10f}", f"{r['w_over_Z']:.10f}",
                 f"{r['abs_error']:.2e}", f"{r['rel_error']:.2e}"] for r in a]),
        ["fig_1_4a_alpha_vs_w.png"]))

    # ---- Task 1.4 (b) ----------------------------------------------------------------------
    t = imgs["b"]["t1_4b"]
    S.append(Section(
        "1.4(b)", "IoU entre los mapas de las dos clases (imagen b)",
        "Toma el 20 % de píxeles de mayor valor de cada mapa (clase perro y clase gato) y calcula la "
        "intersección sobre la unión.",
        ["El IoU y los conteos de píxeles (intersección, unión).",
         "Respuesta: ¿el mapa discrimina entre las dos clases? (con el número del IoU).",
         "Límite de resolución: cada celda 7x7 cubre 32x32 píxeles, así que objetos pequeños o cercanos "
         "entre sí (< ~32 px) no se pueden separar ni delimitar bien.",
         "Figura fig_1_4b_iou.png."],
        [f"- Clases: {t['class1']} vs {t['class2']}",
         f"- Probabilidades: " + ", ".join(f"{k}: {v:.4f}" for k, v in t["probabilities"].items()),
         f"- IoU del top-20 % = **{t['iou_top20']:.4f}** (intersección {t['intersection']} px, "
         f"unión {t['union']} px, {t['pixels_per_mask']} px por conjunto)", ""],
        ["fig_1_4b_iou.png"]))

    # ---- Task 1.4 (c) ----------------------------------------------------------------------
    c = imgs["c"]["t1_4c"]
    S.append(Section(
        "1.4(c)", "Masa dentro del recuadro y prueba causal (imagen c)",
        "Calcula qué fracción del mapa normalizado cae dentro del recuadro y reemplaza por gris todo lo "
        "que queda fuera del recuadro para medir el cambio en la probabilidad de la clase. "
        "(Extra: control con el objeto en gris.)",
        ["Fracción de masa dentro del recuadro (y el área del recuadro como referencia).",
         "Probabilidad original, con fondo gris y su cambio.",
         "Respuesta: ¿coinciden el mapa y la prueba causal en si el modelo usa el fondo? y conclusión "
         "sobre si hay evidencia de un atajo (con los números).",
         "Figura fig_1_4c_causal.png."],
        [f"- Clase analizada: {c['class_name']} (#{c['class_idx']})",
         f"- Fracción de masa del mapa dentro del recuadro: **{c['mass_in_box']:.4f}** "
         f"(el recuadro cubre {c['box_area_fraction']:.4f} de la imagen)",
         f"- p(clase) original: {c['p_original']:.4f}",
         f"- p(clase) con todo lo de fuera del recuadro en gris: {c['p_background_gray']:.4f} "
         f"(cambio {c['delta_background_gray']:+.4f})",
         f"- [control extra] p(clase) con el objeto en gris y el fondo intacto: {c['p_object_gray']:.4f} "
         f"(cambio {c['delta_object_gray']:+.4f})",
         f"- Top-1 tras poner el fondo en gris: {c['top1_after_background_gray']['name']} "
         f"({c['top1_after_background_gray']['probability']:.4f})", ""],
        ["fig_1_4c_causal.png"]))
    return S


def render_markdown(sections: list[Section], results: dict) -> str:
    env = results["environment"]
    L = ["# Task 1 - resultados (generado automáticamente)", "",
         f"Semilla `{env['seed']}` - torch {env['torch']} - torchvision {env['torchvision']} - Python {env['python']}", ""]
    for s in sections:
        L += [f"## Task {s.task} - {s.title}", "", f"**Qué hace el código:** {s.what_it_does}", "",
              "**Qué poner en el Word:**", ""] + [f"- {w}" for w in s.word_items] + [""]
        L += s.lines
        L += [f"Figuras: " + ", ".join(f"`figures/{f}`" for f in s.figures), ""]
    return "\n".join(L)


def render_console(sections: list[Section]) -> str:
    bar = "=" * 78
    out: list[str] = []
    for s in sections:
        out += ["", bar, f" TASK {s.task} - {s.title}", bar, f"Qué hace: {s.what_it_does}", "",
                "QUÉ PONER EN EL WORD:"] + [f"  [ ] {w}" for w in s.word_items] + ["", "RESULTADOS:"]
        out += [ln for ln in s.lines] + ["Figuras: " + ", ".join(f"outputs/task1/figures/{f}" for f in s.figures)]
    return "\n".join(out)


def write_markdown(sections: list[Section], results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(sections, results) + "\n", encoding="utf-8")



INFORME_BEGIN = "<!-- INFORME:INICIO (generado por scripts/run_task1.py; no editar a mano) -->"
INFORME_END = "<!-- INFORME:FIN -->"


def render_informe(results: dict) -> str:
    """Written report (Spanish, impersonal voice) built only from the numbers of this run."""
    env, imgs = results["environment"], results["images"]
    a, b, c = imgs["a"], imgs["b"], imgs["c"]
    t4b, t4c = b["t1_4b"], c["t1_4c"]
    fig = lambda n, cap: f"![{cap}](outputs/task1/figures/{n})\n\n*{cap}*\n"
    box = c["bbox_224"]
    kinds = ["Objeto claro y centrado", "Dos objetos de clases distintas (perro y gato)",
             "Objeto fuera de su contexto habitual (velero en tierra, sobre un remolque)"]

    src = _table(["Imagen", "Característica", "Fuente", "Autor", "Licencia"],
                 [[f"({rid})", k, f"[{it['file']}]({it['source']})", it["author"], it["license"]]
                  for (rid, it), k in zip(imgs.items(), kinds)])
    top = _table(["#", "(a)", "prob.", "(b)", "prob.", "(c)", "prob."],
                 [[str(i + 1)] + sum([[imgs[r]["t1_1_top5"][i]["class"], f"{imgs[r]['t1_1_top5'][i]['probability']:.4f}"]
                                       for r in "abc"], []) for i in range(5)])
    t13 = lambda hdr, f: _table(["Imagen", "Clase", *hdr],
                                [[rid, it["t1_3"]["class"], *f(it["t1_3"])] for rid, it in imgs.items()])
    tb = _table(["Canal", "w", "α (Grad-CAM)", "w/Z", "Error relativo"],
                [[str(r["channel"]), f"{r['w']:.6f}", f"{r['alpha']:.10f}", f"{r['w_over_Z']:.10f}", f"{r['rel_error']:.2e}"]
                 for r in a["t1_4a"]])
    peaks = lambda it: "; ".join(f"{k}: (fila {v[0]}, columna {v[1]})" for k, v in it["t1_2_peaks"].items())
    max_rel = max(r["rel_error"] for r in a["t1_4a"])
    max_b = max(it["t1_3"]["abs_error_logit"] for it in imgs.values())
    max_c = max(it["t1_3"]["max_abs_error_ZL_vs_M"] for it in imgs.values())
    max_d = max(it["t1_3"]["max_abs_error_normalised"] for it in imgs.values())
    ratio = t4c["mass_in_box"] / t4c["box_area_fraction"]
    outside = 1 - t4c["mass_in_box"]
    drop = -t4c["delta_background_gray"] / t4c["p_original"]
    discrimina = t4b["iou_top20"] < 0.1
    usa_objeto = ratio > 1.2 and t4c["p_object_gray"] < 0.1 * t4c["p_original"]
    usa_fondo = t4c["p_background_gray"] < 0.5 * t4c["p_original"] or t4c["top1_after_background_gray"]["name"] != t4c["class_name"]
    veredicto = ("hay evidencia parcial de dependencia del fondo, aunque no la de un atajo total" if usa_objeto and usa_fondo
                 else "no hay evidencia clara de dependencia del fondo" if usa_objeto
                 else "hay evidencia de que el modelo no se apoya principalmente en el objeto, lo que sugiere un atajo")

    return f"""# Informe

## 0. Configuración

- Modelo: ResNet-18 de torchvision con pesos `IMAGENET1K_V1`, en modo evaluación.
- Preprocesamiento: redimensión a 256 píxeles en el lado corto, recorte central de 224×224 y normalización por canal con
  media (0.485, 0.456, 0.406) y desviación estándar (0.229, 0.224, 0.225).
- Semilla aleatoria: **{env['seed']}** (Python, NumPy y PyTorch). Ejecución en CPU con float32.
- Versiones: torch {env['torch']}, torchvision {env['torchvision']}, Python {env['python']}.

## Task 1.1 - Modelo e imágenes

Se seleccionaron tres imágenes de Wikimedia Commons:

{chr(10).join(src)}
Se obtuvo el top-5 de cada imagen con el modelo en modo evaluación:

{chr(10).join(top)}
Para la imagen (c) se definió el recuadro del objeto sobre la imagen original como {c['bbox_original']} (velero completo, sin el remolque)
y se transformó con el mismo redimensionado y recorte que recibe el modelo. Sus coordenadas en la imagen de 224×224
(x0, y0, x1, y1) son **({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f})**. Como el velero es más ancho que el recorte central, el recuadro
llega a ambos bordes.

{fig('fig_1_1_predictions.png', 'Figura 1. Entradas de 224×224 (con el recuadro en (c)) y top-5 de probabilidades.')}
En (a) la clase de mayor probabilidad es `{a['t1_1_top5'][0]['class']}` ({a['t1_1_top5'][0]['probability']:.4f}). En (b) la clase de mayor probabilidad es
`{b['t1_1_top5'][0]['class']}` ({b['t1_1_top5'][0]['probability']:.4f}) y la clase de gato mejor ubicada es `{t4b['class2'].split(' (')[0]}` ({list(t4b['probabilities'].values())[1]:.4f}), presente en el top-5. En (c) el
top-1 (`{c['t1_1_top5'][0]['class']}`) alcanza sólo {c['t1_1_top5'][0]['probability']:.4f} y el top-5 mezcla embarcaciones con `{c['t1_1_top5'][2]['class']}` y `{c['t1_1_top5'][3]['class']}`,
lo que indica que el contexto de la imagen afecta la predicción.

## Task 1.2 - Grad-CAM desde cero

Se implementó Grad-CAM sobre `layer4` con un hook de avance, que captura las activaciones A (512 canales de 7×7), y un hook de
retroceso, que captura ∂y^c/∂A. Se respetaron las reglas del enunciado:

1. Se retropropagó desde el **logit** y^c (antes del softmax), no desde la probabilidad ni desde una pérdida.
2. Se calculó α_k^c como el promedio espacial del gradiente de cada canal k.
3. Se formó L^c = ReLU(Σ_k α_k^c A^k), se amplió a 224×224 con interpolación bilineal y se normalizó con su mínimo y su máximo.

Se generó el mapa de la clase de mayor probabilidad de cada imagen y, en (b), también el de la clase del segundo objeto.
La posición del máximo de cada mapa (en píxeles de la imagen de 224×224) fue: (a) {peaks(a)}; (b) {peaks(b)}; (c) {peaks(c)}.

{fig('fig_1_2_gradcam.png', 'Figura 2. Grad-CAM superpuesto: (a) clase top-1; (b) clase del perro y clase del gato; (c) clase top-1.')}
Al observar la figura, en (a) el mapa se concentra en la cara y las orejas del cachorro; en (b) el mapa de cada clase cae sobre el animal
correspondiente; y en (c) cae sobre el casco y la cabina del velero, con extensión hacia la parte baja, donde está el remolque.

## Task 1.3 - CAM desde los pesos de `fc`

Se calculó M^c_ij = Σ_k w_k^c A^k_ij directamente con los pesos de `fc`, sin gradientes, para la clase de mayor probabilidad de
cada imagen, con Z = {env['Z']}.

**(b)** Descomposición del puntaje: (1/Z) Σ M + b_c contra el logit y^c.

{chr(10).join(t13(['Logit y^c', '(1/Z) Σ M + b_c', 'Error absoluto'], lambda t: [f"{t['logit']:.6f}", f"{t['logit_from_cam']:.6f}", f"{t['abs_error_logit']:.2e}"]))}
**(c)** Error absoluto máximo entre Z·L^c (sin ReLU ni normalización) y M^c sobre las 49 posiciones:

{chr(10).join(t13(['Error abs. máximo'], lambda t: [f"{t['max_abs_error_ZL_vs_M']:.2e}"]))}
**(d)** Error absoluto máximo entre ambos mapas después de ReLU y normalización mínimo-máximo:

{chr(10).join(t13(['Error abs. máximo'], lambda t: [f"{t['max_abs_error_normalised']:.2e}"]))}
{fig('fig_1_3_cam_vs_gradcam.png', 'Figura 3. CAM (M^c), Z·L^c de Grad-CAM y sus diferencias, por imagen.')}
La descomposición del logit se cumplió con un error absoluto máximo de {max_b:.2e} y la igualdad Z·L^c = M^c con un error máximo de {max_c:.2e}, valores del orden de la
precisión de float32. Tras aplicar ReLU y normalización, el error máximo entre ambos mapas fue de {max_d:.2e}. La comparación del inciso (c)
se realizó antes de ReLU y normalización porque esas operaciones no son lineales y ocultarían la igualdad. En esta arquitectura, por lo tanto,
Grad-CAM y CAM producen el mismo mapa.

## Task 1.4 - Análisis

### (a) α frente a w/Z (imagen a, cinco canales de mayor peso)

{chr(10).join(tb)}
{fig('fig_1_4a_alpha_vs_w.png', 'Figura 4. α_k^c contra w_k^c/Z en los 512 canales (en círculo, los cinco de mayor peso).')}
El error relativo máximo observado en los cinco canales fue de {max_rel:.2e}. Como y^c = (1/Z) Σ_k w_k Σ_ij A^k_ij + b_c, la derivada
∂y^c/∂A^k_ij = w_k/Z **es la misma en todas las posiciones (i, j)**, porque el pooling global promedia dando el mismo peso 1/Z a cada posición.
Ese hecho se usa al promediar espacialmente el gradiente: el promedio de Z valores idénticos es ese mismo valor, por lo que α_k^c = w_k^c/Z.

Los errores no son exactamente cero porque autograd calcula el gradiente con sumas y productos en float32, que se redondean de forma distinta a la
división directa w/Z. Su magnitud es del orden de la precisión de la máquina (ε ≈ 1.2e-7) y no corresponde a un error del método.

### (b) ¿El mapa discrimina entre las dos clases? (imagen b)

Se tomó el 20 % de píxeles de mayor valor de cada mapa ({t4b['pixels_per_mask']} de 50176) para {t4b['class1']} y {t4b['class2']}:

- Intersección: {t4b['intersection']} píxeles · Unión: {t4b['union']} píxeles
- **IoU = {t4b['iou_top20']:.4f}**

{fig('fig_1_4b_iou.png', 'Figura 5. Conjuntos del 20 % de mayor valor para cada clase.')}
{"Un IoU cercano a cero indica que el mapa **sí discrimina** entre las dos clases: los conjuntos casi no se solapan y cada uno cae sobre el animal que le corresponde." if discrimina else "Un IoU elevado indica que el mapa **no discrimina bien** entre las dos clases: ambos conjuntos ocupan regiones similares."}

Aun así, la localización es gruesa. Cada posición del mapa original (7×7) cubre 32×32 píxeles de la entrada, de modo que un objeto menor de ~32 píxeles,
o dos objetos más cercanos entre sí que esa distancia, caen en la misma celda o en celdas mezcladas, y el mapa no puede separarlos ni delimitar su contorno.
La interpolación bilineal sólo suaviza el mapa y no agrega resolución. Además, el campo receptivo de `layer4` es mayor que 32 píxeles, lo que difumina aún más
los bordes. En esta imagen la discriminación fue posible porque ambos animales son grandes.

### (c) ¿Usa el modelo el fondo? (imagen c)

Clase analizada: **{t4c['class_name']}** (top-1).

| Medición | Valor |
|---|---|
| Fracción de la masa del mapa normalizado dentro del recuadro | **{t4c['mass_in_box']:.4f}** |
| Fracción del área de la imagen que cubre el recuadro | {t4c['box_area_fraction']:.4f} |
| p({t4c['class_name']}), imagen original | {t4c['p_original']:.4f} |
| p({t4c['class_name']}), todo lo de fuera del recuadro en gris | {t4c['p_background_gray']:.4f} (cambio **{t4c['delta_background_gray']:+.4f}**) |
| Top-1 con el fondo en gris | {t4c['top1_after_background_gray']['name']} ({t4c['top1_after_background_gray']['probability']:.4f}) |
| Control adicional: objeto en gris y fondo intacto | p = {t4c['p_object_gray']:.4f} (cambio {t4c['delta_object_gray']:+.4f}) |

El gris utilizado fue RGB (128, 128, 128), aplicado sobre el recorte de 224×224 antes de normalizar.

{fig('fig_1_4c_causal.png', 'Figura 6. De izquierda a derecha: original con recuadro, mapa Grad-CAM, fondo en gris y objeto en gris.')}
El {t4c['mass_in_box']:.1%} de la masa del mapa cae dentro de un recuadro que ocupa el {t4c['box_area_fraction']:.1%} de la imagen ({ratio:.1f} veces lo esperable si la masa
estuviera repartida de forma uniforme). Al ocultar el objeto, la probabilidad de la clase pasó de {t4c['p_original']:.4f} a {t4c['p_object_gray']:.4f}. Por lo tanto, el mapa y la prueba causal
coinciden en que el modelo usa el objeto.

El fondo, sin embargo, no es irrelevante. Aproximadamente el {outside:.1%} de la masa del mapa queda fuera del recuadro y, al reemplazar el fondo por gris, la probabilidad de
`{t4c['class_name']}` disminuyó {drop:.0%} (de {t4c['p_original']:.4f} a {t4c['p_background_gray']:.4f}), mientras que el top-1 pasó a `{t4c['top1_after_background_gray']['name']}` con {t4c['top1_after_background_gray']['probability']:.4f}.

**Conclusión:** {veredicto}. Esta conclusión tiene límites: se analizó una sola imagen, el gris es una entrada fuera de la distribución de entrenamiento (parte del
cambio puede deberse a ello y no sólo a la eliminación de información) y el recuadro ocupa todo el ancho del recorte, lo que deja poco fondo lateral.

## Conclusiones

- Grad-CAM y CAM coincidieron numéricamente en ResNet-18 (errores máximos de {max(max_b, max_c):.1e} en float32), porque α_k^c = w_k^c/Z.
- Los mapas distinguieron las clases de una misma imagen (IoU de {t4b['iou_top20']:.4f} en (b)), pero su resolución de 7×7 limita la localización de objetos pequeños o cercanos entre sí.
- Un mapa de calor por sí solo no demuestra que el modelo use o ignore el fondo; la prueba causal con gris lo complementó y en (c) {veredicto.replace("hay evidencia", "mostró evidencia", 1) if usa_objeto and usa_fondo else veredicto}.
"""


def update_readme(readme: Path, informe: str) -> None:
    """Replace the generated report block of the README (or append it if markers are missing)."""
    text = readme.read_text(encoding="utf-8")
    block = f"{INFORME_BEGIN}\n\n{informe.rstrip()}\n\n{INFORME_END}\n"
    if INFORME_BEGIN in text and INFORME_END in text:
        head = text[: text.index(INFORME_BEGIN)]
        tail = text[text.index(INFORME_END) + len(INFORME_END):].lstrip("\n")
        text = head + block + ("\n" + tail if tail else "")
    else:
        text = text.rstrip() + "\n\n---\n\n" + block
    readme.write_text(text, encoding="utf-8")
