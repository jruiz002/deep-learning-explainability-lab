"""Plain-text execution report printed at the end of a run (ASCII tables, self-checks)."""

from __future__ import annotations

from . import config

WIDTH = 78


def _table(header: list[str], rows: list[list[str]], align: str | None = None, indent: int = 2) -> list[str]:
    """Fixed-width table. ``align`` has one char per column: 'l' or 'r' (default: first left, rest right)."""
    align = align or "l" + "r" * (len(header) - 1)
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    fmt = lambda r: " ".join(str(v).ljust(w) if a == "l" else str(v).rjust(w) for v, w, a in zip(r, widths, align))
    pad = " " * indent
    sep = pad + " ".join("-" * w for w in widths)
    return [pad + fmt(header), sep] + [pad + fmt(r) for r in rows]


def _title(text: str) -> list[str]:
    return ["", "-" * WIDTH, f" {text}", "-" * WIDTH]


def _check(ok: bool) -> str:
    return "OK" if ok else "FALLO"


def collect_checks(results: dict) -> list[tuple[str, float, float, bool]]:
    """(description, worst value, tolerance, passed) for every numerical self-check."""
    imgs = results["images"]
    worst = lambda f: max(f(it) for it in imgs.values())
    out = [
        ("1.3(b) |logit - ((1/Z) sum M + b)|", worst(lambda i: i["t1_3"]["abs_error_logit"]), config.TOL_ABS),
        ("1.3(c) max |Z*L - M| (49 posiciones)", worst(lambda i: i["t1_3"]["max_abs_error_ZL_vs_M"]), config.TOL_ABS),
        ("1.3(d) max |CAM - Grad-CAM| tras ReLU+min-max", worst(lambda i: i["t1_3"]["max_abs_error_normalised"]), config.TOL_ABS),
        ("1.4(a) error relativo |alpha - w/Z| / |w/Z|", max(r["rel_error"] for r in imgs["a"]["t1_4a"]), config.TOL_REL),
    ]
    return [(d, v, t, v < t) for d, v, t in out]


def render_console(results: dict, figures: list[str], device: str, elapsed: float) -> tuple[str, bool]:
    env, imgs = results["environment"], results["images"]
    a, b, c = imgs["a"], imgs["b"], imgs["c"]
    L: list[str] = ["=" * WIDTH, " LABORATORIO DE EXPLICABILIDAD - TASK 1", " Grad-CAM y CAM sobre ResNet-18 (ImageNet)", "=" * WIDTH]
    L += [
        f" Modelo      ResNet-18 (IMAGENET1K_V1), modo eval",
        f" Entorno     {device}, float32, semilla {env['seed']}",
        f" Versiones   torch {env['torch']} | torchvision {env['torchvision']} | Python {env['python']}",
        f" Entrada     resize 256 -> recorte central {config.CROP_SIZE}x{config.CROP_SIZE} -> normalización ImageNet",
    ]

    # ---- 1.1 -------------------------------------------------------------------------------
    L += _title("Task 1.1  Predicciones top-5")
    for rid, it in imgs.items():
        L += ["", f" Imagen ({rid})  {it['file']}  [{it['license']}]"]
        L += _table(["#", "Clase", "Prob."], [[str(i), f"{p['class']} (#{p['index']})", f"{p['probability']:.4f}"]
                                              for i, p in enumerate(it["t1_1_top5"], 1)], "rlr")
    bb = c["bbox_224"]
    L += ["", f" Recuadro de (c) en {config.CROP_SIZE}x{config.CROP_SIZE}:  x0={bb[0]:.1f}  y0={bb[1]:.1f}  x1={bb[2]:.1f}  y1={bb[3]:.1f}"]

    # ---- 1.2 -------------------------------------------------------------------------------
    L += _title("Task 1.2  Grad-CAM (layer4, desde el logit)")
    rows = []
    for rid, it in imgs.items():
        for name, (y, x) in it["t1_2_peaks"].items():
            rows.append([f"({rid})", name, f"({y}, {x})"])
    L += ["", " Posición del máximo de cada mapa (fila, columna) en la imagen de 224x224:", ""]
    L += _table(["Img", "Clase", "Máximo"], rows, "llr")

    # ---- 1.3 -------------------------------------------------------------------------------
    L += _title("Task 1.3  CAM desde los pesos de fc")
    L += ["", f" (b) Descomposición del puntaje, Z = {env['Z']}:", ""]
    L += _table(["Img", "Clase", "Logit y^c", "(1/Z)sum M + b", "Error abs."],
                [[f"({rid})", it["t1_3"]["class"], f"{it['t1_3']['logit']:.6f}", f"{it['t1_3']['logit_from_cam']:.6f}",
                  f"{it['t1_3']['abs_error_logit']:.2e}"] for rid, it in imgs.items()], "llrrr")
    L += ["", " (c) y (d) Error absoluto máximo entre mapas:", ""]
    L += _table(["Img", "Z*L vs M (sin ReLU/norm.)", "Tras ReLU + min-max"],
                [[f"({rid})", f"{it['t1_3']['max_abs_error_ZL_vs_M']:.2e}", f"{it['t1_3']['max_abs_error_normalised']:.2e}"]
                 for rid, it in imgs.items()], "lrr")

    # ---- 1.4 -------------------------------------------------------------------------------
    L += _title("Task 1.4(a)  alpha vs w/Z  (imagen a, 5 canales de mayor peso)")
    L += [""] + _table(["Canal", "w", "alpha", "w/Z", "Error rel."],
                       [[str(r["channel"]), f"{r['w']:.6f}", f"{r['alpha']:.8f}", f"{r['w_over_Z']:.8f}", f"{r['rel_error']:.2e}"]
                        for r in a["t1_4a"]])

    t = b["t1_4b"]
    L += _title("Task 1.4(b)  Discriminación entre clases (imagen b)")
    L += ["", f" Clases comparadas   {t['class1']}  vs  {t['class2']}",
          f" Píxeles por conjunto {t['pixels_per_mask']} (20 % de {config.CROP_SIZE ** 2})",
          f" Intersección / unión {t['intersection']} / {t['union']} px",
          f" IoU                  {t['iou_top20']:.4f}"]

    k = c["t1_4c"]
    L += _title("Task 1.4(c)  Uso del fondo (imagen c)")
    L += ["", f" Clase analizada: {k['class_name']} (#{k['class_idx']})", ""]
    L += _table(["Medición", "Valor"], [
        ["Masa del mapa dentro del recuadro", f"{k['mass_in_box']:.4f}"],
        ["Área del recuadro / área de la imagen", f"{k['box_area_fraction']:.4f}"],
        ["p(clase), imagen original", f"{k['p_original']:.4f}"],
        ["p(clase), fondo en gris", f"{k['p_background_gray']:.4f}  ({k['delta_background_gray']:+.4f})"],
        ["Top-1 con fondo en gris", f"{k['top1_after_background_gray']['name']} ({k['top1_after_background_gray']['probability']:.4f})"],
        ["p(clase), objeto en gris (control)", f"{k['p_object_gray']:.4f}  ({k['delta_object_gray']:+.4f})"],
    ], "lr")

    # ---- summary ---------------------------------------------------------------------------
    checks = collect_checks(results)
    L += _title("Verificaciones numéricas")
    L += [""] + _table(["Verificación", "Peor caso", "Tolerancia", "Estado"],
                       [[d, f"{v:.2e}", f"< {tol:.0e}", _check(ok)] for d, v, tol, ok in checks], "lrrr")
    all_ok = all(ok for *_, ok in checks)
    L += ["", f" Resultado: {sum(ok for *_, ok in checks)}/{len(checks)} verificaciones correctas"]
    L += _title("Archivos generados")
    L += [f"  figura   outputs/task1/figures/{f}" for f in figures]
    L += ["  datos    outputs/task1/results.json", "  resumen  outputs/task1/results.md", "  informe  README.md (sección Informe)"]
    L += ["", f" Tiempo total: {elapsed:.1f} s", "=" * WIDTH]
    return "\n".join(L), all_ok
