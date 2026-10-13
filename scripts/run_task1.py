#!/usr/bin/env python
"""Run all of Task 1 and write figures + results to ``outputs/task1``."""

from __future__ import annotations

import argparse
import logging

from xai_lab import config, data, report, viz
from xai_lab.model import load_model
from xai_lab.reproducibility import set_seed
from xai_lab.task1 import run_task1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    set_seed(args.seed)
    model = load_model(args.device)
    runs = run_task1(model, data.load_manifest())

    fig = config.FIGURES_DIR
    viz.fig_predictions(runs, fig / "fig_1_1_predictions.png")
    viz.fig_gradcam(runs, fig / "fig_1_2_gradcam.png")
    viz.fig_cam_vs_gradcam(runs, fig / "fig_1_3_cam_vs_gradcam.png")
    viz.fig_alpha_vs_w(runs["a"], fig / "fig_1_4a_alpha_vs_w.png")
    viz.fig_iou(runs["b"], fig / "fig_1_4b_iou.png")
    viz.fig_causal(runs["c"], fig / "fig_1_4c_causal.png")

    results = report.build_results(runs)
    results["environment"]["seed"] = args.seed
    sections = report.build_sections(results)
    report.write_json(results, config.OUTPUT_DIR / "results.json")
    report.write_markdown(sections, results, config.OUTPUT_DIR / "results.md")

    report.update_readme(config.ROOT / "README.md", report.render_informe(results))

    # One block per lab sub-task: what it did, what goes in the Word report, and the numbers.
    print(report.render_console(sections))
    print(f"\nListo. Resultados en {config.OUTPUT_DIR / 'results.md'}\nInforme actualizado en {config.ROOT / 'README.md'}")


if __name__ == "__main__":
    main()
