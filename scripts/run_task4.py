#!/usr/bin/env python
"""Run all of Task 4 and write figures + results to ``outputs/task4`` (and the report block of the README).

4.3(c) quotes numbers of Tasks 2 and 3, so ``run_task2.py`` and ``run_task3.py`` must have been run first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

from xai_lab import config, console_task4, data, report, report_task3, report_task4, task4, viz_task4
from xai_lab.model import load_model
from xai_lab.reproducibility import set_seed


def _load_prior() -> dict:
    prior = {}
    for key, path in (("t2", config.OUTPUT_DIR_T2 / "results.json"), ("t3", config.OUTPUT_DIR_T3 / "results.json")):
        if not path.exists():
            sys.exit(f"Falta {path.relative_to(config.ROOT)}: corra antes scripts/run_task2.py y scripts/run_task3.py.")
        prior[key] = json.loads(path.read_text(encoding="utf-8"))
    if "t3_4" not in prior["t3"]:
        sys.exit("outputs/task3/results.json no tiene los incisos 3.3 - 3.5: vuelva a correr scripts/run_task3.py.")
    return prior


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--device", default="cpu", choices=["cpu"])  # ponytail: tensors and .numpy() calls assume CPU
    args = ap.parse_args()
    t0 = time.perf_counter()
    sys.stdout.reconfigure(encoding="utf-8")

    warnings.filterwarnings("ignore")
    prior = _load_prior()
    set_seed(args.seed)
    model = load_model(args.device)
    run = task4.run_task4(model, data.load_manifest(), args.seed)

    fig = config.FIGURES_DIR_T4
    viz_task4.fig_curves(run.results, fig / "fig_4_1_curves.png")
    viz_task4.fig_frames(run.extra["prepared"]["a"], run.extra["maps"]["a"], fig / "fig_4_1_frames.png")
    viz_task4.fig_cascade_maps(run.results, run.extra, fig / "fig_4_2_cascade_maps.png")
    viz_task4.fig_cascade_spearman(run.results, fig / "fig_4_2_cascade_spearman.png")
    viz_task4.fig_blur_reference(run.results, fig / "fig_4_3a_blur_reference.png")

    report.write_json(run.results, config.OUTPUT_DIR_T4 / "results.json")
    report.update_readme_block(
        config.ROOT / "README.md", report_task4.TASK4_BEGIN, report_task4.TASK4_END,
        report_task4.render_informe_task4(run.results, prior), after=report_task3.TASK3_END)

    figures = sorted(f.name for f in fig.glob("fig_*.png"))
    text, all_ok = console_task4.render_console(run.results, figures, time.perf_counter() - t0)
    print(text)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
