#!/usr/bin/env python
"""Run all of Task 3 and write figures + results to ``outputs/task3`` (and the report block of the README)."""

from __future__ import annotations

import argparse
import sys
import time
import warnings

from xai_lab import config, console_task3, report, report_task2, report_task3, task3, viz_task3
from xai_lab.reproducibility import set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    t0 = time.perf_counter()
    sys.stdout.reconfigure(encoding="utf-8")

    warnings.filterwarnings("ignore")
    set_seed(args.seed)
    run = task3.run_task3(args.seed)

    fig = config.FIGURES_DIR_T3
    viz_task3.fig_lime_coefs(run.results, fig / "fig_3_1_coefs.png")
    viz_task3.fig_stability(run.results, fig / "fig_3_2_stability.png")
    viz_task3.fig_lime_vs_shap(run.results, fig / "fig_3_3_lime_vs_shap.png")
    viz_task3.fig_counterfactuals(run.results, fig / "fig_3_4_counterfactuals.png")
    viz_task3.fig_jaccard_neff(run.results, fig / "fig_3_5a_jaccard_neff.png")
    viz_task3.fig_disagreement(run.results, fig / "fig_3_5b_disagreement.png")

    report.write_json(run.results, config.OUTPUT_DIR_T3 / "results.json")
    report.update_readme_block(
        config.ROOT / "README.md", report_task3.TASK3_BEGIN, report_task3.TASK3_END,
        report_task3.render_informe_task3(run.results), after=report_task2.TASK2_END)

    figures = sorted(f.name for f in fig.glob("fig_*.png"))
    text, all_ok = console_task3.render_console(run.results, figures, time.perf_counter() - t0)
    print(text)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
