#!/usr/bin/env python
"""Run all of Task 2 and write figures + results to ``outputs/task2`` (and the report block of the README)."""

from __future__ import annotations

import argparse
import sys
import time
import warnings

from xai_lab import config, console_task2, report, report_task2, task2, viz_tabular
from xai_lab.reproducibility import set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    t0 = time.perf_counter()
    sys.stdout.reconfigure(encoding="utf-8")

    warnings.filterwarnings("ignore")
    set_seed(args.seed)
    run = task2.run_task2(args.seed)

    fig = config.FIGURES_DIR_T2
    viz_tabular.fig_importance(run, fig / "fig_2_2_importance.png")
    viz_tabular.fig_waterfall(run, "aprobado", fig / "fig_2_2_waterfall_aprobado.png")
    viz_tabular.fig_waterfall(run, "rechazado", fig / "fig_2_2_waterfall_rechazado.png")
    viz_tabular.fig_beeswarm(run, fig / "fig_2_2_beeswarm.png")
    viz_tabular.fig_duplicate(run, fig / "fig_2_3_duplicate.png")
    viz_tabular.fig_kernel(run, fig / "fig_2_4_kernel_vs_tree.png")
    viz_tabular.fig_own_vs_shap(run, fig / "fig_2_5_own_vs_shap.png")

    report.write_json(run.results, config.OUTPUT_DIR_T2 / "results.json")
    report.update_readme_block(
        config.ROOT / "README.md", report_task2.TASK2_BEGIN, report_task2.TASK2_END,
        report_task2.render_informe_task2(run.results), after=report.INFORME_END)

    figures = sorted(f.name for f in fig.glob("fig_*.png"))
    text, all_ok = console_task2.render_console(run.results, figures, time.perf_counter() - t0)
    print(text)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
