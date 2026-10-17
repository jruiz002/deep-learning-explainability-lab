"""Task 4 experiments: deletion / insertion of Grad-CAM against random orders (4.1), cascading parameter
randomisation (4.2) and the extra runs of the analysis (4.3)."""

from __future__ import annotations

import copy
import platform
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torchvision
from torch import nn

from . import config, faithfulness as fa, sanity
from .data import prepare_image
from .gradcam import GradCAM
from .model import CATEGORIES, predict_topk


@dataclass
class Task4Run:
    """JSON-serialisable ``results`` plus the tensors the figures need."""

    results: dict
    extra: dict = field(default_factory=dict)


def _stats(map_auc: float, rand: list[float], higher_is_better: bool) -> dict:
    """Map AUC against the random-order distribution: mean, std, difference and distance in standard deviations."""
    m, s = float(np.mean(rand)), float(np.std(rand, ddof=1))
    gain = (map_auc - m) if higher_is_better else (m - map_auc)
    return {"map": map_auc, "random_mean": m, "random_std": s, "random_all": [float(v) for v in rand],
            "diff_vs_random": map_auc - m, "sds_better_than_random": gain / s if s > 0 else float("inf")}


def _first_below(c: np.ndarray, frac: float) -> int | None:
    """First step k >= 1 where c_k <= frac * c_0 (None if it never happens)."""
    hit = np.flatnonzero(c[1:] <= frac * c[0])
    return int(hit[0]) + 1 if len(hit) else None


def _perturbation(model: nn.Module, x01: torch.Tensor, saliency: torch.Tensor, cls: int, seed: int,
                  reference: torch.Tensor) -> dict:
    """Deletion towards ``reference`` and insertion from the blurred image, with the map order and R random orders."""
    steps, R = config.PERTURB_STEPS, config.PERTURB_RANDOM_ORDERS
    blurred = fa.gaussian_blur(x01, config.BLUR_SIGMA)
    order = fa.order_from_map(saliency)
    rand_orders = [fa.random_order(order.numel(), seed + r) for r in range(R)]
    d_map = fa.curve(model, x01, reference, order, cls, steps)
    i_map = fa.curve(model, blurred, x01, order, cls, steps)
    d_rand = [fa.curve(model, x01, reference, o, cls, steps) for o in rand_orders]
    i_rand = [fa.curve(model, blurred, x01, o, cls, steps) for o in rand_orders]
    return {
        "deletion": _stats(fa.auc(d_map), [fa.auc(c) for c in d_rand], higher_is_better=False),
        "insertion": _stats(fa.auc(i_map), [fa.auc(c) for c in i_rand], higher_is_better=True),
        "curves": {"deletion_map": d_map.tolist(), "insertion_map": i_map.tolist(),
                   "deletion_random_mean": np.mean(d_rand, axis=0).tolist(), "deletion_random_std": np.std(d_rand, axis=0, ddof=1).tolist(),
                   "insertion_random_mean": np.mean(i_rand, axis=0).tolist(), "insertion_random_std": np.std(i_rand, axis=0, ddof=1).tolist()},
        "half_drop_step_map": _first_below(d_map, 0.5),
        "half_drop_step_random": [_first_below(c, 0.5) for c in d_rand],
        "cells_touched_map": [fa.cells_touched(order, k, steps) for k in range(1, steps + 1)],
        "cells_touched_random_step1": fa.cells_touched(rand_orders[0], 1, steps),
        "largest_step_drop_map": float(np.max(-np.diff(d_map))),
        "largest_step_drop_map_at": int(np.argmax(-np.diff(d_map))) + 1,
        "p_start": float(d_map[0]), "p_reference": float(d_map[-1]), "p_blurred": float(i_map[0]),
        "zero_map_pixels": int((saliency == 0).sum()),
    }


def run_task4(model: nn.Module, specs, seed: int = config.SEED) -> Task4Run:
    t0 = time.perf_counter()
    res: dict = {"environment": {"seed": seed, "torch": torch.__version__, "torchvision": torchvision.__version__,
                                 "python": platform.python_version()},
                 "config": {"steps": config.PERTURB_STEPS, "random_orders": config.PERTURB_RANDOM_ORDERS,
                            "blur_sigma": config.BLUR_SIGMA, "gray_level": config.GRAY_LEVEL,
                            "pixels_per_step": config.CROP_SIZE ** 2 / config.PERTURB_STEPS,
                            "stages": list(config.CASCADE_STAGES), "spearman_low": config.CASCADE_SPEARMAN_LOW}}
    extra: dict = {"prepared": {}, "maps": {}}

    # ---- 4.1 ------------------------------------------------------------------------------
    t41, base_maps = {}, {}
    for spec in specs:
        prep = prepare_image(spec)
        top1 = predict_topk(model, prep.tensor, 1)[0]
        with GradCAM(model, model.layer4) as g:
            gc = g(prep.tensor, top1.index)
        x01 = fa.to_unit(prep.crop)
        r = _perturbation(model, x01, gc.cam_224, top1.index, seed, fa.gray_like(x01))
        r.update({"class_idx": top1.index, "class_name": CATEGORIES[top1.index], "p_top1": top1.probability})
        t41[spec.id] = r
        base_maps[spec.id] = (prep, top1.index, gc)
        extra["prepared"][spec.id] = prep
        extra["maps"][spec.id] = gc.cam_224
    res["t4_1"] = t41

    # ---- 4.3(a) extra: deletion of image (a) towards a blurred copy instead of gray --------------------
    prep, cls, gc = base_maps["a"]
    x01 = fa.to_unit(prep.crop)
    blur_ref = _perturbation(model, x01, gc.cam_224, cls, seed, fa.gaussian_blur(x01, config.BLUR_SIGMA))
    res["t4_3a_blur_reference"] = {"image": "a", "class_name": CATEGORIES[cls], "deletion": blur_ref["deletion"],
                                   "curves": {k: v for k, v in blur_ref["curves"].items() if k.startswith("deletion")},
                                   "deletion_gray": t41["a"]["deletion"]}

    # ---- 4.2 ------------------------------------------------------------------------------
    torch.manual_seed(seed)  # reset_parameters draws from the global generator
    rnd = copy.deepcopy(model).eval()
    stages, maps_by_stage = [], {rid: [] for rid in base_maps}
    fc_check = None
    for i, name in enumerate(config.CASCADE_STAGES):
        n_reset = sanity.reinitialize(getattr(rnd, name))
        row = {"stage": i + 1, "randomized": list(config.CASCADE_STAGES[: i + 1]), "modules_reset": n_reset, "images": {}}
        for rid, (prep_i, cls_i, gc0) in base_maps.items():
            with GradCAM(rnd, rnd.layer4) as g:
                gcr = g(prep_i.tensor, cls_i)
            with torch.no_grad():
                p_cls = float(rnd(prep_i.tensor)[0].softmax(0)[cls_i])
            row["images"][rid] = {
                "spearman": sanity.spearman_maps(gc0.cam_7x7.numpy(), gcr.cam_7x7.numpy()),
                "spearman_raw": sanity.spearman_maps(gc0.raw_map.numpy(), gcr.raw_map.numpy()),
                "positive_cells": int((gcr.cam_7x7 > 0).sum()), "p_original_class": p_cls,
            }
            maps_by_stage[rid].append(gcr.cam_7x7)
            if name == "fc":
                # alpha = w/Z: with layer4 untouched the map must equal sum_k w_rand_k A^k / Z with the ORIGINAL A
                pred = (rnd.fc.weight[cls_i].detach()[:, None, None] * gc0.activations).sum(0) / config.Z
                err = float((pred - gcr.raw_map).abs().max())
                fc_check = err if fc_check is None else max(fc_check, err)
                row["images"][rid]["max_abs_error_vs_prediction"] = err
                row["images"][rid]["activations_unchanged"] = float((gcr.activations - gc0.activations).abs().max())
                # what a random fc keeps: the spatial layout shared by all channels (unweighted sum of the activations)
                row["images"][rid]["spearman_original_vs_sum_A"] = sanity.spearman_maps(
                    gc0.cam_7x7.numpy(), gc0.activations.sum(0).numpy())
                row["images"][rid]["fc_weight_sum"] = float(rnd.fc.weight[cls_i].sum())
        stages.append(row)
    res["t4_2"] = {"stages": stages, "fc_prediction_max_abs_error": fc_check,
                   "first_stage_below": {rid: next((s["stage"] for s in stages if not s["images"][rid]["spearman"] >= config.CASCADE_SPEARMAN_LOW), None)
                                         for rid in base_maps}}
    extra["cascade_maps"] = {rid: [gc0.cam_7x7 for (_, _, gc0) in [base_maps[rid]]] + maps_by_stage[rid] for rid in base_maps}
    res["time_s"] = time.perf_counter() - t0
    return Task4Run(res, extra)
