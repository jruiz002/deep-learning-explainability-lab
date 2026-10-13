"""Task 1 experiments (1.1 - 1.4). Pure computation: returns plain, JSON-serialisable results."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import torch
from torch import nn

from . import config
from .cam import compute_cam, feature_maps
from .causal import gray_out
from .data import PreparedImage, prepare_image
from .gradcam import GradCAM, GradCAMResult
from .maps import iou, mass_in_box, relu_minmax, top_fraction_mask, box_mask
from .model import CATEGORIES, class_probability, predict_topk

log = logging.getLogger(__name__)


@dataclass
class ImageRun:
    """Everything computed for one image; tensors are kept for plotting."""

    prepared: PreparedImage
    top5: list
    gradcams: dict[int, GradCAMResult]  # class index -> Grad-CAM result
    cam: dict  # task 1.3 outputs (tensors + floats)
    extra: dict  # task 1.4 outputs


def pick_group_classes(model: nn.Module, x: torch.Tensor, groups: dict[str, list[int]]) -> dict[str, int]:
    """Highest-probability ImageNet class inside each semantic group (e.g. dog / cat)."""
    with torch.no_grad():
        probs = model(x)[0].softmax(0)
    return {name: idxs[int(probs[idxs].argmax())] for name, idxs in groups.items()}


def task_1_3(model: nn.Module, gc: GradCAMResult) -> dict:
    """CAM from the fc weights and its relation to Grad-CAM for one image / class."""
    cam = compute_cam(model, gc.activations, gc.class_idx)
    z_l = config.Z * gc.raw_map  # Z * L^c (no ReLU, no normalisation)
    diff_b = abs(cam.logit_from_cam - gc.logit)
    diff_c = (z_l - cam.map).abs()
    n_cam, n_gc = relu_minmax(cam.map), relu_minmax(gc.raw_map)
    return {
        "cam": cam,
        "z_times_L": z_l,
        "b_logit": gc.logit,
        "b_cam_logit": cam.logit_from_cam,
        "b_abs_error": diff_b,
        "c_max_abs_error": float(diff_c.max()),
        "c_abs_diff_map": diff_c,
        "d_norm_cam": n_cam,
        "d_norm_gradcam": n_gc,
        "d_max_abs_error": float((n_cam - n_gc).abs().max()),
    }


def task_1_4a(gc: GradCAMResult, cam_w: torch.Tensor, n: int = 5) -> list[dict]:
    """alpha_k^c vs w_k^c / Z for the ``n`` channels with the largest fc weight."""
    top = cam_w.topk(n).indices
    rows = []
    for k in top.tolist():
        ref = float(cam_w[k]) / config.Z
        a = float(gc.alpha[k])
        rows.append(
            {
                "channel": k,
                "w": float(cam_w[k]),
                "alpha": a,
                "w_over_Z": ref,
                "abs_error": abs(a - ref),
                "rel_error": abs(a - ref) / abs(ref),
            }
        )
    return rows


def task_1_4b(map1: torch.Tensor, map2: torch.Tensor) -> dict:
    m1, m2 = top_fraction_mask(map1), top_fraction_mask(map2)
    return {
        "iou": iou(m1, m2),
        "mask1": m1,
        "mask2": m2,
        "pixels_per_mask": int(m1.sum()),
        "intersection": int((m1 & m2).sum()),
        "union": int((m1 | m2).sum()),
    }


def task_1_4c(model: nn.Module, prep: PreparedImage, gc: GradCAMResult) -> dict:
    box = prep.bbox_224
    assert box is not None, "image (c) needs a bounding box"
    cls = gc.class_idx
    p_orig = class_probability(model, prep.tensor, cls)
    x_in, x_out = (gray_out(prep.crop, box, keep=k) for k in ("inside", "outside"))
    p_keep_obj = class_probability(model, x_in, cls)  # background replaced by gray
    p_keep_bg = class_probability(model, x_out, cls)  # object replaced by gray (control)
    top_after = predict_topk(model, x_in, 1)[0]
    return {
        "bbox_224": box,
        "box_area_fraction": float(box_mask(box).float().mean()),
        "mass_in_box": mass_in_box(gc.cam_224, box),
        "class_idx": cls,
        "class_name": CATEGORIES[cls],
        "p_original": p_orig,
        "p_background_gray": p_keep_obj,
        "delta_background_gray": p_keep_obj - p_orig,
        "p_object_gray": p_keep_bg,
        "delta_object_gray": p_keep_bg - p_orig,
        "top1_after_background_gray": {"name": top_after.name, "probability": top_after.probability},
        "tensor_background_gray": x_in,
        "tensor_object_gray": x_out,
    }


def run_task1(model: nn.Module, specs) -> dict[str, ImageRun]:
    """Run every sub-task for every image. Keys are the image ids ("a", "b", "c")."""
    runs: dict[str, ImageRun] = {}
    for spec in specs:
        prep = prepare_image(spec)
        top5 = predict_topk(model, prep.tensor, config.TOP_K)
        top1 = top5[0].index
        log.info("image %s top-1: %s (%.3f)", spec.id, top5[0].name, top5[0].probability)

        classes = [top1]
        group_classes: dict[str, int] = {}
        if spec.class_groups:  # image (b): one class per object
            group_classes = pick_group_classes(model, prep.tensor, spec.class_groups)
            for idx in group_classes.values():
                if idx not in classes:
                    classes.append(idx)

        with GradCAM(model, model.layer4) as g:
            gradcams = {c: g(prep.tensor, c) for c in classes}

        cam_info = task_1_3(model, gradcams[top1])
        extra: dict = {"group_classes": group_classes, "classes": classes}
        if spec.role == "a":
            extra["t14a"] = task_1_4a(gradcams[top1], model.fc.weight[top1].detach())
        if spec.role == "b":
            # Order the two object classes by probability: first = most probable.
            ordered = sorted(group_classes.values(), key=lambda c: -float(gradcams[c].logit))
            extra["t14b"] = task_1_4b(gradcams[ordered[0]].cam_224, gradcams[ordered[1]].cam_224)
            extra["t14b"]["class1"], extra["t14b"]["class2"] = ordered
            extra["t14b"]["probabilities"] = {
                c: class_probability(model, prep.tensor, c) for c in ordered
            }
        if spec.role == "c":
            extra["t14c"] = task_1_4c(model, prep, gradcams[top1])
        runs[spec.id] = ImageRun(prep, top5, gradcams, cam_info, extra)
    return runs
