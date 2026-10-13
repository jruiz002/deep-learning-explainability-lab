import torch

from xai_lab import config
from xai_lab.cam import compute_cam, feature_maps
from xai_lab.gradcam import GradCAM

CLS = 7


def test_gradcam_alpha_equals_w_over_Z(model, x):
    with GradCAM(model, model.layer4) as g:
        r = g(x, CLS)
    assert r.activations.shape == (512, 7, 7)
    torch.testing.assert_close(r.alpha, model.fc.weight[CLS].detach() / config.Z, atol=1e-8, rtol=1e-5)


def test_cam_decomposition_equals_logit(model, x):
    cam = compute_cam(model, feature_maps(model, x), CLS)
    logit = float(model(x)[0, CLS])
    assert abs(cam.logit_from_cam - logit) < 1e-4


def test_Z_times_gradcam_equals_cam(model, x):
    with GradCAM(model, model.layer4) as g:
        r = g(x, CLS)
    cam = compute_cam(model, r.activations, CLS)
    torch.testing.assert_close(config.Z * r.raw_map, cam.map, atol=1e-4, rtol=1e-4)


def test_hook_gradient_matches_autograd(model, x):
    captured = {}
    h = model.layer4.register_forward_hook(lambda _m, _i, o: captured.setdefault("a", o))
    logit = model(x)[0, CLS]
    h.remove()
    (grad,) = torch.autograd.grad(logit, captured["a"])
    with GradCAM(model, model.layer4) as g:
        r = g(x, CLS)
    torch.testing.assert_close(r.gradients, grad[0], atol=1e-6, rtol=1e-5)


def test_hooks_are_removed(model, x):
    with GradCAM(model, model.layer4):
        pass
    assert len(model.layer4._forward_hooks) == 0
    assert len(model.layer4._backward_hooks) == 0


def test_output_map_is_normalised(model, x):
    with GradCAM(model, model.layer4) as g:
        r = g(x, CLS)
    assert r.cam_224.shape == (224, 224)
    assert float(r.cam_224.min()) == 0.0 and abs(float(r.cam_224.max()) - 1.0) < 1e-6
