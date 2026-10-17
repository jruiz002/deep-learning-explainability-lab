"""Project-wide constants and paths."""

from __future__ import annotations

from pathlib import Path

SEED = 42

# ImageNet preprocessing (as required by the lab statement).
RESIZE_SHORT_SIDE = 256
CROP_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Geometry of the last convolutional stage of ResNet-18 (layer4).
FEATURE_CHANNELS = 512
FEATURE_SIZE = 7
Z = FEATURE_SIZE * FEATURE_SIZE  # 49 spatial positions

TOP_K = 5
IOU_TOP_FRACTION = 0.20
GRAY_LEVEL = 0.5  # pixel value in [0, 1] used by the causal (occlusion) test

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
MANIFEST_PATH = DATA_DIR / "images.json"
OUTPUT_DIR = ROOT / "outputs" / "task1"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Numerical tolerances for the self-checks printed at the end of a run (float32 arithmetic).
TOL_ABS = 1e-4  # logit decomposition, Z*L = M and normalised-map comparisons
TOL_REL = 1e-5  # alpha_k vs w_k / Z

# ---- Task 2 (SHAP on German Credit) -------------------------------------------------------
OUTPUT_DIR_T2 = ROOT / "outputs" / "task2"
FIGURES_DIR_T2 = OUTPUT_DIR_T2 / "figures"

CREDIT_TEST_SIZE = 0.2
GBC_N_ESTIMATORS = 200
GBC_MAX_DEPTH = 3
BACKGROUND_SIZE = 100
BACKGROUND_SIZES_KERNEL = (10, 50, 100)
KERNEL_NSAMPLES = 2000
KERNEL_N_INSTANCES = 10
PERM_REPEATS = 10
DUP_NOISE_SCALE = 0.05  # copy = x + 0.05 * sigma_x * eps
SMALL_MODEL_FEATURES = (
    "duration", "credit_amount", "installment_commitment", "residence_since", "age", "existing_credits",
)
SMALL_MODEL_BACKGROUND = 50
SMALL_MODEL_INSTANCES = 5
PROB_APPROVED, PROB_REJECTED = 0.8, 0.2  # p(good) thresholds used to pick the two clients
PROB_DOUBTFUL_BAND = (0.45, 0.55)

# Tolerances for Task 2 self-checks. TreeSHAP (C++ extension) accumulates ~1e-7 errors vs the float64 model output
# (efficiency error observed ~1e-7), so 1e-6 leaves ~10x margin without hiding real bugs (real errors are O(1e-2)).
TOL_SHAP = 1e-6
TOL_SHAP_EXACT = 1e-10  # own Shapley efficiency (float64, 64 coalitions): observed error ~1e-16

# ---- Task 3 (LIME from scratch on German Credit) -------------------------------------------
OUTPUT_DIR_T3 = ROOT / "outputs" / "task3"
FIGURES_DIR_T3 = OUTPUT_DIR_T3 / "figures"

LIME_N = 5000  # perturbed samples per explanation (3.1)
LIME_K = 5  # variables kept in the interpretable model
LIME_LAMBDA = 1.0  # ridge penalty (the intercept is not penalised)
LIME_SIGMA_REL = 0.75  # kernel width = sigma_rel * sqrt(d')
LIME_D_PRIME = 20  # reference dimension of the kernel (the 20 German Credit variables)
LIME_STAB_N = (500, 2000, 5000)
LIME_STAB_SIGMA_REL = (0.5, 0.75, 1.5)
LIME_STAB_RUNS = 20  # seeds 0..19 in every cell of the grid
LIME_STAB_TOP = 5  # how many frequent variables are stored per cell

TOL_LIME = 1e-8  # closed-form ridge vs lstsq / sklearn Ridge (float64; errors are far below this)

# 3.3 LIME vs SHAP and fidelity as a function of K
LIME_K_VALUES = (3, 5, 10, 20)
TOP_COMPARE = 3  # size of the "top variables" sets compared between LIME and SHAP
