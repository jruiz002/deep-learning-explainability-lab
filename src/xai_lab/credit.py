"""German Credit data, split, gradient-boosting model and client selection (shared by Tasks 2 and 3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from . import config


@dataclass
class CreditData:
    """Encoded German Credit: 20 features (original column order), y = 1 for ``good``, 0 for ``bad``."""

    X: pd.DataFrame  # float64; categorical columns hold OrdinalEncoder codes (arbitrary order)
    y: np.ndarray
    feature_names: list[str]
    categorical: list[str]
    numeric: list[str]
    encoder: OrdinalEncoder | None  # fitted on the categorical columns (in ``categorical`` order)

    def decode(self, feature: str, value: float) -> str:
        """Human-readable value: category label for categorical columns, the number otherwise."""
        if feature in self.categorical:
            cats = self.encoder.categories_[self.categorical.index(feature)]
            return str(cats[int(round(value))])
        return f"{value:g}"


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    idx_train: np.ndarray  # row positions in the full dataset
    idx_test: np.ndarray


@dataclass
class Base:
    """Everything Task 3 needs to reuse the exact Task 2 model: data, split, GBC and 100-row background."""

    data: CreditData
    split: Split
    model: GradientBoostingClassifier
    background: pd.DataFrame


def load_credit() -> CreditData:
    frame = fetch_openml("credit-g", version=1, as_frame=True).frame
    y = (frame["class"].astype(str) == "good").astype(int).to_numpy()
    X = frame.drop(columns="class")
    names = list(X.columns)
    categorical = [c for c in names if str(X[c].dtype) in ("category", "object")]
    numeric = [c for c in names if c not in categorical]
    X = X.copy()
    enc = None
    if categorical:
        enc = OrdinalEncoder()
        X[categorical] = enc.fit_transform(X[categorical].astype(str))
    return CreditData(X.astype(float), y, names, categorical, numeric, enc)


def split(data: CreditData, seed: int = config.SEED) -> Split:
    idx = np.arange(len(data.y))
    tr, te = train_test_split(idx, test_size=config.CREDIT_TEST_SIZE, stratify=data.y, random_state=seed)
    return Split(data.X.iloc[tr], data.X.iloc[te], data.y[tr], data.y[te], tr, te)


def train_gbc(X, y, seed: int = config.SEED) -> GradientBoostingClassifier:
    gbc = GradientBoostingClassifier(
        n_estimators=config.GBC_N_ESTIMATORS, max_depth=config.GBC_MAX_DEPTH, random_state=seed)
    return gbc.fit(X, y)


def background(X_train, n: int = config.BACKGROUND_SIZE, seed: int = config.SEED):
    """``n`` training rows sampled without replacement (default_rng(seed)); keeps the DataFrame type."""
    rows = np.random.default_rng(seed).choice(len(X_train), size=n, replace=False)
    return X_train.iloc[rows]


def log_odds(model, X) -> np.ndarray:
    return model.decision_function(X)


def proba_good(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def select_clients(p_test: np.ndarray) -> dict[str, int]:
    """Positional indices into X_test: first p>0.8 ('aprobado'), first p<0.2 ('rechazado'), p in (0.45, 0.55) closest to 0.5."""
    p = np.asarray(p_test)
    lo, hi = config.PROB_DOUBTFUL_BAND
    approved = np.flatnonzero(p > config.PROB_APPROVED)
    rejected = np.flatnonzero(p < config.PROB_REJECTED)
    doubtful = np.flatnonzero((p > lo) & (p < hi))
    out = {"aprobado": int(approved[0]), "rechazado": int(rejected[0])}
    if len(doubtful):
        out["dudoso"] = int(doubtful[np.argmin(np.abs(p[doubtful] - 0.5))])
    return out


def build_base(seed: int = config.SEED) -> Base:
    data = load_credit()
    sp = split(data, seed)
    model = train_gbc(sp.X_train, sp.y_train, seed)
    return Base(data, sp, model, background(sp.X_train, config.BACKGROUND_SIZE, seed))
