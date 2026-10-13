"""Credit helpers that do not need the network (client selection, background sampling, decoding)."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from xai_lab import credit


def test_select_clients_rules():
    p = np.array([0.6, 0.9, 0.1, 0.85, 0.15, 0.47, 0.52, 0.49])
    sel = credit.select_clients(p)
    assert sel == {"aprobado": 1, "rechazado": 2, "dudoso": 7}  # first p>0.8, first p<0.2, closest to 0.5 in (0.45, 0.55)


def test_background_is_seeded_and_without_replacement():
    X = pd.DataFrame({"a": np.arange(50.0)})
    b1, b2 = credit.background(X, 20, 42), credit.background(X, 20, 42)
    assert b1.index.equals(b2.index) and b1.index.is_unique and len(b1) == 20


def test_decode_categorical_and_numeric():
    enc = OrdinalEncoder().fit(pd.DataFrame({"c": ["x", "y", "z"]}))
    data = credit.CreditData(pd.DataFrame(), np.array([]), ["c", "n"], ["c"], ["n"], enc)
    assert data.decode("c", 2.0) == "z" and data.decode("n", 18.0) == "18"
