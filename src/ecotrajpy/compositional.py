"""Compositional-data helpers."""
import numpy as np


def clr(X):
    """Row-wise centred log-ratio transform: log(x) - mean(log(x)) per row.

    Rows must be strictly positive (closure is irrelevant for clr).
    """
    L = np.log(np.asarray(X, dtype=float))
    return L - L.mean(axis=-1, keepdims=True)
