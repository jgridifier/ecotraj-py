"""Increment split of a trajectory into an explained ("market") path and a residual path."""
from __future__ import annotations

import numpy as np


def increment_split(C, X) -> dict:
    """Regress trajectory increments on covariate increments.

    dC = diff(C), dX = diff(X); B = lstsq(dX, dC) without intercept;
    market = C[0] + cumsum(dX @ B); residual = C[0] + cumsum(dC - dX @ B).
    ``step_energy_market`` is the share of squared step length explained
    (sum ||dX B||^2 / sum ||dC||^2).
    """
    C = np.asarray(C, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    dC, dX = np.diff(C, axis=0), np.diff(X, axis=0)
    B, *_ = np.linalg.lstsq(dX, dC, rcond=None)
    fit = dX @ B
    start = C[:1]
    market = np.vstack([start, start[0] + np.cumsum(fit, axis=0)])
    resid = np.vstack([start, start[0] + np.cumsum(dC - fit, axis=0)])
    return dict(market=market, residual=resid, B=B,
                step_energy_market=float(np.sum(fit**2) / np.sum(dC**2)))
