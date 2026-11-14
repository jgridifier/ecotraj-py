"""Sign-flip null model for trajectory directionality.

Flipping the sign of each increment (step) of a trajectory independently
destroys persistence of direction while preserving step lengths in every
coordinate.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from .core import define_trajectories
from .metrics import _directionality_idx


def directionality_coords(C, add=True) -> float:
    """Directionality (De Cáceres et al. 2019) of one trajectory given as an
    (T, p) coordinate array, using Euclidean distances."""
    C = np.asarray(C, dtype=float)
    d = squareform(pdist(C))
    return _directionality_idx(d, np.arange(len(C)), add)


def signflip_path(C, signs):
    """Rebuild a path from C[0] + cumsum(signs[:, None] * diff(C))."""
    C = np.asarray(C, dtype=float)
    signs = np.asarray(signs, dtype=float)
    inc = np.diff(C, axis=0) * signs[:, None]
    return np.vstack([C[:1], C[0] + np.cumsum(inc, axis=0)])


def signflip_test(C, nsim=999, rng=None, signs=None, add=True) -> dict:
    """Sign-flip permutation test of directionality.

    Returns ``dict(stat, p_value, null)`` with p = (1 + #{null >= obs}) / (1 + nsim).
    ``signs`` (nsim, T-1) of +/-1 reproduces given draws exactly.
    """
    C = np.asarray(C, dtype=float)
    obs = directionality_coords(C, add)
    if signs is None:
        rng = np.random.default_rng(rng)
        signs = rng.choice([-1.0, 1.0], size=(nsim, len(C) - 1))
    else:
        signs = np.asarray(signs, dtype=float)
        nsim = signs.shape[0]
    null = np.array([directionality_coords(signflip_path(C, s), add) for s in signs])
    p = (1 + np.sum(null >= obs)) / (1 + nsim)
    return dict(stat=obs, p_value=p, null=null)
