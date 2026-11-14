"""Trajectory data structure and distance-matrix helpers.

Everything in ecotrajpy is computed from a (full) distance matrix between
ecological states, following the Ecological Trajectory Analysis framework of
De Cáceres et al. (2019, Ecol. Monogr. 89:e01350).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def _as_square(d) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    if d.ndim == 1:
        d = squareform(d)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("d must be a square distance matrix or a condensed vector")
    if not np.allclose(d, d.T, equal_nan=True):
        raise ValueError("distance matrix must be symmetric")
    return d.copy()


@dataclass
class Trajectories:
    """Distance matrix plus state metadata (sites, surveys, times)."""

    d: np.ndarray
    sites: np.ndarray
    surveys: np.ndarray
    times: np.ndarray

    @property
    def names(self) -> list:
        """Trajectory names in order of first appearance."""
        return list(dict.fromkeys(self.sites.tolist()))

    @property
    def metadata(self) -> pd.DataFrame:
        return pd.DataFrame({"sites": self.sites, "surveys": self.surveys, "times": self.times})

    def index(self, name) -> np.ndarray:
        """Row indices of the states of a trajectory, in survey order."""
        idx = np.flatnonzero(self.sites == name)
        return idx[np.argsort(self.surveys[idx], kind="stable")]

    def indices(self) -> list:
        return [self.index(n) for n in self.names]


def define_trajectories(d, sites, surveys=None, times=None) -> Trajectories:
    """Define trajectories (cf. ecotraj ``defineTrajectories``).

    ``d`` is a square distance matrix or condensed vector. If ``surveys`` is
    missing it is taken from ``times`` (order of times within site) or else
    assumed in order of occurrence within each site; if ``times`` is missing
    it equals ``surveys``.
    """
    D = _as_square(d)
    sites = np.asarray([str(s) for s in sites], dtype=object)
    n = D.shape[0]
    if len(sites) != n:
        raise ValueError("sites must have one entry per row of d")
    if surveys is None:
        if times is not None:
            tt = np.asarray(times, dtype=float)
            surveys = np.zeros(n, dtype=int)
            for s in dict.fromkeys(sites.tolist()):
                idx = np.flatnonzero(sites == s)
                surveys[idx] = np.argsort(np.argsort(tt[idx], kind="stable"), kind="stable") + 1
        else:
            surveys = np.zeros(n, dtype=int)
            for s in dict.fromkeys(sites.tolist()):
                idx = np.flatnonzero(sites == s)
                surveys[idx] = np.arange(1, len(idx) + 1)
    surveys = np.asarray(surveys)
    if times is None:
        times = surveys.astype(float)
    times = np.asarray(times, dtype=float)
    return Trajectories(D, sites, surveys, times)


def _from_coords(X, sites, surveys=None, times=None) -> Trajectories:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return define_trajectories(squareform(pdist(X)), sites, surveys, times)


define_trajectories.from_coords = _from_coords  # type: ignore[attr-defined]


def is_metric(d, tol: float = 0.0) -> bool:
    """True if all distance triplets satisfy the triangle inequality.

    The tolerance is relative: a triplet is a violation when
    (d_ij - d_ik - d_kj) / d_ij > tol (behaviour inferred from ecotraj)."""
    D = _as_square(d)
    n = D.shape[0]
    for k in range(n):
        # d_ij <= d_ik + d_kj + tol for all i, j
        v = D - D[:, [k]] - D[[k], :]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(D > 0, v / D, 0.0)
        if np.any(rel > tol):
            return False
    return True


def triangle_fix(a, b, c, add: bool = True):
    """Local additive correction of triplets of distances.

    Where the largest of the three distances exceeds the sum of the other two,
    the minimum constant that makes the triplet satisfy the triangle
    inequality is added to all three distances (De Cáceres et al. 2019).
    """
    a, b, c = (np.asarray(v, dtype=float) for v in (a, b, c))
    if not add:
        return a, b, c
    m = np.maximum(np.maximum(a, b), c)
    viol = np.maximum(2 * m - (a + b + c), 0.0)
    return a + viol, b + viol, c + viol


def angle_from_distances(dij, djk, dik, add: bool = True):
    """Change of direction (degrees; 0 = straight line, 180 = reversal) at the
    middle point j of an ordered triplet, from the three pairwise distances."""
    dij, djk, dik = triangle_fix(dij, djk, dik, add)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_int = (dij**2 + djk**2 - dik**2) / (2 * dij * djk)
    cos_int = np.where(np.abs(cos_int) > 1, np.nan, cos_int)  # impossible triangle
    return 180.0 - np.degrees(np.arccos(cos_int))
