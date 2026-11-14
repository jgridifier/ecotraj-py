"""Metrics describing individual trajectories (De Cáceres et al. 2019;
Sturbois et al. 2021)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .core import Trajectories, angle_from_distances


def _frame(rows, names, cols):
    return pd.DataFrame(rows, index=names, columns=cols)


def _pad(rows, k):
    return [list(r) + [np.nan] * (k - len(r)) for r in rows]


def _all_lengths_compat(x, names, npts):
    """Reproduce ecotraj's ``trajectoryLengths(all=TRUE)`` for several trajectories.

    Inferred from black-box output (this is a quirk/bug of ecotraj, not intended
    behaviour): trajectories are taken as consecutive blocks of ``npts`` rows of
    the distance matrix (row order, not survey order) and the k-th column holds
    the distance of the (k mod 3)-th pair of (1,n-1), (1,n), (2,n) within the
    block (correct only for n = 3). Unequal lengths, or n = 5, make ecotraj
    fail with "subscript out of bounds".
    """
    sizes = [int(np.sum(x.sites == n)) for n in names]
    if len(set(sizes)) > 1 or npts == 5:
        raise ValueError("subscript out of bounds (ecotraj all=TRUE with several trajectories)")
    pairs = list(itertools.combinations(range(npts), 2))
    cyc = [(0, npts - 2), (0, npts - 1), (1, npts - 1)]
    rows = []
    for t in range(len(names)):
        off = t * npts
        rows.append([x.d[off + cyc[k % 3][0], off + cyc[k % 3][1]] for k in range(len(pairs))])
    return rows, pairs


def trajectory_lengths(x: Trajectories, relative_to_initial=False, all=False, ecotraj_compat=True) -> pd.DataFrame:
    """Segment lengths and path length per trajectory.

    ``relative_to_initial`` gives distances from the first survey to each later
    one (columns ``Lt1_t2``...); ``all`` gives lengths of all state pairs
    i<j (columns ``Lt{i}Lt{j}``, no Path column) taken in row order. For a single
    trajectory this is the plain set of pairwise distances. For several
    trajectories ecotraj's ``all=TRUE`` is buggy (see ``_all_lengths_compat``);
    it is reproduced by default and ``ecotraj_compat=False`` returns the intended
    pairwise distances per trajectory instead.
    """
    names, idxs = x.names, x.indices()
    if all:
        npts = max(len(i) for i in idxs)
        pairs = list(itertools.combinations(range(npts), 2))
        cols = [f"Lt{i+1}Lt{j+1}" for i, j in pairs]
        if len(names) > 1 and ecotraj_compat:
            rows, _ = _all_lengths_compat(x, names, npts)
            return _frame(rows, names, cols)
        idxs = [np.flatnonzero(x.sites == n) for n in names]
        rows = [[x.d[ix[i], ix[j]] if j < len(ix) else np.nan for i, j in pairs] for ix in idxs]
        return _frame(rows, names, cols)
    k = max(len(i) for i in idxs) - 1
    rows, path = [], []
    for ix in idxs:
        if relative_to_initial:
            v = [x.d[ix[0], ix[j]] for j in range(1, len(ix))]
        else:
            v = [x.d[ix[j], ix[j + 1]] for j in range(len(ix) - 1)]
        rows.append(v)
        path.append(float(np.sum(v)))
    cols = [f"Lt1_t{j+2}" for j in range(k)] if relative_to_initial else [f"S{j+1}" for j in range(k)]
    out = _frame(_pad(rows, k), names, cols)
    out["Path"] = path
    return out


def trajectory_speeds(x: Trajectories) -> pd.DataFrame:
    """Segment speeds (length / elapsed time) and path speed (length / duration)."""
    names, idxs = x.names, x.indices()
    k = max(len(i) for i in idxs) - 1
    rows, path = [], []
    for ix in idxs:
        t = x.times[ix]
        v = [x.d[ix[j], ix[j + 1]] / (t[j + 1] - t[j]) for j in range(len(ix) - 1)]
        rows.append(v)
        path.append(sum(x.d[ix[j], ix[j + 1]] for j in range(len(ix) - 1)) / (t[-1] - t[0]))
    out = _frame(_pad(rows, k), names, [f"S{j+1}" for j in range(k)])
    out["Path"] = path
    return out


def _circular_stats(a_deg):
    a = np.asarray(a_deg, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return np.nan, np.nan, np.nan
    r = np.radians(a)
    C, S = np.mean(np.cos(r)), np.mean(np.sin(r))
    rho = np.sqrt(C * C + S * S)
    mean = np.degrees(np.arctan2(S, C)) % 360.0
    sd = np.sqrt(-2 * np.log(rho)) if rho > 0 else np.inf
    return mean, sd, rho


def trajectory_angles(x: Trajectories, all=False, relative_to_initial=False, stats=True, add=True) -> pd.DataFrame:
    """Angles (degrees) between consecutive segments, in [0, 180].

    0 means segments in a straight line and 180 a reversal. ``all=True`` gives
    angles for all ordered triplets i<j<k (columns A1, A2, ... in lexicographic
    order of triplets); ``relative_to_initial`` gives, for each k>=2, the angle
    at k between initial->k and k->k+1 (columns ``t1-S{k}``). Circular mean
    (degrees), circular standard deviation (radians) and mean resultant length
    rho are appended when ``stats``.
    """
    names, idxs = x.names, x.indices()
    npts = max(len(i) for i in idxs)
    d = x.d
    if all:
        trip = list(itertools.combinations(range(npts), 3))
        cols = [f"A{i+1}" for i in range(len(trip))]
    elif relative_to_initial:
        trip = [(0, j, j + 1) for j in range(1, npts - 1)]
        cols = [f"t1-S{j+1}" for j in range(1, npts - 1)]
    else:
        trip = [(j, j + 1, j + 2) for j in range(npts - 2)]
        cols = [f"S{j+1}-S{j+2}" for j in range(npts - 2)]
    rows = []
    for ix in idxs:
        # valid triplets of this trajectory, in order, padded with NaN on the right
        r = [float(angle_from_distances(d[ix[i], ix[j]], d[ix[j], ix[k]], d[ix[i], ix[k]], add))
             for i, j, k in trip if k < len(ix)]
        rows.append(r + [np.nan] * (len(trip) - len(r)))
    out = _frame(rows, names, cols)
    if stats:
        st = np.array([_circular_stats(r) for r in rows])
        out["mean"], out["sd"], out["rho"] = st[:, 0], st[:, 1], st[:, 2]
    return out


def _directionality_idx(d, ix, add):
    """Directionality of the states ``ix`` (in that order) from a distance matrix."""
    n = len(ix)
    if n < 3:
        return np.nan
    I, J, K = (np.array(v) for v in zip(*itertools.combinations(range(n), 3)))
    a, b, c = d[ix[I], ix[J]], d[ix[J], ix[K]], d[ix[I], ix[K]]
    th = angle_from_distances(a, b, c, add)
    w = a + b
    ok = ~np.isnan(th)
    return float(np.sum((w * (1 - th / 180.0))[ok]) / np.sum(w[ok]))


def trajectory_directionality(x: Trajectories, add=True, nperm=None, rng=None):
    """Directionality of each trajectory (De Cáceres et al. 2019).

    D = sum over ordered triplets i<j<k of (s_ij + s_jk)(1 - theta_ijk/180)
    divided by sum of (s_ij + s_jk). With ``nperm`` a permutation test with
    random order of surveys within each trajectory is performed; the p-value is
    (1 + #{perm >= observed}) / (1 + nperm). Returns a Series, or a DataFrame
    with columns ``directionality`` and ``p.value`` when ``nperm`` is given.
    """
    names, idxs = x.names, x.indices()
    obs = np.array([_directionality_idx(x.d, ix, add) for ix in idxs])
    if nperm is None:
        return pd.Series(obs, index=names)
    rng = np.random.default_rng(rng)
    p = []
    for o, ix in zip(obs, idxs):
        cnt = 0
        for _ in range(nperm):
            cnt += _directionality_idx(x.d, rng.permutation(ix), add) >= o
        p.append((1 + cnt) / (1 + nperm))
    return pd.DataFrame({"directionality": obs, "p.value": p}, index=names)


def trajectory_internal_variation(x: Trajectories, relative_contributions=False) -> pd.DataFrame:
    """Internal variability of states within each trajectory.

    Contribution of each state to the sum of squares (diagonal of the Gower-
    centred matrix of squared distances), total ``internal_ss`` and the unbiased
    ``internal_variance`` = ss / (n - 1).
    """
    names, idxs = x.names, x.indices()
    k = max(len(i) for i in idxs)
    rows = []
    for ix in idxs:
        n = len(ix)
        A = -0.5 * x.d[np.ix_(ix, ix)] ** 2
        G = A - A.mean(0, keepdims=True) - A.mean(1, keepdims=True) + A.mean()
        ss = np.diag(G).copy()
        tot = ss.sum()
        contrib = ss / tot if relative_contributions else ss
        rows.append(list(contrib) + [np.nan] * (k - n) + [tot, tot / (n - 1)])
    pre = "rc_" if relative_contributions else "ss_"
    cols = [f"{pre}{i+1}" for i in range(k)] + ["internal_ss", "internal_variance"]
    return _frame(rows, names, cols)


def trajectory_metrics(x: Trajectories, add=True) -> pd.DataFrame:
    """Several metrics at once (cf. ecotraj ``trajectoryMetrics``)."""
    names, idxs = x.names, x.indices()
    L = trajectory_lengths(x)
    sp = trajectory_speeds(x)
    ang = trajectory_angles(x, add=add)
    dire = trajectory_directionality(x, add=add)
    iv = trajectory_internal_variation(x)
    t0 = np.array([x.times[ix[0]] for ix in idxs])
    t1 = np.array([x.times[ix[-1]] for ix in idxs])
    return pd.DataFrame(
        {
            "trajectory": names,
            "n": [len(ix) for ix in idxs],
            "t_start": t0,
            "t_end": t1,
            "duration": t1 - t0,
            "length": L["Path"].values,
            "mean_speed": sp["Path"].values,
            "mean_angle": ang["mean"].values,
            "directionality": dire.values,
            "internal_ss": iv["internal_ss"].values,
            "internal_variance": iv["internal_variance"].values,
        }
    )
