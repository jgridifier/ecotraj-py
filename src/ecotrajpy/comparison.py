"""Comparison of trajectory segments and whole trajectories.

Segment distances follow Besse et al. (2016, IEEE Trans. Intell. Transp. Syst.
17:3306) as adapted in De Cáceres et al. (2019). Everything is derived from the
distance matrix between states (embedding-free), using the cosine law on
triplets of distances.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .core import Trajectories, triangle_fix


def _point_segment(dpu, dpv, L, add=True):
    """Distance from point p to segment (u, v) given d(p,u), d(p,v), d(u,v)."""
    dpu, dpv, L = (np.asarray(v, dtype=float) for v in (dpu, dpv, L))
    a, b, c = triangle_fix(dpu, dpv, L, add)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (a**2 + c**2 - b**2) / (2 * c)  # position of projection along u->v
        h2 = np.maximum(a**2 - t**2, 0.0)
        res = np.where(t <= 0, dpu, np.where(t >= c, dpv, np.sqrt(h2)))
    return np.where(L == 0, dpu, res)


def _segments(x: Trajectories):
    """(ini, fin, trajectory-position, labels) of all segments."""
    ini, fin, owner, labels = [], [], [], []
    for k, (name, ix) in enumerate(zip(x.names, x.indices())):
        for j in range(len(ix) - 1):
            ini.append(ix[j])
            fin.append(ix[j + 1])
            owner.append(k)
            labels.append(f"{name}[{x.surveys[ix[j]]}-{x.surveys[ix[j+1]]}]")
    return np.array(ini), np.array(fin), np.array(owner), labels


def _hausdorff_matrix(d, ini, fin, add):
    L = d[ini, fin]
    n = len(ini)
    # D[p, s]: distance of point p (ini or fin of any segment) to segment s
    def pt(points):
        dpu = d[np.ix_(points, ini)]
        dpv = d[np.ix_(points, fin)]
        return _point_segment(dpu, dpv, L[None, :], add)
    pi, pf = pt(ini), pt(fin)  # (n_points, n_segments)
    # H[i,j] = max(dist of ends of i to segment j, dist of ends of j to segment i)
    H = np.maximum.reduce([pi, pf, pi.T, pf.T])
    return H, pi, pf


def _segment_geometry(d, ini, fin, add):
    L = d[ini, fin]
    dii = d[np.ix_(ini, ini)]
    dff = d[np.ix_(fin, fin)]
    dif = d[np.ix_(ini, fin)]
    dfi = dif.T
    dot = 0.5 * (dif.T**2 + dif**2 - dii**2 - dff**2)  # (B-A).(D-C), A,B=seg i
    # dot[i,j] = 0.5*(|A_i-D_j|^2 + |B_i-C_j|^2 - |A_i-C_j|^2 - |B_i-D_j|^2)
    dot = 0.5 * (dif**2 + dfi**2 - dii**2 - dff**2)
    return L, dii, dff, dif, dfi, dot


def segment_distances(x: Trajectories, distance_type="directed-segment", add=True) -> dict:
    """Distances between all pairs of directed segments of all trajectories.

    ``distance_type``: ``"directed-segment"`` (default), ``"Hausdorff"`` or
    ``"PPA"``. Returns DataFrames ``Dseg`` (segment distance), ``Dini``,
    ``Dfin`` (distances between initial / final points), ``Dinifin`` (initial
    point of the row segment to final point of the column segment) and
    ``Dfinini``, labelled ``site[s1-s2]``.

    Hausdorff is the exact Hausdorff distance between the two geometric
    segments. For angles between segments below 90 degrees the directed-segment
    distance equals it; for obtuse angles ecotraj adds a direction penalty whose
    exact form is not reproduced here (an approximation is used; see docs).
    """
    d = x.d
    ini, fin, _, labels = _segments(x)
    L, dii, dff, dif, dfi, dot = _segment_geometry(d, ini, fin, add)
    H, pi, pf = _hausdorff_matrix(d, ini, fin, add)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosang = dot / (L[:, None] * L[None, :])
    if distance_type == "Hausdorff":
        Dseg = H
    elif distance_type == "directed-segment":
        Dseg = _directed_segment(H, L, cosang, pi, pf)
    elif distance_type == "PPA":
        Dseg = _ppa(d, ini, fin, L, cosang, add)
    else:
        raise ValueError("distance_type must be 'directed-segment', 'Hausdorff' or 'PPA'")
    np.fill_diagonal(Dseg, 0.0)

    def df(m):
        return pd.DataFrame(m, index=labels, columns=labels)

    return dict(Dseg=df(Dseg), Dini=df(dii), Dfin=df(dff), Dinifin=df(dif))


def _directed_segment(H, L, cosang, pi, pf):
    """Directed-segment distance.

    Equal to the Hausdorff distance when the angle between segments is < 90
    degrees (exact, verified against ecotraj). For obtuse angles ecotraj adds
    a penalty; empirically (about 74% of random cases to 1e-6) it is
    max(H, Lmax + min distance from the endpoints of the longer segment to the
    shorter one). This branch is an approximation of ecotraj's behaviour.
    """
    n = len(L)
    out = H.copy()
    near = np.minimum(pi, pf)  # near[i, j]: closest endpoint of segment i to segment j
    for i in range(n):
        for j in range(n):
            if i != j and cosang[i, j] < 0:
                x = near[i, j] if L[i] >= L[j] else near[j, i]
                out[i, j] = max(H[i, j], max(L[i], L[j]) + x)
    return out


def _proj_geometry(d, ini, fin, L, i, j, add):
    """Shorter segment s projected on the line of the longer one l.
    Returns (l_perp_start, l_perp_end, par_start, par_end)."""
    s, l = (i, j) if L[i] <= L[j] else (j, i)
    out = []
    for p in (ini[s], fin[s]):
        a, b, c = triangle_fix(d[p, ini[l]], d[p, fin[l]], L[l], add)
        t = (a**2 + c**2 - b**2) / (2 * c) if c > 0 else 0.0
        hperp = np.sqrt(max(a**2 - t**2, 0.0))
        out.append((hperp, min(abs(t), abs(c - t))))
    return out[0][0], out[1][0], out[0][1], out[1][1], s


def _perp(d, ini, fin, L, i, j, add):
    l1, l2, _, _, _ = _proj_geometry(d, ini, fin, L, i, j, add)
    return (l1**2 + l2**2) / (l1 + l2) if l1 + l2 > 0 else 0.0


def _ppa(d, ini, fin, L, cosang, add):
    """Perpendicular + parallel + angle distance (Besse et al. 2016)."""
    n = len(ini)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            l1, l2, p1, p2, s = _proj_geometry(d, ini, fin, L, i, j, add)
            perp = (l1**2 + l2**2) / (l1 + l2) if l1 + l2 > 0 else 0.0
            par = min(p1, p2)
            c = cosang[i, j]
            theta = np.arccos(np.clip(c, -1, 1))
            dth = L[s] * (np.sin(theta) if theta < np.pi / 2 else 1.0)
            out[i, j] = perp + par + dth
    return out


def trajectory_distances(x: Trajectories, distance_type="DSPD", symmetrization="mean", add=True) -> pd.DataFrame:
    """Distances between trajectories (De Cáceres et al. 2019; Besse et al. 2016).

    * ``"DSPD"`` - directed segment path distance: mean over the segments of T1
      of the minimum directed-segment distance to the segments of T2.
    * ``"SPD"`` - segment path distance: mean over the states of T1 of the
      distance from the state to the nearest point of the path T2.
    * ``"Hausdorff"`` - maximum of the Hausdorff segment distances between all
      segments of T1 and all segments of T2 (inferred from ecotraj output; it
      is symmetric). Self-comparison excludes a segment with itself, so the
      diagonal is non-zero, as in ecotraj.

    The asymmetric matrix D[i, j] = D(Ti -> Tj) is symmetrized with
    ``"mean"``, ``"max"`` or ``"min"``; ``None`` returns it unsymmetrized.
    For ``"SPD"`` any non-None symmetrization is the mean (as in ecotraj).
    """
    names, idxs = x.names, x.indices()
    d = x.d
    ini, fin, owner, _ = _segments(x)
    nt = len(names)
    M = np.zeros((nt, nt))
    if distance_type in ("DSPD", "Hausdorff"):
        t = "directed-segment" if distance_type == "DSPD" else "Hausdorff"
        Dseg = segment_distances(x, t, add)["Dseg"].values.copy()
        np.fill_diagonal(Dseg, np.nan)  # note: also removes zeros within own trajectory
        for a in range(nt):
            ra = np.flatnonzero(owner == a)
            for b in range(nt):
                rb = np.flatnonzero(owner == b)
                sub = Dseg[np.ix_(ra, rb)]
                if a != b:
                    sub = np.where(np.isnan(sub), 0.0, sub)
                with np.errstate(all="ignore"):
                    m = np.nanmin(sub, axis=1) if sub.size else np.array([np.nan])
                if distance_type == "DSPD":
                    M[a, b] = 0.0 if a == b else np.nanmean(m)
                else:
                    M[a, b] = np.nanmax(sub)
    elif distance_type == "SPD":
        for a in range(nt):
            for b in range(nt):
                if a == b:
                    continue
                rb = np.flatnonzero(owner == b)
                dp = _point_segment(d[np.ix_(idxs[a], ini[rb])], d[np.ix_(idxs[a], fin[rb])], d[ini[rb], fin[rb]][None, :], add)
                M[a, b] = np.mean(np.min(dp, axis=1))
    else:
        raise ValueError("distance_type must be 'DSPD', 'SPD' or 'Hausdorff'")
    if symmetrization is None:
        return pd.DataFrame(M, index=names, columns=names)
    if distance_type == "SPD":
        symmetrization = "mean"  # ecotraj ignores the option for SPD
    fn = {"mean": lambda A, B: (A + B) / 2, "max": np.maximum, "min": np.minimum}[symmetrization]
    S = fn(M, M.T)
    np.fill_diagonal(S, 0.0)
    return pd.DataFrame(S, index=names, columns=names)


def _mk(seq):
    seq = np.asarray(seq, dtype=float)
    if len(seq) < 2 or np.any(np.isnan(seq)):
        return np.nan, np.nan
    r = kendalltau(np.arange(len(seq)), seq)
    return float(r.statistic), float(r.pvalue)


def trajectory_convergence(x: Trajectories, type="pairwise.asymmetric", add=True) -> dict:
    """Mann-Kendall tests of convergence/divergence between trajectories.

    * ``pairwise.asymmetric``: trend in the distance from each state of the row
      trajectory to the path of the column trajectory (row approaching column).
    * ``pairwise.symmetric``: trend in distances between states of the same
      survey of two trajectories (equal numbers of surveys).
    * ``multiple``: trend in the variability among trajectories' states at the
      same survey (synchronous trajectories).

    Positive tau means divergence, negative convergence. Returns
    ``dict(tau=..., p_value=...)``.
    """
    names, idxs = x.names, x.indices()
    d = x.d
    nt = len(names)
    if nt < 2 and type != "multiple":
        raise ValueError("at least two trajectories are needed for pairwise convergence tests")
    if type == "multiple":
        n = len(idxs[0])
        if any(len(i) != n for i in idxs):
            raise ValueError("trajectories must be synchronous")
        if any(not np.allclose(x.times[ix], x.times[idxs[0]]) for ix in idxs):
            raise ValueError("trajectories must be synchronous")
        v = []
        for k in range(n):
            pts = np.array([ix[k] for ix in idxs])
            sub = d[np.ix_(pts, pts)]
            v.append(np.sum(sub**2) / 2 / nt / (nt - 1))
        tau, p = _mk(v)
        return dict(tau=tau, p_value=p)
    tau = np.full((nt, nt), np.nan)
    pv = np.full((nt, nt), np.nan)
    for a in range(nt):
        for b in range(nt):
            if a == b:
                continue
            if type == "pairwise.symmetric":
                m = min(len(idxs[a]), len(idxs[b]))  # truncated to common surveys (as ecotraj)
                seq = d[idxs[a][:m], idxs[b][:m]]
            elif type == "pairwise.asymmetric":
                ib = idxs[b]
                u, w = ib[:-1], ib[1:]
                dp = _point_segment(d[np.ix_(idxs[a], u)], d[np.ix_(idxs[a], w)], d[u, w][None, :], add)
                seq = dp.min(axis=1)
            else:
                raise ValueError("unknown type")
            tau[a, b], pv[a, b] = _mk(seq)
    return dict(
        tau=pd.DataFrame(tau, index=names, columns=names),
        p_value=pd.DataFrame(pv, index=names, columns=names),
    )
