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


def _point_segment_geometry(dpu, dpv, L, add=True):
    """For a point p and segment (u, v): position t of the projection of p on the
    line u->v (distance from u), altitude h to the line and distance e to the segment."""
    a, b, c = triangle_fix(*(np.asarray(v, dtype=float) for v in (dpu, dpv, L)), add)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (a**2 + c**2 - b**2) / (2 * c)
        h = np.sqrt(np.maximum(a**2 - t**2, 0.0))
    # beyond the ends the distance to the endpoint is the original (uncorrected) one
    ra, rb, rc = (np.asarray(v, dtype=float) for v in (dpu, dpv, L))
    e = np.where(t <= 0, ra, np.where(t >= c, rb, h))
    if add:  # segment longer than the two other sides together: nearest endpoint, uncorrected
        e = np.where(rc > ra + rb, np.minimum(ra, rb), e)
    return t, h, e, c


def _hausdorff_matrix(d, ini, fin, add):
    L = d[ini, fin]
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
    return L, dii, dff, dif, dfi


def segment_distances(x: Trajectories, distance_type="directed-segment", add=True) -> dict:
    """Distances between all pairs of directed segments of all trajectories.

    ``distance_type``: ``"directed-segment"`` (default), ``"Hausdorff"`` or
    ``"PPA"``. Returns DataFrames ``Dseg`` (segment distance), ``Dini``,
    ``Dfin`` (distances between initial / final points), ``Dinifin`` (initial
    point of the row segment to final point of the column segment) and
    ``Dfinini``, labelled ``site[s1-s2]``.

    As in ecotraj (whose ``Dseg`` is an R ``dist`` object) only the lower triangle
    of the segment-by-segment computation is retained and mirrored: the value for
    segments ``i < j`` is computed with ``i`` as the first segment and ``j`` as
    the second one (matters only for asymmetric definitions and for ties in length).

    PPA follows Lee et al. (2007) with parallel/angle terms evaluated from the
    triangles formed by the endpoints (see ``_ppa_pair``). The directed-segment
    distance is the Hausdorff distance with a direction penalty for end points
    whose projection runs against the other segment (see ``_directed_segment``).
    """
    d = x.d
    ini, fin, _, labels = _segments(x)
    L, dii, dff, dif, dfi = _segment_geometry(d, ini, fin, add)
    H, pi, pf = _hausdorff_matrix(d, ini, fin, add)
    n = len(ini)
    if distance_type == "Hausdorff":
        Dseg = H.copy()
    elif distance_type == "directed-segment":
        Dseg = _directed_segment(d, ini, fin, L, H, pi, pf, add)
    elif distance_type == "PPA":
        Dseg = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                Dseg[j, i] = _ppa_pair(d, ini, fin, L, i, j, add)
    else:
        raise ValueError("distance_type must be 'directed-segment', 'Hausdorff' or 'PPA'")
    Dseg = np.tril(Dseg, -1)
    Dseg = Dseg + Dseg.T  # as.dist() in R keeps the lower triangle

    def df(m):
        return pd.DataFrame(m, index=labels, columns=labels)

    return dict(Dseg=df(Dseg), Dini=df(dii), Dfin=df(dff), Dinifin=df(dif))


def _obtuse(d, ini, fin, i, j):
    """True if the angle between directed segments i and j is obtuse (dot < 0)."""
    A, B, C, D = ini[i], fin[i], ini[j], fin[j]
    return 0.5 * (d[A, D] ** 2 + d[B, C] ** 2 - d[A, C] ** 2 - d[B, D] ** 2) < 0


def _directed_segment(d, ini, fin, L, H, pi, pf, add):
    """Directed-segment distance, De Cáceres et al. (2019) eqs. 8-10.

    D_DS(si, sj) = max{Dps(x_i, sj), D'ps(x_i+1, sj), Dps(x_j, si), D'ps(x_j+1, si)}
    where D'ps(x_i+1, sj) = Dps(x_i+1, sj) if R(x_i, sj) <= R(x_i+1, sj) (R is the
    relative position, 0..1, of the projection on the segment; 0 / 1 when it falls
    beyond the initial / final end point) and otherwise a penalty equal to the
    segment length plus the distance of the nearer end point of si to sj,
    min(Dps(x_i, sj), Dps(x_i+1, sj)) + L(si). (The paper writes Dps(x_i, sj) + L(si);
    the min form is what ecotraj 1.2.2 computes, established by black-box probing.)
    ``pi[p, s]`` / ``pf[p, s]`` are Dps of the initial / final point of segment p
    to segment s.
    """
    def rel(points):
        t, _, _, c = _point_segment_geometry(d[np.ix_(points, ini)], d[np.ix_(points, fin)], L[None, :], add)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(t <= 0, 0.0, np.where(t >= c, 1.0, t / c))
        return np.where(L[None, :] == 0, 0.0, r)
    ri, rf = rel(ini), rel(fin)  # R of initial / final point of segment p onto segment s
    fin_mod = np.where(ri > rf, np.minimum(pi, pf) + L[:, None], pf)  # D'ps(x_p+1, s)
    T = np.maximum(pi, fin_mod)  # row p, column s: end points of p against s
    return np.maximum(T, T.T)


def _lehmer(x, y):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (x * x + y * y) / (x + y)


def _ppa_pair(d, ini, fin, L, i, j, add):
    """Perpendicular-parallel-angle distance between segments i (first) and j (second).

    The longer segment (i on ties) is the reference, the shorter one is projected
    on it. For each endpoint of the shorter segment, with ``e`` its distance to the
    reference segment and ``par`` its parallel offset (distance of the projection to
    the nearest end of the reference when it falls inside it, else 0):
    ``PPA = Lehmer(e1, e2) + min(par) + angle`` where the angle term is
    ``|e1 - e2|`` for acute angles (or when no projection falls inside) and
    the length of the shorter segment for obtuse angles (as in TRACLUS).
    """
    if L[i] >= L[j]:
        lo, sh = i, j
    else:
        lo, sh = j, i
    e, par, inside = [], [], []
    for p in (ini[sh], fin[sh]):
        t, h, ee, c = _point_segment_geometry(d[p, ini[lo]], d[p, fin[lo]], L[lo], add)
        ok = bool(0 <= t <= c)
        inside.append(ok)
        e.append(float(ee))
        par.append(float(min(t, c - t)) if ok else 0.0)
    if not any(inside) or not _obtuse(d, ini, fin, lo, sh):
        angle = abs(e[0] - e[1])
    else:
        angle = L[sh]
    return float(_lehmer(np.float64(e[0]), np.float64(e[1]))) + min(par) + angle


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
        reference_times = np.unique(x.times[idxs[0]])
        if any(not np.array_equal(np.unique(x.times[ix]), reference_times) for ix in idxs):
            raise ValueError("trajectories must be synchronous")
        # Public R-oracle observations: use times in the first site's input
        # row order, without sorting. Match all rows by time, not by survey;
        # repeated times contribute repeated tests of the same group.
        order = x.times[x.sites == x.sites[0]]
        v = []
        for time in order:
            pts = np.flatnonzero(x.times == time)
            sub = d[np.ix_(pts, pts)]
            v.append(np.sum(sub**2) / 2 / len(pts) / (nt - 1))
        tau, p = _mk(v)
        return dict(tau=tau, p_value=p)
    tau = np.full((nt, nt), np.nan)
    pv = np.full((nt, nt), np.nan)
    for a in range(nt):
        for b in range(nt):
            if a == b:
                continue
            if type == "pairwise.symmetric":
                if len(idxs[a]) != len(idxs[b]):  # ecotraj gives NA for unequal numbers of surveys
                    continue
                seq = d[idxs[a], idxs[b]]
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
