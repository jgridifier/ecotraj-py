"""Optional Numba acceleration for Euclidean directionality (add=True).

No fast-math: roundoff and invalid triangles follow the reference arithmetic.
Without Numba, triplets are evaluated in bounded NumPy chunks.
"""
from itertools import combinations, islice

import numpy as np

try:
    from numba import njit, prange
except ImportError:
    njit = None
    prange = range


def _directionality_kernel(C):
    n, p = C.shape
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ss = 0.0
            for q in range(p):
                delta = C[i, q] - C[j, q]
                ss += delta * delta
            d[i, j] = d[j, i] = np.sqrt(ss)
    numerator = 0.0
    denominator = 0.0
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                a, b, c = d[i, j], d[j, k], d[i, k]
                w = a + b
                correction = max(2 * max(a, b, c) - (a + b + c), 0.0)
                a, b, c = a + correction, b + correction, c + correction
                den = 2 * a * b
                if den == 0:
                    continue
                cosine = (a * a + b * b - c * c) / den
                if abs(cosine) > 1 or np.isnan(cosine):
                    continue
                theta = 180.0 - np.arccos(cosine) * (180.0 / np.pi)
                numerator += w * (1 - theta / 180.0)
                denominator += w
    return numerator / denominator if denominator != 0 else np.nan


def _directionality_numpy(C):
    C = np.asarray(C, dtype=float)
    n, p = C.shape
    d = np.zeros((n, n))
    # Sequential coordinate accumulation matches scipy's Euclidean pdist.
    for q in range(p):
        delta = C[:, q, None] - C[None, :, q]
        d += delta * delta
    np.sqrt(d, out=d)
    triples = combinations(range(n), 3)
    numerator = denominator = 0.0
    while True:
        chunk = list(islice(triples, 32768))
        if not chunk:
            break
        i, j, k = np.asarray(chunk).T
        a, b, c = d[i, j], d[j, k], d[i, k]
        w = a + b
        correction = np.maximum(2 * np.maximum(np.maximum(a, b), c) - (a + b + c), 0.0)
        a, b, c = a + correction, b + correction, c + correction
        with np.errstate(divide="ignore", invalid="ignore"):
            cosine = (a**2 + b**2 - c**2) / (2 * a * b)
            cosine = np.where(np.abs(cosine) > 1, np.nan, cosine)
            theta = 180.0 - np.degrees(np.arccos(cosine))
        ok = ~np.isnan(theta)
        numerator += np.sum((w * (1 - theta / 180.0))[ok])
        denominator += np.sum(w[ok])
    return float(numerator / denominator) if denominator != 0 else np.nan


if njit is not None:
    directionality_coords_fast = njit(cache=True)(_directionality_kernel)
else:
    directionality_coords_fast = _directionality_numpy


def _null_kernel(C, signs):
    nsim = signs.shape[0]
    n, p = C.shape
    out = np.empty(nsim)
    for draw in prange(nsim):
        path = np.empty((n, p))
        for q in range(p):
            path[0, q] = C[0, q]
            total = 0.0
            for t in range(1, n):
                total += signs[draw, t - 1] * (C[t, q] - C[t - 1, q])
                path[t, q] = C[0, q] + total
        out[draw] = directionality_coords_fast(path)
    return out


if njit is not None:
    signflip_null_fast = njit(cache=True, parallel=True)(_null_kernel)
else:
    signflip_null_fast = _null_kernel


def signflip_test_fast(C, nsim=999, rng=None, signs=None):
    """Return stat, null and (1 + count(null >= stat)) / (1 + nsim).

    Explicit signs override nsim. Near-ties are resolved with the reference
    arithmetic so parallel/scalar reduction roundoff cannot alter the p-value.
    Kernels accept arrays; this convenience wrapper also accepts array-likes.
    """
    C = np.ascontiguousarray(C, dtype=float)
    if C.ndim != 2 or len(C) == 0:
        raise ValueError("C must be a nonempty (T, p) coordinate array")
    if signs is None:
        signs = np.random.default_rng(rng).choice([-1.0, 1.0], size=(nsim, len(C) - 1))
    else:
        signs = np.ascontiguousarray(signs, dtype=float)
    if signs.ndim != 2 or signs.shape[1] != len(C) - 1:
        raise ValueError("signs must have shape (nsim, T-1)")
    obs = directionality_coords_fast(C)
    null = signflip_null_fast(C, signs)
    close = np.flatnonzero(np.abs(null - obs) <= 1e-12)
    if len(close):
        from .nulls import directionality_coords, signflip_path
        obs = directionality_coords(C)
        for i in close:
            null[i] = directionality_coords(signflip_path(C, signs[i]))
    return dict(stat=obs, p_value=(1 + np.sum(null >= obs)) / (1 + len(signs)), null=null)
