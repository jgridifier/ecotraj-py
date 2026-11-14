"""Extension 2: increment-level market/residual split with sign-flip tests.

Implements the procedures of prereg/EXT2_PREREG.md (P1, P1-naive, L, W, Mk).
All statistics are functions of clr coordinates (Aitchison geometry), so they are
invariant to part order and to an orthonormal ilr change of basis.
"""
from __future__ import annotations

import numpy as np

from .fast import directionality_coords_fast
from . import fast as _fast

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None

B_DRAWS = 199
ALPHA_COUNT = 9          # reject iff #{null >= obs} <= 9  (p <= 0.05 with B = 199)


# ---------------------------------------------------------------- compositional
def softmax(C):
    C = np.asarray(C, dtype=float)
    e = np.exp(C - C.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def clr(S):
    L = np.log(np.asarray(S, dtype=float))
    return L - L.mean(axis=-1, keepdims=True)


def ilr_basis(D, rng=None):
    """(D, D-1) matrix V with orthonormal columns orthogonal to 1.

    Helmert basis by default; if ``rng`` is given, rotated by a random orthogonal
    (D-1)x(D-1) matrix, giving another orthonormal ilr basis.
    """
    V = np.zeros((D, D - 1))
    for j in range(1, D):
        V[:j, j - 1] = 1.0 / j
        V[j, j - 1] = -1.0
        V[:, j - 1] *= np.sqrt(j / (j + 1.0))
    if rng is not None:
        Q, _ = np.linalg.qr(rng.standard_normal((D - 1, D - 1)))
        V = V @ Q
    return V


# ---------------------------------------------------------------- split
def split_increments(C, X):
    """Increment split with component paths from zero (prereg section 2).

    Returns dict with dC, dX, B (k x D), fit = dX B, E = dC - fit, Mmat (T-1 x T-1),
    m, r (T x D, both starting at 0).
    """
    C = np.asarray(C, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    dC, dX = np.diff(C, axis=0), np.diff(X, axis=0)
    G = np.linalg.solve(dX.T @ dX, dX.T)          # (k, T-1)
    B = G @ dC
    fit = dX @ B
    E = dC - fit
    Mmat = np.eye(len(dC)) - dX @ G
    zero = np.zeros((1, C.shape[1]))
    return dict(dC=dC, dX=dX, B=B, fit=fit, E=E, Mmat=Mmat,
                m=np.vstack([zero, np.cumsum(fit, axis=0)]),
                r=np.vstack([zero, np.cumsum(E, axis=0)]))


def level_residual_path(C, X):
    """Procedure L: regress clr levels on [1, X] column by column; residual path =
    level residuals + column means of C."""
    C = np.asarray(C, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    Z = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(Z, C, rcond=None)
    return C - Z @ coef + C.mean(axis=0)


# ---------------------------------------------------------------- null kernel
def _null_dir_kernel(c0, inc):
    """inc: (nsim, T-1, D) already-flipped (and projected) increments.
    Path = c0 + cumsum(inc) with a leading c0 row; returns dir of each path."""
    nsim, steps, p = inc.shape
    out = np.empty(nsim)
    path = np.empty((steps + 1, p))
    for draw in range(nsim):
        for q in range(p):
            path[0, q] = c0[q]
            total = 0.0
            for t in range(steps):
                total += inc[draw, t, q]
                path[t + 1, q] = c0[q] + total
        out[draw] = _dir_point(path)
    return out


# The jitted kernel calls fast's raw jitted directionality (not its Python wrapper).
# cache=False: numba's on-disk cache does not track changes in callees, and a stale
# cache is exactly what hid the wrapper change during the first M4 run.
if njit is not None:
    _dir_point = _fast._directionality_impl
    null_dir_kernel = njit(cache=False)(_null_dir_kernel)
else:
    _dir_point = directionality_coords_fast
    null_dir_kernel = _null_dir_kernel


def directionality(path):
    return float(directionality_coords_fast(np.ascontiguousarray(path, dtype=float)))


def draw_signs(rng, nsim=B_DRAWS, steps=None):
    """i.i.d. fair signs, with replacement; shape (nsim, steps)."""
    return rng.integers(0, 2, size=(nsim, steps)).astype(float) * 2.0 - 1.0


def null_values(c0, inc, signs, Mmat=None):
    """Null directionalities. Flip: s * inc; if Mmat given, re-project: Mmat @ (s * inc)."""
    flipped = signs[:, :, None] * np.asarray(inc, dtype=float)[None]
    if Mmat is not None:
        flipped = np.matmul(Mmat[None], flipped)
    return null_dir_kernel(np.ascontiguousarray(c0, dtype=float), np.ascontiguousarray(flipped))


def exceed_count(obs, null):
    """Literal #{null >= obs} (prereg section 3); no tolerance."""
    return int(np.sum(np.asarray(null) >= obs))


def path_test(path, signs, Mmat=None, inc=None):
    """Sign-flip test of dir(path). Increments default to diff(path)."""
    path = np.asarray(path, dtype=float)
    if inc is None:
        inc = np.diff(path, axis=0)
    obs = directionality(path)
    null = null_values(path[0], inc, signs, Mmat)
    if not (np.isfinite(obs) and np.all(np.isfinite(null)) and np.all(np.isfinite(path))):
        # invalid replication: never counted as a rejection or a non-rejection
        return dict(stat=obs, count=None, p=np.nan, null=null, valid=False)
    n = exceed_count(obs, null)
    return dict(stat=obs, count=n, p=(1 + n) / (1 + len(signs)), null=null, valid=True)


# ---------------------------------------------------------------- procedures
def run_procedures(S, X, signs):
    """Run P1, P1n, L, W, Mk on shares S (T x D) and covariate levels X (T x k),
    re-using the same sign draws. Returns dict name -> test dict, plus 'info'."""
    C = clr(S)
    sp = split_increments(C, X)
    c1 = C[0]
    res_path = c1 + sp["r"]
    res = {}
    res["P1"] = path_test(res_path, signs, Mmat=sp["Mmat"], inc=sp["E"])
    res["P1n"] = path_test(res_path, signs, inc=sp["E"])
    res["L"] = path_test(level_residual_path(C, X), signs)
    res["W"] = path_test(C, signs)
    mk_path = c1 + sp["m"]
    res["Mk"] = path_test(mk_path, signs)
    fit = sp["fit"]
    drift = fit.mean(axis=0)
    noise = np.sqrt(np.mean(np.sum((fit - drift) ** 2, axis=1)))
    info = dict(mk_drift_to_noise=float(np.linalg.norm(drift) / noise) if noise > 0 else np.nan,
                share_fit=float(np.sum(fit**2) / np.sum(sp["dC"] ** 2)),
                M1_retained=float(np.sum(sp["Mmat"]) / len(sp["dC"])),  # 1' M 1 / (T-1)
                E=sp["E"])
    res["info"] = info
    return res
