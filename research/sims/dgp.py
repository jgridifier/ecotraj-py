"""Data-generating process of research/prereg/EXT2_PREREG.md section 4."""
from __future__ import annotations

import numpy as np

from ecotrajpy.ext2 import clr, softmax

D = 14
K = 4
SIGMA = 0.4 / np.sqrt(D - 1)
TAU2 = 0.35 * SIGMA**2 / (0.65 * K)
A_DRIFT = 0.10
PHI = (0.90, None, 0.97, 0.5)   # Z2 is i.i.d.
SCENARIOS = ("i", "ii", "i-b", "i-c", "i-d")


def start_composition():
    s = 0.75 ** np.arange(D)
    return s / s.sum()


def ar1(rng, phi, T):
    x = np.empty(T)
    x[0] = rng.normal(0.0, 1.0 / np.sqrt(1.0 - phi**2))
    e = rng.standard_normal(T - 1)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + e[t - 1]
    return x


def raw_covariates(rng, T):
    z1 = ar1(rng, 0.90, T)
    z2 = rng.standard_normal(T)
    z3 = ar1(rng, 0.97, T)
    z4 = ar1(rng, 0.5, T) + 0.02 * np.arange(1, T + 1)
    return np.column_stack([z1, z2, z3, z4])


def centring(z):
    """P z for rows of z: subtract the row mean."""
    return z - z.mean(axis=-1, keepdims=True)


def planted_direction(Bmat):
    """Unit clr contrast: e1 - e2 projected off the constant vector and rowspace(B)."""
    v = np.zeros(D)
    v[0], v[1] = 1.0, -1.0
    basis = np.vstack([np.ones(D), Bmat])
    Q, _ = np.linalg.qr(basis.T)       # D x (k+1), spans 1 and the rows of B
    u = v - Q @ (Q.T @ v)
    return u / np.linalg.norm(u)


def simulate(scenario, T, rng):
    """One replication. Returns dict: S (T x D shares), X (T x k levels), plus truth
    (A, B, eta, u, delta, Z). Draw order: Z, B, residual noise."""
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    Z = raw_covariates(rng, T)
    dZ = np.diff(Z, axis=0)
    A = dZ / dZ.std(axis=0, ddof=1)          # no column-mean subtraction
    X = np.vstack([np.zeros(K), np.cumsum(A, axis=0)])
    Bmat = rng.normal(0.0, np.sqrt(TAU2), size=(K, D))
    Bmat -= Bmat.mean(axis=1, keepdims=True)

    C1 = clr(start_composition())
    if scenario == "i-b":
        rho = (SIGMA / np.sqrt(2.0)) * centring(rng.standard_normal((T, D)))
        eta = np.diff(rho, axis=0)
        C1 = C1 + rho[0]
    else:
        if scenario == "i-c":
            z = rng.standard_t(5, size=(T - 1, D)) * np.sqrt(3.0 / 5.0)
        else:
            z = rng.standard_normal((T - 1, D))
        eta = SIGMA * centring(z)
        if scenario == "i-d":
            z1 = Z[:T - 1, 0]
            q = (z1 - z1.mean()) / z1.std(ddof=1)
            h = np.exp(0.5 * q) / np.sqrt(np.mean(np.exp(q)))
            eta = eta * h[:, None]
    u = None
    delta = np.zeros(D)
    if scenario == "ii":
        u = planted_direction(Bmat)
        delta = A_DRIFT * u
    mkt = A @ Bmat
    C = np.vstack([C1, C1 + np.cumsum(mkt + eta + delta, axis=0)])
    return dict(S=softmax(C), X=X, A=A, B=Bmat, eta=eta, u=u, delta=delta, Z=Z,
                realized_share=float(np.sum(mkt**2) / np.sum((mkt + eta) ** 2)))
