"""Reproducible compositional panels; no generated data are committed.

A multiplicative Dirichlet random walk starts at the uniform composition.
At each survey it is perturbed by an independent Dirichlet(100,...,100)
vector, multiplied componentwise and closed to sum one. Return its clr.
"""
import numpy as np

SEED = 20261114


def make_panel(T, D, seed=SEED):
    rng = np.random.default_rng(seed)
    log_steps = np.log(rng.dirichlet(np.full(D, 100.0), size=T - 1))
    log_parts = np.vstack([np.zeros(D), np.cumsum(log_steps, axis=0)])
    # Subtracting the row mean gives clr(closure(exp(log_parts))) without
    # underflow from explicitly multiplying many small compositions.
    return log_parts - log_parts.mean(axis=1, keepdims=True)


def make_signs(T, nsim=999, seed=SEED + 1):
    return np.random.default_rng(seed).choice([-1., 1.], size=(nsim, T - 1))
