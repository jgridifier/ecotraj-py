import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sims"))
import dgp  # noqa: E402
from ecotrajpy import ext2  # noqa: E402
from ecotrajpy.nulls import signflip_test  # noqa: E402
from ecotrajpy.split import increment_split  # noqa: E402


def sim(scn="i", T=40, seed=5):
    return dgp.simulate(scn, T, np.random.default_rng(seed))


def test_zero_row_sums():
    s = sim()
    C = ext2.clr(s["S"])
    sp = ext2.split_increments(C, s["X"])
    for k in ("fit", "E", "m", "r", "dC"):
        assert np.abs(sp[k].sum(axis=1)).max() < 1e-10


def test_components_reconstruct_shares():
    s = sim()
    C = ext2.clr(s["S"])
    sp = ext2.split_increments(C, s["X"])
    S = s["S"]
    rec = ext2.softmax(ext2.clr(S[0]) + 0) # S_1
    for t in range(len(S)):
        # perturbation: closure of elementwise product
        prod = S[0] * ext2.softmax(sp["m"][t]) * ext2.softmax(sp["r"][t])
        np.testing.assert_allclose(prod / prod.sum(), S[t], atol=1e-10)
    assert rec.shape == S[0].shape


def _all_stats(S, X, signs):
    out = ext2.run_procedures(S, X, signs)
    return {k: (v["stat"], v["null"]) for k, v in out.items() if k != "info"}


def test_invariance_permutation_and_ilr():
    s = sim("ii", 30, 11)
    S, X = s["S"], s["X"]
    rng = np.random.default_rng(0)
    signs = ext2.draw_signs(rng, 25, 29)
    base = _all_stats(S, X, signs)
    perm = rng.permutation(S.shape[1])
    pm = _all_stats(S[:, perm], X, signs)
    # ilr: run on coordinates C V (statistics are functions of clr coords only through
    # Euclidean geometry, so feed ilr coordinates through the same code paths)
    V = ext2.ilr_basis(S.shape[1], rng)
    C = ext2.clr(S)
    sp = ext2.split_increments(C, X)
    spi = ext2.split_increments(C @ V, X)
    np.testing.assert_allclose(spi["E"], sp["E"] @ V, atol=1e-10)
    c1 = C[0]
    pairs = [
        (c1 + sp["r"], sp["E"], sp["Mmat"], (C[0] @ V) + spi["r"], spi["E"]),
        (c1 + sp["r"], sp["E"], None, (C[0] @ V) + spi["r"], spi["E"]),
        (C, np.diff(C, axis=0), None, C @ V, np.diff(C @ V, axis=0)),
        (c1 + sp["m"], sp["fit"], None, (C[0] @ V) + spi["m"], spi["fit"]),
        (ext2.level_residual_path(C, X), None, None, ext2.level_residual_path(C @ V, X), None),
    ]
    for p, inc, M, pi, inci in pairs:
        a = ext2.path_test(p, signs, Mmat=M, inc=inc)
        b = ext2.path_test(pi, signs, Mmat=M, inc=inci)
        assert abs(a["stat"] - b["stat"]) < 1e-10
        np.testing.assert_allclose(a["null"], b["null"], atol=1e-10)
    for k in base:
        assert abs(base[k][0] - pm[k][0]) < 1e-10
        np.testing.assert_allclose(base[k][1], pm[k][1], atol=1e-10)


def test_p1_naive_equals_scout_m1b():
    s = sim("ii", 40, 3)
    C = ext2.clr(s["S"])
    signs = ext2.draw_signs(np.random.default_rng(2), 60, 39)
    out = ext2.run_procedures(s["S"], s["X"], signs)
    ref = signflip_test(increment_split(C, s["X"])["residual"], signs=signs)
    assert abs(out["P1n"]["stat"] - ref["stat"]) < 1e-12
    np.testing.assert_allclose(out["P1n"]["null"], ref["null"], atol=1e-10)
    assert out["P1n"]["p"] == pytest.approx(ref["p_value"])


def test_exact_null_with_known_B_toy():
    # known B, no estimation: residual increments are iid symmetric -> exact test.
    T, Dd, N = 6, 3, 1500
    rng = np.random.default_rng(42)
    allsigns = np.array([[1.0 if (i >> j) & 1 else -1.0 for j in range(T - 1)]
                         for i in range(2 ** (T - 1))])
    rej = 0
    for _ in range(N):
        path = np.vstack([np.zeros(Dd), np.cumsum(rng.standard_normal((T - 1, Dd)), axis=0)])
        t = ext2.path_test(path, allsigns)
        rej += t["p"] <= 3 / 32 + 1e-12
    rate = rej / N
    # s and -s give the same statistic (reflected path), so p takes even multiples of 1/32;
    # p <= 3/32 <=> the observed is the (weakly) top of 16 equivalence classes: size 1/16.
    assert rate <= 3 / 32
    assert abs(rate - 1 / 16) < 0.025         # SE ~ 0.006


def test_dgp_energy_share_and_geometry():
    shares = [dgp.simulate("i", 62, np.random.default_rng(1000 + i))["realized_share"]
              for i in range(400)]
    assert abs(np.mean(shares) - 0.35) < 0.03
    s = dgp.simulate("ii", 40, np.random.default_rng(7))
    u = s["u"]
    assert abs(np.linalg.norm(u) - 1) < 1e-12
    assert abs(u.sum()) < 1e-12
    assert np.abs(s["B"] @ u).max() < 1e-12
    assert np.abs(s["B"].sum(axis=1)).max() < 1e-12


def test_dgp_ar_stationary_init_and_A_not_centred():
    T = 6
    x0 = np.array([dgp.ar1(np.random.default_rng(i), 0.97, T)[0] for i in range(4000)])
    assert abs(x0.var() - 1 / (1 - 0.97**2)) / (1 / (1 - 0.97**2)) < 0.08
    s = dgp.simulate("i", 62, np.random.default_rng(9))
    A = s["A"]
    # column sd (ddof=1) is 1, but columns are not mean-centred
    # scaled by the sd of dZ (ddof=1) only; the trend column keeps a positive mean
    assert np.abs(A.mean(axis=0)).min() > 1e-8
    m4 = np.mean([dgp.simulate("i", 62, np.random.default_rng(100 + i))["A"][:, 3].mean()
                  for i in range(300)])
    assert 0.0173 - 0.008 < m4 < 0.0173 + 0.008   # 0.02 / sd(dZ4) ~ 0.02/1.155
    assert np.abs(A.mean(axis=0)).max() > 1e-8
    np.testing.assert_allclose(np.diff(s["X"], axis=0), A, atol=1e-12)
    assert np.all(s["X"][0] == 0)
