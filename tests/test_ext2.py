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


# ---------------------------------------------------------------- literal count / validity
def test_exceed_count_is_literal():
    obs = 0.5
    null = np.array([0.5, 0.5 - 1e-13, 0.5 + 1e-13, 0.4])
    assert ext2.exceed_count(obs, null) == 2          # no tolerance: 0.5 - 1e-13 is not counted


def test_nonfinite_stat_is_invalid_not_rejection():
    path = np.vstack([np.zeros(3), np.cumsum(np.random.default_rng(0).standard_normal((6, 3)), axis=0)])
    signs = ext2.draw_signs(np.random.default_rng(1), 20, 6)
    ok = ext2.path_test(path, signs)
    assert ok["valid"] and 0 <= ok["count"] <= 20
    bad = path.copy()
    bad[3, 0] = np.nan
    t = ext2.path_test(bad, signs)
    assert t["valid"] is False and t["count"] is None and np.isnan(t["p"])
    flat = ext2.path_test(np.zeros((7, 3)), signs)      # every triple degenerate -> non-finite statistic
    assert flat["valid"] == bool(np.isfinite(flat["stat"])) and (flat["valid"] or flat["count"] is None)


# ---------------------------------------------------------------- summarize validation
import json  # noqa: E402

import summarize as sm  # noqa: E402

_PROCS = sm.PROCS


def _write_cell(rd, scn, T, R, rej_frac, rng, reps=None, bad_count=False):
    cols = ["rep"] + [f"{p}_count" for p in _PROCS] + [f"{p}_p" for p in _PROCS] \
        + [f"{p}_stat" for p in _PROCS] + [f"{p}_valid" for p in _PROCS] \
        + ["realized_share", "mk_drift_to_noise", "M1_retained", "net_disp_u"]
    rows = []
    for r in range(R):
        cnt = [int(rng.integers(0, 9)) if rng.random() < rej_frac else int(rng.integers(10, 200))
               for _ in _PROCS]
        if bad_count:
            cnt[0] = 250
        row = [r if reps is None else reps[r]] + cnt + [(1 + c) / 200 for c in cnt] \
            + [0.1] * 5 + [1] * 5 + [0.35, 1.0, 0.9 if scn == "ii" else np.nan,
                                      0.1 if scn == "ii" else np.nan]
        rows.append(row)
    np.savetxt(rd / f"{scn}_T{T}.csv", np.array(rows), delimiter=",", header=",".join(cols),
               comments="", fmt="%.10g")


def _synthetic(rd, rej_i=0.05, rej_ii=0.9, small=None, skip=None, meta_over=None):
    rng = np.random.default_rng(0)
    meta = dict(master_seed=2026111401, B=199, preregistered_configuration=True, cells={})
    for scn, T in sm.ORDER:
        R = sm.expected_R(scn) if small is None else small
        meta["cells"][f"{scn}_T{T}"] = dict(R=R)
        if (scn, T) == skip:
            continue
        _write_cell(rd, scn, T, R, rej_ii if scn == "ii" else rej_i, rng)
    meta.update(meta_over or {})
    (rd / "run_meta.json").write_text(json.dumps(meta))


def _run(tmp_path):
    out = sm.summarize(tmp_path, tmp_path / "s.json", tmp_path / "s.md")
    return out, (tmp_path / "s.md").read_text()


def test_summary_valid_run_gets_verdict(tmp_path):
    _synthetic(tmp_path)
    out, md = _run(tmp_path)
    assert out["binding"] and "DEVELOPMENT RUN" not in md
    assert "Criterion 1" in md and "SECONDARY MISSES" not in md


def test_summary_dev_run_banner_no_verdict(tmp_path):
    _synthetic(tmp_path, small=3, meta_over=dict(preregistered_configuration=False, master_seed=1))
    out, md = _run(tmp_path)           # also exercises tiny CSVs
    assert not out["binding"] and "DEVELOPMENT RUN - NOT A PREREGISTERED RESULT" in md
    assert "Verdict" not in md and out["criteria"].startswith("withheld")


def test_summary_one_row_csv(tmp_path):
    _synthetic(tmp_path, small=1, meta_over=dict(preregistered_configuration=False))
    out, _ = _run(tmp_path)
    assert out["cells"]["i/40"]["R"] == 1


def test_summary_missing_cell_and_bad_counts(tmp_path):
    _synthetic(tmp_path, skip=("ii", 40))
    out, md = _run(tmp_path)
    assert not out["binding"] and any("ii/40" in x for x in out["integrity_problems"])
    assert "Verdict" not in md


def test_summary_integrity_checks(tmp_path):
    _synthetic(tmp_path)
    rng = np.random.default_rng(1)
    _write_cell(tmp_path, "i", 40, 2000, 0.05, rng, bad_count=True)
    _write_cell(tmp_path, "i", 62, 2000, 0.05, rng, reps=[0] * 2000)
    out, _ = _run(tmp_path)
    probs = " ".join(out["integrity_problems"])
    assert "not integers in [0, 199]" in probs and "not unique" in probs and not out["binding"]


def test_summary_invalid_reps_excluded_from_rates(tmp_path):
    d = {"P1_count": np.array([0.0, np.nan, 50.0, 3.0]), "P1_valid": np.array([1, 0, 1, 1])}
    r = sm.rate(d, "P1")
    assert (r["rejections"], r["R"], r["invalid"]) == (2, 3, 1)


def test_summary_reports_secondary_misses_and_scout_sentence(tmp_path):
    _synthetic(tmp_path, rej_i=0.05, rej_ii=0.95)
    rng = np.random.default_rng(2)
    _write_cell(tmp_path, "i-c", 40, 1000, 0.2, rng)
    out, md = _run(tmp_path)
    assert out["binding"] and "SECONDARY MISSES: i-c/40" in md
    assert "scout's diagnosis was not reproduced" in md or not out["criteria"]["L"]["passes_both"]
