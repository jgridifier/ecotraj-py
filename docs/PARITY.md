# M1 parity against ecotraj 1.2.2

Oracle: R 4.5.3 + ecotraj 1.2.2 (conda env `rscout`), called as a black box.

## 1. Scout targets (2026-11-12 run, seed 20261112, 999 draws)

Panels are private-derived and not in this repo; checked locally with
`scripts/parity_scout_panels.py --data-dir ... --signs-json ...`.

| Quantity | ecotraj 1.2.2 | ecotrajpy | Match |
|---|---:|---:|---|
| Panel A (T=40, 14 parts) directionality | 0.577301 | 0.577301 | yes (diff < 5e-7, i.e. rounding of the 6-digit target) |
| Panel E4 (T=62, 4 parts) directionality | 0.431413 | 0.431413 | yes |
| A residual-increment directionality | 0.595092 | 0.595092 | yes |
| A market-increment directionality | 0.377861 | 0.377861 | yes |
| E4 residual / market directionality | 0.513292 / 0.329615 | 0.513292 / 0.329615 | yes |
| A market share of step energy | 0.161606 | 0.161606 | yes |
| A residual sign-flip p (R's 999 sign draws replayed) | 0.008 | 0.008 | exact; max null diff 7.6e-15 |
| A market sign-flip p | 0.918 | 0.918 | exact |
| E4 residual / market sign-flip p | 0.365 / 0.928 | 0.365 / 0.928 | exact |

R's own draws were recorded by rerunning the scout script with the same seed
(`oracle/scout_record_signs.R`); it reproduced the published p-values exactly. With
numpy's RNG the p-values are draw-dependent (the null distribution matches, the
individual draws do not), which is expected.

## 2. Randomized oracle suite (40 synthetic cases, 1,280 checks, seed 20261114)

Euclidean 2/5/14-D, squared-Euclidean and Bray-Curtis inputs, 1-5 trajectories,
3-12 surveys, unequal lengths, shuffled rows, irregular times. Tolerance rtol 1e-8,
atol 1e-10. Plus 5 sign-flip fixtures (T = 20/40/62, 199 R draws each): all exact.

| Function (ecotraj name) | Oracle checks | Exact match (rtol 1e-8) | Known deviation | Categories |
|---|---:|---:|---:|---|
| `segment_distances` | 120 | 113 | 7 | NONMETRIC 6, DS 1 |
| `trajectory_angles` | 240 | 237 | 3 | NONMETRIC 3 |
| `trajectory_convergence` | 120 | 112 | 8 | CONV_ROWORDER 6, NONMETRIC 2 |
| `trajectory_directionality` | 80 | 80 | 0 | — |
| `trajectory_distances` | 480 | 468 | 12 | SPD_ASYM 3, NONMETRIC 5, DS 4 |
| `trajectory_internal_variation` | 40 | 40 | 0 | — |
| `trajectory_lengths` | 120 | 120 | 0 | — |
| `trajectory_metrics` | 40 | 40 | 0 | — |
| `trajectory_speeds` | 40 | 40 | 0 | — |
| **Total** | **1280** | **1250** | **30** | |

"Exact match" includes the 38 cases where ecotraj itself errors and ecotrajpy raises.
Known deviations are listed one by one in `tests/known_deviations.json` and run as
non-strict xfail, so they stay visible in CI.

| Category | What differs |
|---|---|
| DS (5) | Directed-segment distance now follows De Cáceres et al. (2019) eqs. 8–10 (open-access manuscript, University of Lleida repository), with the penalty using the nearer endpoint as ecotraj does; 102 of 107 fixed. The 5 left are all one squared-Euclidean case (case40: Dseg and four DSPD variants). |
| NONMETRIC (16) | Only non-metric inputs (squared-Euclidean, Bray-Curtis). Round 4 reproduced ecotraj's angle arithmetic from black-box probes (project the middle point onto the outer side, sum the two base angles with acos, clamp only the upper cosine bound; NaN for coincident states), which fixed 56 of 72; the rest are segment/trajectory distances and convergence on corrected triangles. |
| (sd floor) | Circular sd of a single angle is 0 or 1.49e-08 depending on last-bit rounding (platform dependent). The comparator treats sd values both ≤ sqrt(2·eps) as equal; no xfail needed. |
| CONV_ROWORDER (6) | `trajectoryConvergence(type="multiple")` on rows not sorted by survey: ecotraj's tau depends on input row order (random row permutations change it); two candidate rules each explain only some cases. Ours sorts by survey and equals ecotraj on sorted input. |
| SPD_ASYM (3) | Raw asymmetric SPD (`symmetrization=NULL`) on 3 cases: some directed entries differ by up to 5%, though every symmetrized SPD matches. |


Everything the scout run and Extension 2 rely on (lengths, directionality on
Euclidean/clr data, the sign-flip null, the increment split) matches exactly.

### Round 3–4 additions (2026-11-14)
- Exactly collinear paths (1-D and collinear 2-D): new R-oracle fixture `tests/fixtures/collinear_cases.json`
  (`oracle/collinear_fixtures.R`), matched by the new angle arithmetic.
- The numba kernels (`ecotrajpy.fast`) mirror the reference arithmetic; M4's pre-registered results were
  re-checked with the new kernels on 10 replications (all five procedures' exceedance counts identical), and
  the scout targets and sign-flip p-values are unchanged.
