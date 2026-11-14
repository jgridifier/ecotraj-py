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
| `segment_distances` | 120 | 113 | 7 | DS 1, NONMETRIC 6 |
| `trajectory_angles` | 240 | 201 | 39 | NONMETRIC 33, ULP_SD 6 |
| `trajectory_convergence` | 120 | 112 | 8 | CONV_ROWORDER 6, NONMETRIC 2 |
| `trajectory_directionality` | 80 | 72 | 8 | NONMETRIC 8 |
| `trajectory_distances` | 480 | 468 | 12 | DS 4, SPD_ASYM 3, NONMETRIC 5 |
| `trajectory_internal_variation` | 40 | 40 | 0 | — |
| `trajectory_lengths` | 120 | 120 | 0 | — |
| `trajectory_metrics` | 40 | 29 | 11 | NONMETRIC 11 |
| `trajectory_speeds` | 40 | 40 | 0 | — |
| **Total** | **1280** | **1195** | **85** | |

"Exact match" includes the 38 cases where ecotraj itself errors and ecotrajpy raises.
Known deviations are listed one by one in `tests/known_deviations.json` and run as
non-strict xfail, so they stay visible in CI.

| Category | What differs |
|---|---|
| DS | Directed-segment distance now follows De Cáceres et al. (2019) eqs. 8-10 (selector R(x_i) > R(x_i+1); penalty = L + distance of the nearer end point to the other segment, black-box verified), matching ecotraj on 39/40 cases (3,624 segment pairs). The remaining case (squared-Euclidean `case40`) is non-metric: the DS pairs differ where the correction constant interacts with the penalty term. |
| NONMETRIC | Only on non-metric inputs (squared-Euclidean, Bray-Curtis). Reproduced: cosine clamped to [-1, 1] on corrected triplets (never NaN), NaN when d_ik = 0. Not reproduced: in near-degenerate (collinear after correction) triplets ecotraj returns 0, ~1e-6 deg (1-4 ulp from ±1) or 180 depending on its exact operation order; none of the ~60 operation orders tried reproduced more than about half of ~14,000 crafted probe triplets bit-for-bit; the rest differ by an ulp or two of the cosine. A sum-of-base-angles form explains ecotraj's NaN at d_ik = 0 and its precision on sharp turns but not the last bits. No Euclidean (or Aitchison/clr) input is affected. |
| ULP_SD | Circular sd of a single angle: 1.49e-08 vs 0 (last-bit rounding of rho). |
| CONV_ROWORDER | `trajectoryConvergence(type="multiple")` on rows not sorted by survey: ecotraj's result depends on row order in a way not identified (grouping rows with stride, or surveys in order of first appearance, each explain only 3 and 2 of the 6 cases); ours sorts by survey and equals ecotraj on sorted input. |
| SPD_ASYM | Raw asymmetric SPD (`symmetrization=NULL`) on 3 cases: some directed entries differ by up to 5% though every symmetrized SPD matches. Not investigated further in round 3. |


Everything the scout run and Extension 2 rely on (lengths, directionality on
Euclidean/clr data, the sign-flip null, the increment split) matches exactly.
