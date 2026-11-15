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
| `trajectory_convergence` | 120 | 118 | 2 | NONMETRIC 2 |
| `trajectory_directionality` | 80 | 80 | 0 | — |
| `trajectory_distances` | 480 | 472 | 8 | DS 4, NONMETRIC 4 |
| `trajectory_internal_variation` | 40 | 40 | 0 | — |
| `trajectory_lengths` | 120 | 120 | 0 | — |
| `trajectory_metrics` | 40 | 40 | 0 | — |
| `trajectory_speeds` | 40 | 40 | 0 | — |
| **Total** | **1280** | **1260** | **20** | |

"Exact match" includes the 38 cases where ecotraj itself errors and ecotrajpy raises.
Known deviations are listed one by one in `tests/known_deviations.json` and run as
non-strict xfail, so they stay visible in CI.

| Category | What differs |
|---|---|
| DS (5) | Directed-segment distance now follows De Cáceres et al. (2019) eqs. 8–10 (open-access manuscript, University of Lleida repository), with the penalty using the nearer endpoint as ecotraj does; 102 of 107 fixed. The 5 left are all one squared-Euclidean case (case40: Dseg and four DSPD variants). Round 6 isolated them to 3 segment pairs (reproduced in R with 4-point sub-matrices) and then to a rounding effect: on 6,000 random squared-Euclidean 4-point pairs, 51 differ from ecotraj, and every one involves a corrected triplet that is exactly collinear in exact arithmetic (a^2 - t^2 about +-1e-14, either sign). There ecotraj's penalised term is sometimes `L + 0` and sometimes `L + d(endpoint, endpoint)` (e.g. L1 + d12), which looks like the sign of that noise, i.e. the same floating-point-order class as NONMETRIC; no deterministic rule found (5,407 passing pairs are also degenerate), so it is left as is. |
| NONMETRIC (15) | Only non-metric inputs (squared-Euclidean, Bray-Curtis). Round 4 reproduced ecotraj's angle arithmetic from black-box probes (project the middle point onto the outer side, sum the two base angles with acos, clamp only the upper cosine bound; NaN for coincident states), which fixed 56 of 72; the rest are segment/trajectory distances and convergence on corrected triangles. |
| (sd floor) | Circular sd of a single angle is 0 or 1.49e-08 depending on last-bit rounding (platform dependent). The comparator treats sd values both ≤ sqrt(2·eps) as equal; no xfail needed. |
| (CONV_ROWORDER, fixed) | `trajectoryConvergence(type="multiple")` uses the times in the input-row order of the first row's site as its Mann-Kendall axis, matching observations by exact time value. It does not sort by time. The survey-based rule holds when surveys and times have a shared one-to-one mapping across sites (including default times). Independent round-6 review matched 240/240 shuffled inputs (2–5 sites, 3–8 surveys, 2D/5D, Euclidean and non-Euclidean), plus 300 stress cases with site-specific survey/time mappings and repeated times. Synchrony requires equal sets of times, allowing unequal multiplicities; each time group's dispersion scales by its row count. Permanent R-oracle coverage is in `tests/test_convergence_roworder.py`. Ordinary survey-sorted inputs are unchanged; all 6 original cases remain exact. |
| (SPD, fixed) | The raw asymmetric SPD differences were rounding on the diagonal computation, now computed like R; all SPD checks match. |


Everything the scout run and Extension 2 rely on (lengths, directionality on
Euclidean/clr data, the sign-flip null, the increment split) matches exactly.

### Round 3–4 additions (2026-11-14)
- Exactly collinear paths (1-D and collinear 2-D): new R-oracle fixture `tests/fixtures/collinear_cases.json`
  (`oracle/collinear_fixtures.R`), matched by the new angle arithmetic.
- The numba kernels (`ecotrajpy.fast`) mirror the reference arithmetic; M4's pre-registered results were
  re-checked with the new kernels on 10 replications (all five procedures' exceedance counts identical), and
  the scout targets and sign-flip p-values are unchanged.


## Round 6b: NONMETRIC element-level evidence (15 ids)

All differences are on squared-Euclidean cases, where the local additive correction makes violated triplets exactly collinear (a + b = c after correction, so the altitude h = sqrt(a^2 - t^2) is sqrt of +-1e-15 rounding noise).

* **Tiny (1e-9 to 2e-7) PPA differences** on 4-point random squared-Euclidean probes (e.g. ours 2.19738780 vs R 2.19738774, 5.6e-8; 2.12116717 vs 2.12116718): h = sqrt(noise) is about 1e-8 in R and differs in the last bits. R quirk (floating-point noise on exactly-degenerate corrected triplets).
* **NaN vs finite** (R NaN where ours is finite, and vice versa): R's sqrt of a slightly *negative* a^2 - t^2 gives NaN (no clamp) and Lehmer(0, 0) is 0/0; the sign of the noise decides. Seen in case04 PPA (3 pairs, ours 3.96/5.26/3.89 vs R NaN), case28 PPA (9,10), case40 PPA (7 pairs, e.g. ours 3.006 vs R NaN), case28/case40 `trajectory_angles` mean (ours 90 / 89.99999983 vs R NaN: a single angle exactly at the +-1 cosine bound).
* **Large PPA differences** (e.g. case24 (2,8): 179.44 vs 163.54; case40 (19,22): 6.28 vs 0.91): in 4-point probes with R, a violated triplet (d_endpoint1 + d_endpoint2 < L) gives for the point-to-segment distance sometimes the *uncorrected* nearer-endpoint distance and sometimes the *corrected* one (r + violation) for near-identical parameters (v/L = 0.34 gives both). No deterministic formula was found. Dropping the DS "longer than the two other sides" override from PPA agrees with R on 2,062 of 3,000 random 4-point probes (vs 1,373 now) but regresses two currently exact oracle cases (case08, case32 PPA), so it was not adopted.
* **Convergence pairwise.asymmetric** (case04 (2,1): tau -0.138 vs 0.0; case40 (0,1): 0.894 vs 1.0, (1,3): 0.733 vs 0.6): tau/p are of the per-survey vector of mean squared distances; the vector involves the corrected distances, whose ties (exact equality vs 1-ulp difference) change Kendall's tau-b ties. Rounding-tie class; not reproduced.
* **SPD (case40, mean/max/min/NULL)**: e.g. (0,1) 2.46789 vs 2.47296, (1,3) 1.96847 vs 2.08521; the same point-to-segment degenerate-triplet quantity as PPA/DS.
