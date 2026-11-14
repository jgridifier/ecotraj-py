# Status log

Newest first. One entry per working session: date (ET), milestone, what passed, next step.

## 2026-11-14 — Parity rounds 3–4 (after M4)
- Directed-segment distance now follows De Cáceres et al. 2019 eqs. 8–10 (open-access manuscript found in the University of Lleida repository; penalty uses the nearer endpoint, as ecotraj does) [Claude].
- ecotraj's angle arithmetic reproduced from black-box probes (sum of base angles, upper-bound clamp, NaN for coincident states); numba kernels mirror it; collinear 1-D/2-D R fixtures added [Codex, interrupted by its usage limit; finished by the lab lead].
- Circular-sd ulp noise handled by a documented sd-only floor in the comparator (fixed a CI flake on Python 3.11).
- Randomized oracle: **1,254 / 1,280 exact** (was 1,086). SPD diagonal now computed like R (fixed the 3 raw-asymmetric SPD and one more). Remaining 26: non-metric 15, convergence row order 6, DS on one squared-Euclidean case 5.
- Both CLIs hit their usage limits at ~13:45 ET (Claude resets 3:40 PM, Codex 3:41 PM ET); the last 26 were not attempted further.
- Re-checked: scout targets and sign-flip p unchanged; M4 exceedance counts identical on 10 replications with the new kernels.
- Next: optional last 30 deviations; Extension 2 power curve for smaller swaps (would need its own prereg); daily check routine (parent).

## 2026-11-14 — M4 done (Extension 2 simulations, pre-registered run)
- Ran exactly the pre-registered configuration (seed 2026111401, B=199, R=2000/1000, 10 cells) at commit c297c65 (M2 + Addendum 1); 1,415 s on 8 cores; 0 invalid replications.
- P1 (increment split, re-projected sign flip): size 0.052 (T=40) and 0.053 (T=62), inside [0.030, 0.070]; power 0.997 at T=62 (0.937 at T=40). **Extension 2 PASSES.**
- L (old level procedure): size 0.001 / 0.001 and power 0.013 -> misses both criteria. McNemar (ii, T=62): 1,966 vs 0 discordant, so "P1 has more power than L" is permitted.
- Secondary nulls (P1): heavy tails 0.046 / 0.047, heteroskedastic 0.040 / 0.038 (in band); stationary levels 0.000 (conservative, allowed). P1-naive (the scout's exact m1b procedure): 0.068 / 0.060 in scenario i, reported without a verdict.
- Provenance: run 1 used a stale numba cache; fixed (53b8086) and re-ran the full pre-registered configuration with caches cleared: all CSVs byte-identical (results/PROVENANCE.md).
- Spot-check: three replications recomputed independently with the slow reference statistic; statistics and counts identical.
- Next: parity round 3 PR (DS rule found), results page.

## 2026-11-14 — M2 done (speed, Extension 1)
- numba kernels `ecotrajpy.fast` (directionality, parallel sign-flip null), equal to the reference to <= 1e-12, exact p on R fixtures; Claude reviewed Codex's kernels, Codex fixed all points.
- Benchmark (999 sign-flip draws, five-run medians): T=62 x 14 parts: R ecotraj naive loop 98.8 s*, Python reference 8.89 s, numba 1 thread 0.593 s, 8 threads 0.106 s. T=120 x 50 parts: R 614 s*, Python 74.1 s, numba 1 thread 3.21 s, 8 threads 0.424 s. (*R extrapolated from warm batches.) Cold JIT 1.16 s.
- Prereg Addendum 1 merged (PR #3) before the run.
- Next: parity round 3 (DS obtuse rule, degenerate triangles, convergence row order), then the pre-registered M4 run.

## 2026-11-14 — M1 done (port + parity)
- Clean-room port (Claude Code: core + parity fixes; Codex: R-oracle fixtures, parity suite, code review).
- Scout targets reproduced: A dir 0.577301, E4 dir 0.431413, A residual-increment dir 0.595092; sign-flip p exact (0.008 / 0.918 / 0.365 / 0.928) when replaying R's seed-20261112 draws.
- Randomized oracle: 1,086 / 1,280 checks exact at rtol 1e-8; 194 documented deviations (DS obtuse rule 107, non-metric degenerate triangles 72, convergence row order 6, 1-ulp sd 6, raw asymmetric SPD 3). See docs/PARITY.md.
- Extension 2 pre-registration merged (PR #1, 8362004) before any simulation.
- Next: merge M2 (numba kernels, benchmark), then run the pre-registered M4 simulations.

## 2026-11-14 — M0
- Scaffold: package skeleton, MIT license with credit to the ecotraj authors, CI (pytest on 3.11/3.13), Pages skeleton.
- Oracle available locally: R 4.5.3 + ecotraj 1.2.2 (conda env `rscout`).
- Checked again for an existing Python port: none on PyPI (`ecotraj`, `pyecotraj`, `ecotraj-py` all 404) or GitHub search.
- Next: M1 port + parity.
