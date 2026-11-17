# P2 pre-registration: Extension 2 power curve for smaller swaps and fewer parts

**Status:** pre-registered. Written and merged on 2026-11-17 (ET) **before any P2 simulation was run**.
Requested by Jared on research-lab issue #50 ("All 3 extensions", 2026-11-17). Any change after merge is a dated
addendum below; the original text is never edited. Self-written by the lab agent; not independently reviewed before merge.

## What is unchanged from EXT2_PREREG.md

Statistic, procedure P1 (increment split with re-projected sign flip), P1-naive, L, W, Mk, B = 199 sign draws,
reject iff #{null ≥ obs} ≤ 9, the DGP of section 4 (k = 4 covariates, loadings, noise, start composition rule
0.75^j, σ = 0.4/√(D−1), τ² = 0.35σ²/(0.65k)), and the code (`sims/dgp.py`, `ecotrajpy.ext2`, numba kernels at the
current main commit). The only things that vary are the drift a and the number of parts D, set through module constants
by a thin runner `sims/run_p2.py`.

## Cells

- Scenario ii (planted sticky swap) with a ∈ {0.025, 0.05, 0.075, 0.10} × D ∈ {14, 6} × T ∈ {40, 62}: 16 cells.
- Scenario i (null) at D = 6 × T ∈ {40, 62}: 2 cells (D = 14 size is already known from M4).
- R = 1,000 replications per cell. Master seed **2026111701**; `SeedSequence(2026111701).spawn(18)`, one child per
  cell in the order: i/D6/T40, i/D6/T62, then ii in the order D (14, 6) × a (0.025, 0.05, 0.075, 0.10) × T (40, 62).

## Outputs and criteria (decided now)

- Rejection rate of every procedure per cell with Clopper–Pearson 95% intervals; retained-drift fraction and net
  displacement along u for scenario ii.
- **Size check (the only pass/fail):** P1 at D = 6 in scenario i within [0.030, 0.070] at T = 40 and T = 62.
- **Power curve (descriptive):** P1 power against a, by D and T. The minimum detectable swap is the smallest a in the grid
  with P1 power ≥ 0.80 at T = 62, reported per D ("none in grid" if no a qualifies).
- No forecast metric. No tuning after results. A coding error found after the run is fixed and the full run repeated,
  with both results reported in an addendum.
