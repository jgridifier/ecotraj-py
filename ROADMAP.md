# Roadmap

| Milestone | Scope | Done when | Status |
|---|---|---|---|
| **M0 Scaffold** | Repo, MIT license, credit, CI (pytest), Pages skeleton, ROADMAP/STATUS | CI green on `main`, Pages live | ✅ 2026-11-14 |
| **M1 Port + parity** | Clean-room port of `defineTrajectories`, `trajectoryLengths`, `trajectoryAngles`, `trajectoryDirectionality` (+ random-order permutation test), `trajectoryConvergence` (3 types), `segmentDistances`, `trajectoryDistances` (Hausdorff, SPD, DSPD), clr, increment sign-flip null, increment market/residual split | Randomized parity vs ecotraj 1.2.2 fixtures; scout targets A dir 0.577301, E4 dir 0.431413, A residual-increment dir 0.595092 (seed 20261112, 999 draws) reproduced; sign-flip p reproduced exactly when fed R's sign draws | ✅ 2026-11-14 (PR #2; parity rounds 3–4 PR #7, SPD PR #9: 1,254/1,280; round 6 2026-11-15: 1,260/1,280 exact) |
| **M2 Speed (Extension 1)** | Vectorized / numba kernels (small Rust core only if needed) for directionality, sign-flip null, DSPD | Wall-time benchmark vs ecotraj on a stated larger synthetic panel; results identical to M1 to 1e-10 | ✅ 2026-11-14 (PR #4) |
| **M3 Extension 2 pre-registration** | Increment-level, clr-valid market/residual split with a sign-flip test; scenarios, alpha band, power bar, T=40/62, rejection rule for the level-residualization procedure | `prereg/EXT2_PREREG.md` committed and merged **before** any Extension 2 simulation is run | ✅ 2026-11-14 (PR #1, Addendum 1 PR #3, both before the run) |
| **M4 Extension 2 simulations + results** | Run the pre-registered simulations exactly as written | Results page with size (scenario i), power (scenario ii, T=62 and T=40), and the level procedure's outcome; pass/fail reported against the prereg. No forecast metrics | ✅ 2026-11-14 (PR #5): Extension 2 passes; L misses |

Out of scope: promotion into analysis-skills (needs a separate dated approval), real private data.

## Next (proposed, not started)

| Item | Scope | Note |
|---|---|---|
| P1 | Remaining 20 parity deviations (non-metric 15, DS on one squared-Euclidean case 5); convergence row order and raw asymmetric SPD fixed | In progress (1,260/1,280 exact, 2026-11-15); all 20 are on exactly-degenerate corrected triplets and look like R floating-point noise (documented in docs/PARITY.md); none affects Euclidean/clr directionality, the sign-flip null or the split |
| P2 | Extension 2 power curve for smaller swaps (a = 0.025–0.10) and fewer parts | Done 2026-11-17: size PASS (D = 6: 0.051 / 0.042); minimum detectable swap at T = 62: 0.075 (D = 14), 0.10 (D = 6). `results/P2_RESULTS.md` |
| P3 | Scout plan 3: rerun the increment split unchanged on activity-like count panels | Plans only; data must be public or local-only |
| P4 | Package release (PyPI) | Only if the lab wants it |
