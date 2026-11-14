# Roadmap

| Milestone | Scope | Done when |
|---|---|---|
| **M0 Scaffold** | Repo, MIT license, credit, CI (pytest), Pages skeleton, ROADMAP/STATUS | CI green on `main`, Pages live |
| **M1 Port + parity** | Clean-room port of `defineTrajectories`, `trajectoryLengths`, `trajectoryAngles`, `trajectoryDirectionality` (+ random-order permutation test), `trajectoryConvergence` (3 types), `segmentDistances`, `trajectoryDistances` (Hausdorff, SPD, DSPD), clr, increment sign-flip null, increment market/residual split | Randomized parity vs ecotraj 1.2.2 fixtures; scout targets A dir 0.577301, E4 dir 0.431413, A residual-increment dir 0.595092 (seed 20261112, 999 draws) reproduced; sign-flip p reproduced exactly when fed R's sign draws |
| **M2 Speed (Extension 1)** | Vectorized / numba kernels (small Rust core only if needed) for directionality, sign-flip null, DSPD | Wall-time benchmark vs ecotraj on a stated larger synthetic panel; results identical to M1 to 1e-10 |
| **M3 Extension 2 pre-registration** | Increment-level, clr-valid market/residual split with a sign-flip test; scenarios, alpha band, power bar, T=40/62, rejection rule for the level-residualization procedure | `prereg/EXT2_PREREG.md` committed and merged **before** any Extension 2 simulation is run |
| **M4 Extension 2 simulations + results** | Run the pre-registered simulations exactly as written | Results page with size (scenario i), power (scenario ii, T=62 and T=40), and the level procedure's outcome; pass/fail reported against the prereg. No forecast metrics |

Out of scope: promotion into analysis-skills (needs a separate dated approval), real private data.
