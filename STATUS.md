# Status log

Newest first. One entry per working session: date (ET), milestone, what passed, next step.

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
