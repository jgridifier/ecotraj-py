# AGENTS.md — working on ecotrajpy

This file is for coding agents (and humans) picking up the repo.

## Purpose

`ecotrajpy` is a **clean-room Python port** of Ecological Trajectory Analysis
(ETA), with optional Numba kernels (Extension 1) and a clr-valid increment
market/residual split plus sign-flip tests (Extension 2). The R package
**ecotraj 1.2.2 is the oracle**. No ecotraj source code is in this tree, and
none should be read, printed, or browsed (see the clean-room rule below).

Live pages: https://jgridifier.github.io/ecotraj-py/
Results: https://jgridifier.github.io/ecotraj-py/results.html
Repo: https://github.com/jgridifier/ecotraj-py

The package is not on PyPI yet. Do not publish it, create a PyPI account, or
add a publish workflow unless the owner has provided a token and an explicit go.

## Repo map

```
src/ecotrajpy/     Installable package (core, metrics, comparison, compositional,
                   nulls, split, fast, ext2)
tests/             Pytest suite, including R-oracle fixtures and known deviations
docs/              GitHub Pages site (served from main, /docs) plus API and parity docs
docs/assets/       Logo (logo.svg, logo.png) and favicons
oracle/            R scripts that generate black-box fixtures (do not open ecotraj sources)
scripts/           Local-only helpers (e.g. scout-panel parity; data is not in the repo)
bench/             Speed comparison vs R (Extension 1)
research/          Frozen pre-registrations, simulation runners, and published results
  prereg/          EXT2 and P2 pre-registrations (scientific text is frozen)
  sims/            DGP and runners; defaults write under research/results/
  results/         Published CSVs, summaries, provenance
```

Top-level docs worth reading first: `README.md`, `docs/api.md`, `docs/PARITY.md`,
`docs/API_SPEC.md` (clean-room rule and intended API), `ROADMAP.md`, `STATUS.md`,
`research/README.md`.

## Setup, tests, and parity

Python 3.10+ (CI runs 3.11 and 3.13). From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,fast]"
pytest -q -m "not slow"
```

That is the same command CI runs (`.github/workflows/ci.yml`). The `slow`
marker is reserved for long simulations; do not drop `-m "not slow"` in CI.

Parity suite (subset of the tests above):

```bash
pytest -q tests/test_parity.py tests/test_core.py tests/test_collinear.py tests/test_convergence_roworder.py tests/test_fast.py
```

Regenerating R-oracle fixtures (optional, needs a local R + ecotraj 1.2.2;
historically the `rscout` conda env). The scripts live in `oracle/` and write
under `tests/fixtures/`. Do not regenerate fixtures to "make a test pass"
without recording why the oracle output changed.

Scout-panel parity (private data, not in the repo):

```bash
python scripts/parity_scout_panels.py --data-dir <local> --signs-json <local>
```

Extension 2 unit tests (use the DGP in `research/sims/`):

```bash
pytest -q tests/test_ext2.py
```

## Invariants

1. **R is the oracle.** ecotraj 1.2.2, called as a black box, is the source of
   truth for the ported metrics. Python must not be "fixed" to look nicer if
   that breaks agreement with R on Euclidean / documented cases.
2. **Parity tolerances.** The randomized suite uses rtol 1e-8, atol 1e-10.
   Numba kernels must match the reference to ≤ 1e-12, and sign-flip p-values
   must be exact when R's sign draws are replayed. Do not loosen tolerances
   to hide a disagreement.
3. **Do not change numerics without a parity run.** Any change to distances,
   angles, directionality, convergence, segment/trajectory distances, clr,
   the increment split, or the sign-flip null requires `pytest -q -m "not slow"`
   (and `tests/test_parity.py` in particular) before you conclude it is safe.
4. **The 20 documented deviations stay documented.** 1,260 / 1,280 checks
   match. The remaining 20 are R floating-point quirks on non-metric inputs,
   listed in `tests/known_deviations.json` and `docs/PARITY.md`. Do not delete
   xfails to green-wash the suite, and do not change the comparator to swallow
   them.
5. **`research/` is frozen pre-registered material.** Do not edit a
   pre-registration's hypotheses, seeds, margins, or pass/fail rules after
   its run. Path-only updates that follow a directory move are layout, not a
   new study. New studies get a new pre-registration, merged before any run.
6. **No neural nets.** This package is classical trajectory metrics
   (distances, angles, permutation tests). Do not add learned models.
7. **Clean-room rule.** Do not read, print, or browse ecotraj source code
   (no `print(ecotraj::fn)`, no `body()` / `deparse()` of package functions,
   no files inside the installed package library, no browsing
   `github.com/emf-creaf/ecotraj` `R/` or `src/`). Allowed: the published
   papers, help pages / vignettes, and black-box calls that observe outputs.
   Details: `docs/API_SPEC.md`.
8. **Synthetic data only in the repo.** Do not commit private or real-panel
   data. `local_data/` is gitignored.
9. **License.** ecotraj is GPL (≥ 2); this package is MIT on the clean-room
   claim above. Do not copy GPL source, and do not change the license
   silently.

## Common tasks

```bash
# full default test suite (what CI runs)
pytest -q -m "not slow"

# one file or test
pytest -q tests/test_parity.py
pytest -q tests/test_ext2.py::test_zero_row_sums

# speed benchmark (writes bench/results.json; needs R for the R side)
python bench/run.py

# Extension 2 / P2 simulations (long; defaults are the pre-registered runs)
python research/sims/run_ext2.py
python research/sims/summarize.py
python research/sims/run_p2.py
python research/sims/summarize_p2.py

# development-only short sim (marks the run as not pre-registered)
python research/sims/run_ext2.py --reps 8 --out /tmp/ext2-dev
```

## Where the results page is built

GitHub Pages is served from the `main` branch, folder `/docs`
(`https://jgridifier.github.io/ecotraj-py/`). There is no separate build
step: edit the HTML/CSS in `docs/` and merge to `main`.

| File | Role |
|---|---|
| `docs/index.html` | Teaching page |
| `docs/results.html` | Results (parity, speed, Extension 2, P2 power curve) |
| `docs/style.css` | Site theme (keep the existing theme; logo lives in the header) |
| `docs/assets/logo.svg` | Logo (PNG render: `docs/assets/logo.png`) |
| `docs/PARITY.md` | Element-level parity notes |
| `docs/api.md` | API reference |
| `docs/API_SPEC.md` | Intended API and the clean-room rule |

The numbers on `docs/results.html` come from `research/results/` (and
`docs/PARITY.md` / `bench/`). Update the page when a new pre-registered
result is merged; do not invent numbers.

If a new page has no theme of its own, use a Google Research blog-like
look (clean, white, sans-serif). This site already has a theme in
`docs/style.css` — keep it, and keep the logo in the header.

## Pull requests

Use a short-lived branch, open a PR, and wait for CI (pytest on Python 3.11
and 3.13) to pass before merging. Add a line to `STATUS.md` (newest first)
and, if user-visible, `CHANGELOG.md`.
