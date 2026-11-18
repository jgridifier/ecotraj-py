# Contributing to ecotrajpy

Thanks for helping. This package is a clean-room port with a strict parity
contract. Please read [AGENTS.md](AGENTS.md) as well; the points below are
the ones that most often matter in a pull request.

## Setup

```bash
git clone https://github.com/jgridifier/ecotraj-py.git
cd ecotraj-py
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,fast]"
pytest -q -m "not slow"
```

CI (`.github/workflows/ci.yml`) runs that same pytest command on Python 3.11
and 3.13. A PR should stay green there.

## Clean-room rule

ecotraj (R) is GPL (≥ 2). This repository is MIT and must remain a
clean-room reimplementation of the *published methods*, not a transcription
of ecotraj source.

**Do not** read, print, or browse ecotraj source code (no `body()` /
`deparse()` / `print(ecotraj::fn)`, no files from the installed package
library, no browsing `R/` or `src/` on GitHub or CRAN). Allowed sources are
the papers, the published help pages and vignettes, and black-box calls that
only observe outputs. The intended API is in [docs/API_SPEC.md](docs/API_SPEC.md).

## Numerics and parity

- ecotraj 1.2.2 is the oracle. Do not change arithmetic so it "looks nicer"
  if that breaks agreement with R on Euclidean / documented cases.
- Any change to metrics, distances, clr, the increment split, or the
  sign-flip null needs `pytest -q -m "not slow"` (especially
  `tests/test_parity.py`) before you conclude it is safe.
- Tolerances stay at rtol 1e-8, atol 1e-10 in the randomized suite. The 20
  known deviations in `tests/known_deviations.json` are documented R
  floating-point quirks; do not drop those xfails or widen the comparator
  to hide them. Notes: [docs/PARITY.md](docs/PARITY.md).

## Research material

[`research/`](research/README.md) holds frozen pre-registrations, runners,
and published results. Do not edit a pre-registration's hypotheses, seeds,
margins, or pass/fail rules after its run. New work gets a new
pre-registration, merged before any simulation.

## Other invariants

- No neural nets; this is classical trajectory analysis.
- Only synthetic data belongs in the repo. `local_data/` is gitignored.
- Do not publish to PyPI or add a publish workflow without an explicit go
  from the owner and a token.
- Do not change the license silently.

## Pull requests

Use a short-lived branch off `main`. Describe what changed and how you
checked it. Add a newest-first entry to `STATUS.md` and, if the change is
user-visible, a note in `CHANGELOG.md`.
