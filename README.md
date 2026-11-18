<p align="center">
  <img src="docs/assets/logo.svg" width="128" height="128" alt="ecotrajpy logo: a directed trajectory through points in a 2-D ordination space, ending in a leaf">
</p>

# ecotrajpy

A clean-room Python port of **Ecological Trajectory Analysis (ETA)**: path
lengths, angles, directionality, convergence, and segment and trajectory
distances (including DSPD). It also includes a clr-valid increment
market/residual split and a sign-flip null for directionality. The R package
[ecotraj](https://cran.r-project.org/package=ecotraj) 1.2.2 is the test oracle.

Docs and results: https://jgridifier.github.io/ecotraj-py/

This package is **not on PyPI yet**. Install it from source (see below).

## Credit

The methods are by the ecotraj authors. Please cite them (and this software;
see [CITATION.cff](CITATION.cff)):

- De Cáceres M, Coll L, Legendre P, Allen RB, Wiser SK, Fortin M-J, Condit R & Hubbell S (2019).
  Trajectory analysis in community ecology. *Ecological Monographs* 89(2):e01350.
  [doi:10.1002/ecm.1350](https://doi.org/10.1002/ecm.1350)
- Sturbois A, De Cáceres M, Sánchez-Pinillos M, Schaal G, Gauthier O, Le Mao P, Ponsero A & Desroy N (2021).
  Extending community trajectory analysis: New metrics and representation. *Ecological Modelling* 440:109400.
  [doi:10.1016/j.ecolmodel.2020.109400](https://doi.org/10.1016/j.ecolmodel.2020.109400)
- Djeghri N, Sturbois A, Hélaouët P, Geissmann Q, Rimmelin-Maury P, Boyé A, Gauthier O & De Cáceres M (2026).
  Going round in cycles but going somewhere: Deciphering cyclical dynamics using Ecological Trajectory Analysis.
  *Ecological Monographs* 96(2):e70058.
  [doi:10.1002/ecm.70058](https://doi.org/10.1002/ecm.70058)
- De Cáceres M, Djeghri N, Sturbois A. ecotraj: Ecological Trajectory Analysis. R package 1.2.2 (CRAN).

## Clean-room note

ecotraj is licensed **GPL (≥ 2)**. This package is **MIT** and contains **no
ecotraj source code**. It was written from the papers and the package's
published documentation (help pages, vignettes), and checked against ecotraj
1.2.2 run as a black box: R produces JSON fixtures on synthetic trajectories
(`oracle/`), and `tests/` compares against them. See [LICENSE](LICENSE) and
[CONTRIBUTING.md](CONTRIBUTING.md).

## Install (from source)

Requires Python 3.10+ and NumPy, SciPy, and pandas. The optional `fast` extra
pulls in Numba for the directionality and sign-flip kernels.

```bash
git clone https://github.com/jgridifier/ecotraj-py.git
cd ecotraj-py
pip install -e ".[test,fast]"
pytest -q -m "not slow"
```

## 30-second quickstart

```python
import numpy as np
from ecotrajpy import define_trajectories, trajectory_directionality, trajectory_lengths, clr

# 30 surveys of a 5-part composition, as clr coordinates (Aitchison geometry)
X = np.random.default_rng(0).dirichlet(np.ones(5), size=30)
x = define_trajectories.from_coords(clr(X), sites=["s"] * 30)
print(trajectory_lengths(x))
print(trajectory_directionality(x))
```

A sign-flip test of directionality (optional Numba path):

```python
from ecotrajpy import clr
from ecotrajpy.fast import signflip_test_fast

shares = np.random.default_rng(0).dirichlet(np.ones(6) * 50, size=40)
res = signflip_test_fast(clr(shares), nsim=999, rng=1)
print(res["stat"], res["p_value"])
```

## Mapping to R ecotraj

| Python | ecotraj |
|---|---|
| `define_trajectories` | `defineTrajectories` |
| `trajectory_lengths`, `trajectory_speeds` | `trajectoryLengths`, `trajectorySpeeds` |
| `trajectory_angles`, `trajectory_directionality` | `trajectoryAngles`, `trajectoryDirectionality` |
| `trajectory_internal_variation`, `trajectory_metrics` | `trajectoryInternalVariation`, `trajectoryMetrics` |
| `trajectory_convergence` | `trajectoryConvergence` |
| `segment_distances`, `trajectory_distances` | `segmentDistances`, `trajectoryDistances` |
| `is_metric` | `is.metric` |
| `clr`, `signflip_test`, `increment_split` | (extensions; not in ecotraj) |
| `ecotrajpy.fast` | Numba kernels for directionality and the sign-flip null (Extension 1) |
| `ecotrajpy.ext2` | Extension 2 market/residual split and tests |

Full signatures: [docs/api.md](docs/api.md).

## Parity status

Against ecotraj 1.2.2, a randomized oracle suite of 40 synthetic cases
(Euclidean, squared-Euclidean, and Bray-Curtis inputs; 1–5 trajectories;
unequal lengths; shuffled rows) runs **1,280** checks at rtol 1e-8, atol
1e-10. **1,260 match exactly.** The remaining **20** are documented R
floating-point quirks on non-metric inputs: after the local triangle
correction a triplet is exactly collinear, and ecotraj's result follows the
sign of ~1e-14 rounding noise (NaN vs finite, corrected vs uncorrected
endpoint distance, Kendall ties). None of them affects Euclidean / clr
directionality, lengths, the sign-flip null, or the increment split — the
quantities this package's studies use.

The 20 cases are listed one by one in `tests/known_deviations.json` and run
as non-strict xfails, so they stay visible in CI. Element-level notes:
[docs/PARITY.md](docs/PARITY.md).

## Status and plan

See [ROADMAP.md](ROADMAP.md) and [STATUS.md](STATUS.md). Pre-registrations,
simulation runners, and published results live under [`research/`](research/README.md)
and are frozen scientific material.

## For coding agents

Start at [AGENTS.md](AGENTS.md) (setup, invariants, exact commands) and
[llms.txt](llms.txt).

## Data policy

Only synthetic data lives in this repo. Parity on private scout panels is run
locally with `scripts/parity_scout_panels.py` against a local data directory;
only the resulting summary numbers are published.
