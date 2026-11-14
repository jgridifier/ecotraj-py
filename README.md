# ecotrajpy

A clean-room Python port of **Ecological Trajectory Analysis (ETA)**: path
lengths, angles, directionality, convergence, segment and trajectory distances
(including DSPD), plus the increment sign-flip null used in our 2026-11-12
methods-scout run. The R package [ecotraj](https://cran.r-project.org/package=ecotraj)
1.2.2 is the test oracle.

Docs and results: https://jgridifier.github.io/ecotraj-py/

## Credit

The methods are by the ecotraj authors. Please cite them:

- De Cáceres M, Coll L, Legendre P, Allen RB, Wiser SK, Fortin M-J, Condit R & Hubbell S (2019).
  Trajectory analysis in community ecology. *Ecological Monographs* 89:e01350.
  [doi:10.1002/ecm.1350](https://doi.org/10.1002/ecm.1350)
- Sturbois A, De Cáceres M, Sánchez-Pinillos M, et al. (2021). Extending community trajectory
  analysis: new metrics and representation. *Ecological Modelling* 440:109400.
  [doi:10.1016/j.ecolmodel.2020.109400](https://doi.org/10.1016/j.ecolmodel.2020.109400)
- Djeghri N, et al. (2026). *Ecological Monographs* 96(2):e70058.
  [doi:10.1002/ecm.70058](https://doi.org/10.1002/ecm.70058)
- De Cáceres M, Djeghri N, Sturbois A. ecotraj: Ecological Trajectory Analysis. R package 1.2.2 (CRAN).

## Clean-room note

ecotraj is GPL. This package is MIT and contains **no ecotraj source code**. It was
written from the papers and the package's published documentation (help pages,
vignettes), and checked against ecotraj 1.2.2 run as a black box: R produces JSON
fixtures on synthetic trajectories (`oracle/`), and `tests/` compares against them.

## Install

```bash
pip install -e ".[test,fast]"
pytest -q -m "not slow"
```

## Quick use

```python
import numpy as np
from ecotrajpy import define_trajectories, trajectory_directionality, clr

X = np.random.default_rng(0).dirichlet(np.ones(5), size=30)   # 30 surveys of a 5-part composition
x = define_trajectories.from_coords(clr(X), sites=["s"] * 30)
print(trajectory_directionality(x))
```

## Status and plan

See [ROADMAP.md](ROADMAP.md) and [STATUS.md](STATUS.md).

## Data policy

Only synthetic data lives in this repo. Parity on our real scout panels is run
locally with `scripts/parity_scout_panels.py` against a local data directory; only
the resulting summary numbers are published.
