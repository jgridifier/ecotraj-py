# ecotrajpy API spec (M1)

Clean-room rule for everyone working on this repo: **do not read, print, or browse
ecotraj source code** (no `print(ecotraj::fn)`, no `body()`/`deparse()` of package
functions, no reading files inside the installed package library, no browsing
github.com/emf-creaf/ecotraj/blob/... or cran/ecotraj R/ or src/ files). Allowed sources:
the help pages in /workspace/ecotraj-docs/*.txt (exported Rd text), the ecotraj
vignettes/articles on https://emf-creaf.github.io/ecotraj/articles/ , the papers
(De Cáceres et al. 2019 Ecol Monogr e01350; Sturbois et al. 2021; Djeghri et al. 2026;
Besse et al. 2016 for segment distances), and **black-box calls** to ecotraj 1.2.2 via
`/home/box/mamba/envs/rscout/bin/Rscript` (R 4.5.3) to observe outputs.

Package: `src/ecotrajpy/`. numpy/scipy/pandas only (numba optional later, M2).
Return pandas objects labelled like ecotraj where natural.

```python
from ecotrajpy import (
    Trajectories, define_trajectories,            # define_trajectories(d, sites, surveys=None, times=None)
                                                  # d: full square (n,n) distance matrix or condensed vector
                                                  # define_trajectories.from_coords(X, sites, surveys=None, times=None)  (Euclidean)
    trajectory_lengths,        # (x, relative_to_initial=False, all=False) -> DataFrame, cols S1..Sk,Path (or Lt1_t2.., Path)
    trajectory_speeds,         # (x) -> DataFrame
    trajectory_angles,         # (x, all=False, relative_to_initial=False, stats=True, add=True) -> DataFrame (S1-S2.., mean, sd, rho)
    trajectory_directionality, # (x, add=True, nperm=None, rng=None) -> Series (or DataFrame with columns directionality, p.value)
    trajectory_internal_variation,  # optional
    trajectory_metrics,        # (x, add=True) -> DataFrame like ecotraj trajectoryMetrics
    trajectory_convergence,    # (x, type="pairwise.asymmetric"|"pairwise.symmetric"|"multiple", add=True) -> dict(tau=..., p_value=...)
    segment_distances,         # (x, distance_type="directed-segment"|"Hausdorff"|"PPA", add=True) -> dict(Dseg, Dini, Dfin, Dinifin[, Dfinini]) of DataFrames
    trajectory_distances,      # (x, distance_type="DSPD"|"SPD"|"Hausdorff", symmetrization="mean"|"max"|"min"|None, add=True) -> DataFrame (square)
    is_metric,                 # (d, tol=0) -> bool
)
from ecotrajpy.compositional import clr                   # row-wise centred log-ratio
from ecotrajpy.nulls import (
    signflip_path,      # (C, signs) -> path rebuilt from C[0] + cumsum(signs[:,None]*diff(C))
    signflip_test,      # (C, nsim=999, rng=None, signs=None) -> dict(stat, p_value, null) ; p=(1+#{null>=obs})/(1+nsim)
                        #   signs: optional (nsim, T-1) array of +/-1 to reproduce a given draw exactly
    directionality_coords,  # (C, add=True) fast single-trajectory directionality from coordinates
)
from ecotrajpy.split import increment_split   # (C, X) -> dict(market=path, residual=path, B=coef, step_energy_market=float)
          # dC = diff(C); dX = diff(X); B = lstsq(dX, dC) (no intercept)
          # market = C[0] + cumsum(dX @ B); residual = C[0] + cumsum(dC - dX @ B)
```

Directionality (De Cáceres et al. 2019): for all ordered triplets i<j<k within a
trajectory, with segment lengths s_ij, s_jk and angle θ_ijk (degrees, change of
direction at j, from the triangle of distances, with ecotraj's local additive
triangle-inequality correction when `add=True`):
D = Σ (s_ij + s_jk)(1 − θ_ijk/180) / Σ (s_ij + s_jk).
