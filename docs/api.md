# ecotrajpy API reference

The public import surface is re-exported from `ecotrajpy`. Optional kernels
live in `ecotrajpy.fast` (requires Numba). Extension 2 procedures live in
`ecotrajpy.ext2`. This page is a readable copy of the package docstrings;
the source is authoritative if the two ever differ.

```python
from ecotrajpy import (
    Trajectories, define_trajectories, is_metric,
    trajectory_lengths, trajectory_speeds, trajectory_angles,
    trajectory_directionality, trajectory_internal_variation, trajectory_metrics,
    trajectory_convergence, segment_distances, trajectory_distances,
    clr, signflip_path, signflip_test, directionality_coords, increment_split,
)
```

## Construction

### `Trajectories(d, sites, surveys, times)`

Distance matrix plus state metadata (sites, surveys, times). Built with
`define_trajectories` rather than by hand.

### `define_trajectories(d, sites, surveys=None, times=None) -> Trajectories`

Define trajectories (cf. ecotraj `defineTrajectories`).

`d` is a square distance matrix or condensed vector. If `surveys` is
missing it is taken from `times` (order of times within site) or else
assumed in order of occurrence within each site; if `times` is missing
it equals `surveys`.

`define_trajectories.from_coords(X, sites, surveys=None, times=None)`
builds the same object from an `(n, p)` coordinate array using Euclidean
distances.

### `is_metric(d, tol=0.0) -> bool`

True if all distance triplets satisfy the triangle inequality.

The tolerance is relative: a triplet is a violation when
`(d_ij - d_ik - d_kj) / d_ij > tol` (behaviour inferred from ecotraj).

## Metrics of a single trajectory

### `trajectory_lengths(x, relative_to_initial=False, all=False, ecotraj_compat=True) -> DataFrame`

Segment lengths and path length per trajectory.

`relative_to_initial` gives distances from the first survey to each later
one (columns `Lt1_t2`…); `all` gives lengths of all state pairs i<j
(columns `Lt{i}Lt{j}`, no Path column) taken in row order. For a single
trajectory this is the plain set of pairwise distances. For several
trajectories ecotraj's `all=TRUE` is buggy (see `_all_lengths_compat`);
it is reproduced by default and `ecotraj_compat=False` returns the intended
pairwise distances per trajectory instead.

### `trajectory_speeds(x) -> DataFrame`

Segment speeds (length / elapsed time) and path speed (length / duration).

### `trajectory_angles(x, all=False, relative_to_initial=False, stats=True, add=True) -> DataFrame`

Angles (degrees) between consecutive segments, in [0, 180].

0 means segments in a straight line and 180 a reversal. `all=True` gives
angles for all ordered triplets i<j<k (columns A1, A2, … in lexicographic
order of triplets); `relative_to_initial` gives, for each k>=2, the angle
at k between initial→k and k→k+1 (columns `t1-S{k}`). Circular mean
(degrees), circular standard deviation (radians) and mean resultant length
rho are appended when `stats`.

### `trajectory_directionality(x, add=True, nperm=None, rng=None)`

Directionality of each trajectory (De Cáceres et al. 2019).

D = sum over ordered triplets i<j<k of `(s_ij + s_jk)(1 - theta_ijk/180)`
divided by sum of `(s_ij + s_jk)`. With `nperm` a permutation test with
random order of surveys within each trajectory is performed; the p-value is
`(1 + #{perm >= observed}) / (1 + nperm)`. Returns a Series, or a DataFrame
with columns `directionality` and `p.value` when `nperm` is given.

### `trajectory_internal_variation(x, relative_contributions=False) -> DataFrame`

Internal variability of states within each trajectory.

Contribution of each state to the sum of squares (diagonal of the
Gower-centred matrix of squared distances), total `internal_ss` and the
unbiased `internal_variance` = ss / (n - 1).

### `trajectory_metrics(x, add=True) -> DataFrame`

Several metrics at once (cf. ecotraj `trajectoryMetrics`).

## Comparing trajectories

### `trajectory_convergence(x, type="pairwise.asymmetric", add=True) -> dict`

Mann-Kendall tests of convergence/divergence between trajectories.

- `pairwise.asymmetric`: trend in the distance from each state of the row
  trajectory to the path of the column trajectory (row approaching column).
- `pairwise.symmetric`: trend in distances between states of the same
  survey of two trajectories (equal numbers of surveys).
- `multiple`: trend in the variability among trajectories' states at the
  same survey (synchronous trajectories).

Positive tau means divergence, negative convergence. Returns
`dict(tau=..., p_value=...)`.

`type="multiple"` takes its Mann-Kendall axis from the first site's times
in **input-row order** and groups rows by exact time value, matching
ecotraj. It does not sort by time.

### `segment_distances(x, distance_type="directed-segment", add=True) -> dict`

Distances between all pairs of directed segments of all trajectories.

`distance_type`: `"directed-segment"` (default), `"Hausdorff"` or `"PPA"`.
Returns DataFrames `Dseg` (segment distance), `Dini`, `Dfin` (distances
between initial / final points), `Dinifin` and `Dfinini`, labelled
`site[s1-s2]`.

As in ecotraj (whose `Dseg` is an R `dist` object) only the lower triangle
of the segment-by-segment computation is retained and mirrored.

### `trajectory_distances(x, distance_type="DSPD", symmetrization="mean", add=True) -> DataFrame`

Distances between trajectories (De Cáceres et al. 2019; Besse et al. 2016).

- `"DSPD"` — directed segment path distance: mean over the segments of T1
  of the minimum directed-segment distance to the segments of T2.
- `"SPD"` — segment path distance: mean over the states of T1 of the
  distance from the state to the nearest point of the path T2.
- `"Hausdorff"` — maximum of the Hausdorff segment distances between all
  segments of T1 and all segments of T2 (symmetric). Self-comparison
  excludes a segment with itself, so the diagonal is non-zero, as in
  ecotraj.

The asymmetric matrix `D[i, j] = D(Ti → Tj)` is symmetrized with
`"mean"`, `"max"` or `"min"`; `None` returns it unsymmetrized. For `"SPD"`
any non-None symmetrization is the mean (as in ecotraj).

## Compositional helpers and extensions

### `clr(X)`

Row-wise centred log-ratio transform: `log(x) - mean(log(x))` per row.
Rows must be strictly positive (closure is irrelevant for clr).

### `directionality_coords(C, add=True) -> float`

Directionality (De Cáceres et al. 2019) of one trajectory given as a
`(T, p)` coordinate array, using Euclidean distances.

### `signflip_path(C, signs)`

Rebuild a path from `C[0] + cumsum(signs[:, None] * diff(C))`.

### `signflip_test(C, nsim=999, rng=None, signs=None, add=True) -> dict`

Sign-flip permutation test of directionality.

Returns `dict(stat, p_value, null)` with
`p = (1 + #{null >= obs}) / (1 + nsim)`.
`signs` of shape `(nsim, T-1)` with values ±1 reproduces given draws
exactly (used to replay ecotraj's draws in the parity tests).

### `increment_split(C, X) -> dict`

Regress trajectory increments on covariate increments.

`dC = diff(C)`, `dX = diff(X)`; `B = lstsq(dX, dC)` without intercept;
`market = C[0] + cumsum(dX @ B)`; `residual = C[0] + cumsum(dC - dX @ B)`.
`step_energy_market` is the share of squared step length explained
(`sum ||dX B||² / sum ||dC||²`).

## `ecotrajpy.fast` (optional Numba)

Requires the `fast` extra (`pip install -e ".[fast]"`). No fast-math:
roundoff and invalid triangles follow the reference.

- `directionality_coords_fast(C)` — Euclidean directionality; array-likes
  are coerced to C-contiguous float64.
- `signflip_null_fast(C, signs)` — null statistics.
- `signflip_test_fast(C, nsim=999, rng=None, signs=None)` — returns
  `stat`, `null` and `(1 + count(null >= stat)) / (1 + nsim)`. Explicit
  `signs` override `nsim`. Near-ties are resolved with the reference
  arithmetic so parallel reduction roundoff cannot alter the p-value.

Kernels match the reference to ≤ 1e-12.

## `ecotrajpy.ext2` (Extension 2)

Implements the procedures of `research/prereg/EXT2_PREREG.md` (P1,
P1-naive, L, W, Mk). All statistics are functions of clr coordinates
(Aitchison geometry). See that pre-registration and `docs/results.html`
for the study design; this module is the implementation, not the result.

Principal entry points: `split_increments`, `run_procedures`,
`path_test`, `level_residual_path`, `draw_signs`, `ilr_basis`.
