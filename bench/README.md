# M2 sign-flip speed benchmark

## Measured results

Wall times in seconds for 999 draws; warm timings are five-run medians.

| Panel (T × D) | R loop (1 thread)* | Python (1 thread) | Numba (1 thread) | Numba (8 threads) | Bare R seconds/draw (1 thread) |
|---|---:|---:|---:|---:|---:|
| 62 × 14 | 98.800* | 8.893 | 0.593 | 0.106 | 0.08515 |
| 120 × 50 | 614.185* | 74.070 | 3.212 | 0.424 | 0.60840 |

| Panel | Python / Numba (1 thread) | Python / Numba (8 threads) | R / Numba (1 thread)* | R / Numba (8 threads)* |
|---|---:|---:|---:|---:|
| 62 × 14 | 15.0× | 83.7× | 166.5×* | 930.4×* |
| 120 × 50 | 23.1× | 174.8× | 191.2×* | 1449.4×* |

All Python times are medians of five complete 999-draw tests. R times are
projections from the median of five warm batches: 20 draws per batch for
62 × 14 and 10 for 120 × 50. An asterisk marks **both extrapolated times and
ratios**; these assume linear scaling across draws. Neither complete R test
was run, so neither R p-value is verified. Bare R times are per draw, measured
on prebuilt trajectory objects, and are not complete sign-flip-test times.

Cold first-call JIT: **1.155 s**; median warm tiny call: 0.00001489 s; estimated compilation/initialization overhead: 1.155 s.

## Validation

Full suite, using the requested command
`PYTHONPATH=src /workspace/ecotraj-py/.venv/bin/python -m pytest -q`:
**1,213 passed, 196 xfailed, 0 failed**, 187 warnings, in 20.03 s.
These are the current merged-M1 branch results, replacing the obsolete pre-merge
failure counts. Targeted fast suite: **101 passed**, 75 warnings, in 12.95 s
(`NUMBA_NUM_THREADS=8 PYTHONPATH=src /workspace/ecotraj-py/.venv/bin/python -m pytest tests/test_fast.py -q`).
Warnings include reference arithmetic on degenerate/invalid angles and NaNs.

Coverage includes T=3 through 70, dimensions spanning 1 through 20, R fixtures,
seeded draws, ties, degenerate paths, empty draws, shape validation, integer and
list inputs, Fortran order, noncontiguous arrays, NaN coordinates, and simulated
missing Numba. Input cases run against both backends. A crafted collinear
Euclidean path has a positive triangle correction of 3.55e-15 from roundoff;
the test asserts that violation and checks reference parity. Exact Euclidean
coordinates cannot violate the triangle inequality, but floating-point
distances can. Separate subprocesses with NUMBA_NUM_THREADS=1 and 4 produce
bitwise-identical null arrays; the benchmark also checks 1 versus 8 threads.

- 62 × 14: p=0.874; maximum absolute Numba/reference null error 1.5e-14; measured R/reference error 9.38e-15.
- 120 × 50: p=0.644; maximum absolute Numba/reference null error 3.38e-14; measured R/reference error 1.76e-14.

## Method

The complete Python sign-flip test includes the observed statistic and 999
null draws, with seed **20261114**, on two synthetic compositional panels.
`panel.py` defines a multiplicative Dirichlet random walk: start uniformly,
multiply by independent Dirichlet(100, …, 100) perturbations, close to sum one,
and take clr in log space. The sign matrix uses seed 20261115 and is identical
for R, reference Python, and Numba. Data interchange uses temporary CSV files.

Reference Python and R execute serially. Numba is measured with both one and
eight worker threads; the eight-thread ratios include parallelism gains.
`OPENBLAS_NUM_THREADS=1` prevents nested BLAS parallelism. Host CPU, versions,
affinity, raw samples, statistics, p-values, and errors are in `results.json`.
Five-run medians reduce noise but are not confidence intervals or isolated
hardware measurements. R and Python timing runs execute sequentially.

The R baseline is a **naive user loop**, not a native ecotraj sign-flip test.
For each draw it reconstructs the path, calls `dist`, `defineTrajectories`,
and `trajectoryDirectionality` through public ecotraj 1.2.2 APIs. The additional
bare-call measurement excludes reconstruction, distances, and object creation;
each draw's trajectory is built before timing. No installed ecotraj source was
inspected. Startup, CSV IO, and an explicit warm-up draw are excluded.

The old three-draw pilot and adaptive threshold have been removed. Its roughly
3× discrepancy with the old 50-draw sample cannot be attributed confidently to
warm-up from the recorded data. The replacement warms the API first and uses
five equal-sized batches with all raw times recorded, eliminating the comparison
of differently sized pilot and measurement batches. The same signs are reused
in each batch; only the first 20/10 distinct R null values are validated against
Python. Median batch time is multiplied by 999/batch size and the median
observed-statistic time is added once.

Python timers include reconstruction, distances, directionality, and p-value
computation; imports and panel generation are excluded. The first call uses a
tiny 4 × 3 panel with two draws. The runner creates a fresh temporary
`NUMBA_CACHE_DIR` before importing Numba; the cold time includes compilation,
thread initialization, and tiny computation. Its difference from the median
warm tiny call estimates compilation overhead. Warm panel runs exclude JIT.

Both JIT kernels deliberately disable disk caching, preventing stale compiled
callees and avoiding writes to read-only package directories. Compilation occurs
once per process. Numba uses no fast-math and precomputes one distance matrix per
path. Triplet work is O(T³); null draws run through `prange`. The optional NumPy
fallback promises **bounded triplet memory, not speed**: it evaluates at most
32,768 triples per chunk, with an additional O(T²) distance matrix. Near-ties
are recomputed with the reference implementation to preserve exact p-values
and the `>=` rule on returned values.

## Reproduction and API

From the repository root, using a Python environment with the fast extra:

```bash
NUMBA_NUM_THREADS=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src python bench/run.py
PYTHONPATH=src /workspace/ecotraj-py/.venv/bin/python -m pytest -q
```

Rscript is selected from `RSCRIPT`, then `shutil.which('Rscript')`, then the
local default `/home/box/mamba/envs/rscout/bin/Rscript`. Override `RSCRIPT` for
another installation. The benchmark checks all Python nulls and measured R
nulls to absolute tolerance 1e-12, and exact Python/reference p-values.

```python
from ecotrajpy.fast import directionality_coords_fast, signflip_null_fast, signflip_test_fast
result = signflip_test_fast(C, nsim=999, rng=20261114)
```

All three public functions accept numeric array-likes and coerce coordinates
(and explicit signs) to C-contiguous float64 before either backend. Numba is
optional (`pip install '.[fast]'`); without it the same API uses NumPy.
All functions use the reference's `add=True` behavior.
