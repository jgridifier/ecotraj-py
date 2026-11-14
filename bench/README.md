# M2 sign-flip speed benchmark

## Measured results

Wall time in seconds, 999 null draws; eight CPU cores available and eight Numba threads.

| Panel (T × D) | R ecotraj 1.2.2 | Reference Python | Warm Numba | Python / Numba | R / Numba |
|---|---:|---:|---:|---:|---:|
| 62 × 14 | 88.162 | 8.788 | 0.102 | 86.2× | 865.1× |
| 120 × 50 | 622.966* | 70.809 | 0.555 | 127.7× | 1123.1× |

*Large-case R time is extrapolated: 50 draws took 31.146 s; multiply by 999/50 and add the observed-statistic time. The 999-draw R test was not run.

Cold first-call JIT time: **1.289 s** (warm tiny call: 0.000032 s; approximate compilation/initialization overhead: 1.289 s). This is separate from the warm times above.

## Validation

The final targeted run (`NUMBA_NUM_THREADS=8 PYTHONPATH=src /workspace/ecotraj-py/.venv/bin/python -m pytest tests/test_fast.py -q`) passed **88 tests** in 8.37 s. Three warnings come from the reference dividing zero by zero for degenerate trajectories; the fast implementation returns the same NaN. Coverage includes every T from 3 through 70, dimensions spanning 1 through 20, fixed R sign-flip fixtures with exact p-values, seeded draws, ties, degenerate paths, empty draws, input shape validation, and a simulated missing-Numba import.

- T=62, D=14: p=0.874; maximum absolute fast/reference null difference 1.5e-14; maximum measured R/reference null difference 1.5e-14.
- T=120, D=50: p=0.644; maximum absolute fast/reference null difference 3.38e-14; maximum measured R/reference null difference 2.38e-14.

An initial full-suite run, with the first 18 fast tests, reported **1,026 passed, 298 failed, 2 xfailed** in 14.52 s. The failures are existing R-parity mismatches in the untouched core/metrics/comparison functionality (including non-Euclidean angles, directionality, convergence, and segment/trajectory distances). No existing source files were changed for this milestone. The later 70 added tests were verified in the final targeted run above.

The benchmark compares a complete sign-flip test (observed statistic plus 999
null draws), with seed **20261114**, on two synthetic compositional panels:
62 surveys × 14 parts (scout scale), and 120 surveys × 50 parts (larger case).
`panel.py` defines a multiplicative Dirichlet random walk: start uniformly,
multiply by independent Dirichlet(100, …, 100) perturbations, close to sum one,
and take clr. It computes clr in log space for numerical stability. The sign
matrix uses seed 20261115 and is identical for R, reference Python, and Numba.
Only the generator is saved; panels and signs are exchanged through temporary
CSV files and removed after the run.

Eight cores/Numba worker threads are requested. Reference Python and R execute
serially; this comparison includes the parallelism benefit of the fast API.
`OPENBLAS_NUM_THREADS=1` prevents nested BLAS parallelism. Host CPU and library
versions, affinity, raw times, observed statistics, p-values, and maximum
numerical differences are recorded in `results.json`. These are single-run
wall-clock measurements on a shared host, not confidence intervals or isolated
hardware measurements.

The R script calls only public ecotraj 1.2.2 APIs:
`defineTrajectories(dist(path), sites=…, surveys=…)` followed by
`trajectoryDirectionality`, for the observation and each reconstructed null
path. Path reconstruction and distance computation are included in every draw.
No ecotraj source was inspected. R package startup, CSV IO, and a three-draw
pilot are excluded. If the large-case pilot predicts more than 600 seconds,
50 draws are measured and the draw time is multiplied by 999/50; the observed
statistic's measured time is added once. An extrapolated test has no reported
R p-value; its 50 null statistics are still checked against Python.

Python timers include path reconstruction, distances, directionality, and
p-value computation; imports and synthetic panel generation are excluded.
Cold JIT timing uses a separate tiny 4×3 panel with two draws and a fresh cache;
it includes compilation, thread initialization, and that tiny computation.
The warm tiny time is recorded separately, with their difference reported as
an approximate compilation overhead. Warm benchmark runs exclude compilation.
Numba uses no fast-math and precomputes one distance matrix per path. Triplet
work is O(T³); draws run through `prange`. The NumPy fallback evaluates triplets
in chunks of at most 32,768. Near-ties in the test wrapper are recomputed with
the reference implementation, preserving both exact p-values and the stated
`>=` rule on the returned values.

Reproduce from the repository root (use a fresh cache directory each time):

```bash
NUMBA_CACHE_DIR=$(mktemp -d /tmp/ecotraj-m2-jit.XXXXXX) \
NUMBA_NUM_THREADS=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src /workspace/ecotraj-py/.venv/bin/python bench/run.py
```

The R executable is `/home/box/mamba/envs/rscout/bin/Rscript`.
The benchmark checks all Python null values and measured R null values to
absolute tolerance 1e-12 and checks exact p-values wherever all 999 draws ran.

Use the new API without changing existing package imports:

```python
from ecotrajpy.fast import directionality_coords_fast, signflip_null_fast, signflip_test_fast
result = signflip_test_fast(C, nsim=999, rng=20261114)
```

The two kernels accept NumPy arrays; the test wrapper also accepts array-like
inputs. Numba is optional (`pip install '.[fast]'`); without it, the same API
uses the NumPy fallback. All functions use the reference's `add=True` behavior.
