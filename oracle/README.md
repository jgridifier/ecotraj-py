# Randomized R oracle

From the repository root, regenerate both committed fixtures with:

```sh
/home/box/mamba/envs/rscout/bin/Rscript oracle/generate_fixtures.R
.venv/bin/python -m pytest tests/test_parity.py -q
```

The generator uses R 4.5.3, ecotraj 1.2.2, and jsonlite (the exact versions
are recorded in each fixture). It requires no vegan installation: Bray-Curtis
is computed directly from positive synthetic abundances. Output paths are
relative to the script, so generation also works from another directory.

`oracle_cases.json` contains 40 cases seeded with `set.seed(20261114)`:
Euclidean random walks in 2, 5, and 14 dimensions, squared-Euclidean and
Bray-Curtis dissimilarities, 1–5 sites, 3–12 surveys, unequal lengths,
shuffled survey rows, and explicit irregular times. Each case has 32 calls,
covering all requested length/angle modes, both angle/directionality `add`
settings, speeds, internal variation, metrics, three convergence types,
three segment distances, and three trajectory distances with four
symmetrization settings. Convergence is attempted even on invalid cases
so its errors are recorded too. `triangle_violation` records whether any
input triangle fails the inequality (allowing 1e-10 numerical noise).

`signflip_cases.json` contains five clr paths with 20, 40, or 62 states and
4–14 parts, seeded separately with `set.seed(20261112)`. For each path,
199 draws use exactly `s <- sample(c(-1,1), nrow(d), TRUE)` with `d = diff(C)`.
The complete signs, reconstructed-path oracle directionality null vector,
observed statistic, and `(1 + sum(null >= observed)) / 200` are stored.
Supplying these signs to Python avoids dependence on its random generator.

Serialization uses `digits = NA` and JSON null for R NA/NaN. Numerical
outputs carry explicit kinds and labels; `dist` outputs become full square
matrices. Identical output leaves are stored once in each case's `pool`;
`ref` is a one-based pool index. This deduplication is lossless and keeps
the main fixture below 5 MB. Warnings and errors are captured per call.
The pytest comparator translates `p.value` to `p_value` and checks labels,
shapes, and values at rtol=1e-8, atol=1e-10, including NaNs. Sign-flip nulls
and observed statistics use absolute tolerance 1e-10; p-values are exact.
Recorded R errors require Python to raise ValueError, TypeError, or
RuntimeError; unfinished APIs and NotImplementedError fail explicitly.
No tests are skipped just because an export is missing.

Clean-room rule: only exported ecotraj calls and public help documentation
were used. Never read, print, or browse ecotraj implementation source,
including function bodies or installed package files. See
[the API specification](../docs/API_SPEC.md) for the full rule.

The seeded fixtures record 38 R errors among 1,280 calls:

| R error | Function | Count |
| --- | --- | ---: |
| `Trajectory states must be different` | convergence | 8 |
| `not enough finite observations` | convergence | 8 |
| `Trajectories need to be synchronous for a global convergence test.` | convergence | 8 |
| `subscript out of bounds` | lengths | 8 |
| `replacement has 3 items, need 10` | lengths | 6 |

These observations are retained rather than dropped or repaired. The other
1,242 calls succeeded; all five sign-flip simulations completed. Twelve
input matrices violate the triangle inequality.
