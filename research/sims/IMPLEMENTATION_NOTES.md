# Ext2 implementation notes

## (a) Choices where the prereg is silent

1. **Within-cell RNG.** Each replication r of a cell uses the child
   `SeedSequence(entropy, spawn_key=cell_key + (r,))` (identical to `cell_ss.spawn(R)[r]`), one
   `default_rng` per replication. Draw order: Z1, Z2, Z3, Z4 (each AR: initial value, then T-1
   innovations), loadings B, residual noise (i-b: rho; i-c: Student-t; else Normal), then the 199 x (T-1)
   sign matrix (`rng.integers(0,2)` mapped to ±1). Same signs for P1, P1n, L, W, Mk.
2. **Z4 trend index** t = 1..T (constant offset is irrelevant after differencing).
3. **Student-t scaling:** t5 * sqrt(3/5) (unit variance), then centred by P and scaled by sigma.
4. **Null paths** for all procedures are rebuilt as c0 + cumsum of flipped (and, for P1, re-projected) increments.
   For P1 the observed path is C_1 + r_t; for L/W/Mk the null starts from the path's own first point.
5. **L:** OLS of C on [1, X] via lstsq; residual path = residuals + column means of C (clr of the shares).
6. **Mk drift-to-noise ratio** (undefined in prereg): ||mean market increment|| / sqrt(mean_t ||fit_t - mean fit||^2).
7. **ii net displacement along u:** (sum_t E_t) . u with E the residual increments; the retained fraction is
   1' M 1 / (T-1). Both averaged over replications in the summary.
8. **Realized market share** is stored per replication and averaged (planted drift excluded).
9. **i-d q_t:** Z1 raw levels at t = 1..T-1 (before the step), standardized with ddof=1 over those T-1 values.
10. **i-b:** starting clr = clr(start composition) + rho_1 (rho_1 is not re-centred; it is already centred by P).
11. **Planted u:** QR basis of [1, rows of B] to project e1 - e2; normalized.
12. **Seeds/dev:** `--seed`/`--reps` only mark `preregistered_configuration=false` in run_meta.json.
    Workers run with NUMBA_NUM_THREADS=1 (parallelism over replications, 8 processes).
13. **McNemar:** exact two-sided binomial on discordant pairs; "more power" also requires b > c.
14. **Secondary i-b criterion** is rate <= 0.070 only (no lower bound).
15. **Invalid replications.** If the observed or any null statistic is non-finite, the replication is
   flagged invalid for that procedure (`<proc>_valid = 0`, count and p-value NaN). It is never counted as a
   rejection or a non-rejection; the summary reports the number of invalid replications per cell and procedure
   and computes rates over valid replications only, showing the count.
16. **Summary validation.** `research/sims/summarize.py` checks `run_meta.json` (preregistered_configuration true,
   seed 2026111401, B = 199, R per cell) and CSV completeness/integrity before issuing a verdict. Anything else
   is labelled `DEVELOPMENT RUN - NOT A PREREGISTERED RESULT` and gets no binding verdict.
17. **Scenario-ii-only outputs.** Retained-drift fraction and net displacement along u are saved only for
   scenario ii (NaN elsewhere).

## (b) Items covered by Addendum 1 (research/prereg/, written by the lab lead)

These are fixed or touched by the prereg text and are covered by Addendum 1, not decided here:

- **Statistic implementation.** The runner uses `ecotrajpy.fast.directionality_coords_fast` (numba) rather than
  `nulls.directionality_coords` named in section 3.
- **Zero-length-leg convention (theta := 0).** The fast kernel skips degenerate triples; see Addendum 1 for the
  convention that applies.

The exceedance count is the literal `#{null >= obs}` of section 3 (no tolerance); reject iff count <= 9;
p = (1 + count)/200.
