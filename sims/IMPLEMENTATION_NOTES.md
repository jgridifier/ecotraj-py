# Ext2 implementation choices where the prereg is silent

1. **Within-cell RNG.** Each replication r of a cell uses the child
   `SeedSequence(entropy, spawn_key=cell_key + (r,))` (identical to `cell_ss.spawn(R)[r]`), one
   `default_rng` per replication. Draw order: Z1, Z2, Z3, Z4 (each AR: initial value, then T-1
   innovations), loadings B, residual noise (i-b: rho; i-c: Student-t; else Normal), then the 199 x (T-1)
   sign matrix (`rng.integers(0,2)` mapped to ±1). Same signs for P1, P1n, L, W, Mk.
2. **Z4 trend index** t = 1..T (constant offset is irrelevant after differencing).
3. **Student-t scaling:** t5 * sqrt(3/5) (unit variance), then centred by P and scaled by sigma.
4. **Ties:** count = #{null >= obs - 1e-12}; the tolerance only absorbs roundoff for the identity/reversal draws
   (s and -s give the same statistic). Reject iff count <= 9; p = (1+count)/200.
5. **Statistic implementation:** `ecotrajpy.fast.directionality_coords_fast` (numba, distance-based with the
   triangle correction, which is a no-op for Euclidean distances up to roundoff) rather than the
   scipy `nulls.directionality_coords`; the zero-length rule (theta := 0) is not implemented (kernel skips
   degenerate triples; probability-zero event).
6. **Null paths** for all procedures are rebuilt as c0 + cumsum of flipped (and, for P1, re-projected) increments.
   For P1 the observed path is C_1 + r_t; for L/W/Mk the null starts from the path's own first point.
7. **L:** OLS of C on [1, X] via lstsq; residual path = residuals + column means of C (clr of the shares).
8. **Mk drift-to-noise ratio** (undefined in prereg): ||mean market increment|| / sqrt(mean_t ||fit_t - mean fit||^2).
9. **ii net displacement along u:** (sum_t E_t) . u with E the residual increments; the retained fraction is
   1' M 1 / (T-1). Both averaged over replications in the summary.
10. **Realized market share** is stored per replication and averaged (planted drift excluded).
11. **i-d q_t:** Z1 raw levels at t = 1..T-1 (before the step), standardized with ddof=1 over those T-1 values.
12. **i-b:** starting clr = clr(start composition) + rho_1 (rho_1 is not re-centred; it is already centred by P).
13. **Planted u:** QR basis of [1, rows of B] to project e1 - e2; normalized.
14. **Seeds/dev:** `--seed`/`--reps` only mark `preregistered_configuration=false` in run_meta.json.
    Workers run with NUMBA_NUM_THREADS=1 (parallelism over replications, 8 processes).
15. **McNemar:** exact two-sided binomial on discordant pairs; "more power" also requires b > c.
16. **Secondary i-b criterion** is rate <= 0.070 only (no lower bound).
