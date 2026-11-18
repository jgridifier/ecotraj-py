# Extension 2 pre-registration: increment-level market/residual trajectory split with a sign-flip test

**Status:** pre-registered. Written and merged on 2026-11-14 (ET) **before any Extension 2 simulation was run**.
Any change after merge must be a dated addendum below the original text; the original text is never edited.
Simulation code may be written after this file is merged, but must implement exactly what is written here.
Drafting: written by the lab lead; reviewed before merge by an independent agent (Codex CLI), whose 12 points
(4 blockers, 8 should-fix) were all addressed in this text. No Extension 2 simulation had been run at merge time.

## 1. Question

A composition (shares of a total across D parts, observed at T time points) moves for two reasons:
a common "market" driver measured by k observed covariates, and part-specific ("sticky") movement
that the market does not explain. Does the residual part of the path move in a persistent direction?

The 2026-11-12 scout found that residualizing clr **levels** on market covariates hid a planted-looking
sticky walk (residual sign-flip p = 0.994), while residualizing **increments** found it (residual
directionality 0.595, p = 0.008). Extension 2 turns the increment split into a procedure with a
demonstrated size and power on simulated panels where the truth is known.

## 2. Compositional setting (why this is valid on the simplex)

- Shares S_t in the open simplex are mapped to clr coordinates C_t = log S_t − mean(log S_t).
  Euclidean distance between clr vectors is the Aitchison distance, so ETA directionality computed
  on clr coordinates is a statistic of the compositions themselves, invariant to the order of parts
  and to the choice of ilr basis.
- Increments ΔC_t = C_{t+1} − C_t lie in the zero-sum hyperplane. The split is additive in clr,
  which is perturbation (⊕) in the simplex: S_t = S_1 ⊕ M_t ⊕ R_t, where M is the market sub-path and
  R the residual sub-path. Both sub-paths map back to valid compositions by the inverse clr (softmax).
- The least-squares fit below is done column by column on clr increments; because every row of ΔC
  sums to zero, the fitted market and residual increments also sum to zero in every row.
  The unit tests check: zero row sums, invariance of every statistic to permuting the parts, and to
  replacing clr by an orthonormal ilr basis (to 1e-10).
- Component paths are defined from zero: m_1 = r_1 = 0, m_{t+1} = m_t + (ΔX B̂)_t, r_{t+1} = r_t + Ê_t.
  Then S_t = S_1 ⊕ softmax(m_t) ⊕ softmax(r_t) exactly. The residual trajectory that is tested is
  C_1 + r_t (T points, starting at the observed first composition).

## 3. Procedures compared

**Statistic.** For a trajectory with points y_1..y_T (clr coordinates), s_ij = ‖y_j − y_i‖ (Euclidean =
Aitchison). For every triple i < j < k, θ_ijk ∈ [0°, 180°] is the angle between the vectors y_j − y_i and
y_k − y_j (0° = same direction). Then
dir = Σ_{i<j<k} (s_ij + s_jk)(1 − θ_ijk/180) / Σ_{i<j<k} (s_ij + s_jk)
(De Cáceres et al. 2019; equal to ecotraj 1.2.2 `trajectoryDirectionality` on Euclidean distances, checked
by the M1 parity suite). If s_ij or s_jk is 0, θ_ijk := 0 (a probability-zero event under these DGPs).
Implementation: `ecotrajpy` at the commit that merges M2, function `ecotrajpy.nulls.directionality_coords`.

**p-value.** B = 199 independent sign vectors per test, drawn with replacement (duplicates and the identity
draw are kept). p = (1 + #{null ≥ obs}) / 200; **reject iff #{null ≥ obs} ≤ 9**, i.e. p ≤ 0.05 (α = 0.05).
The plus-one formula does not by itself make an approximate null transformation exact; calibration below
refers to this 199-draw algorithm and is checked empirically.

Let X_t be the k market covariates, ΔX the (T−1)×k increments, ΔC the (T−1)×D clr increments.

- **P1 (primary: increment split, re-projected sign flip).**
  B̂ = (ΔXᵀΔX)⁻¹ΔXᵀΔC (no intercept, as in the scout run). Residual increments
  Ê = ΔC − ΔX B̂ = M ΔC with M = I − ΔX(ΔXᵀΔX)⁻¹ΔXᵀ. Residual path R = C_1 + cumsum(Ê). Statistic dir(R).
  Null draw: s ∈ {−1,+1}^{T−1} i.i.d. fair signs; E* = M · diag(s) · Ê (flip, then re-project onto the
  residual space, so the null paths live in the same space as the observed residual path);
  null path C_1 + r* with r*_1 = 0, r*_{t+1} = r*_t + E*_t; null value dir of that path.
  P1 is an **approximate** residual sign-flip procedure (Freedman–Lane / ter Braak style; Winkler et al.
  2014): under a Gaussian null the residual's temporal covariance is ∝ M, while the transformed one is
  ∝ M S M S M, so re-projection does not guarantee finite-sample validity. Its calibration is what
  this pre-registration tests.
- **P1-naive (secondary).** As P1 but without re-projection: E* = diag(s) Ê. This is exactly the
  2026-11-12 m1b procedure.
- **L (the old level procedure, m1).** Regress clr levels on [1, X_t] column by column; residual path
  = level residuals + column means of C; sign-flip test on that path's increments (naive flips, path
  rebuilt from its first point).
- **W (descriptive only).** No split: sign-flip test of dir(C) on the whole path. W rejection may reflect
  market drift or temporal dependence; frequent rejection is not assumed.
- **Mk (descriptive only).** dir of the market component path C_1 + m_t, with its sign-flip p, and the
  drift-to-noise ratio of the market increments.

## 4. Data-generating process

Common to all scenarios: D = 14 parts, k = 4 market covariates, T ∈ {40, 62}.

- Starting composition: shares ∝ 0.75^j, j = 0..13, then clr → C_1.
- Raw market covariates (each replication draws new, mutually independent paths of length T):
  Z1 AR(1) φ = 0.90 (volatility-like), Z2 i.i.d. N(0,1) (return-like), Z3 AR(1) φ = 0.97 (spread-like),
  Z4 = AR(1) φ = 0.5 plus a deterministic trend 0.02·t added outside the recursion (growth-like).
  Innovations N(0,1); each AR component starts from its stationary distribution N(0, 1/(1−φ²)).
- Generation and fitting use the same matrix: A_tj = ΔZ_tj / s_j, with s_j the sample sd (ddof = 1) of
  ΔZ_·j, **without subtracting column means** (so the trend survives). The covariate levels handed to the
  procedures are X_1 = 0, X_{t+1} = X_t + A_t (so ΔX = A exactly; L regresses on these levels).
- Market loadings: B (k×D) entries i.i.d. N(0, τ²), each row centred to sum zero.
- Residual noise: η_t = σ · P z_t with z_t ~ N(0, I_D) and P = I − 11ᵀ/D (centring), σ = 0.4/√(D−1),
  so the expected residual step norm² is 0.16 (≈ the 0.42 step RMS seen on the scout panel).
- τ² = 0.35 σ² / (0.65 k), which approximately targets a ratio of expected market step energy to total
  null step energy of 0.35 (scout panels: 0.16 and 0.45). It is approximate because ‖A‖²_F/(T−1) is not
  exactly k. The realized share Σ‖A_t B‖² / Σ‖A_t B + η_t‖² is reported per cell (planted drift excluded).
- Loadings, market innovations and residual noise are drawn independently of each other and across
  replications.
- Path: C_{t+1} = C_t + A_t B + η_t + δ, t = 1..T−1. Shares are softmax(C_t); all procedures start from the shares
  (re-applying clr), so the round trip through the simplex is part of the test.

Scenarios:

- **(i) Null, primary: common market factor, no residual walk direction.** δ = 0. All persistent
  movement comes from the market (X4's trend makes the whole path directional, so W is expected to reject;
  that is the trap).
- **(ii) Planted sticky swap, orthogonal to the market loadings.** δ = a·u with a = 0.10 per step (the scout
  panel's residual net 3.81 over 39 steps ≈ 0.098/step; signal-to-noise per step 0.25). u is the unit clr
  contrast obtained by projecting the part-1-versus-part-2 contrast e_1 − e_2 away from the constant vector
  and the row space of the replication's B; other parts may move a little. Orthogonality is in part space;
  the time-space regression can still attenuate the drift (the planted drift enters the residual as
  a·(M1)uᵀ), so for scenario ii we report the retained fraction 1ᵀM1/(T−1) and the residual net
  displacement along u.
- **(i-b) Secondary null, dependence stress test: stationary residual levels.** δ = 0; residual levels
  ρ_t = (σ/√2)·P z_t, t = 1..T, i.i.d.; residual increments ρ_{t+1} − ρ_t (same expected step energy as (i),
  negative lag-1 dependence); C_1 is the starting composition plus ρ_1. One-sided criterion only
  (a conservative test is acceptable here).
- **(i-c) Secondary null: heavy tails.** z_t entries i.i.d. Student-t with 5 df scaled to unit variance.
- **(i-d) Secondary null: market-linked heteroskedasticity.** η_t multiplied by
  h_t = exp(0.5 q_t) / sqrt(mean_t exp(q_t)), where q_t is Z1 at the start of step t standardized
  (mean 0, sd 1, ddof = 1) over t = 1..T−1; this keeps the average step energy.

## 5. Replications, seeds, and outputs

- Primary cells (scenarios i and ii × T ∈ {40, 62}): **R = 2000** replications each.
- Secondary null cells (i-b, i-c, i-d × T ∈ {40, 62}): R = 1000 each.
- Master seed **2026111401**, numpy `SeedSequence(2026111401).spawn(...)` one child per
  (scenario, T) cell in the fixed order i/40, i/62, ii/40, ii/62, i-b/40, i-b/62, i-c/40, i-c/62, i-d/40, i-d/62.
  Within a replication, the same sign draws are used for all procedures.
- Outputs: rejection rate per procedure per cell, with a Clopper-Pearson 95% interval; the market
  step-energy share actually realized; for scenario ii only, the retained drift fraction and the mean
  residual net displacement along u. Raw per-replication p-values are
  saved as CSV in the results folder.

## 6. Pass/fail criteria (decided now)

Rejection rate = share of replications with p ≤ 0.05.

1. **Size (P1, scenario i):** observed rejection rate in **[0.030, 0.070]** at T = 40 **and** at T = 62.
   (With R = 2000 the Monte Carlo SE at 0.05 is 0.0049; the band is about ±4 SE.) Passing means the observed
   rate is within this pre-set tolerance, not that the true size is proven to be 0.05; the Clopper-Pearson
   interval is reported beside it.
2. **Power (P1, scenario ii):** observed rejection rate **≥ 0.80 at T = 62** (interval reported). At T = 40 the rate is reported with its
   interval and no threshold.
3. **Extension 2 passes** if and only if criteria 1 and 2 both hold for P1.
4. **Level procedure L** is judged by the same two criteria. Our prediction, from the scout run, is that L
   fails criterion 2 (it hides the walk). L "passes" only if it meets both criteria. If L passes,
   we report that the scout's diagnosis was not reproduced on this design. In every case results are
   described as each procedure meeting or missing its criteria; no superiority claim is made beyond the
   paired comparison in item 7.
5. **Secondary nulls (P1):** i-c and i-d use the same band [0.030, 0.070]; i-b only requires ≤ 0.070.
   A secondary miss is reported prominently but does not change the primary verdict.
6. **P1-naive, W and Mk** are reported for all cells without pass/fail.
7. **Paired comparison (descriptive, scenario ii, T = 62):** with the same replications and sign draws,
   the counts of replications where P1 rejects and L does not, and vice versa, with an exact McNemar
   p-value. A statement that P1 has more power than L is made only if P1 meets criterion 1 and this
   McNemar p < 0.05.

No forecast metric is computed or reported. Effect size a, noise σ, market share 0.35, B, R, the seed, and
the band are fixed here and will not be tuned after seeing any result. If a coding error is found after the
run, the fix and a full re-run are reported as an addendum, with both results shown.

## 7. What a result does and does not mean

A pass means the increment split's sign-flip test holds its size when all directed motion is market-driven
and finds a planted orthogonal swap of the scout panel's size at T = 62 most of the time. It does not
establish that the scout panel's residual walk is real, and it says nothing about forecasting.

## Addenda

### Addendum 1 — 2026-11-14 (ET), before any pre-registered simulation was run

Written after an independent audit (Codex) of the simulation code against this text, and before the
pre-registered configuration was executed. No rejection rate from any run had been seen; only smoke runs
with a development seed and at most 10 replications were executed, and their rates were not printed.

1. **Statistic implementation.** §3 names `ecotrajpy.nulls.directionality_coords`. The run uses
   `ecotrajpy.fast.directionality_coords_fast` / the numba null kernel instead, because the reference
   implementation is about 86× slower and the pre-registered run would not finish in a session. The two
   compute the same formula with the same triangle arithmetic; M2's tests show agreement to ≤ 1e-12 on
   random paths (T = 3..70) and exact p-values on the R sign-flip fixtures. The comparison rule is
   unchanged and literal: count = #{null ≥ obs}, reject iff count ≤ 9.
2. **Zero-length legs.** §3 says θ := 0 when a leg has zero length. The implementation (like ecotraj)
   instead drops such a triple from both sums. Under every pre-registered DGP this is a probability-zero
   event; the run treats any non-finite statistic as an invalid replication (never a rejection) and the
   number of invalid replications is reported per cell and procedure.
3. **Retained-drift and displacement outputs** are recorded for scenario ii only, as §5 says.

Nothing else changes: DGPs, constants, seed, B, R, procedures, and pass/fail criteria are as originally written.
