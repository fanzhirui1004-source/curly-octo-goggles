# Stage 2b verification: T3 accuracy evidence and evaluation independence

Verifier: adversarial check of I-16, I-17, I-18, I-24, I-30, I-39, I-42 and I-48 (CONSOLIDATED.md).
Method: read-only inspection of the manuscript, the evidence JSONs and the archived evaluation code. Scratch script: `/tmp/claude-0/.../scratchpad/t3.py`. It recomputes the population statistics from `evidence/newval2_*.json` and `evidence/valmeta.json`. Nothing in the manuscript, evidence or code was changed.

Line numbers refer to the current HEAD (`693de8e`). "MS" means MANUSCRIPT_EN.md, "SUPP" means SUPPLEMENTARY_EN.md and "APP" means APPENDICES_EN.md.

## Summary

| Issue | Verdict | Action class | Priority | Cost (one RTX 5090) |
|---|---|---|---|---|
| I-16 "80 unseen" | PARTLY. The disclosure and the 60-geometry number already exist (MS 357, 379). The overclaim is in three places (MS 5, 387, 566). "Two checkpoints per predictor" is wrong for B. The cut strata are worse on the 60. | (a) text, plus (b) a trivial table (numbers below) | must | 0 |
| I-17 pairs on development cells | CONFIRMED for the pair tests. Every pair cell is in the 20-geometry selection subset. None of A3's 13 worst geometries was assembled. The H2/y pair was run and failed for every arm, and this is not disclosed. REFUTED for the lattices: the 16 lattice cells are newly generated geometries. | (c) plus (a) | must if the Abstract claim stays; otherwise should | about 1–2 GPU-h plus under 1 CPU-h for neighbour packets |
| I-18 worst direction | CONFIRMED. Per-direction errors for the 80 geometries are not archived: `eval_views.py` stores only `e.mean()`. Per-direction max and p90 exist only for 6 detailed cells. | (b) rerun with per-direction output. Optional: Lanczos ε\* | should (text: must) | about 1.5 GPU-h for three arms; Lanczos 2–5 h optional |
| I-24 CutFEM reference | CONFIRMED and strengthened. On H1, the successive compliance increments do not decrease (n=40→48 is larger than 32→40), so no observed order exists and "of order one percent" is a lower estimate. | (c) small, plus (a) | must (refinement plus wording); should (one body-fitted check) | CPU hours; body-fitted: 1–3 person-days |
| I-30 lattices | PARTLY. The cells are fresh, and 3 iid Gaussian load vectors already probe non-uniform loading. Still true: only 8 cells, τ ∈ [0.2516, 0.5350], no sliver, no figure, no bending case, 3×3×1 layout not archived. | (c) plus a figure | should (merge with §6.11) | about 1–3 GPU-h plus layout work |
| I-39 δ²κ | CONFIRMED (text). §6.4, ST19 and `p1_checks.py` use a Jacobi-weighted interior norm with a mixed denominator. B.1 and J.7 state the full-field Euclidean norm. | (a) | should | 0 |
| I-42 surrogate vs discretisation | PARTLY. MS 371 already says the comparisons measure the discrete operator. Disagree with the matched-accuracy cost point. The "order of magnitude" gap holds for energy but not for sensitivity. | (a) plus (d) | should | 0 |
| I-48 cut-surface loads | REFUTED as a data gap. A3 and B+W cut-traction results are in the gate JSONs, and the rows are given below. ST06 is also selectively incomplete: C on L1/y reaches 3.70% and A2b on M2/y 3.92%, both unlisted. | (b) trivial table | should | 0 |

---

## I-16: "80 unseen geometries"

**How many geometries entered selection.** Exactly 20 of the 80 (`valmeta.json`, `sel=true`): `fresh_val_2000`–`2019`. By stratum this is 6 uncut, 4 light, 5 moderate and 5 heavy, so 14 cut. All eight detailed cells U1, U2, M1, M2, H1, H2, H3 and L1 (`fresh_val_2000`–`2006` and `2010`; key in SUPP 19–26) are inside this subset.

**Where selection is described.**
- MS 357: "selected on a validation list whose first 40 geometries include 20 of the 80 … selection compared two checkpoints per predictor."
- SUPP ST12 (384–393): the score J_sel, views 0 and 17, eight classes. "B is evaluated every 10,000 updates and becomes eligible at update 10,000; C, S8 and A3 are evaluated every 7,500 … A2b is evaluated … at update 15,000."

**Attempted refutation.**
1. The disclosure and the 60-geometry number are already in MS 357 and MS 379 ("0.077% against 0.074% … 0.0591% against 0.0579%"). I reproduced both: 0.0767% vs 0.0737% (force_c) and 0.0591% vs 0.0579% (force). The concern is therefore not an undisclosed leak. It is the unqualified word "unseen" in the Abstract (MS 5), the Fig. 5 title (MS 387) and §8 (MS 566).
2. For every continued predictor the selected checkpoint is the final step. `ckpt_step` is 15000 in newval2_A0_ctrl, A2_tail8, A2b_tail8 and A3_2grid. Selection between steps 7,500 and 15,000 therefore had no effect relative to "take the last checkpoint". This is a strong mitigating fact that the paper does not state.
3. The selection did matter for B: step 30,000 was chosen from four eligible checkpoints (10k, 20k, 30k, 40k). A3, C, S8, A2b and B+W all inherit B's selected weights. So "two checkpoints per predictor" (MS 357) is wrong for B, and the influence on A3 is indirect.
4. The overall mean is unchanged, but the cut strata are worse on the 60-geometry subset. The subset used for selection and development is the easier one.

A3, force_c, geometry-mean energy excess (%), selection subset vs held-out subset:

| Stratum | 20 selected (n) mean / max | 60 held-out (n) mean / max | Reported (all 80) |
|---|---|---|---|
| Uncut | (6) 0.0081 / 0.016 | (14) 0.0200 / 0.062 | 0.0164 |
| Light | (4) 0.176 / 0.278 | (16) 0.054 / 0.287 | 0.0786 |
| Moderate | (5) 0.051 / 0.127 | (15) 0.105 / **0.651** | 0.091 |
| Heavy | (5) 0.059 / 0.174 | (15) **0.126** / 0.540 | 0.109 |
| All | (20) 0.065 / 0.278 | (60) **0.0767** / 0.651 | 0.0737 |

Other held-out (60-geometry) numbers for A3:
- Median 0.029%, p90 0.213%, p95 0.312%, max 0.651%.
- Bootstrap 95% confidence interval of the mean: all 80 [0.050, 0.102]%; 60 held-out [0.047, 0.112]%.
- The five worst A3 geometries are all held out: 2051 (moderate, 0.651%), 2045 (heavy, 0.540%), 2074 (heavy, 0.488%), 2021 (heavy, 0.303%) and 2063 (light, 0.287%).

B+W, all 80 vs 60 held-out: 0.0965% vs 0.101%; max 0.905% (2045, held out).

The full 60-geometry tables for all arms and classes are already in `evidence/RESULTS_TABLES.md` lines 71ff. They are not in the SUPP.

A remaining, milder dependency: A3 was designated the principal predictor after comparing arms on all 80 geometries and on the pairs. This is arm-level selection on the full set. Mention it in one clause.

**Verdict: PARTLY confirmed.** The substance is already disclosed. The wording overclaims in three places, and one factual statement ("two checkpoints per predictor") is wrong.

**Action (a), replacement wording.**
- Abstract (MS 5): replace "on 80 unseen geometries, with at most 0.65% for any geometry" with
  "on 80 validation geometries not used in training (0.077% on the 60 of them that also played no part in weight selection); the largest geometry mean is 0.65%."
- Fig. 5 title (MS 387): "Directional energy error of the learned substructures on the 80 validation geometries." Add to the caption: "Twenty of them (6 uncut, 14 cut) entered weight selection; statistics on the other 60 are given in Table ST01d."
- §8 (MS 566): replace "keeps the mean energy error of every one of 80 unseen geometries at or below 0.65%" with
  "keeps the geometry-mean energy error under consistent tractions at or below 0.65% on all 80 validation geometries, the largest occurring on a geometry that entered neither training nor weight selection."
- MS 357: replace "selection compared two checkpoints per predictor" with
  "each continued predictor was selected between steps 7,500 and 15,000 and in every case the final step was retained; B's weights, from which all continued predictors start, were selected among four checkpoints (steps 10,000–40,000) on the same list."
- MS 379: add one clause. "On the 60 geometries outside the selection list, the cut-stratum means are higher (0.105% moderate, 0.126% heavy) and the five largest geometry means all occur."
- (b) Add Table ST01d, the 60-geometry version of ST01. It is already generated in RESULTS_TABLES.md.

---

## I-17: Assembled accuracy rests on development cells

**Pair cells.** The pairs use U1, U2, M1, H1, M2, H3 and L1 (`gate_*_fresh_val_{2000,2001,2002,2003,2004,2005,2006}_{x,y}.json`), which gives the 14 A3 configurations. All of these are `sel=true`. MS 357 already says they "served repeatedly as development cases".

**Where the pair cells rank in A3's force_c distribution (rank 1 = worst of 80).** M1 is 14, M2 33, H3 48, L1 51, U1 52, H2 57, H1 62 and U2 73. None of the 13 hardest geometries was assembled. The development cells were chosen to be hard for B, not for A3.

**New finding: the H2/y pair was run and dropped silently.**
- `gate_{A0_ctrl,A2_tail8,A2b_tail8,A3_2grid}_fresh_val_2010_d0_v0_y.json` exist.
- For every arm, including C, they show NaN compliance on the three neighbour-face loads. PCG hit the 400-iteration cap, and the energy participations are negative (`w` = −0.15 and −0.21).
- `evidence/RESULTS_TABLES.md:155` lists it as "fail" for C, S8 and A3.
- MS 444 and SUPP 471 speak of "all fourteen configurations evaluated" and do not mention it.
- The pattern points to an ill-posed pair: the 2.7% retained sliver is presumably disconnected from, or singular with, the clamped neighbour (I-40). It is not a predictor failure, but it is an evaluated configuration that was excluded without comment. H2/x has no file.

**Lattice cells (partial refutation).**
- `hlat221a.json` and `hlat221b.json` (the z-layers of hlat222) list cases `hlat222_000`…`hlat222_111`, generated by `gen_hlat.py` (seed 0, θ = 20°, graded field) in stage S4.
- `lat_hetero331_A3.json` lists `hlat331_*`.
- The `template` argument (`fresh_val_2003_d1_v1`) appears to supply the discretisation template, not the geometry. The τ corners differ.
- These cells are therefore newly generated geometries outside training, validation and selection. The main text never says so, and it should (after the authors confirm).

**Can the pair results be extrapolated to the worst geometries? No.**
- U1 has a geometry mean of 0.016% and gives the largest pair sensitivity error, 0.67%.
- M1 has 0.127% and gives only 0.16–0.21%.
- Sensitivity error is governed by participation and derivative coupling (§6.7), not by the geometry mean. This is the strongest argument that only new runs can answer R4-04.

**Could fresh held-out pairs be run cheaply? Yes.**
- `lat_full.py --custom <ckpt>:<case>:: --nb-mode explicit` runs the pair protocol for any case that has neighbour packets `<case>_nbmx` and `<case>_nbmy`. These come from `gen_new.py` (CPU geometry pipeline).
- Archived A3 run times are 14–160 s per configuration: 13.6 s for H1, 141 s for L1 with its host factor.

**Verdict: CONFIRMED for pairs, REFUTED for lattice provenance.**

**Action (c), specification.**
- Held-out cells:
  - the 5 worst held-out A3 geometries: 2051, 2045, 2074, 2021, 2063;
  - one randomly drawn held-out geometry per stratum (4, fixed seed, drawn before any run);
  - that is 9 cells in configurations x and y, giving 18 configurations.
- Arms: A3 and B+W, with C optional.
- Loads: six face loads plus three cut loads.
- Report every outcome, including failures and ill-posed configurations, as a distribution.
- Existing checkpoints suffice. Neighbour packets must be generated (18 × `gen_new.py`, well under 1 CPU-h with 16 processes).
- GPU: 36–54 runs × 0.3–2.5 min ≈ 1–2 h.
- Priority: must while the Abstract keeps "below 0.7% in all fourteen configurations". If it is reworded (below), it becomes should.

**Action (a).**
- Abstract (MS 5): "In two-cell assemblies of seven development cells (fourteen configurations) with an exact continuous-thickness neighbour, it keeps …". If the new runs are done, add "and below X% on eighteen configurations of nine held-out cells, including the five with the largest single-cell errors".
- MS 444 / SUPP 471: add "A fifteenth configuration, H2/y, is ill-posed: the retained sliver … [authors state the reason]; the reference solve does not converge and it is excluded for all predictors."
- §6.9 (MS 484): add "The sixteen lattice cells were generated for these examples and entered neither training nor weight selection." Confirm this first.

---

## I-18: Worst-direction error

**Is per-direction data available? No, for the 80-geometry set.**
- `src_v2_wip/eval_views.py` computes the per-direction vector `e` but stores only `r[k][c] = float(e.mean())`. `newval2_*.json` therefore holds one scalar per geometry and class.
- The number of directions per class and geometry is not stored.
- Only view 0 was evaluated for A3. ST01c shows that for B, view 17 raises force_c by 13% (6.62 → 7.49 on 20 geometries), so view dependence is another dimension the 0.65% figure does not cover.

**Available per-direction evidence: `p1_checks_cpu.json` and `p1_checks_u2.json`** (6 detailed cells, 32 validation directions). Values are A3 mean / p90 / max, in %.

| Cell | force_c | max/mean | force | max/mean |
|---|---|---|---|---|
| U2 | 0.0046 / 0.0051 / 0.0055 | 1.2 | 0.0070 / 0.0080 / 0.0108 | 1.5 |
| M1 | 0.131 / 0.198 / 0.228 | 1.7 | 0.084 / 0.121 / 0.218 | 2.6 |
| M2 | 0.036 / 0.055 / 0.063 | 1.8 | 0.040 / 0.073 / 0.081 | 2.0 |
| H1 | 0.0115 / 0.0171 / 0.0193 | 1.7 | 0.0187 / 0.0372 / 0.0478 | 2.6 |
| H2 | 0.0148 / 0.0240 / 0.0295 | 2.0 | 0.0145 / 0.0200 / 0.0368 | 2.5 |

The worst sampled direction is 1.2–2.6 times the geometry mean. By analogy, the worst sampled consistent-traction direction on geometry 2051 is plausibly about 1.1–1.7%. This is an extrapolation, not a measurement, and all these maxima are still lower bounds on ε\*.

**Distribution of geometry means (computed; A3, all 80).** Median, p90, p95 and max, in %:

| Class | Median | p90 | p95 | Max | Worst geometry |
|---|---|---|---|---|---|
| force_c | 0.028 | 0.213 | 0.288 | 0.651 | 2051 |
| force | 0.033 | 0.117 | 0.182 | 0.326 | |
| glued | 0.021 | 0.172 | 0.239 | 0.384 | |
| support_k | 0.020 | 0.169 | 0.204 | 0.468 | |
| face_c | 0.014 | 0.078 | 0.115 | 0.210 | |
| face | 0.013 | 0.056 | 0.067 | **0.724** | 2021 |

**Is "at most 0.65% for any geometry" correct, and what does it bound?**
- The value is 0.6507% (geometry 2051, moderate cut, held out). "At most 0.65%" is a rounding-down (R6-20).
- It bounds only the mean over the sampled consistent-traction directions of one geometry, in the identity view.
- It bounds none of the following: individual directions, ε\*, other loading classes (the `face` class reaches 0.724% on 2021, ST01), rotated views, assembled responses, or sensitivities.
- §8 (MS 566) drops the class qualifier ("every one of 80 … at or below 0.65%"). Read literally, this is false for the `face` class reported in ST01.

**Verdict: CONFIRMED.**

**Action (b).**
- Code change: in `eval_views.py`, store `n=len(e)`, `max`, `p95` and `p99` alongside the mean. This is three lines.
- Rerun A3, B+W and C on all 80 geometries, view 0. The archived A3 run took 1,636 s, so the three arms take about 1.5 GPU-h in total.
- Report the per-class p95, p99 and max of directional errors, and the direction counts.
- Priority: should. It is cheap, and it removes a Major item.
- Optional, but recommended for the heavy-cut 20 plus the 10 worst: estimate λ_max(S⁻¹Ŝ) − 1 by LOBPCG or Lanczos on the rigid complement, 50–100 iterations, applying S⁻¹ with the existing exact Neumann solve (App. G.3). Roughly 1–5 min per geometry per arm, so 2–5 GPU-h.

**Action (a).** In the Abstract: "…with a largest geometry-mean error of 0.65% (mean over the sampled directions of one geometry; individual sampled directions reach about twice the geometry mean, Table STxx)." In §8, add "under consistent tractions".

---

## I-24: Verification of the CutFEM reference

**What exists.** `ref_valid.json` covers M1, M2 and U1 at n = 24, 32, 40, 48 (U1 only to 40). H1 failed there (all NaN) and was redone in `ref_valid_h1.json` with clamp x=0 and load y=0. Both files include tests of γ (10⁻⁵ to 10⁻³), integration refinement and finite-difference step. Their `observed_order` field is "unequal ratios: see vs_finest", i.e. no order was computed.

**Computed convergence behaviour (compliance per load direction).**
- **H1**, y-load: 170.87, 172.82, 173.49, 174.55. The increments are +1.95, +0.67, +1.07, so the 40→48 increment exceeds the 32→40 one. Every direction increases monotonically without clear decay. A three-level fit gives no positive order (ratio 0.63–0.83 with h ratios of 0.8), so Richardson extrapolation is meaningless. The n=48 value is not converged, which makes "difference from finest = 0.99%" a lower estimate of the n=32 error.
- **M1**: the increments change sign (+0.26, +0.50, −0.16 for x), so the behaviour is non-monotone and has no asymptotic regime.
- **M2**: mostly non-monotone.
- **U1**: consistent with a high order (p ≈ 3.3 for x and z). The n=32 error vs the extrapolated value is about 0.05%.
- **Sensitivities vs finest level at n=32**: M1 ≤ 0.087%, M2 ≤ 0.068%, U1 ≤ 0.107%, H1 ≤ 0.971%. The text's claim that sensitivities "converge at the same rate" (MS 371) is not supported, because no rate exists.

**Penalty and text facts.**
- The ghost share at γ = 10⁻³ is 1.2×10⁻⁴ to 5.6×10⁻⁴ (U1 dir 1: 5.58×10⁻⁴). R6-21 is correct.
- The γ variation changes compliance by ≤ 0.053% (M1 γ=10⁻³: 0.039%; U1: 0.053%).
- Changing the integration rule changes compliance by ≤ 0.010%.
- The H1 worst case is the y-load on face y=0, i.e. normal to the loaded face. R6-22 is correct: "in-plane load" is wrong.

**Not covered by the existing check:**
- H2 (2.7% retained);
- the thinnest-wall geometry (corner τ near 0.176);
- A3's worst geometries (2051, 2045);
- any body-fitted reference;
- the τ_c of the detailed cells, which is not reported.

**Mitigation that limits the severity.** All NICE claims are relative to the same discrete model, and MS 371 says so explicitly. Reference verification therefore bears on the physical credibility of the model, not on NICE's accuracy.

**Verdict: CONFIRMED, and for H1 worse than stated.**

**Minimal proportionate additions for CMAME.**
1. (must, CPU only) Extend `ref_valid` to n = 56 and 64 for H1, and add H2, 2051 and the minimum-τ validation cell at n = 24–64. Host PARDISO handles it: the largest case is about 1.3M DOFs at n=64 for a dense cell, minutes per solve, tens of GB of RAM. Report the observed increments honestly. If H1 still does not settle, say that the heavy-cut reference error at n=32 is "at least about 1%".
2. (should) One body-fitted P2-tetrahedral comparison for compliance only, on U1 and H1. Use an implicit-surface mesher (gmsh/CGAL on the band plus the cut) and any standard FE code. Cost: 1–3 person-days and negligible compute. This is the most convincing independent check and the one reviewers ask for.
3. (optional) cond(D⁻¹A) or λ_min over a sweep of cut positions for γ ∈ [10⁻⁵, 10⁻¹] on H1.

**Action (a), MS 371.**
- Replace "is thus of order one percent" with "is at least of order one percent: its successive increments do not yet decrease between n = 40 and 48, so the difference from n = 48 is a lower estimate".
- Delete "The eight-corner sensitivities converge at the same rate;" and replace it with "The eight-corner sensitivities behave similarly;".
- Replace "about 4×10⁻⁴" with "1.2×10⁻⁴ to 5.6×10⁻⁴".
- Replace "the in-plane load that bends the remaining wall segments" with "the load normal to the loaded face".
- Move the modular-reference statement (APP 566) into §2.
- Fix the Fig. S06 axis label ("background elements per axis").

---

## I-30: Lattice examples

**Refutable parts.**
1. Provenance: the cells are fresh (see I-17).
2. The runs already include three iid Gaussian load vectors on the retained DOFs (`lat_multi.py` line 12: "n_random iid Gaussian"). These excite every cell and give sensitivity errors of 0.45% (2×2×2) and 0.43% (3×3×1) against 0.14% under face loads. This is a partial answer to "near-uniform participation".

**Confirmed parts.**
- τ range: the 2×2×2 range computed from `hlat221a/b` is 0.2516–0.5350, against a validation range of 0.176–0.698. MS 484's "0.25 to 0.56" is not reproducible from the archive. R6-19 is correct.
- The 3×3×1 layout (hlat331) is not archived. Only the case names appear, in `lat_hetero331_A3.json`, `learned_hlat331.json` and `lat_direct_hlat331.json`.
- Retained volumes are 62% and 25%, with no sliver.
- There is no figure, only eight cells, and no bending-dominated or localised load.

**Verdict: PARTLY.**

**Action (c).**
- One lattice with at least 27 cells in bending: a cantilever clamped on one face, with a localised patch load at the far edge.
- τ spanning [0.18, 0.69] with steep gradients, cut cells from every stratum including a retained volume below 5%, all cells A3, compared with the exact per-cell reference (the `lat_hetero.py` route).
- Cost estimate: exact reference 8 cells took 60–86 s; learned run 110 s per iteration for 8 cells. For 27 cells, about 1–3 GPU-h including cell setup, plus the time to write the layout.
- Merge this with §6.11.
- Priority: should. The figure of the existing lattices and archiving the hlat331 layout are must, at no cost.

---

## I-39: δ²κ definitions

**Evidence.**
- MS 405 and `src_v2_wip/p1_checks.py:77–86`, which produced ST19 and the §6.4 numbers, both use δ² = d_Iᵀ D d_I / u_Iᵀ D u_I and κ = (dᵀKd / d_Iᵀ D d_I) / (uᵀKu / u_Iᵀ D u_I). This is a Jacobi-weighted interior norm with a mixed denominator.
- APP B.1 (123) states "We use the full-field Euclidean norm, W = I", with B.7 requiring W ≻ 0.
- APP J.7 (819) states that the identity "uses the same full-field displacement norm for both Rayleigh quotients".
- Both appendix statements contradict what was computed. The §6.4 choice corresponds to W = diag(0_P, D), which is only positive semidefinite. The identity still holds because d_P = 0.
- The identity is tautological for any W (verified: `eps_check` in `p1_checks`). The observed magnitude of κ is informative, but the "bending" attribution is not measured.

**Verdict: CONFIRMED (text only).**

**Action (a).**
- B.1: after B.8, add "Section 6.4 and Table ST19 use W = diag(0, D) with D = diag(K_II), for which the identity still holds because d vanishes on the retained coordinates; the rigid gauge is then not needed."
- Delete "We use the full-field Euclidean norm, W = I_{n_a}."
- J.7 (819): replace "uses the same full-field displacement norm for both Rayleigh quotients" with "uses the same displacement weighting in both Rayleigh quotients".
- MS 405: replace "which in these thin-walled cells deforms mainly through soft bending of the walls" with "(an interpretation consistent with, but not established by, the spectral locations of Section 6.4)". Also add "This identity is a diagnostic decomposition, not an independent explanation."

---

## I-42: Surrogate error vs discretisation error

**Refutation.** MS 371 already states that "the comparisons in this section measure the approximation of the discrete operator, not of the continuum". The matched-accuracy cost point proposed by R7-15 would compare a different discrete design problem. A conventional route at n=24 changes the model, and NICE itself would need retraining at n=24 to be compared at equal continuum accuracy. Design sensitivities must also be consistent with the discrete compliance being optimised.

**Nuance to add.** The order-of-magnitude gap holds for energy (A3 geometry means ≤ 0.65% against a reference error of about 1% or more for heavy cuts). It does not hold for sensitivity:
- A3's pair sensitivity errors reach 0.67%.
- The reference sensitivity changes between n=32 and the finest level are 0.07–0.11% (M1, M2, U1) and 0.97% (H1).

**Verdict: PARTLY.**

**Action (a) plus (d).** Add after "not of the continuum" in MS 371, or in §7.4:
"Both routes are compared on this same discrete model, which also defines the design problem; a coarser reference would change the model rather than the cost of reaching it. The surrogate's energy error is well below the reference's discretisation error in heavily cut cells, whereas its largest assembled sensitivity errors are comparable to the reference's sensitivity change under refinement."

No experiment is needed.

---

## I-48: Cut-surface loads for A3 and B+W

**Refutation of the data gap.** The existing gate JSONs contain `cut_compliance_max` and `cut_sens_max` for A3 and B+W. Maximum over the three cut tractions, compliance / sensitivity (%):

| Cell / config | A3 | B+W |
|---|---|---|
| M1/x | 0.035 / 0.161 | 0.049 / 0.217 |
| M1/y | **0.101** / 0.158 | 0.124 / **0.305** |
| M2/x | 0.0060 / 0.072 | 0.0047 / 0.053 |
| M2/y | 0.027 / 0.154 | 0.020 / 0.091 |
| H1/x | 0.0034 / 0.061 | 0.0044 / 0.096 |
| H1/y | 0.0059 / 0.126 | 0.0071 / 0.022 |
| H3/x | 0.0031 / 0.125 | 0.0039 / 0.038 |
| H3/y | 0.0098 / 0.129 | 0.011 / 0.042 |
| L1/x | 0.0044 / 0.087 | 0.0048 / 0.039 |
| L1/y | 0.0105 / 0.071 | 0.011 / 0.069 |

**Additional findings.**
1. ST06 (SUPP 260–279) omits several recorded rows:
   - C on L1/x: 0.35 / 1.66;
   - C on L1/y: 1.14 / **3.70**, above the 3% reference;
   - every A2b row, including M1/y 2.42 / 5.19 and M2/y 1.12 / **3.92**;
   - A3 and B+W.

   MS 444 says C "meets the reference" on L1. That is true for the face loads only.
2. Under cut tractions, A3's compliance error reaches 0.10% (M1/y), above the "below 0.06%" in the Abstract. That figure applies to face loads only, which the Abstract should say or which should be changed to 0.11%.

**Verdict: REFUTED as a missing-data concern. CONFIRMED as a reporting gap.**

**Action (b).**
- Complete ST06 with all recorded arm/configuration rows. The data are in `evidence/gate_*_fresh_val_*.json`; SUPP 471 already distinguishes the pair counts.
- Add A3 and B+W markers to Fig. 10, or state that Fig. 10 is B/C/S8 only.
- A cut-surface load in one lattice is optional (c): minutes of GPU time.

---

## Top three decisions for the authors

1. **Abstract assembly claim (I-17, plus the H2/y omission).** Either run the pair protocol on about 9 held-out cells, including the 5 worst A3 geometries (about 2 GPU-h plus neighbour packets), or reword the claim to "seven development cells". In either case, disclose the failed or ill-posed H2/y configuration.
2. **How to state single-cell accuracy (I-16 and I-18).** Replace "unseen" and "at most 0.65% for any geometry" with "not used in training", the 60-geometry figure, and "largest geometry mean". Decide whether to spend about 1.5 GPU-h to store per-direction p95/p99/max, which the current archive cannot supply. Correct "two checkpoints per predictor", since B chose among four.
3. **Reference verification scope (I-24).** H1 shows no convergence regime up to n=48. Decide between CPU-only refinement to n=64 on H1, H2 and the worst geometry plus honest wording (minimum), and adding one body-fitted P2 check (1–3 person-days), which is what a CMAME reviewer is most likely to accept.
