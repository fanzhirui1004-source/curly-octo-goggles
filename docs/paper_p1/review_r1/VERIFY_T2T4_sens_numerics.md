# Stage 2b verification: T2 (sensitivity / design derivative) and T4 (numerics / precision / solver)

**Scope:** I-03, I-08, I-25, I-31, I-50 (T2); I-04, I-20, I-23, I-38 (T4).
**Method:** For each issue I tried to refute the reviewers' concern using the manuscript, the appendices, the code mirror (`docs/data/newmachine_20260924/src_v2_wip/`), the evidence folder (`docs/paper_p1/evidence/`) and earlier project records under `docs/data/` and `docs/*.md`. The verdict is "confirmed" only where no refutation was found.
**Constraints:** No manuscript, evidence or code file was changed. There was no server access; all cost estimates are my own and assume one RTX 5090 plus the existing checkpoint `A3_2grid/best.pt`, the bodies, the banks and the layouts on the server.

---

## 0. Summary

| Issue | Verdict | Action class | Priority | Rough cost (1×RTX 5090) |
|---|---|---|---|---|
| I-03 complete surrogate derivative never evaluated | **CONFIRMED** (code computes only s̃_c) | (b)+(a): new computation on existing checkpoints and configurations | **must** | 3–6 h GPU (fixed-q̂ central differences), plus about 1 day of scripting |
| I-08 sensitivity metric not what the optimiser sees | **CONFIRMED** | (b): needs a re-run that dumps vectors, because the per-cell s-vectors are not saved | **must** | about 2 h GPU (shared with the I-04 re-run bundle) |
| I-25 FD-step statement; PSD of numerical K,c | **CONFIRMED** (text); PSD **partly refuted** (an unarchived element-level check exists) | (a), plus small optional (b) | **must** (text), should (PSD rerun) | minutes |
| I-31 non-smoothness across discrete switches | **CONFIRMED** in substance; the jump that the surrogate *adds* is bounded by the error level (partial rebuttal) | (a)+(c) small | should (sweep); **must** (text and switch logging in §6.11) | 2–6 h GPU for one or two 1-D sweeps |
| I-50 design-dependent loads frozen | **CONFIRMED** (already disclosed in App. H) | (a) | **must** once §6.11 exists | none |
| I-04 finite-precision residual floor | **PARTLY**: floor and missing ρ records confirmed; existing data indicate that the headline numbers are not contaminated at the reported digits; one precision statement is wrong | (a) must; (b) instrumented re-run should; (c) fp64-network pair optional | **must** (text), should (re-run) | 2–3 h GPU re-run; fp64 pair about 0.5–1 h; fp64 lattice 5–20 h (optional) |
| I-20 no runtime error control / exploitation | **CONFIRMED** (no deployed indicator); partly mitigated (a label-free lower-bound certificate is implemented and timed, and the exploitation gain is bounded by β) | (a) must; (b) certificate effectivity should | should | about 1 h GPU |
| I-23 two-grid bound only (near) non-expansion | **CONFIRMED** (ρ₈⁴ ≈ 0.980–0.986) | (a) must; (b) ‖T‖_A by Lanczos plus multi-cycle, should | **must** (wording) | under 1 h GPU (all 80 cells) |
| I-38 Chebyshev interval and coarse-solve verification | **PARTLY REFUTED**: the GPU endpoint *is* verified on all 80 geometries, and the §6.2 sentence is the inaccurate one; probe = Galerkin is already checked on 3 cells | (a) must; (b) small, should | must (text) | under 30 min |

### Top 3 decisions for the authors

1. **Which gradient drives §6.11, and how it is verified (I-03/I-08).** The code computes only the field-based estimate s̃_c. The "reverse-mode sensitivities" of S7 are reverse mode through the *moment integrals* at fixed fields, not through the network or the correction. The authors should decide whether (i) to keep s̃_c and state explicitly that it is an estimate of the exact C,c that is inconsistent with Ĉ, or (ii) to implement Ĉ,c. In both cases they should run the cheap fixed-q̂ central-difference check (§1.1 below) on all 14 pairs and both lattices before §6.11 is run. The exact identity derived in §1.1 shows that the answer depends on ‖H_{,c}q‖_A versus ‖E_{I,c}q‖_A, and nothing in the paper controls that ratio.
2. **Algebraic accuracy of the assembled solves (I-04).** The authors must decide whether to (a) run the cheap instrumented re-run, which computes Ūᵀρ, J(Ū), a rigorous dual-norm bound via the exact dense lattice operator, and dumps vectors for I-08/I-03; and (b) additionally run one fp64-network pair to substantiate "set by the network's single precision". Independently of that choice, Table 1, §6.2 and S7 must be corrected: the timed deployment route of Table 6 runs the correction and coarse solve in **fp32** (`lat_scale.py:95`, `teacher.py:398,486`), while the accuracy runs use fp64. An existing streamed-fp32 rerun of the 2×2×2 block (`docs/data/newmachine_20260924/p2/lat_hetero222_stream.json`) reproduces the fp64 errors to about 3×10⁻⁹. It should be moved into `evidence/` and cited.
3. **How strongly to word "controllable" (I-23/I-20).** The theory guarantees SPSD, S ⪯ Ŝ, and non-increase under repeated cycles, but only an energy contraction of at most 1.4–2.0% per 8/Q1/8 cycle. It does *not* guarantee monotonicity in the smoothing degree k, and there is no a-posteriori upper bound. Either the Abstract and contribution (i) are reworded as empirical (wording in §2.3), or a Lanczos ‖T‖_A study and a certificate-effectivity study are added to support a stronger statement.

---

## 1. T2: Sensitivity and design derivative

### 1.1 I-03: The complete derivative of the surrogate objective is never evaluated

**Verdict: CONFIRMED.**

**Code evidence (what is actually computed):**
- `moments_ad.py` docstring (lines 14–18) and `cell_sens` (end of file) compute `s[c,k] = -u_k^T (dK/dtau_c) u_k` for **fixed fields** u (energy density `g = cell.energy_density(u)`, then `moments_vjp` backpropagates only through the moment integration). This is s̃_c of Eq. (13) when u = F q̂.
- `lat_scale.py` (the S7 / Table 6 "deployed" route): `--sens ad` calls `MA.cell_sens(C, g, ...)` with g from u = F q̂ (lines 229–239). `lat_hetero.py` lines 276–291 (the §6.9 route): `u = ops[i].field(...)`, then `sh = C.sens(u)` (central-difference moments, Eq. H.6).
- No autograd path from τ to the network or the correction exists. The only `autograd.grad` calls on inputs are with respect to q (transpose checks in `ops.py`, `t_fast*.py`, `trainlib.py:639`) or network parameters (training). `FastNet` "freezes" the geometry path (`bench_deploy.py` lines 11–13, 136). App. H (last paragraph) already says that training differentiates only the field-based loss.
- Conclusion: every reported sensitivity, including the S7 "reverse-mode sensitivities", is s̃_c. The S7 wording is misleading.

**Existing evidence on complete derivatives (none for NICE/A3):**
- `docs/TAU_DERIVATIVE_PASSES_20260919.md` and `docs/data/tau_derivative_20260919/predict_notf32.json` contain a finite-difference test of the complete derivative of a *predecessor* surrogate (models MULTI_AUGMENT and WIDE_BOX, not NICE). It covers one box cell and one uniform-scaling direction, and finds derivative errors of 1.2–3.7%, comparable to the value errors.
- This shows that the protocol is cheap and workable (bitwise-reproducible predictions with TF32 off). It says nothing about A3.

**Can the concern be refuted analytically? Partly. The debate is sharper than §7.2 or R4-01 state.**

At a fixed retained displacement q, with H = F − E, (KEq)_I = 0, r_I = A(Hq)_I and AE_{I,c} = −J_I K_{,c} E, Eqs. (13), (14) and (H.4) give the following *exact* identities (my derivation, checked term by term):

```
s~_c  - s_c  = +2 (Hq)^T A E_{I,c} q        - (Hq)^T K_{,c} (Hq)
C^_,c - C_,c = -2 (H_{I,c} q)^T A (Hq)      - (Hq)^T K_{,c} (Hq)
```

- The residual term of Eq. (14) removes the linear field-error term exactly and replaces it with one weighted by the design derivative of the *extension error*.
- The quadratic term −ζ_c is common to both expressions.
- Which gradient is more accurate is therefore decided by ‖H_{I,c}q‖_A versus ‖E_{I,c}q‖_A (and the alignment of each with Hq). The energy error does not control this.
- At assembly, the trace change q̂ − q adds the terms of Eq. (H.2) to both expressions.

Consequences for the reviewer arguments:
- **R4-01's "at least about 5–16% per component".** This is the Cauchy–Schwarz bound 2‖E_{I,c}q‖_A√Δ with F_{I,c} ≈ E_{I,c}. The *same* bound applies to the linear part of s̃_c − s_c, whose measured total is ≤ 0.7% (and whose linear share is small, per §7.2). The bound is therefore pessimistic by at least an order of magnitude *for the E_{I,c} part*. It says nothing about the H_{I,c} part, which is unmeasured.
- **§7.2's argument ("about 2.7% of ‖û‖_K")** does not settle the question either. The normalisation uses ‖û‖_K rather than |s_c| (R1-02), B's value is 25.4% rather than 26% (R6-14), and it uses √(mean ε) instead of mean √ε (R1-17d).
- **The internal pre-analysis says the opposite of App. J.5.** `docs/data/newmachine_20260924/arch_analysis_0926/WORKFLOW_RESULT.json` (item "SENSITIVITY ERROR STRUCTURE") recommends "keep the field formula … autodiff of Ĉ adds a FIRST-order term". The two positions are reconciled only by measurement.

**A network with binary node flags cannot be assumed C¹-small in τ.** `models.py:435` uses hard `is_port/is_box/is_cut/weak` flags, and `models.py:431` uses `log(vf+1e-6)/10`. Hence ‖H_{,c}‖ may be large locally.

**Action class:** (b) new computation on existing checkpoints and configurations, plus (a) text.

**Specification (cheap route; no reverse-mode implementation needed):**
1. Use the surrogate's own equilibrium: with a fixed load, Ĉ,c = −Σ_m q̂_mᵀ Ŝ_{m,c} q̂_m at the assembled solution q̂. Evaluate Ŝ_{m,c}q̂_m by **central differences of the per-cell learned operator at fixed q̂_m**. For each affected cell and each corner c, rebuild the cell at τ ± h e_c (moments, assembly, `netdata`, `Geo`, `FastNet`, warm-up application, i.e. the per-cell front end of `lat_scale.py` lines 129–160) and apply it once to q̂_m. This requires no assembled re-solve.
2. Use h ∈ {10⁻⁴, 3×10⁻⁴, 10⁻³}τ_c. Check the Richardson ratio, and check that the retained set, the active-element count and the flag vector are unchanged at ±h. Otherwise report the case as a switch (links I-31).
3. Validate the fixed-q̂ route on 2–3 variables of one pair by full re-solve central differences of Ĉ(τ) (one pair solve takes 15–50 s).
4. Report per configuration, for all 14 pairs (target cell) and both lattices (all 8 cells; also with shared-corner aggregation, I-08):
   - exact C,c;
   - s̃_c;
   - Ĉ,c;
   - the extension term Ĉ,c − s̃_c relative to |C,c| and to ‖∇C‖;
   - per-variable error distributions and sign agreement.
5. Optional diagnostic that explains the result: ‖E_{I,c}q‖_A (one interior solve per corner with the teacher factor) and ‖(F_{I,c} − E_{I,c})q‖_A (central differences of the learned field minus exact field) on the five detailed cells.
6. Scripts to adapt: `lat_scale.py` (per-cell front end at arbitrary τ), `lat_hetero.py` and the pair-gate driver (to obtain q̂, s, s̃), `ref_valid.py` (step study pattern).
7. **Cost:** a per-cell rebuild takes about 3–6 s (`time_setup.json`: moments 0.9 s, assembly 0.8 s, model cache 0.2 s, freeze 0.1 s, plus the correction warm-up). One corner at three step sizes takes about 6 rebuilds, i.e. 20–40 s.
   - 14 pairs × 8 corners: about 1 h.
   - 2 lattices × 8 cells × 8 corners: about 1.5 h.
   - Re-solving to obtain q̂ (shared with I-04/I-08): about 2 h.
   - Total: 3–6 GPU-h plus about 1 day of scripting.
   - Existing checkpoints and data suffice.
8. **Full reverse mode** (optional, for §6.11 if Ĉ,c is chosen):
   - The τ-differentiable pieces exist: `moments_ad` for the moments; the features vf, log vf, M/vol and log(d3/median) are differentiable given moments; the Galerkin probe and Cholesky are torch ops.
   - The binary flags are non-differentiable and would get zero derivative.
   - The Chebyshev b obtained from 40 power steps is differentiable but noisy.
   - `FastNet` would have to be replaced by the autograd model.
   - Estimate: several days of engineering. It is not needed for the verification above.

**What §6.11 needs (answer to question 1):**
- **(i)** An explicit statement of which gradient drives the optimiser (s̃ or Ĉ,c), and why.
- **(ii)** Verification of that gradient at the initial, an intermediate and the final design against the exact C,c (exact condensation plus `C.sens`) *and* against fixed-q̂ central differences of Ĉ. Report the global relative error, cosine, per-variable median/95th percentile/max and sign agreement (M1 of CONSOLIDATED).
- **(iii)** The load model (I-50).
- **(iv)** A log of discrete switches per iteration (I-31).
- **(v)** Surrogate compliance error at the start and at the optimum, with the per-cell certificate (I-20).
- **(vi)** The recomputed residual and Ūᵀρ per iteration, and the stopping rule (I-04, M6).
- **(vii)** If MMA/GCMMA or any line search is used with s̃, a sentence on the gradient inconsistency.

With OC or plain MMA, which have no line search, the inconsistency matters only through convergence tests.

**Replacement wording** (§7.2, MANUSCRIPT_EN.md line 544, from "Its residual factor in Eq. (H.5) …" to "… left to subsequent work."), to be finalised after the measurement:

> At a fixed retained displacement, with \(H=F-E\), Eqs. (13) and (14) give \(\widetilde s_c-s_c=2(Hq)^TAE_{I,c}q-(Hq)^TK_{,c}Hq\) and \(\widehat C_{,c}-C_{,c}=-2(H_{I,c}q)^TA\,Hq-(Hq)^TK_{,c}Hq\): the residual term of Eq. (14) removes the linear field-error term and replaces it by one weighted by the design derivative of the extension error. Which of the two is the more accurate gradient therefore depends on whether \(\|H_{I,c}q\|_A\) is small compared with \(\|E_{I,c}q\|_A\), which the energy error does not control. Section 6.11 compares both with the exact derivative [numbers].

Also replace in S7 (SUPPLEMENTARY_EN.md line 655) "and reverse-mode sensitivities for the three loads" with "and the field-based sensitivities \(\widetilde s_c\) of Eq. (13) for the three loads, obtained by reverse-mode differentiation of the moment integrals at fixed recovered fields".

---

### 1.2 I-08: The sensitivity metric is not the one an optimiser sees

**Verdict: CONFIRMED.**
- `lat_hetero.py` line 291 defines `serr = ||sh - S[i]|| / ||S[i]||` per cell. The pair gates store `sens_vec_rel_err` per cell (e.g. `evidence/gate_A3_2grid_fresh_val_2000_full_x.json`).
- No global-gradient, cosine or per-variable metric is computed anywhere. The shared-variable map ∇_{τg}C is never evaluated.
- The data needed to compute it after the fact are **not saved**: the pair and lattice JSONs store relative errors, energy shares, ε and β, but neither s nor s̃ vectors.
- The only per-cell weights available are the energy shares, which the reviewers rightly note are not the relevant normalisation.

**Partial mitigation.** For U1/x N-z, where the local error is 0.598% with an energy share of 0.0013, a global metric will almost certainly show that the error is negligible. That strengthens, not weakens, the paper's separation argument, but it still has to be shown.

**Action class:** (b), requiring a re-run with vector output.

**Specification:**
- In the same instrumented re-run as I-04, dump per cell and per load: s (exact, `C.sens(E q)`), s̃ (`C.sens(F q̂)`), energies, and the corner map (layout JSON `cells[].position`, `hlat*.json`).
- Compute:
  - ‖∇̃C − ∇C‖/‖∇C‖ over all cell-corner variables, and over **shared lattice-vertex variables** after aggregation Σ_m (∂τ_m/∂τ_g)ᵀ s_m;
  - cosine;
  - per-variable relative error median/95th percentile/max, with a floor such as 10⁻³ max|∂C| to avoid division by zero;
  - sign agreement;
  - each cell's ‖s_m‖/‖∇C‖ next to its local error.
- Scripts: add `np.save` of S and sh in `lat_hetero.py` (the loop at lines 276–291) and in the pair driver.
- **Cost:** pairs about 14 × 1 min; lattices 25–55 min each (A3 run times 1,410 s and 3,240 s in the evidence, shorter if more cells stay resident). **About 2 GPU-h**, shared.
- **Priority: must.** This is cheap, and M1 in CONSOLIDATED requires it.

**Wording** (§6.6, MANUSCRIPT_EN.md line 444): replace "using 3% on compliance and on each cell's eight-parameter sensitivity vector as a common accuracy reference" with

> using 3% on compliance and on each cell's eight-parameter sensitivity vector as a common visual reference; the threshold is not derived from an optimiser tolerance, and Section 6.x reports the error of the assembled gradient itself.

---

### 1.3 I-25: Contradictory finite-difference step statement; PSD of the numerical K,c

**Verdict: CONFIRMED (text). The PSD part is PARTLY REFUTED by an unarchived check.**

**Step-study evidence (`evidence/ref_valid.json`, `evidence/ref_valid_h1.json`, key `per_case.<case>.fd.rel_to_h1e_5`, relative to h = 10⁻⁵τ_c):**

| step/τ_c | M1 (2003) | M2 (2006) | U1 (2000) | H1 (2005) |
|---|---|---|---|---|
| 10⁻³ | 9.1e-8 | **2.56e-7** | 6.4e-8 | **2.19e-7** |
| 10⁻⁴ | 8.9e-10 | 2.53e-9 | 6.3e-10 | 2.17e-9 |
| 10⁻⁶ | 1.2e-10 | 7.4e-11 | 7.5e-11 | 1.6e-10 |

- `direct_vs_sens`, which compares against central differences of the re-solved compliance, is at most 3.84e-8 (M1).
- The step study falls by 100× per decade of h, the O(h²) signature of central differences. This is **positive evidence** that no integration branch switches within ±10⁻³τ_c at a fixed active set on these four cells. It is worth stating, and it is relevant to I-31.
- App. J.5 (APPENDICES_EN.md line 762), "No derivative eigenvalues or step-refinement study are saved", contradicts §6.2.

**PSD evidence.**
- `docs/data/newmachine_20260924/arch_analysis_0926/work/mech2/psd.py` computed the eigenvalues of the element derivative matrices K_{e,c} = Σ_m ∂M_em/∂τ_c T_m for 300 random cut elements × 8 corners on `fresh_train_0021_cover01_r1`. `WORKFLOW_RESULT.json` records the result as "dK_e/dtau_c is PSD, with min eigenvalue about 0 to roundoff".
- Element-wise PSD implies global PSD, because K_{,c} = Σ_e P_eᵀ K_{e,c} P_e.
- The output file is not archived, the check covers one training cell only, and the ghost term is excluded (it is constant, so its derivative is zero).

**Action class:** (a) must; (b) optional but cheap.

**Specification (b):** Rerun `psd.py`-style batched `eigvalsh` of all 81×81 K_{e,c} for **all** partially filled elements of the five detailed cells. Use both the central-difference dM (Eq. H.6) and the `moments_ad` reverse-mode dM. Report min λ/max|λ|. Cost: a few thousand elements × 8 corners of 81×81 `eigvalsh`, i.e. seconds to minutes. Existing data suffice.

**Replacement wording:**
- §6.2 (MANUSCRIPT_EN.md line 371), from "The central-difference stiffness derivative …" to the end of the sentence:
  > The central-difference stiffness derivative used for the reference sensitivities agrees with central differences of the re-solved compliance to within \(4\times10^{-8}\) (relative). Relative to the step \(10^{-5}\tau_c\) used throughout, it changes by at most \(2.6\times10^{-7}\) at \(10^{-3}\tau_c\), \(2.5\times10^{-9}\) at \(10^{-4}\tau_c\) and \(1.6\times10^{-10}\) at \(10^{-6}\tau_c\) on M1, M2, U1 and H1; the hundredfold reduction per decade of step is the second-order truncation error of the central difference and shows no change of integration branch within \(\pm10^{-3}\tau_c\) at the fixed active set.
- App. J.5 (APPENDICES_EN.md line 762): replace "No derivative eigenvalues or step-refinement study are saved in the selected sensitivity records." with
  > Section 6.2 reports a step-refinement study of the numerical derivative on four cells. [If the PSD check is run:] On the five detailed cells, the smallest eigenvalue of every element derivative matrix is at least \(-X\) times its largest, so the numerical derivatives preserve the semidefiniteness to rounding. [Otherwise:] Eigenvalues of the numerical derivative matrices are not part of the selected records.
- §4.3 (MANUSCRIPT_EN.md line 250): replace "exact integration gives \(K_{,c}\succeq0\)" with
  > the exact integral gives \(K_{,c}\succeq0\) (Eq. J.6); a numerical derivative inherits this only while its quadrature preserves the nesting of the material domains (Appendix J.5)
- App. H (APPENDICES_EN.md line 495): replace "Eq. (J.6) in Appendix J.5" with "Eq. (J.6) (Section J.5)".

---

### 1.4 I-31: Non-smoothness of Ĉ(τ) across discrete switches

**Verdict: CONFIRMED in substance. There is a partial rebuttal on magnitude.**

**Switches present in the code:**
- Hard node flags `is_port, is_box, is_cut, weak` (`models.py:435`). The weak flag uses the 0.01 × median threshold (App. G.1).
- The `log(vf+1e-6)` element feature (`models.py:431`), continuous but steep as vf → 0.
- The median normalisation `log(d3/median)` (`models.py:434`), which jumps when the active set changes.
- Element activation and ghost-face membership (App. A.1).
- The retained cut-band set P.
- The Cholesky shift picked from {0, 10⁻¹², …} (App. F.2; `evidence/time_setup.json`: shift 0 on 6 cells and 10⁻¹² on H1, so the switch does occur across geometries).
- The finite power estimate of b. It is deterministic given the seed and device, and continuous but not smooth.
- The cubic-view frame, if a canonicalisation by argmax is used in deployment. The predecessor model had one (`docs/TAU_DERIVATIVE_PASSES_20260919.md` §5). This should be confirmed for A3 (links I-26).

**Existing τ-smoothness evidence:**
1. *Current pipeline:* the step study above shows smoothness of K(τ) within ±10⁻³τ_c at a fixed active set. It says nothing about activation.
2. *Predecessor teacher, no network* (`docs/CUT_CELL_TAU_GATE_20260920.md`; `docs/data/cut_tau_20260920/SLOPES_100032.json`, `SLOPES2_100000_straddle.json`):
   - A realistically sized cut cell (3,651 active cells) is smooth over ±0.1% uniform τ scaling (Richardson ratios 1.0001–1.018; no active-set change).
   - A 19-cell sliver loses an element between ε = −2.5×10⁻⁴ and −5×10⁻⁴, and its condensed compliances **jump by 8–58%** (`box_load_0` 9.58e7 → 1.16e8; `cut_coordinates` 142 → 134).
3. *Predecessor surrogate* (`docs/TAU_DERIVATIVE_PASSES_20260919.md`): an element birth between ε = 0 and 10⁻⁴ produced a 3.5× kink in free-cell compliance slopes, but the clamped observable stayed smooth.

These are not NICE results, and the teacher may differ from the manuscript's reference. They do demonstrate that (a) the **reference C(τ) itself** is only piecewise smooth, and (b) jumps can be large for slivers.

**Partial rebuttal (rigorous, text-level).** With exact assembled solves, Eq. (12) gives 0 ≤ C − Ĉ ≤ βC at every design. Writing Ĉ = C(1 − e) with 0 ≤ e ≤ β, any jump the surrogate adds beyond the jump of C satisfies |[Ĉ] − [C]| ≤ β_max C. In the reported configurations this is at most about 0.06% (pairs) and 0.015% (lattices) of C. Surrogate-specific non-smoothness, such as that from the network's binary features, is therefore bounded by the surrogate's own error level. This is below typical early per-iteration objective changes but can matter near convergence. The remaining, and larger, non-smoothness belongs to the fixed-background CutFEM model itself, and exact condensation shares it.

**Action class:** (a) must; (c) small, should.

**Specification (c):**
- 1-D sweeps of one corner τ_c from 0.18 to 0.70 for M1 and H1 (optionally U1), with about 60 coarse points plus refinement to 10⁻⁴ around every detected switch.
- At each point record: C, Ĉ, s_c, s̃_c; active elements, ghost faces, |P|, the number of weak flags and the Cholesky shift. Add fixed-q̂ central differences of Ĉ at the refined points.
- Needs body regeneration at each τ, because the active set changes (`fast_prep4`/`gen_*`; `lat_scale.py` notes that bodies are not regenerated in its τ loop). Plus the learned front end (3–6 s) and the exact factor (about 3 s).
- **Cost:** 2–6 GPU-h, dominated by body generation, for which there is no timing in the evidence.
- Also require per-iteration logging of the same switch counters in §6.11. This costs nothing.

**Wording** (new paragraph at the end of §4.3, MANUSCRIPT_EN.md after line 264):

> The derivatives above hold on intervals where the active elements, ghost faces, retained coordinates, the network's binary node features and the coarse factor shift are fixed. Across such a switch the discrete reference compliance itself can jump, by an amount of the order of the energy carried by the element that enters or leaves; this non-smoothness is shared by exact condensation on the same background mesh. Because \(0\le C-\widehat C\le\beta C\) at every design (Eq. 12), any jump that the surrogate adds to that of the reference is bounded by its own error level, \(\beta_{\max}C\). Section 6.11 records the switches crossed along the optimisation and the corresponding objective changes.

---

### 1.5 I-50: Design-dependent loads frozen in the sensitivities

**Verdict: CONFIRMED, but it is a disclosed modelling choice, not an error.**
- App. H (APPENDICES_EN.md lines 486–487 and 511) states that the labels hold the base-design nodal load fixed and gives the omitted term 2f_{g,c}ᵀÛ.
- `lat_multi.MultiLattice(loads='consistent')` integrates consistent tractions over the current material patches and normalises them to a unit resultant. The load therefore depends on the design.

**Action class:** (a). **Priority: must** (for §6.11).

**Wording:**
- In §6.11 (placeholder, MANUSCRIPT_EN.md line 524):
  > Loads and supports are applied [through non-design cells / as nodal vectors fixed at the initial design], so the load-derivative term \(2f_{g,c}^T\widehat U\) of Appendix H is absent [or: is included, with \(f_{g,c}\) obtained by differentiating the traction integration].
- Add to §6.9 after "loaded by unit consistent tractions on the opposite face":
  > the reported sensitivities hold this nodal load fixed at the evaluated design (Appendix H).
- If loads remain on design cells, including the term is cheap: f_{g,c} by central differences of the face-patch integration, one dot product per variable.

---

## 2. T4: Numerics, precision and solver

### 2.1 I-04: Finite-precision residual floor in the assembled solves

**Verdict: PARTLY CONFIRMED.**

**Confirmed:**
- The floor exists. `evidence/lat_hetero222_A3.json` gives `true_residual` 3.63e-4 against recursive 9.9e-11. `lat_hetero331_A3.json` gives 3.22e-3 against 9.2e-11.
- The pair gate JSONs store no true residual.
- Neither Ūᵀρ nor ω is stored anywhere. J.6 says so itself.
- The attribution "set by the network's single-precision arithmetic" (MANUSCRIPT_EN.md line 486) is not demonstrated.

**Precision actually used (question 2):**

| Run | Network | Correction (smoothing, coarse) | K in condensed product / energies | Evidence |
|---|---|---|---|---|
| §6.9 lattices (`lat_hetero222_A3.json`, `lat_hetero331_A3.json`) | fp32; true fp32 convolutions (`conv_tf32: False`, `OPL_CONV_FP32=1`) | **fp64** (no `--deploy`, so `C._Kc` absent and `trainlib._Kc` returns fp64 `C.K`; `OPL_COARSE_FP32` unset); output returned to fp32 (F.3) | fp64 | `args` in both JSONs; `trainlib.py:59–62, 395–402` |
| Streamed deploy rerun of 2×2×2 (`docs/data/newmachine_20260924/p2/lat_hetero222_stream.json`, `deploy: true`) | fp32 | **fp32** (`_Kfp32`, `OPL_COARSE_FP32=1`) | fp64 | `lat_hetero.py:166, 177`; `teacher.py:398` |
| S7 / Table 6 learned route (`learned_hlat221a/b.json`, `learned_hlat331.json`) | fp32 | **fp32**; `lat_scale.py:95` forces `OPL_COARSE_FP32=1`; `Cell(deploy=True)` builds `_Kfp32` (`teacher.py:486`) | fp64 | `lat_scale.py` |
| Table 5 cell benchmarks (`bench_deploy.py`) | fp32 | fp64 (no `lean(deploy)`) | fp64 | `bench_deploy.py` |

- `OPL_KE_SYMV` applies only to the fp32 correction stiffness with `ke_moments` (deploy); `OPL_KE_SYMV64` applies only to the fp64 deploy-mode K product (still fp64). Neither affects the §6.9 runs.
- Environment variables are not logged in any JSON; only the convolution flags are. The authors should log them.
- Matmul TF32 is never enabled; PyTorch's default is off, and grep finds no override. No fp16/bf16 appears anywhere.

**Therefore Table 1 (MANUSCRIPT_EN.md line 103) and §6.2 (line 373), "smoothing and coarse solve in double precision", are wrong for the timed deployment route of Table 6, and incomplete for all routes because fields are returned to fp32 (F.3).**

**What refutes the contamination concern (at the reported digits):**
1. **Correction precision does not matter.** The fp32-correction rerun reproduces the fp64 run:

   | | fp64 correction | fp32 correction |
   |---|---|---|
   | true residual | 3.629e-4 | 3.657e-4 |
   | max consistent compliance error | 1.370919e-4 | 1.370942e-4 |
   | max sensitivity error | 1.44128e-3 | 1.44023e-3 |
   | solution error | 1.29888e-3 | 1.29887e-3 |

   The floor is therefore not caused by the correction arithmetic. This is consistent with, but does not prove, the network-fp32 attribution.
2. **β−err is tiny and has the right sign.** β − (reported error) is +1.5 to +4.9×10⁻⁷ of C on all six consistent loads of both lattices (0.13–0.33% of the error). This holds although the 3×3×1 floor is 10× higher.
   - Theory: βC − (C − Ĉ) = (Û−U)ᵀK̂(Û−U) ≥ 0, and reported error = (C − Ĉ)/C + Ûᵀρ/C.
   - Hence Ûᵀρ/C = (β−err) − gap ≥ −4.9×10⁻⁷ rigorously.
   - The observed agreement is what one expects if the residual work is negligible and β−err is the second-order trace term. A residual work comparable to the reported errors (~10⁻⁵–10⁻⁴) would have had to be offset by an equal and opposite gap in all six loads.
3. **Pairs.** Out of the 114 load cases of the 14 A3 configurations, β−err is negative in only five, each between 2×10⁻¹¹ and 8×10⁻¹⁰ of C (e.g. `gate_A3_2grid_fresh_val_2004_d0_v2_y.json` load 5: err 1.98e-7, β−err ≈ −7.7e-10). Algebraic contamination in the pairs is therefore at most of order 10⁻⁹ of C, visible only on errors of order 10⁻⁷.

**What does not refute it.** The J.6 Cauchy–Schwarz bound |Ūᵀρ|/C ≤ (‖Ū‖₂‖f‖₂/C)(‖ρ‖₂/‖f‖₂) cannot certify anything. For a face load on about 1.4×10⁵ free DOFs the constant is plausibly well above 10, which makes the bound 0.4–4% (2×2×2) or 3–30% (3×3×1), far above the 0.01% errors. So, strictly, **the reported compliance error is not bounded by the reported residual**. The residual must be measured in the dual energy norm, or Ūᵀρ computed directly.

**Would fp64 re-runs change the headline numbers?** Evidence 1–2 indicates no: changes are expected in the third significant digit at most (the fp64/fp32 correction swap moved compliance errors by 2×10⁻⁹ and the max sensitivity error from 0.1441% to 0.1440%). This remains inference until the re-run below is done.

**Action class:** (a) must; (b) should; (c) optional.

**Specification (b), "instrumented re-run bundle"** (also serves I-03 and I-08):
- Re-run `lat_hetero.py` (both layouts, A3 only, `--exact-dense`) and the 14 pair gates. Save:
  - Ū, U, f;
  - ρ = f − K̂Ū (one extra matvec);
  - Ūᵀρ and J(Ū) = fᵀŪ + Ūᵀρ;
  - ω = ŪᵀK̂Ū − Σ_m ū_mᵀK_mū_m;
  - the true residual every 10 PCG iterations (pass `snaps`, or add a callback in `lat_precond.pcg`);
  - the vectors needed for I-08.
- **Rigorous bound:** because K̂ ⪰ 𝕂, ρᵀK̂⁻¹ρ ≤ ρᵀ𝕂⁻¹ρ. Hence |Ūᵀρ| ≤ √(ŪᵀK̂Ū)·√(ρᵀ𝕂⁻¹ρ), and the latter needs one extra PCG solve with the *exact* dense condensed lattice operator already formed for the reference (about 60–90 s). Report Ūᵀρ/C, the bound, ω/C and the error of J(Ū).
- Report the iteration at which the true residual reaches the floor. This addresses the inflated 183/186 counts (R3-05); note that the exact route iterated to the same tolerance.
- **Cost:** pairs about 15 min; lattices about 1–1.5 h (A3) plus about 5 min (exact); total **about 2–3 GPU-h**. Existing checkpoints, bodies and layouts suffice.

**Specification (c), optional:**
- Convert the network to fp64 for one pair (M1/x) with `t_fast64.to64` (it already builds an fp64 copy of the model and caches). Run the pair with an fp64 correction and without casts through the autograd model (`ops.py` gives the transpose by autograd). Show that the true residual falls to about 10⁻¹⁰ and that the errors are unchanged.
- Consumer-GPU fp64 is slow (roughly 1/64 of fp32 throughput). Estimate 0.5–1 h for a pair and 5–20 h for a lattice (skip the lattice).

**Replacement wording:**
- Table 1, "Arithmetic" row (MANUSCRIPT_EN.md line 103):
  > Network, rigid reconstruction and retained entries in single precision (true fp32 convolutions, no TF32); stiffness actions of the condensed product and energies in double precision; correction (smoothing and coarse solve) in double precision in the accuracy studies of Sections 6.2–6.9 and in single precision in the timed route of Table 6, with the corrected field returned to single precision (Appendix F.3)
- §6.2 (line 373): replace "the deployed implementation evaluates the network in single precision and the stiffness actions, smoothing and coarse solve in double precision" with
  > the implementation used for the accuracy results evaluates the network in single precision and the stiffness actions, smoothing and coarse solve in double precision, returning the corrected field in single precision (Appendix F.3)
- §6.9 (line 486): replace "while its recomputed residual stagnates at \(3.6\times10^{-4}\), the level set by the network's single-precision arithmetic; the errors above are measured at this solution." with
  > while its recomputed residual \(\|f-\widehat{\mathbb K}\bar U\|/\|f\|\) stagnates at \(3.6\times10^{-4}\). Repeating the solve with the correction in single precision changes this residual by less than 1% and the compliance errors by less than \(3\times10^{-9}\), so the stagnation does not come from the correction arithmetic; it is consistent with the rounding of the single-precision network action. The errors above are measured at this solution. The solve-independent bound \(\beta\) exceeds them by only \(1.5\)–\(4.9\times10^{-7}\) of the compliance, the size of the second-order trace term; [after re-run:] the signed residual work of Eq. (18) is at most \(X\) of the compliance.
- S7 (SUPPLEMENTARY_EN.md line 655): add after "the deployed implementation with A3":
  > with the correction smoothing and coarse solve in single precision
- App. J.6 (APPENDICES_EN.md lines 790–797): fix the dangling "which are not stored in those records" (see I-28), and state which records now contain Ūᵀρ and ω.
- Move `p2/lat_hetero222_stream.json` into `evidence/`.

**Rebuttal paragraph (for the response letter):**
> We agree that the recomputed residual of the learned lattice solves stagnates and that the signed residual work was not reported. Two existing results bound its effect on the reported numbers: re-running the \(2\times2\times2\) block with the correction in single precision changes the recomputed residual by 1% and the compliance errors by at most \(3\times10^{-9}\); and the solve-independent participation bound \(\beta\) exceeds the measured compliance error by only \(1.5\)–\(4.9\times10^{-7}\) of the compliance on all consistent loads of both lattices, which leaves no room for a residual work comparable to the reported errors. In the revision we report \(\bar U^T\rho\), the residual-corrected functional \(J(\bar U)\) and a rigorous dual-norm bound computed with the exact lattice operator for every assembled result, and we state the precision of each evaluation route.

---

### 2.2 I-20: No runtime error control; exploitation of the one-sided bias

**Verdict: CONFIRMED** (no indicator in the reported routes). **Partly mitigated:**
- **(i)** A rigorous, label-free lower-bound certificate is already implemented. `src_v2_wip/cert.py` gives Δ ≥ L_z with z from Jacobi (m = 0) or a block Jacobi–Krylov space (m ≥ 1), all in fp64. It is timed in `evidence/bench_A3_cells.jsonl` (`cert_m0_B16_s` about 0.015 s, `cert_m8_B16_s` about 0.096 s for a 177k-DOF cell), but its **effectivity is never reported**.
- **(ii)** The optimiser's possible gain from exploitation is bounded by e ≤ β ≤ β_max within the validated domain. That domain is only the danger zone out of distribution, where no evidence exists.
- **(iii)** The Chebyshev and variational structure (SPSD, Ŝ ⪰ S) holds regardless of b (see I-38), so an out-of-distribution geometry cannot make the operator unsafe, only inaccurate.

**Action class:** (a) must; (b) should.

**Specification (b):**
- Run `cert.py` (m = 0, 2, 8) on the 80 validation geometries with A3 at existing bank directions, where Δ is known from the stored teacher energies. Report the effectivity L_z/Δ (min, median) and the resulting empirical upper estimate Δ ≲ L_z/eff_min.
- In the re-run lattices, and later in §6.11, report the per-cell certificate at the solution trace q̂_m.
- Define a policy: flag cells with L_z/(q̂ᵀŜq̂) above a threshold for a larger budget or for exact condensation.
- **Cost:** about 1 GPU-h. Data and checkpoints exist.

**Wording** (new sentence in §7.4 or the Limitations section):

> The one-sided error \(\widehat C\le C\) could in principle be exploited by an optimiser; within the validated domain the gain is bounded by \(\beta\), but for geometries outside it no a-posteriori upper bound is available. A label-free lower bound on each cell's energy error (Appendix I) costs about one correction cycle and is used in Section 6.11 to [flag / monitor] cells; its effectivity on the validation geometries is [x–y].

---

### 2.3 I-23: The two-grid bound proves only (near) non-expansion; "controllable" is empirical

**Verdict: CONFIRMED.**
- With λ₁ ≈ 4–6×10⁻⁴, b ≈ 5.1 and a = b/30, the energy factor of App. D is ρ_k⁴:
  - k = 8: 0.986 (λ₁ = 4×10⁻⁴) and 0.980 (6×10⁻⁴);
  - k = 32: 0.921–0.946 (my calculation of p_k(λ₁) from Eq. D.1).
- It is a strict contraction, because p_k < 1 on (0, b], but it guarantees at most a 1.4–2.0% energy reduction per 8/Q1/8 cycle. The observed reduction is 73× (M1: 13.5% → 0.186%).
- The proof uses no approximation property of V (‖C_V‖_A = 1).
- **A further nuance the reviewers missed:** repeating a cycle is guaranteed not to increase the energy error, but *increasing k* is not, because Chebyshev polynomials of different degree are not ordered pointwise. Monotonicity in k was observed (`LATE_RESULTS.md`: energy monotone in k; sensitivity not monotone, e.g. U1 force_c with B and smoothing only, 0.41 → 0.44 → 0.53% at k = 2, 4, 8) but is not guaranteed.
- There is no multi-cycle or ‖T‖_A evidence in the repository.

**Action class:** (a) must; (b) should.

**Specification (b):**
- For the five detailed cells, and ideally all 80, estimate ‖T‖_A for T = P_kC_VP_k.
  - T is A-self-adjoint, so use Lanczos in the A-inner product: 40–80 steps, each one cycle application plus one A-product.
  - Use the existing differentiable correction routines in `trainlib` (`_cheb`, `_coarse_solve`).
  - Report ‖T‖_A and ‖T^m‖_A^{1/m}, and compare with the directional reductions of §6.5.
- Show multi-cycle curves (m = 1–4) from B's field and from zero on M1 and H1.
- **Cost:** under 1 s per cycle, so about 1–2 min per cell and about 1 h for all 80. Existing data suffice.

**Replacement wording:**
- App. D (APPENDICES_EN.md line 227): replace "which proves the \(\rho_k^4\) energy estimate and operator ordering in Eq. (17)" with
  > Because \(\rho_k<1\) whenever the positive spectrum lies in \((0,b]\), the cycle is an energy contraction and yields the operator ordering in Eq. (17). For the spectra of the detailed cells (\(\lambda_1\approx4\)–\(6\times10^{-4}\) against \(a\approx0.17\)), however, \(\rho_8^4\ge0.98\), so the estimate guarantees little more than non-expansion; it uses no approximation property of the coarse space, and the reductions by one to two orders of magnitude in Section 6.5 are empirical. Repeating the cycle cannot increase the energy error; changing the smoothing degree changes the polynomial and is not guaranteed to be monotone. A two-grid convergence estimate would require an approximation property of the \(Q_1\) space for walls about one element thick [Xu & Zikatanov (2002); Falgout, Vassilevski & Zikatanov (2005)].
- Abstract (MANUSCRIPT_EN.md line 5): replace "Its error is confined to the interior extension and can be reduced through both the network and the correction budget." with
  > Its error is confined to the interior extension; the correction cannot increase it, and in the examples a better initial field or a larger correction budget reduced it.
- Contribution (i) (line 25): replace "and its error can be reduced through both the initial field and the correction budget" with
  > and the correction cannot increase its error; in all examples, improving the initial field or enlarging the correction budget reduced it
- §5.2 (line 310): append
  > this ordering is guaranteed, whereas the size of the reduction is not (Appendix D).

---

### 2.4 I-38: Chebyshev interval and coarse-solve verification

**Verdict: PARTLY REFUTED.** The reviewers' main premise is wrong, but the text is inconsistent, and in the direction opposite to what R1-11 assumed.

**Evidence:**
- `trainlib.tail_bounds` (lines 65–82) seeds the generator with `torch.Generator(device=dev).manual_seed(0)`. The start vector is therefore deterministic *per device type*.
  - `lam_check.py` (the 80-geometry check) calls this same function.
  - `evidence/lam_check_val80.json` M1 `op_lmax` = 5.198581037, **identical** to `evidence/p1_checks.json` (GPU) `b` = 5.19858. H1 matches as well (5.18084).
  - So the 80-geometry check verifies **the operational GPU endpoint used by all pair and lattice runs** (RTX 5090). Margins are 2.07–5.00%, and 3.7–4.4% on U2/M1/M2/H1/H2.
- Table ST19's b values for H2, H1, M2 and M1 come from `evidence/p1_checks_cpu.json`, a **host run** whose CPU generator gives a different start (M1 5.19326, H1 5.19078). U2 comes from `p1_checks_u2.json` (GPU).
- Hence §6.2's "These margins refer to the interval estimates of the deployed runs; the 80-geometry check in Appendix D uses its own power-iteration start" is **backwards** for four of the five cells.
- **Probe = Galerkin** is already checked on 3 lattice cells. `docs/data/newmachine_20260924/p2/coarse_test.jsonl` gives `L_rel` (probed against element/face-assembled Galerkin factor) of 2.3e-16, 1.3e-11 and 2.2e-16, and Ŝq differences of 2.4–3.5e-8.
- **Shifts** are recorded for 7 cells in `evidence/time_setup.json`: 0, except H1 with 10⁻¹² on the Jacobi-scaled matrix.
- **Different implementations in Table 3** (B columns from the CPU PARDISO study, A3 column from probed shifted Cholesky): confirmed, but the above evidence says the difference is at rounding level.
- **Certified upper bound:** a Lanczos Ritz value plus its residual certifies an eigenvalue near θ, not that θ is λ_max. R3-11's proposal is therefore not rigorous either. The 2.1% minimum margin against a maximum relative Ritz residual of 9.6×10⁻⁴ is the practical argument.
- **Rebuttal point to add:** SPSD, symmetry and Ŝ ⪰ S hold for *any* b, because Ŝ = FᵀKF with J_P F = I. Containment affects only the ordering Ŝ_tg ⪯ Ŝ₀ and the accuracy.

**Action class:** (a) must; (b) should.

**Specification (b):**
- Extend `coarse_test.py` to the 80 validation geometries and the five detailed cells, recording `L_rel`, Ŝq differences and the selected shift. About 7 s per cell, so about 10 min.
- Optionally, add a b/a ∈ {10, 30, 100} sensitivity study on the five cells with fixed weights (minutes).
- Confirm that every GPU run used `device='cuda'` for the generator (it does in the code).

**Replacement wording:**
- §6.2 (MANUSCRIPT_EN.md line 373): replace "These margins refer to the interval estimates of the deployed runs; the 80-geometry check in Appendix D uses its own power-iteration start." with
  > Four of these margins come from a host evaluation whose seeded generator gives a different power-iteration start; the endpoint used by all GPU runs is the one verified on all 80 validation geometries in Appendix D (margins 2.1–5.0%, and 3.7–4.4% on these five cells). Symmetry, positive semidefiniteness and \(\widehat S\succeq S\) do not depend on this containment; it governs only the ordering in Eq. (17) and the accuracy.
- App. D: keep "operational". Add after "caches the interval for the geometry":
  > The seed fixes the start vector for a given device type; the host verification in Table ST19 therefore uses a different start from the GPU runs.
- §5.1: add after "a=b/30":
  > (a choice that places the slowest targeted modes near the coarse-space resolution; Figure 8d shows the budget dependence)
  
  or give the b/a study.
- App. F.2: add
  > On three lattice cells, the probed factor agrees with the element-assembled Galerkin factor to \(\le1.3\times10^{-11}\) (relative); the selected shift was 0 on six of seven recorded cells and \(10^{-12}\) on H1.

---

## 3. Side findings (outside my themes, flagged for the clerk)

1. **H2 two-cell configurations.** `evidence/gate_*_fresh_val_2010_d0_v0_y.json` (all predictors, including A3) shows PCG at maxit = 400, NaN neighbour errors and **negative exact energy shares** (−0.15, −0.21). The reference problem is ill-posed. `RESULTS_A3_CN.md` line 7 records that both H2 configurations are ill-posed (the target cell has no material connection to the supported neighbour) and excluded. The manuscript says "fourteen configurations evaluated" but never mentions the exclusion. One sentence in §6.6 or the S-notes is needed; a reviewer who opens `evidence/` will find these files.
2. **Unlogged environment.** No JSON records `OPL_*` environment variables (`OPL_KE_SYMV`, `OPL_KE_SYMV64`, `OPL_COARSE_FP32`, `OPL_TAILT_FUSED`), only the convolution flags. Recommend logging them in every driver (`TL.conv_precision()` could be extended).
3. **The fp32 correction route (S7) is untested for pair-level accuracy,** except on the 2×2×2 block (`lat_hetero222_stream.json`). That file should move into `evidence/`, because it is the only accuracy record of the timed route.
4. **Internal analyses** (`arch_analysis_0926/WORKFLOW_RESULT.json`) contain the element-level PSD result and a "regret toy" (3/6/15% gradient error giving 0.045/0.13/0.73% regret). Neither is archived as evidence. The PSD result should be rerun and archived (I-25). The toy is not publishable as it stands but could motivate the 3% reference if redone properly (I-08).

---

## 4. Consolidated experiment plan (one RTX 5090)

| Block | Serves | What | Existing inputs | GPU time | Priority |
|---|---|---|---|---|---|
| R1 instrumented re-run | I-04, I-08, I-03 (q̂) | 14 A3 pairs + 2 lattices; dump Ū, U, ρ, Ūᵀρ, J(Ū), ω, dual-norm bound, s/s̃ vectors, true-residual history | checkpoints, bodies, layouts, dense exact S cache | 2–3 h | must (I-08), should (I-04) |
| R2 fixed-q̂ complete derivative | I-03 | per-cell rebuild at τ ± h, 3 step sizes, all corners; validate on 2–3 variables by full re-solve | R1 outputs | 3–4 h | must |
| R3 certificate effectivity | I-20 | `cert.py` m = 0/2/8 on the 80 validation geometries; per-cell certificate in lattices | banks, teacher energies | about 1 h | should |
| R4 two-grid norm | I-23 | A-Lanczos on P_kC_VP_k, multi-cycle curves | trainlib routines | about 1 h | should |
| R5 probe/shift/b-a audit | I-38 | extend `coarse_test.py`; optional b/a sweep | none new | under 0.5 h | should |
| R6 PSD of K_{e,c} | I-25 | batched `eigvalsh` on all cut elements, FD and AD derivatives | moments | minutes | should |
| R7 τ sweeps | I-31 | 1–2 cells × 1 corner, about 60 points plus refinement, switch counters | body generator | 2–6 h | should |
| R8 fp64 network pair | I-04 | `t_fast64.to64` + autograd operator on M1/x | checkpoint | 0.5–1 h | optional |

**Total for the must/should items:** about 10–16 GPU-h plus 2–3 days of scripting. No retraining is needed.
