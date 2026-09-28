# Review R2 (machine learning for computational mechanics, operator learning, learned substructures)

Manuscript: "Learned static condensation for cut thin-walled TPMS cells with equilibrium correction" (CMAME submission)

Materials reviewed: MANUSCRIPT_EN.md, APPENDICES_EN.md, SUPPLEMENTARY_EN.md and the figures. As instructed, I did not review Section 6.11 or the "[Placeholder ...]" sentences. At the end of the review I state what that example must demonstrate.

---

## 1. Summary

The paper approximates the static condensation of cut, thin-walled P-type TPMS cells in a stabilised Q2 CutFEM model. It keeps the full retained space (all box-face coefficients plus a cut band) and approximates only the interior extension. A geometry-conditioned network maps the non-rigid retained displacement to an interior field. The map is linear in the displacement, and a nonlinear hypernetwork-like geometry branch supplies the coefficients of local gather/scatter interactions and of a U-shaped latent hierarchy. A fixed, geometry-specific correction then reduces the field's interior equilibrium residual: 8 Chebyshev steps, a trilinear Galerkin coarse solve, then 8 more steps. The condensed stiffness is defined variationally as F^T K F. It is applied matrix-free with the full transpose, which makes it symmetric positive semidefinite, exact on rigid modes and bounded below by the exact Schur complement. Ritz and energy arguments connect the interior error to compliance (through energy participation) and to field-based thickness sensitivity (through K_{,c}). The principal predictor (A3) is trained through the correction with an energy + sensitivity objective. On 80 validation geometries it reaches a mean directional energy excess of 0.074%. In 14 two-cell configurations it gives compliance errors below 0.06% and sensitivity errors below 0.7%, and on two eight-cell lattices it shows small errors. It is also faster and uses less memory than CPU PARDISO condensation.

## 2. Recommendation

**Major revision.** The variational construction is careful and correct: symmetric, lower-bounded, rigid-exact, matrix-free, with the complete transpose. Using the energy-residual identity as a common criterion for the network and the correction is a sound design, and the compliance-versus-sensitivity analysis is a useful message for the learned-substructure community. The manuscript is also unusually candid about its own limitations, for example the selection overlap, the development cells and the fp32 residual floors.

From the machine-learning side, however, the evidence does not yet support the specific learning claims the paper makes:
- **Training through the correction is confounded.** Its benefit is measured against B+W, which differs from A3 in training data and training steps as well as in correction-aware training.
- **The sensitivity term and the architecture are not ablated.**
- **The strongest non-learned competitor is missing.** That competitor is the same two-grid correction run longer, or started from a cheap extension, at matched accuracy and on the same GPU.
- **The offline cost of data generation and training is not reported**, yet the paper claims the method is cheaper "for any number of queries".
- **The assembly results have no independent test.** They come from cells used during development.

Most of these points can be fixed by re-evaluating existing checkpoints plus a few targeted runs. None of them undermines the mechanical framework.

---

## 3. Findings

### R2-01 — Positioning with respect to operator learning, learned solvers and component-based ROM is incomplete, and the ML contribution needs sharper framing
- **Severity:** Major
- **Location:** Section 1, paragraphs 3–5, and contributions (i)–(ii): "Hybrid neural solvers combine learned prediction with relaxation…", "Learned substructures have been developed along a related line."; reference list (18 entries).
- **Issue:**
  - **What the results show.** Per Section 6.3/6.5 and Tables 3/ST13, most of the accuracy gain comes from the classical two-grid correction. The network acts as a learned, linear warm start for a fixed-iteration interior solver.
  - **Closest prior work omitted.** Several lines of work that are the closest methodological neighbours are not discussed:
    - the static-condensation reduced basis element method (SCRBE; Huynh, Knezevic & Patera 2013), which approximates interior "bubble" responses on full port spaces with a posteriori error control;
    - learned multigrid and learned coarse spaces (e.g. Greenfeld et al. 2019; Luz et al. 2020; ML-adaptive coarse spaces in FETI-DP/overlapping Schwarz by Heinlein, Klawonn and co-workers);
    - solver-in-the-loop / differentiable-solver training (e.g. Um et al. 2020), which is exactly the "training through the correction" idea;
    - geometry-aware neural operators and mesh graph networks (e.g. Geo-FNO, GINO, MeshGraphNets, MgNO), against which the linear-in-q, geometry-conditioned architecture should be contrasted;
    - learning linear solution operators and Green's functions (e.g. Boullé & Townsend), which is the natural class for a map that is linear in q;
    - learned preconditioners for CG.
  - **Missing prior art for specific design choices.** The geometry branch is in effect a hypernetwork, and the energy loss is a Ritz/Deep-Ritz-type objective; neither is related to prior art.
- **Why it matters:** CMAME readers need to see what is genuinely new. As I read it, the new elements are:
  - (a) a learned extension that is exactly linear in the retained displacement and embedded in a variational F^T K F operator on a large cut-band retained space;
  - (b) a correction that keeps symmetry and the Schur lower bound;
  - (c) the compliance/sensitivity error analysis.

  The current text instead implies novelty for hybrid learning + relaxation and for training through a solver, which exist elsewhere.
- **Suggested fix:**
  - Add a related-work paragraph that covers the literatures above.
  - Reframe contribution (ii) as "a learned linear warm start for a fixed variational correction".
  - State explicitly, in the abstract and introduction, that the correction delivers most of the accuracy (B+W already reaches 0.0965% and all assembled criteria), while the network lowers the accuracy that a practical correction budget can reach (Table 3).
- **Fix type:** text only
- **Confidence:** high

### R2-02 — The benefit of "training through the correction" is confounded with additional data and steps
- **Severity:** Major
- **Location:** Table 2; Section 6.3: "training through it reduces the remaining mean by a further factor of 1.3"; Section 6.6; Section 7.3: "it lowers them by factors of 1.35 to 2.2"; Conclusions.
- **Issue:**
  - **Confound.** A3 = B + 15,000 extra steps on a larger population (591 versus 305 geometries) + correction in the loop. B+W = B's weights + the same correction, with no extra training. The A3/B+W difference therefore mixes three effects.
  - **Clean control available but not evaluated.** C is already the control for "extra steps + data without correction". The clean comparison is C+W: C's checkpoint evaluated with the 8/Q1(17)/8 correction. It needs no new training.
  - **Seed variability unknown.** All arms have a single seed. The claimed factors (1.3 in mean, 1.35–2.2 on two cells' sensitivity maxima) are small enough to be within seed-to-seed variability, which is unknown.
- **Why it matters:** The abstract and contribution (ii) present training through the correction as a key ingredient. With the present evidence, the effect could come entirely from continued training on more geometries.
- **Suggested fix:**
  - Evaluate C+W on the 80 geometries and the 14 assembly configurations.
  - Train A3 and C with at least three seeds, and report mean ± spread.
  - Report paired per-geometry differences (A3 − C+W) with a bootstrap confidence interval.
  - If C+W ≈ A3, rephrase the claim accordingly.
- **Fix type:** re-analysis of existing data (C+W); new experiment (seeds)
- **Confidence:** high

### R2-03 — No ablation of the sensitivity term in the loss or of the architectural components
- **Severity:** Major
- **Location:** Eq. (8) and Section 3.3: "the reported configurations use w_s=1"; Abstract: "Trained through the correction with an objective that includes thickness sensitivity"; Section 3.1 and Appendix G.1 (multilevel hierarchy, ghost-face interactions, weak-region layers, stiffness-weighted scatter Eq. (G.1), coefficient bounds).
- **Issue:**
  - **Sensitivity term.** Every arm uses w_s = 1, so the paper never shows that the sensitivity term helps. This is notable because Section 6.4 finds the quadratic term dominant, in which case energy training alone might already control sensitivity.
  - **Architecture.** The architecture is elaborate (603k parameters, 24 local interaction layers with 4 heads, 12 coarse convolutions, a separate weak-region block), and no component is ablated. Since the correction does most of the work (R2-01), it is an open question whether a much simpler linear-in-q network, or even a geometry-agnostic one, reaches similar accuracy after correction and correction-aware training.
- **Why it matters:** Contribution (ii) and the architectural description in Section 3 are the main ML content. Without ablations, a reader cannot tell which design decisions are necessary. This also matters for transfer to other cell families.
- **Suggested fix:**
  - Run a small ablation study, at minimum with A3's training protocol:
    - (a) w_s = 0;
    - (b) no latent hierarchy (local interactions only);
    - (c) no weak-region layers;
    - (d) geometry branch replaced by fixed, non-conditioned coefficients (i.e. a linear map depending only on mesh incidences);
    - (e) a reduced-width variant.
  - Report energy excess on the 80 geometries and sensitivity maxima on the assembly set, together with parameter counts and application cost.
- **Fix type:** new experiment
- **Confidence:** high

### R2-04 — The key non-learned baseline at matched accuracy and matched hardware is missing
- **Severity:** Major
- **Location:** Section 6.5 and Table 3 ("Quadrupling the smoothing budget to 32 steps per stage does not close the gap"); Section 6.10 and Tables 5–6: "The conventional route runs on the 16 host cores… the learned route runs on one RTX 5090".
- **Issue:**
  - **Matched-accuracy iterative baseline.** The only competitor at matched hardware is exact condensation (CPU PARDISO), which delivers a different accuracy (exact versus ~0.07%). The obvious alternative is an iterative interior solve on the same GPU, stopped at the accuracy A3 reaches. Two linear variants are natural:
    - a fixed number m of repeated two-grid cycles (8/Q1(17)/8)^m, or a cycle with a Q2(17) coarse space (ST04 shows Q2 is stronger), started from zero or harmonic;
    - a Chebyshev-accelerated two-grid iteration of fixed degree.
  - **What Table 3 does and does not show.** Table 3 increases only the smoothing per stage, with a single coarse solve. It shows that the harmonic start plus one cycle is worse. It does not show how many cycles, and how much time, the harmonic start needs to match A3.
  - **Hardware confound.** Section 7.4 says application time is "dominated by the 32 stiffness actions of the correction, not by the network". The trade-off between network cost and extra cycles is therefore the central cost question. Tables 5–6 also compare a GPU method with a CPU solver, which confounds algorithm and hardware; the front end is also run on the GPU for the learned route and on the CPU for the conventional one.
- **Why it matters:** For practitioners, the question is whether the trained network beats a classical two-grid interior solver at equal accuracy and equal hardware. The 9–29× speed-ups do not answer this.
- **Suggested fix:**
  - Add an accuracy–time Pareto plot (energy excess and sensitivity error against wall-clock time per cell and per application, on the GPU) for {zero, harmonic, B, A3} × {1, 2, 4, … two-grid cycles}, on at least the 20 heavy-cut geometries.
  - If feasible, add a GPU sparse direct solver (e.g. cuDSS) or GPU AMG-preconditioned interior solves as the matched-hardware exact baseline.
  - Report CPU and GPU front ends separately, so that hardware and method effects can be told apart.
  - Stop calling the learned setup "condensing a cell". No condensed matrix is formed, and the operator is approximate.
- **Fix type:** new experiment
- **Confidence:** high

### R2-05 — Offline cost (training data generation and training) is not reported, and the amortisation claim ignores it
- **Severity:** Major
- **Location:** Section 6.1: "A3's continuation took 2.7 h on one NVIDIA GeForce RTX 5090"; Section 6.10: "the learned route is cheaper for any number of queries per design iteration"; Conclusions: "so it is cheaper for any number of queries"; Appendix G.2 (direction banks).
- **Issue:**
  - **Data generation.** Each training geometry needs a teacher: CutFEM assembly, an interior factorisation, Neumann/pinned solves to generate mechanically defined directions (force, support, glued), energy normalisation q^T S q, and finite-difference sensitivity labels for 8 corners. The number of directions per geometry and the total number of exact solves are not given. Neither is the wall-clock or CPU/GPU-hour cost of generating the banks for 148 + 305 + 591 geometries.
  - **Training.** Only A3's 2.7 h continuation is reported. The full pipeline is P0 → B (40,000 steps) → A3, and its total cost is not given.
  - **Rough break-even estimate.** Per Table 5, the learned route saves roughly 7–53 s per cell ("ready for queries" 3.2–7.6 s versus 9.9–61 s). The 2.7 h continuation alone therefore needs about 180–1,450 cell preparations to amortise, before counting B's training and all data generation.
  - **Retraining.** Because the model is tied to one family, one resolution and one material (R2-07), this offline cost recurs for every new setting.
- **Why it matters:** "Cheaper for any number of queries" holds only per query after the model exists. For a CMAME audience evaluating learned substructures, the end-to-end cost is essential. It is also a main criticism of the PIML-type approaches the paper contrasts with.
- **Suggested fix:**
  - Report, per training geometry: the number of directions per class, the exact solves needed, and the time to generate banks and sensitivity labels.
  - Report the total offline cost (data + all training stages) in GPU-h and CPU-h.
  - Give a break-even count in cell preparations and design iterations, for example for the 8- and 27-cell optimisation of Section 6.11.
  - Qualify the "any number of queries" statements in the abstract, Section 6.10 and the Conclusions.
- **Fix type:** text only + re-analysis of existing logs
- **Confidence:** high

### R2-06 — The evaluation is not fully independent of development and selection; "80 unseen geometries" overstates this
- **Severity:** Major
- **Location:** Abstract: "on 80 unseen geometries"; Figure 5 caption; Section 6.1: "the weights of the continued predictors were selected on a validation list whose first 40 geometries include 20 of the 80 reported validation geometries… the cells used in the assembly examples served repeatedly as development cases"; Supplementary R1 (U1…L1 are `fresh_val_2000–2010`, i.e. members of the 80-geometry validation set); ST12 (selection uses views 0 and 17 and includes a sensitivity term).
- **Issue:**
  - **Checkpoint selection.** 20 of the 80 geometries entered checkpoint selection.
  - **Design and hyperparameter choices.** The detailed cells (U1, U2, M1, M2, H1, H2, H3, L1) all belong to the validation set and were development cases. They underlie the fixed-weight correction study, the spectral analysis, the choice of correction schedule (8/Q1(17)/8, a = b/30) and all 14 assembly configurations. The 60-geometry subset guards only against checkpoint selection, not against these decisions.
  - **Lattices.** The lattice cells are newly generated but in-distribution, and their provenance is not stated in the main text.
  - **Transparency.** The paper is transparent about all of this, but the headline claims still read as test-set results.
- **Why it matters:** The central assembled-accuracy claim ("below 0.7% in all fourteen configurations") rests entirely on development cells. Generalisation of the assembled sensitivity accuracy is therefore not established.
- **Suggested fix:**
  - Generate a sealed test set with the same generator and a new random stream, e.g. 40 geometries stratified as before. Evaluate A3, B+W, C+W and the classical baselines once.
  - Draw assembly target cells and configurations at random from this set (e.g. 20 random pairs, both orientations), and report the full distribution of compliance and sensitivity maxima.
  - Replace "80 unseen" with "80 validation geometries not used in training (20 of them used in checkpoint selection)".
  - State the provenance of the lattice cells.
- **Fix type:** new experiment + text
- **Confidence:** high

### R2-07 — The generalisation scope is narrow, not tested out of distribution, and not reflected in the title or abstract
- **Severity:** Major
- **Location:** Eq. (1); Section 6.1: "Canonical cut normals are (cos ϑ, sin ϑ, 0), with 0<ϑ<π/4… multiple cuts per cell and curved boundaries are outside the present study"; Table 1 (n = 32, E_Y = 1, ν = 0.3, γ = 1e-4 for all geometries); Appendix A.1 and G.1 (network hierarchy fixed to 65/33/17/9 positions per axis); Section 6.2: "For heavily cut cells the discretisation error of the reference at n=32 is thus of order one percent".
- **Issue:**
  - **Single family and single cut.** All training and test data come from a single TPMS family (P-type), a single planar cut, fixed material and stabilisation, and a fixed background resolution.
  - **Uncovered cut normals.** Under the 48 cube symmetries, the canonical normals map only to normals with at least one zero Cartesian component. General oblique cuts, with all three components non-zero, were never seen.
  - **Uncovered cell types.** Edge and corner cells of a cut lattice (two or three planes), which any real finite TPMS part contains, are excluded.
  - **Fixed resolution.** The architecture is tied to the n = 32 grid, where the reference itself has about 1% error for heavy cuts. Refining the discretisation therefore requires retraining.
  - **No out-of-distribution test.** Every test (80 geometries, lattices with corner parameters 0.25–0.56) lies inside the training domain [0.1752, 0.6993].
- **Why it matters:** The title and abstract speak of "cut thin-walled TPMS cells" generally. Practitioners need to know what the trained operator can and cannot be applied to without retraining, and how it fails outside its domain. Silent degradation matters for design, where an optimiser may drive thicknesses to the bounds.
- **Suggested fix:**
  - State the scope explicitly in the title or abstract (P-type, single planar cut, n = 32, fixed material) and in a dedicated limitations paragraph.
  - Add at least a small out-of-distribution stress test, e.g.:
    - (a) oblique normals with three non-zero components;
    - (b) corner parameters 10–20% outside the training range;
    - (c) a different ν;
    - (d) a doubly cut corner cell.
  - Report whether the residual-based correction degrades gracefully. The residual/energy identity (Eq. 10) also provides a label-free a posteriori indicator that could flag out-of-distribution inputs; please demonstrate it.
- **Fix type:** text + new experiment
- **Confidence:** high

### R2-08 — Statistical reporting: geometry-mean aggregation, no worst-direction or operator-norm error, no uncertainty
- **Severity:** Major
- **Location:** Section 6.1: "The population maximum is consequently the largest geometry-level direction mean"; Abstract: "with at most 0.65% for any geometry"; Supplementary R2: "The aggregate records do not retain the realised direction count for every geometry and class"; Appendix J.7 (coverage argument).
- **Issue:**
  - **Worst direction is what assembly sees.** All population statistics are geometry means over sampled directions. The assembled solve selects its own direction, and Appendix J.7 correctly notes that sampled means give no bound on unsampled directions. The paper has the tools to estimate the actual operator error: the adversarial block search of Appendix G.3, and Lanczos on the pencil (Ŝ, S) using the teacher factorisation. Yet no worst-direction or λ_max(S⁻¹Ŝ) − 1 statistic is reported for A3.
  - **Missing counts and uncertainty.** The number of evaluation directions per geometry and class is not recorded. There are no confidence intervals, and there is no seed variability.
  - **Ambiguous wording.** "At most 0.65% for any geometry" is easily read as a worst-case bound.
- **Why it matters:** Readers cannot assess robustness from means and 90th percentiles of means. For a substructure operator, the relevant quantity is the energy-norm operator error, as the authors' own theory (Eqs. 9–12, J.8) makes clear.
- **Suggested fix:**
  - For A3 (and B+W / C+W), report per geometry the largest generalised Rayleigh quotient of (Ŝ − S, S) on the non-rigid complement, estimated by a few Lanczos iterations or by the existing adversarial search at evaluation time. Report it at least for the 20 heavy-cut geometries, ideally for all 80.
  - Record and report direction counts.
  - Give bootstrap confidence intervals for population means and paired differences.
  - Reword the abstract as "largest geometry-mean directional error 0.65%".
- **Fix type:** re-analysis of existing data (checkpoints and teachers exist)
- **Confidence:** high

### R2-09 — Predictor labels are hard to follow; some arms add little
- **Severity:** Minor
- **Location:** Table 2; Section 6.1: "S8 continues B with eight smoothing steps at evaluation; only part of its training steps passed through the smoothing stage"; "The historical computational examples use a separate uncorrected predictor D"; Appendix G.3 uses internal run IDs ("The v2L1 (B), A0 (C), A2 (S8)…"); Supplementary R1.
- **Issue:**
  - **Opaque labels.** Eight labels (P0, B, C, S8, A2b, B+W, A3, D) plus internal run identifiers make the ablation logic hard to follow. The labels do not encode the factors varied (data size, extra steps, correction in training, correction at evaluation). "W" in B+W is never tied to the correction 𝒲.
  - **Ill-defined arm.** S8 has an unspecified, partial training protocol and is explicitly "not interpreted".
  - **Marginal arms.** P0 appears only in ST01. D appears only in the historical supplement.
- **Why it matters:** A clean factorial design is essential for the ablation claims (R2-02).
- **Suggested fix:**
  - Use descriptive labels, e.g. N305, N591, N591⊕Cheb8, N305+𝒲, N591+𝒲 (C+W), N591⊕𝒲.
  - Present the arms as a 2×2(×2) grid: {305, 591 continuation} × {correction at evaluation: no/yes} × {correction in training: no/yes}.
  - Drop S8, or specify the fraction of steps that passed through smoothing.
  - Move P0 and D to the supplement.
  - Remove internal run IDs (v2L1, A0_ctrl, B2grid, …) from the appendices.
- **Fix type:** text only (plus cut or move)
- **Confidence:** high

### R2-10 — The architecture is well described but not fully reproducible; there is no code or data availability statement
- **Severity:** Minor
- **Location:** Appendix G.1: "Bounds are fixed model-state arrays indexed by coefficient type, layer and gather/scatter role"; "The B, C and S8 configurations set k_b=0.5"; "their optional five-feature extension is disabled"; "Each layer has its own learned λ_mix"; Section 3.1.
- **Issue:**
  - **Unspecified details.** The following are not given:
    - the numerical values of the coefficient bounds a_max;
    - the initialisation of λ_mix, σ_ℓ and the weights;
    - the definition and dimension source of the "eight-dimensional slot embedding" (learned? per stencil type?);
    - whether k_b = 0.5 and the disabled feature extension also apply to A2b and A3;
    - how the 24 local layers split between element and face interactions per "pair";
    - a parameter breakdown per block.
  - **Training-loop details.** Pool sampling (with or without replacement; view sampling per step) and the adversarial-direction refresh schedule during training are not fully specified.
  - **Availability.** No statement on code, trained weights or the geometry generator is given, although the supplement's archive identifiers indicate that these exist.
- **Why it matters:** Reproducibility is a CMAME requirement, and the architecture cannot be re-implemented from the text alone.
- **Suggested fix:**
  - Add a complete hyperparameter table (all bounds, initialisations, per-block parameter counts, per-arm flags).
  - Add a code and data availability statement, and release the model, generator and teacher.
  - A pseudo-code listing of one forward pass, with tensor shapes, would help.
- **Fix type:** text only
- **Confidence:** high

### R2-11 — "591 training geometries" may overstate the data each continuation actually used
- **Severity:** Minor
- **Location:** Appendix G.3: "keep three geometries in the device pool, and replace a pool entry every 100 steps"; Table 2 "Training geometries 591"; Section 6.1: "A3 continues B on the larger training population".
- **Issue:**
  - **Arithmetic.** A pool of three with one replacement per 100 steps admits at most about 150 new geometries in a 15,000-step continuation. That is roughly a quarter of 591, each seen under a limited number of cube views. B (40,000 steps) can have visited all 305.
  - **Consequence.** The "larger population" label and the implicit data-scaling reading (P0 148 → B 305 → C 591) are therefore uncertain. The P0/B/C differences are further confounded by different step counts and by continuation.
- **Why it matters:** Reviewers and readers will draw data-scaling conclusions from Table 2. The real sample complexity is also needed for R2-05.
- **Suggested fix:**
  - Report the number of distinct geometries (and geometry–view pairs) actually visited by each arm.
  - If available, add a simple learning curve: validation energy excess against distinct training geometries seen.
- **Fix type:** re-analysis of existing logs
- **Confidence:** medium (depends on the actual pool sampling, which is not fully specified)

### R2-12 — Checkpoint selection is described inconsistently, and view dependence of the principal model is not evaluated
- **Severity:** Minor
- **Location:** Appendix G.3: "The weights are selected at step 15,000 on the validation list"; ST12: "C, S8 and A3 are evaluated every 7,500 updates… the lowest finite score is selected" and "A2b is evaluated with its EMA weights at update 15,000"; Section 6.1: "selection compared two checkpoints per predictor"; ST01c (view 17 only for P0 and B, on an earlier evaluation with 15–20 geometries for the enriched classes); Appendix J.7: "Training on transformed examples encourages this relation but does not prove…".
- **Issue:**
  - **Inconsistency.** The three descriptions of selection disagree or are incomplete. A2b uses a different protocol from A3.
  - **Views.** Selection used views 0 and 17, but A3 is reported only in view 0. Since the network is not equivariant by construction, frame dependence is a real error source. B's consistent-traction mean changes from 6.62% to 7.49% between views.
- **Why it matters:** Arms should share a protocol for fair comparison. The frame sensitivity of A3 after correction is unknown.
- **Suggested fix:**
  - Harmonise the text.
  - Report A3 (and B+W) on all 80 geometries in several of the 48 views, e.g. 4–8 random views, and report the view-to-view spread per geometry.
- **Fix type:** text + re-analysis of existing data
- **Confidence:** high

### R2-13 — The per-cell relative sensitivity metric overweights weakly participating cells; report a gradient-level metric as well
- **Severity:** Minor
- **Location:** Section 2.3, e_s = ‖s̃ − s‖/‖s‖ per cell; Section 6.6 and Table 4 (U1/x N-z: target energy share 0.0013, sensitivity error 3.89–4.89%); Section 7.2: "normalising by a small reference-vector norm can amplify the relative discrepancy"; the "3% accuracy reference".
- **Issue:**
  - **Normalisation.** The headline "compliance accurate, sensitivity not" examples are cells whose sensitivity norm is tiny relative to the structural gradient.
  - **What optimisers use.** OC/MMA updates depend on the assembled gradient with shared variables (Section 4.3, last paragraph), not on per-cell relative errors.
  - **Arbitrary threshold.** The 3% threshold is not motivated.
- **Why it matters:** The mechanical message is valid, but its practical significance for design is best shown at the level of the assembled shared-variable gradient.
- **Suggested fix:**
  - Additionally report ‖∇_τg Ĉ − ∇_τg C‖ / ‖∇_τg C‖ and the cosine similarity of the assembled gradient, in the pairs and lattices.
  - Justify, or drop, the 3% threshold. Alternatively, relate it to the optimiser's tolerance in Section 6.11.
- **Fix type:** re-analysis of existing data
- **Confidence:** medium

### R2-14 — The complete surrogate derivative (second term of Eq. 14) is never measured
- **Severity:** Minor (it becomes Major once Section 6.11 uses the operator for optimisation)
- **Location:** Eq. (14): "The reported field-based estimates contain only the first term"; Section 7.2: "Measuring this term within a complete design optimisation is left to subsequent work."
- **Issue:**
  - **Which derivative is reported.** All reported sensitivities evaluate the exact K_{,c} on the learned field. The derivative of the surrogate objective also contains −2 (F_{I,c} q̂)^T r_I. The paper argues it is small from the energy excess (≈ 2.7% of ‖û‖_K for A3), but F_{I,c} of a network composed with a geometry-dependent correction is uncontrolled.
  - **Which derivative an optimiser needs.** An optimiser using the field-based estimate is inconsistent with the objective it evaluates. This can affect line search and MMA convergence.
- **Why it matters:** Section 6.11 will depend on it.
- **Suggested fix:** Measure the second term for A3 on the assembly cells. Use central differences of Ĉ in τ_c with the full pipeline re-evaluated (encoder + correction setup), and compare with s̃.
- **Fix type:** new experiment (small)
- **Confidence:** high

### R2-15 — fp32 network arithmetic sets a residual floor in assembled solves
- **Severity:** Minor
- **Location:** Section 6.9: "its recomputed residual stagnates at 3.6×10⁻⁴… (3.2×10⁻³ in the 3×3×1 layer)"; Table 1 ("Network in single precision"); ST14d (historical D: recomputed residual 2e-2, compliance error 3.4%).
- **Issue:** The forward and transpose actions are consistent only to fp32 precision, so the assembled residual stagnates at a floor. That floor differs by an order of magnitude between two eight-cell lattices: 3.6e-4 for the 2×2×2 block and 3.2e-3 for the 3×3×1 layer. Its dependence on layout and size, its effect on sensitivities, and its scaling to 27 or more cells are not analysed.
- **Why it matters:** In larger lattices or optimisation loops, this floor may dominate the approximation error, and Eq. (18) says the residual work enters the compliance comparison.
- **Suggested fix:**
  - Report the results with the network in fp64 (inference only) as a check.
  - Report the residual floor against the number of cells.
  - Evaluate the Eq. (18) residual-work term for the lattice results.
- **Fix type:** re-analysis / small new experiment
- **Confidence:** medium

### R2-16 — The learned-start versus harmonic-start comparison rests on five development cells, and the harmonic baseline is weak
- **Severity:** Minor
- **Location:** Abstract: "the learned starting field is 7 to 290 times more accurate than a harmonic extension"; Section 6.5 and Table 3 (five cells, 32 directions, B's field); "The harmonic extension solves a graph Laplacian on the element connectivity weighted by material volume".
- **Issue:**
  - **Scope.** A population-level claim in the abstract is based on five development cells.
  - **Baseline strength.** The scalar graph-Laplacian extension ignores elasticity and the cut geometry's stiffness. A vector elastic "discrete harmonic" start, e.g. a few AMG V-cycles on A, or the Q1(17) coarse solve alone as the start, would be a stronger and still cheap comparator.
  - **Cost.** Evaluating the non-learned starts over all 80 geometries needs no training.
- **Why it matters:** This is the main quantitative evidence that the network is worth having (see also R2-04).
- **Suggested fix:**
  - Run the zero, harmonic and elastic-coarse starts with the same correction on all 80 geometries, and report population statistics.
  - Qualify the abstract to state the population and the predictor (B) used.
- **Fix type:** re-analysis of existing data / small new experiment
- **Confidence:** high

### R2-17 — The energy loss needs almost no labels; clarify what requires exact solves and discuss data-free variants
- **Severity:** Minor
- **Location:** Eq. (8) and Section 3.3: "each retained direction is normalised to q_j^T S q_j = 1 using the reference substructure solution".
- **Issue:**
  - **The energy term needs only normalisation.** Because Ŝ ⪰ S, log(q^T Ŝ q) − log(q^T S q) has the same θ-gradient as log(q^T Ŝ q). The energy term is therefore label-free apart from the direction weighting, and minimising it is a Ritz principle.
  - **Where exact solves are really needed.** They are needed to generate the mechanically defined directions (force, support and glued responses require Neumann solves) and the sensitivity labels.
  - **Not discussed.** The paper does not spell this out, nor relate it to data-free PIML training (Huang et al. 2024).
- **Why it matters:** It bears directly on training cost (R2-05) and on how easily the method transfers to new cell families.
- **Suggested fix:**
  - State which labels require exact solves.
  - Consider, or at least discuss, a variant that normalises by a cheap proxy (e.g. q^T K_PP q, or the corrected energy itself) and uses only prescribed-displacement directions. Report its accuracy if feasible.
- **Fix type:** text only (optional new experiment)
- **Confidence:** high

---

## 4. Requirements for the design-optimisation example (Section 6.11, not reviewed)

For the example to support the paper's claims about design, it should:
- **(a) State which derivative drives the optimiser** (field-based s̃ or the complete Ĉ_{,c}) and quantify the neglected term along the optimisation path (R2-14).
- **(b) Compare the final design and the objective history with an exact-condensation run from the same start,** and verify the learned final design with the exact model.
- **(c) Report whether iterates leave the training domain** (thickness range, active-set changes), and how the operator behaves if they do (R2-07). Use the residual indicator of Eq. (10) as a monitor.
- **(d) Include the full cost:** online wall-clock per iteration, and total, against the exact route on matched hardware, plus the amortised offline cost (R2-05).
- **(e) Use at least a 27-cell lattice,** so that the fp32 residual floor (R2-15) and the error accumulation across many learned cells are tested.
- **(f) Include cut cells in the lattice,** including edge or corner cells if the scope is extended.

---

## 5. Top three concerns

1. **The ML contributions are not isolated (R2-02, R2-03, R2-04).**
   - The benefit of training through the correction is confounded with extra data and steps; C+W and multiple seeds are needed.
   - Neither the sensitivity loss term nor any architectural component is ablated.
   - The paper does not compare against the obvious non-learned competitor: more two-grid cycles from a cheap start on the same GPU, at matched accuracy.
2. **Independence and scope of the evaluation (R2-06, R2-07, R2-08).**
   - The assembled-accuracy claims rest on development cells from the validation set.
   - 20 of the "80 unseen" geometries entered checkpoint selection.
   - There is no worst-direction or operator-norm error and no uncertainty quantification.
   - Generalisation is shown only in distribution, for one TPMS family, one planar cut with restricted normals, one resolution and one material.
3. **Cost accounting (R2-05, R2-04).**
   - The claim of being cheaper for any number of queries ignores the offline cost of generating exact training data and of training.
   - The claim compares a GPU method with a CPU direct solver.
   - A break-even analysis and a matched-hardware, matched-accuracy comparison are needed.
