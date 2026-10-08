# Review R4: structural/topology optimisation and graded-lattice perspective

**Manuscript:** "Learned static condensation for cut thin-walled TPMS cells with equilibrium correction" (CMAME submission)
**Reviewer focus:** thickness-sensitivity treatment (Sections 4, 6.6, 6.7, 7.2; Appendices H, J.4–J.5, J.9), design parametrisation, realism of lattice examples, and what a practitioner would need to use this in graded-TPMS design. Section 6.11 and the "[Placeholder]" sentences are not reviewed. Section 4 below sets out what that example must show.

---

## 1. Summary

The manuscript approximates the Schur complement of cut, thin-walled P-type TPMS cells on the full retained space. A geometry-conditioned network, linear in the retained displacement, supplies an initial interior field. A fixed, geometry-specific Chebyshev/Galerkin correction then reduces the interior equilibrium residual, and the condensed stiffness F^T K F is applied matrix-free. The variational analysis is correct and clearly presented: the Ritz identity (9), the compliance identity (11), the participation bound (12) and the sensitivity expansions (13)–(14). The operator-level verification (symmetry, rigid modes, Chebyshev interval, reference convergence) is careful. For the design use that motivates the paper, the evidence is much thinner. The sensitivity that is validated is the field-based estimate −û^T K_,c û on a per-cell, locally normalised 8-vector. It is not the gradient of the surrogate objective. The global, shared-variable gradient that an optimiser actually uses is never assessed. Differentiability of the discrete model across the topology events that an optimiser crosses is not examined. The assembled examples are few, small, loaded near-uniformly, and use development cells. The cost comparison omits the baselines a topology-optimisation practitioner would use: matrix-free GPU multigrid on the full model, and homogenisation-based graded design.

## 2. Recommendation: **Major revision**

The paper makes a sound and useful methodological contribution. Keeping the full retained space and putting all approximation error in an interior extension whose error can be reduced is a good idea, and the error analysis is a real strength. However, the paper repeatedly positions the method for thickness design: in the title of the sensitivity analysis, contribution (iv), Sections 7.2 and 7.4, and the planned Section 6.11. The design-relevant claims are not yet supported by the right quantities:

- the complete derivative of the surrogate is not measured;
- the global gradient error is not measured;
- the behaviour across active-set, ghost-face and feature switches is not characterised.

Most of these issues can be settled by computations on existing configurations plus a well-designed optimisation example (Section 4 of this review). None of them undermines the analysis itself. I would be glad to see a revised version.

---

## 3. Findings

### R4-01: The complete derivative of the surrogate compliance is never computed, and the argument that its extra term is small does not hold
- **Severity:** Major
- **Location:** Section 4.3, Eq. (14): "The reported field-based estimates contain only the first term." Section 7.2: "at A3's population-mean excess of 0.074% it is about 2.7% of ‖û‖_K … so the extension term is correspondingly small unless F_{I,c}q̂ greatly exceeds the field's own design derivative. Measuring this term within a complete design optimisation is left to subsequent work." Appendix H, Eq. (H.5).
- **Issue:** All reported sensitivities are the field-based s̃_c. The gradient of the objective that the surrogate actually evaluates, Ĉ_,c = s̃_c − 2(F_{I,c}q̂)^T r_I, is not computed anywhere. The Section 7.2 estimate does not show that the second term is small. Using (H.5):
  - |Ĉ_,c − s̃_c| ≤ 2‖F_{I,c}q̂‖_A ‖r_I‖_{A⁻¹}, with ‖r_I‖_{A⁻¹} ≈ √ε ‖u‖_K ≈ 0.027‖u‖_K.
  - If F_{I,c} ≈ E_{I,c}, then ‖E_{I,c}q‖_A = ‖J_I K_,c u‖_{A⁻¹}, so the relative bound is roughly 2·0.027·‖u‖_K‖K_,c u‖_{A⁻¹}/(u^T K_,c u).
  - The last ratio is at least of order one. For a corner parameter whose K_,c acts on only part of the cell it is larger (roughly the square root of total over locally affected energy).
  - The bound is therefore at least about 5% per component, and about 16% at the worst-geometry excess of 0.65%. That is an order of magnitude above the 0.7% field-based errors the paper reports.
  
  The bound may be pessimistic. Appendix J.5 shows that the residual term can cancel the linear part of the field-estimate error, so Ĉ_,c could be *more* accurate than s̃_c. The paper does not know which case applies, and the text currently implies the matter is settled.
- **Why it matters:** An optimiser driven by s̃_c while evaluating Ĉ gets gradients that are inconsistent with its own objective. Line-search, GCMMA inner loops and KKT-based convergence checks assume consistency. Conversely, s̃_c might be the better estimate of the *true* sensitivity C_,c, which is what matters for the final design. The paper needs to say which quantity it recommends for optimisation, and why.
- **Suggested fix:**
  1. Compute Ĉ_,c by reverse-mode differentiation through network, rigid split and correction. The machinery exists: training already backpropagates through the correction (Appendix G.3).
  2. Verify Ĉ_,c against central differences of Ĉ over several step sizes.
  3. On all 14 two-cell configurations and both lattices, report C_,c (exact), s̃_c and Ĉ_,c side by side: error of each against C_,c, and the discrepancy |Ĉ_,c − s̃_c|.
  4. Rewrite the 7.2 paragraph according to the result.
  5. Report the cost of the complete derivative per design iteration.
- **Fix type:** re-analysis of existing data (new computations on existing configurations)
- **Confidence:** High that the term is unmeasured and the argument inconclusive; medium on the numerical size of the bound.

### R4-02: Differentiability across the design changes an optimiser actually crosses
- **Severity:** Major
- **Location:** Section 4.3: "On a differentiable design interval with fixed active and retained coordinates…". Appendix A.1: "The ghost-face set … internal background faces whose two neighbouring elements are active and are not both certified as completely filled with material." Appendix G.1: "A node is designated weakly supported when this norm is less than 0.01 times its median over active nodes." Appendix F.2: shifts tried "in the order 0, 10⁻¹², …, 10⁻⁴". Appendix H: "The moment quadrature is recomputed at each perturbed thickness, so its clipping and integration branches may change."
- **Issue:** Every sensitivity statement assumes fixed active, retained and ghost sets. When τ changes over an optimisation, several things switch discretely:
  - elements activate and deactivate;
  - ghost-face membership changes, and the penalty γ g_h does not scale with the material fraction, so the stiffness jumps when a face enters or leaves the set;
  - retained cut-band nodes appear and disappear;
  - the network's binary node features (weak-support threshold, retained and cut-band membership) switch;
  - the adaptive subcell refinement changes branches;
  - the coarse-factorisation shift is chosen from a discrete list;
  - the Chebyshev endpoint b comes from a finite power iteration.
  
  So C(τ) and Ĉ(τ) are only piecewise smooth, and Ĉ may have jumps that C does not. The ghost energy share is small under consistent tractions (below 0.05%), but 49–81% under nodal forces (Section 6.3). The size of these jumps relative to the gradient step of an optimiser is unknown.
- **Why it matters:** Gradient-based optimisers (MMA and OC with move limits) tolerate small non-smoothness, but can stall or oscillate if Ĉ is noisy at the level of the design change per iteration. This matters most near convergence, where the paper's claimed 0.1–0.7% sensitivity accuracy is supposed to count.
- **Suggested fix:**
  - Show one-dimensional sweeps of C(τ_c), Ĉ(τ_c), s̃_c and Ĉ_,c over the design range (for example 0.18→0.70) for two or three cells, including a cut cell. Use fine enough increments to cross element activation and ghost-set changes.
  - Report the jump magnitudes against typical per-iteration objective changes.
  - Discuss remedies: smooth weak-support features, a ghost penalty scaled by volume fraction or kept on a fixed extended set, and a fixed seed and iteration count for the spectral estimate (already done), with a check of its smoothness.
- **Fix type:** new experiment (inexpensive)
- **Confidence:** High that the switches exist (they follow from the stated definitions); medium on their practical size.

### R4-03: The sensitivity metric is not what an optimiser sees, and the separation claim partly reflects local normalisation
- **Severity:** Major
- **Location:** Section 2.3: "e_s = ‖s̃ − s‖₂/‖s‖₂". Section 4.3, last paragraph: "∇_{τ_g}C = Σ_m (∂τ_m/∂τ_g)^T s_m". Section 6.9: "The largest eight-corner sensitivity error over all cells is 0.14%". Contribution (iv); Figure 10(d); Section 7.2: "normalising by a small reference-vector norm can amplify the relative discrepancy."
- **Issue:**
  - All sensitivity errors are relative errors of each cell's own 8-vector. The shared-variable gradient of Section 4.3 is defined but never evaluated, not even for the lattices of Section 6.9 where every corner is shared by up to eight cells. Errors from neighbouring cells may add or cancel in that gradient.
  - Figure 10(d) shows that the local relative sensitivity error is essentially independent of participation, while the compliance error scales with it (Fig. 10c). This "separation" is largely what one expects from dividing by a local, participation-dependent reference norm. In the headline example (U1/x, N-z; target energy share 0.13%), the target's 8-vector is presumably a very small part of the global gradient, so a 5.8% local error may not matter for the design update.
  - The 2-norm over eight components also hides componentwise errors. An OC update uses per-variable ratios, so per-variable relative errors matter there, especially for variables near the Lagrange-multiplier threshold.
  - The observation that an accurate objective does not imply an accurate gradient is well known for approximate reanalysis and inexact solvers in topology optimisation (e.g. Amir, Bendsøe & Sigmund 2009, IJNME; Amir, Stolpe & Sigmund 2010, SMO). The paper should connect to that literature rather than present the point as new.
- **Why it matters:** Contribution (iv) is framed as a principal mechanical finding with design consequences. Without a gradient-level metric, the reader cannot tell whether the consequences are real (whether C's 11% local errors would change an optimised design) or an artefact of the metric.
- **Suggested fix:**
  - For every assembled case, report the error of the assembled shared-variable gradient: ‖∇̃C − ∇C‖/‖∇C‖, cosine similarity, the distribution of per-variable relative errors (median, 95th percentile, max) and sign agreement.
  - Report each cell's 8-vector norm as a fraction of the global gradient norm, alongside the local relative error.
  - Temper contribution (iv) and Section 8 to the scope actually shown, and cite the approximate-reanalysis and inexact-solve sensitivity literature.
  - Show the design consequence explicitly in Section 6.11 (see Section 4 of this review).
- **Fix type:** re-analysis of existing data (plus text)
- **Confidence:** High

### R4-04: Assembled sensitivity accuracy rests on seven development cells; the worst population geometries are never assembled
- **Severity:** Major
- **Location:** Section 6.1: "the cells used in the assembly examples served repeatedly as development cases during method development, so the assembly results characterise these configurations rather than an independent test sample." Abstract: "below 0.7% in all fourteen configurations". Table ST13.
- **Issue:**
  - The 14 "configurations" are 7 target cells in two orientations, all development cases, each with an exact neighbour.
  - The population study shows that the largest geometry-mean energy excess of A3 (0.65%) is about five times M1's value (0.13%). Section 6.4 and Table ST02a show that the sensitivity error can exceed the energy error by a factor of two or more (H2: 35% energy, 75% sensitivity for B).
  - Assembled sensitivity errors on the worst validation geometries are therefore likely to be larger than 0.7%, and they were never measured.
  - The weight-selection score (Table ST12) also includes a sensitivity term evaluated on a list that overlaps 20 of the 80 validation geometries.
- **Why it matters:** An optimiser will visit many geometries, and it will be drawn to atypical ones (see R4-09). The headline claim needs to hold for held-out and worst-case cells, not only for development cells.
- **Suggested fix:** Run the two-cell assembly protocol (six face loads plus cut loads) on the 60 validation geometries that were not used for selection, or at least on the 10–15 with the largest A3 energy error in each stratum. Report the distribution of compliance and sensitivity errors (local and global-gradient versions, R4-03). Rewrite the abstract claim as a distributional statement.
- **Fix type:** new experiment (reuses the existing pipeline)
- **Confidence:** High

### R4-05: The design parametrisation is not translated into quantities a lattice designer can use
- **Severity:** Major
- **Location:** Section 2.1: "They control the implicit band width … they are not pointwise physical wall thicknesses." Section 6.9: "Corner parameters range from 0.25 to 0.56". Table ST11: "every corner parameter to [0.1752, 0.6993], the corner span to at most 0.47, and the maximum reference-coordinate gradient norm to at most 0.47"; "centre volume fraction … between 0.1 and 0.4". Section 6.11 placeholder: "the eight corner thicknesses of each cell".
- **Issue:**
  - The paper speaks of "thickness sensitivity" but never gives the map from τ to physical wall thickness, relative density, or homogenised stiffness.
  - My own estimate: at the point (¼,¼,¼), |∇φ| = 2π√3 ≈ 10.9, so the band at τ ≈ 0.18 is about 0.03 box lengths thick. That is roughly one background element (h = 1/32) with Q2 displacements. The thinnest admissible walls are therefore resolved by about one element.
  - Section 6.2 shows a 1% reference error for H1 at n = 32. The authors should check whether the error is larger at the lower τ bound, which is exactly where volume-constrained compliance optimisation pushes lightly loaded regions.
  - The training generator also limits the within-cell span and gradient of τ to 0.47. An optimiser with the admissible bounds [0.18, 0.70] can create corner spans up to 0.52, and the shared-corner map produces in-cell gradients set by neighbouring cells. Designs will then leave the training domain unless an explicit constraint is added.
  - The trilinear, C⁰ shared-corner parametrisation is fine, but its design resolution (one trilinear field per cell) should be stated as a modelling choice.
- **Why it matters:** A practitioner needs:
  - wall thickness in physical units, for the minimum printable feature size;
  - relative density, for the volume constraint and cost;
  - the admissible design domain, including gradient limits, so the surrogate is not used out of distribution.
- **Suggested fix:**
  - Add a figure or table relating τ to minimum and mean wall thickness (in units of h and of cell size) and to relative density, for uniform τ, with and without cuts.
  - Report reference convergence (as in Fig. S06) at the τ bounds.
  - State the design-variable bounds and the gradient/span constraint that an optimiser must impose to stay inside the training domain, and how the Section 6.11 example enforces them.
  - Use "band parameter" consistently, or define "corner thickness" once.
- **Fix type:** text only + re-analysis of existing data (geometry post-processing; one convergence run)
- **Confidence:** High that the information is missing; medium on my thickness estimate.

### R4-06: The lattice examples are small, favourable and not shown
- **Severity:** Major
- **Location:** Section 6.9: "a 2×2×2 block of eight distinct cells, four of them cut with retained volumes of 62% and 25%, and a 3×3×1 layer … clamped on one face and loaded by unit consistent tractions on the opposite face". Figure 4 (two-cell configurations only).
- **Issue:**
  - Only two lattices, each of eight cells. Both are clamped on one face and loaded by uniform consistent traction on the opposite face, which produces near-uniform stress and participation.
  - τ lies in 0.25–0.56, well inside the training range, and no heavily cut sliver (like H2, 2.7% retained volume) appears.
  - There is no figure of the lattices, their τ fields or their cuts.
  - Loads act directly on open wall sections at box faces. In practice, load introduction and support go through solid skins or plates, and cut boundaries are often capped.
  - Bending-dominated structures (cantilever, three-point bending, MBB-type), localised patch loads, and supports that create stress concentrations are the cases where participation varies strongly and where the paper's own theory predicts the largest local errors. None is tested with every cell learned.
- **Why it matters:** The conclusion that "the cells' energy errors combine through their shares of the assembled energy rather than accumulate" rests on two benign cases. A designer cannot infer the behaviour for realistic loads or larger lattices.
- **Suggested fix:**
  - Add a figure of both lattices (geometry, τ field, cut cells, supports and loads).
  - Add at least one bending-dominated lattice of ≥27 cells with a localised load and support, cut cells from all three strata including a sliver, and τ near both bounds with steep gradients.
  - Report the per-cell participation distribution and the global-gradient error (R4-03).
  - If the method supports solid non-design regions (skins, plates), show one. If it does not, state this as a limitation.
- **Fix type:** new experiment
- **Confidence:** High

### R4-07: Cost claims omit the baselines relevant to design, and the scaling is not shown
- **Severity:** Major
- **Location:** Section 6.10: "Since both preparation and every application are cheaper, the learned route is cheaper for any number of queries per design iteration." Section 8: "so it is cheaper for any number of queries." Section 7.4: "The learned applications are dominated by the 32 stiffness actions of the correction". Tables 6 and ST20c (CG iterations 114→165; "operator state of four cells on the GPU and stream the remainder from host memory"). Supplementary Note S7: "The host was shared with other jobs during these runs (one-minute load average between 11 and 37 …)".
- **Issue:**
  1. **Baseline.** Each global CG iteration applies, for every cell, the network forward and transpose plus about 32–34 stiffness actions on the full cell. So one retained-level iteration costs roughly as much as about 30 matrix-vector products of the full, uncondensed lattice. With 114–165 iterations per design iteration, one design iteration costs several thousand full-model matvec equivalents. A matrix-free GPU geometric-multigrid-preconditioned CG on the full cut-FE model is the standard tool in large-scale topology optimisation (typically O(10–100) iterations, each costing a few matvec equivalents). By my estimate it could be competitive or cheaper, and it needs no training. The comparison is with sparse direct factorisation on a CPU, which is not what a practitioner would use at 10⁶–10⁷ DOFs.
  2. **Scaling.** CG iterations grow from 114 (4 cells) to 165 (8 cells). GPU memory already forces host streaming at 8 cells. Nothing shows how iterations, memory and time scale to 27, 64 or 1000 cells.
  3. **Attribution.** Much of the route advantage in Table 6 comes from the front end: 140–321 s on the host versus 5–11 s on the GPU for the same cut integration. That is an implementation difference, not a consequence of learning.
  4. **Timing fairness.** The direct runs were made on a shared host with a load average of 11–37.
  5. **Accuracy baseline.** The paper does not compare against homogenisation-based graded-lattice design (effective C(τ) interpolated from unit-cell homogenisation), the de facto standard for graded TPMS (e.g. Panesar et al. 2018, Additive Manufacturing). That comparison would show the value of full resolution for cut and boundary cells.
- **Why it matters:** The paper's value for optimisation depends on cost per design iteration at a prescribed gradient accuracy, as Section 7.4 itself says. The "cheaper for any number of queries" statement is only true relative to the chosen CPU direct baseline.
- **Suggested fix:**
  - Add a full-model matrix-free GPU multigrid-CG baseline, even a simple one, on the same lattices, at matched compliance and gradient accuracy.
  - Add a scaling study (cells vs CG iterations, time and memory; at least 8/27/64 cells).
  - Separate front-end implementation speed-up from condensation speed-up in the summary claims.
  - Rerun the direct timings on an idle host, or state the contamination in the table caption.
  - Add a homogenisation baseline for accuracy (exact compliance of designs, or error of the homogenised prediction on the Section 6.9 lattices).
  - Qualify the "any number of queries" statements.
- **Fix type:** new experiment + text
- **Confidence:** Medium-high. The matvec-count estimate is mine, from Sections 7.4 and ST20c.

### R4-08: Single TPMS family, fixed discretisation and material, no limitations section, and unreported offline cost
- **Severity:** Major
- **Location:** Section 2.1 (P-type only), Table 1 (n = 32, ν = 0.3, γ = 10⁻⁴), Section 6.1: "All geometries carry a single planar cut with normal (cos ϑ, sin ϑ, 0); multiple cuts per cell and curved boundaries are outside the present study." Section 6.1: "A3's continuation took 2.7 h" (training time of B's 40,000 steps, reference-label generation for 591 geometries, and the direction searches are not reported).
- **Issue:**
  - The network is trained for one TPMS family (Schwarz P), one cubic cell aspect ratio, one mesh density, one Poisson ratio and one ghost coefficient, with at most one planar cut per cell.
  - Gyroid and diamond lattices, which are more common in metal additive manufacturing because they are self-supporting, are not addressed.
  - Corner and edge cells of a real part (two or three cuts), curved part boundaries, stretched or conformal cells, and other materials would require retraining.
  - The full offline cost (label generation with exact Schur solutions and sensitivity labels, B's training, direction searches, GPU-hours) is not reported, so the break-even number of cell condensations cannot be estimated.
  - There is no explicit limitations section. Limitations are scattered through Sections 6.1, 6.2 and 7.
- **Why it matters:** A practitioner deciding whether to adopt the method needs the retraining cost per new cell family and a sense of how far the correction compensates for an out-of-family network. B+W shows that the correction does most of the work, and Table 3 shows that a harmonic start with the same correction is 7–290 times worse.
- **Suggested fix:**
  - Add a limitations subsection.
  - Report the complete offline cost and a break-even estimate.
  - Optionally, a zero-shot test: apply the P-trained network plus correction to a few gyroid or two-cut cells, compared with the harmonic start plus the same correction. This would show how much of the method transfers without retraining, which is a practically important question.
- **Fix type:** text only (limitations, cost) + optional new experiment
- **Confidence:** High

### R4-09: One-sided surrogate bias invites exploitation by the optimiser, and there is no runtime error control
- **Severity:** Major
- **Location:** Eq. (12) and Appendix C: "Ĉ ≤ C"; Section 6.2: "The contraction condition is therefore verified per cell rather than guaranteed a priori."; Eq. (10) and Appendix I (residual lower bounds).
- **Issue:**
  - The surrogate always underestimates compliance, and its error is largest for heavily cut, thin-walled or out-of-distribution cells.
  - A compliance minimiser is therefore systematically rewarded for moving towards designs where the surrogate is most over-stiff, the classic exploitation of surrogate error.
  - In deployment, thousands of new geometries are created, none of them validated. The Chebyshev-interval containment is verified only offline for the 80 validation geometries and 5 deployed cells.
  - The paper already has the ingredients for a cheap runtime indicator: the interior residual r_I, the Appendix I lower bound, and the recomputed global residual. None is used in the assembled workflow.
- **Why it matters:** Without runtime error control, a converged "optimum" may be an artefact of surrogate error. This matters most for Section 6.11.
- **Suggested fix:**
  - Add an a-posteriori per-cell indicator to the deployed route, for example L_z/(û^T K û) from Appendix I with z from one extra coarse or Krylov step, or a few interior CG iterations to estimate ‖r_I‖_{A⁻¹}.
  - Define a policy for flagged cells: increase the correction budget, which the paper's own Section 6.5 supports, or fall back to exact condensation.
  - Add a runtime check of the Chebyshev endpoint (for example a Lanczos or Gershgorin-tightened check at low cost).
  - In Section 6.11, report surrogate error at the start and at the optimum (see Section 4 of this review).
- **Fix type:** new experiment (implementation of an indicator) + text
- **Confidence:** High on the mechanism; its practical size is unknown until tested.

### R4-10: Design-dependent loads are silently frozen
- **Severity:** Minor
- **Location:** Section 6.1: "For sensitivity evaluation, the assembled nodal load is held fixed at its base-design value." Supplementary Note S1.2: "integrates the Q2 surface shape functions, normalizes each nodal load to unit resultant". Appendix H: "If the load depends on design, the full compliance derivative also contains 2 f_{g,c}^T Û".
- **Issue:** Consistent tractions are integrated over the material patches on a face and then normalised to a unit resultant. The patch areas depend on τ, so the physical load distribution depends on the design. The reported sensitivities omit f_,c. That is legitimate as a definition, but for a real optimisation with loads applied to lattice faces the true gradient differs.
- **Why it matters:** In an optimisation with loads on design cells, the gradient is inconsistent with the objective unless the load term is included, or the load is applied through non-design regions.
- **Suggested fix:** State which load model the Section 6.11 example uses. Either include 2 f_{g,c}^T Û, or apply loads and supports through non-design (fixed-τ or solid) regions.
- **Fix type:** text only (or re-analysis if the term is added)
- **Confidence:** Medium

### R4-11: Contradictory statements about finite-difference step refinement, and the positive semidefiniteness of the numerical K_,c is unchecked
- **Severity:** Minor
- **Location:** Section 6.2: "changes by less than 10⁻⁷ when the step is varied between 10⁻⁶τ_c and 10⁻³τ_c" and Fig. S06(c), versus Appendix J.5: "No derivative eigenvalues or step-refinement study are saved in the selected sensitivity records." Appendix J.1: "present frozen sensitivity implementation".
- **Issue:** The appendix contradicts the main text on the step study. It also leaves open whether the finite-difference K_,c computed with adaptive subcell refinement is actually PSD. The paper's sign arguments (J.5–J.7), and the sign guarantee of s̃_c that makes OC updates well defined, rely on this.
- **Suggested fix:** Update J.5 to match Section 6.2. Report the smallest eigenvalue (Lanczos) of the numerical K_,c for a few cells, or explain why nesting is preserved by the integration scheme.
- **Fix type:** text only + re-analysis of existing data
- **Confidence:** High

### R4-12: Single-precision residual floor in the assembled solve
- **Severity:** Minor
- **Location:** Section 6.9: "its recomputed residual stagnates at 3.6×10⁻⁴ … 3.2×10⁻³, the level set by the network's single-precision arithmetic". Table ST20c: CG to recursive 10⁻⁶.
- **Issue:** A relative residual of 3×10⁻³ is large for gradient-based optimisation. By Eq. (18), compliance is affected through Ū^T ρ, and the sensitivities through the unconverged field. Near convergence, when design changes are small, this floor may set the achievable gradient accuracy. The effect on s̃ and Ĉ_,c is not reported, and a slightly non-symmetric fp32 action can also disturb CG.
- **Suggested fix:** Report sensitivity errors against residual level. Use the residual-corrected functional J(Ū) (Appendix J.6). Consider fp64 for the network displacement path, or iterative refinement in the outer solve, and report the cost.
- **Fix type:** re-analysis of existing data
- **Confidence:** Medium

### R4-13: How sensitivities are computed in the deployed route is unclear
- **Severity:** Minor
- **Location:** Supplementary Note S7: "reverse-mode sensitivities for the three loads"; Table 6 "(… sensitivities)". Appendix H, Eq. (H.6): central moment differences with h_c = 10⁻⁵τ_c.
- **Issue:** It is not stated whether the deployed "reverse-mode sensitivities" are s̃_c or Ĉ_,c, how K_,c is formed in the learned route (finite differences of moments requiring 16 extra moment evaluations per cell, or automatic differentiation through the integration), or how the cost scales with the number of design variables and load cases.
- **Suggested fix:** Specify the quantity, the method and the per-cell cost. Confirm that the deployed K_,c matches the teacher's to the stated tolerance.
- **Fix type:** text only
- **Confidence:** High

### R4-14: Connectivity and rigid-mode assumptions for heavily cut slivers and thinning designs
- **Severity:** Minor
- **Location:** Section 2.2: "we additionally assume that K has exactly six rigid-body modes whose restriction to P has rank six." Figure 1(d) (H2, 2.67% macro volume, which appears as an isolated ring).
- **Issue:** Heavily cut cells can contain disconnected material islands, or material that touches no shared box face, and thinning during optimisation can disconnect walls. Either case violates the six-mode assumption, gives floating bodies in the assembled system, and breaks the rigid split C_R. The paper does not say how the generator excludes such cases or what the deployed route does when they occur.
- **Suggested fix:** State the connectivity check used in geometry generation and the treatment of disconnected components (removal, or a per-component rigid split). Add an optimisation-time safeguard, for example a lower bound on τ tied to connectivity.
- **Fix type:** text only
- **Confidence:** Medium (the figure alone cannot settle whether H2 touches a box face)

### R4-15: The comparison with reduced-boundary (PIML-type) substructures ignores their cost advantage
- **Severity:** Minor
- **Location:** Section 1: "restricting every box face to corner-linear displacements gives compliance errors of 78–85% even with exact cell operators". Section 7.1: "For the local thickness sensitivity of the cut cells examined, the boundary restriction produces larger errors than the interior approximation of the corrected operator".
- **Issue:** The Bernstein study is useful, but it compares accuracy only. Reduced-boundary methods trade accuracy for coarse models with tens to hundreds of DOFs per cell, which is what makes lattice optimisation with thousands of cells feasible. NICE keeps 17k–26k retained DOFs per cell, a reduction of only about 13 times relative to the full model (Table ST20a). For optimisation, the relevant comparison is accuracy against cost per design iteration.
- **Suggested fix:** Add the cost side to Figure 11 or Table ST07 (retained DOFs, solve time), or a short accuracy-versus-cost discussion, and soften the framing in Section 1.
- **Fix type:** text only (or re-analysis of existing data)
- **Confidence:** Medium

### R4-16: Reuse across design iterations is dismissed too quickly
- **Severity:** Minor
- **Location:** Section 7.4: "In design optimisation every iteration changes the thickness parameters … so neither preparation can be reused across iterations".
- **Issue:** In practice, late iterations change τ very little. Previous factorisations can precondition new systems (approximate reanalysis), and the previous global solution is an effective CG warm start. The learned route could reuse the previous correction set-up (b, coarse factor) as long as the active set is unchanged. The statement biases the cost comparison.
- **Suggested fix:** Discuss warm starts and reuse for both routes, and state which the Section 6.11 timings use.
- **Fix type:** text only
- **Confidence:** Medium

### R4-17: Presentation of the sensitivity results
- **Severity:** Minor
- **Location:** Sections 4.3, 6.6, 6.7 and Appendix J.9.
- **Issue:** The matrix examples of J.9 (examples 2–5) are instructive and could support a short main-text paragraph on what they mean for optimisation. Example 5 in particular (the surrogate's complete derivative can have the wrong sign while s̃ remains negative) is a strong argument for using s̃ in OC. The main text never states which sensitivity the authors recommend for optimisation, or why.
- **Suggested fix:** Add a short paragraph ("Recommendation for design use") at the end of Section 4.3 or 7.2, stating which sensitivity should be used with which optimiser, supported by R4-01's results.
- **Fix type:** text only
- **Confidence:** High

---

## 4. Requirements for the design-optimisation example (Section 6.11)

The example should do more than show that an optimisation runs. It should show that the learned operator produces **the same design quality as exact condensation, at lower cost, with verified gradients**, and it should make visible the practical consequence of the compliance–sensitivity separation. Concretely:

**Problem definition**
1. Minimum compliance of a graded P-type lattice with a relative-density (volume) constraint. Design variables are the shared lattice-vertex band parameters τ_g, which give C⁰ continuity across faces via the Section 4.3 map.
2. Bounds and constraints that keep designs inside the training domain: τ ∈ [0.18, 0.70], plus the generator's in-cell span and gradient limits (0.47, Table ST11) as explicit constraints. Alternatively, report how often and by how much designs leave the domain.
3. A bending-dominated case (cantilever, MBB or three-point bending) with a localised load patch and a localised support, not uniform face traction. Loads and supports go through non-design regions, or else the load-derivative term is included (R4-10).
4. A non-box design domain, so that the lattice contains cut cells from all three strata, including at least one heavily cut sliver (<1/3 retained volume).
5. Preferably a second load case (multi-load compliance) to show that the per-load participation argument holds in aggregate.

**Scale**
6. At least 27 cells (3×3×3). Preferably a second, larger lattice (≥64 cells, e.g. 8×4×2) to show scaling of CG iterations, memory (host streaming) and time per iteration. Report the number of design variables.

**Optimiser**
7. State the optimiser (OC with move limit and damping, or MMA/GCMMA), all parameters, the convergence criterion (change in τ, KKT residual) and the maximum iterations. State which sensitivity is used: s̃_c or Ĉ_,c (R4-01, R4-17). If MMA is used, show that the objective and gradient are consistent, or that inconsistency does no harm.

**Verification (mandatory)**
8. **Gradient check** at the initial design, an intermediate design and the final design, for all variables or a random subset of at least 50:
   - exact C_,c from exact condensation;
   - field-based s̃_c;
   - complete Ĉ_,c;
   - central finite differences of Ĉ at several steps.
   
   Report global relative error, cosine, the per-variable relative-error distribution (median, 95th percentile, max) and sign agreement.
9. **Final-design verification** with the exact reference: compliance error, exact gradient, and the KKT or stationarity residual computed with *exact* gradients. This shows whether the design is an optimum of the true problem, not only of the surrogate.
10. **Exploitation check (R4-09):** per-cell energy excess and assembled compliance error of the surrogate at the start and at the optimum. Report whether any cell at the optimum lies outside the validated parameter domain, and whether error grew during optimisation.
11. **Smoothness check (R4-02):** Ĉ and C along one line in design space through the optimum, crossing at least one active-set or ghost-set change.

**Comparisons (mandatory)**
12. **Twin optimisation with exact condensation** from the same start with identical settings. Compare:
   - objective, volume and design-change histories;
   - iterations to convergence;
   - final τ fields (max and RMS difference, plus a visual comparison);
   - exact compliance of both final designs (target: relative difference well below 1%).
   
   If exact condensation over the full history is too expensive for the larger lattice, do it for the 27-cell case and use exact checks at selected iterates for the larger one.
13. **Consequence of the separation claim:** repeat the optimisation with an uncorrected predictor (C, which passes compliance but fails sensitivity) or with S8, and with B+W. Show whether their final designs and exact compliances differ from A3 and the exact twin. This is the missing experiment that turns contribution (iv) from a diagnostic observation into a design-relevant finding. If the designs do not differ, say so; that is also informative.
14. **Practitioner baselines:**
   - a uniform-density lattice of equal volume (improvement ratio);
   - a homogenisation-based graded design (effective C(τ) from periodic unit-cell homogenisation), with its final design evaluated by the exact full model. This quantifies what full-resolution condensation of cut cells buys over the standard approach.

**Cost metrics**
15. Wall-clock per design iteration split into front end, correction set-up, global solve (with iteration count), sensitivities and design update. Also report total time, peak GPU and host memory, and the same for the exact-condensation twin. Ideally include a full-model GPU multigrid-CG route (R4-07). State any reuse or warm starts (R4-16).

**Presentation and manufacturability**
16. Figures of the design domain, loads and supports, and the initial and final τ fields rendered as TPMS geometry. Report minimum and maximum wall thickness in units of h and of cell size, and the final relative-density field (R4-05).
17. The abstract and conclusion sentences should report the quantitative outcomes of items 9, 12 and 15, not just that the optimisation ran.

---

## 5. Top three concerns

1. **The sensitivity actually needed for optimisation is not validated** (R4-01, R4-02, R4-03). The paper validates a field-based, per-cell, locally normalised sensitivity at fixed topology. It does not measure the complete derivative of its own surrogate objective, the global shared-variable gradient, or the behaviour of Ĉ(τ) across the active-set, ghost-set and feature switches that an optimiser crosses. The Section 7.2 argument that the missing term is small does not follow from the stated bound.
2. **The evidence base for assembled accuracy is narrow and favourable** (R4-04, R4-06, R4-09). Seven development cells with an exact neighbour and two eight-cell lattices under uniform face tractions do not show accuracy on the worst validation geometries, under bending-dominated or localised loads, at practical lattice sizes, or at designs an optimiser is drawn to because the surrogate always underestimates compliance.
3. **Practical value for graded-lattice design is not established against the right baselines** (R4-05, R4-07, R4-08). The paper gives no τ→thickness/density map and no admissible design domain. It does not compare with full-model GPU multigrid or homogenisation-based design, does not report offline training cost, and does not discuss limitations (single P family, fixed resolution and material, single planar cut). Section 6.11 should be designed to answer these points (Section 4 above).
