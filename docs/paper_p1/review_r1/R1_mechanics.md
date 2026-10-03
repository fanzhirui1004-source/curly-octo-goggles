# Review R1: Computational mechanics, condensation theory and CutFEM

**Manuscript:** "Learned static condensation for cut thin-walled TPMS cells with equilibrium correction" (CMAME submission)
**Reviewer focus:** correctness and rigour of Sections 2, 4, 5 and Appendices A–J; how classical results are attributed; notation; the CutFEM reference and its verification (Section 6.2); whether the claims follow from the evidence.
**Not reviewed:** Section 6.11 and the "[Placeholder]" sentences, as instructed. What that example must show is set out at the end.

---

## 1. Summary

The paper approximates the static condensation of cut, thin-walled P-type TPMS cells, discretised by a ghost-penalty-stabilised Q2 CutFEM model, on the complete retained space. That space consists of all active box-face coefficients plus a band of coefficients from the elements that carry the macro-cut. A geometry-conditioned network gives an initial interior field. The network is linear in the retained displacement, reproduces rigid motion exactly and restores the retained values. A fixed, geometry-specific correction then acts on this field: 8 Chebyshev steps, a trilinear Galerkin coarse solve, and 8 more Chebyshev steps. The condensed stiffness is the Ritz form Ŝ = FᵀKF. It is applied matrix-free through the complete transpose and is never formed.

The analysis collects the Ritz identity Ŝ − S = HᵀAH, an exact energy decomposition of the compliance gap at assembled equilibrium, a participation-weighted two-sided compliance bound, and expansions of the field-based thickness sensitivity and of the complete surrogate derivative. The numerical study covers:
- 80 validation cells;
- two-cell assemblies with one learned cell;
- 8-cell lattices in which every cell is learned;
- a Bernstein boundary-restriction comparison;
- cost comparisons against host PARDISO.

The central empirical message is that the correction, rather than the network alone, brings energy, compliance and eight-corner sensitivity errors below 1%, whereas uncorrected predictors can combine small compliance errors with sensitivity errors of several percent.

## 2. Recommendation

**Major revision.** Most of the algebra is correct. I checked Eqs. (9)–(18), (B.1)–(B.9), (C.1)–(C.4), (D.1)–(D.5), (E.1)–(E.4), (F.1)–(F.3), (H.1)–(H.5), the J.3 bounds, (J.4), (J.5), the J.5 cancellation argument and J.9 examples 2–5 by hand, and found no errors in them. The implementation checks (symmetry, action/energy consistency, rigid energy) are careful, and the paper is unusually explicit about its assumptions.

However, five problems need to be fixed before the claims can be accepted:
1. Much of the stated "error analysis" contribution is classical Ritz, Galerkin, Chebyshev, two-grid and compliance-sensitivity theory, and it is neither attributed nor separated from what is new.
2. The design-relevant quantity, the derivative of the surrogate objective, is never evaluated. The surrogate is also not a smooth function of the design.
3. The exact-equilibrium identities are compared with measured errors that lie below the level of the algebraic (mixed-precision) solve residual, and that residual is not reported.
4. The verification of the CutFEM reference is too thin for thin-walled, heavily cut cells.
5. Only direction-averaged accuracy is reported for the principal predictor, and the non-learned baselines are weak.

Most fixes are re-analysis or small additional computations.

---

## 3. Findings

### R1-01: Classical results are presented as contributions and are not attributed
- **Severity:** Major
- **Location:** Section 1, contributions (iii) and (iv) ("An error analysis that follows the interior extension error …", "The demonstration that accurate compliance does not imply accurate local sensitivity"). Also Eqs. (9), (11), (12), (16), (17), (18); Appendices B, C, D, J.2, J.6; Section 4.3 ("… [Giles & Pierce (2000)]").
- **Issue:** The following results are standard:
  - the energy-minimising property of the discrete harmonic (Schur) extension and the Ritz identity Ŝ − S = HᵀAH for any trace-admissible extension (Eq. 9, B.1);
  - the fact that any admissible N gives NᵀKN ⪰ S, together with symmetry and PSD (already exploited by PIML, as the paper notes);
  - the A-orthogonal Galerkin projection (Eq. 16, D.5);
  - Chebyshev semi-iteration and its energy bound (D.1–D.4);
  - the pre-smooth/coarse/post-smooth two-grid error operator (Eq. 17);
  - compliance ordering under stiffness ordering (C.2–C.3);
  - the residual-corrected compliance functional (J.6);
  - the self-adjoint compliance sensitivity −uᵀK,cu and S,c = EᵀK,cE (H.1).

  The only citations are Xu (1992), Adams et al. (2003) and Giles & Pierce (2000). Giles & Pierce is an aerodynamic-adjoint tutorial and is not the natural source for structural compliance sensitivity. The abstract and Section 8 say "classical", but contribution (iii) presents the analysis as new. Contribution (iv), that a small energy/compliance error need not imply a small error in a local functional, is the standard distinction between energy-norm and goal-oriented error (Becker & Rannacher 2001; Oden & Prudhomme 2001).
- **Why it matters:** CMAME readers will recognise these results, and the paper's real contributions are obscured. As far as I can see, those contributions are:
  - applying a learned initial field plus a trained-through two-grid correction to CutFEM condensation on the full retained space;
  - the load-specific two-sided participation bound β/(1+ρ) ≤ e_C ≤ β/(1+β) (J.3);
  - the global-to-local sensitivity bound (J.4);
  - the O(t) cancellation between the field estimate and the residual-chain term (J.5);
  - the empirical study itself.
- **Suggested fix:**
  - Add a short "Classical background" paragraph at the start of Section 4 and label each relation as classical (with citation) or new.
  - Cite the static-condensation and CMS origins: Guyan 1965; Irons 1965; Craig & Bampton 1968.
  - Cite the discrete-harmonic extension and energy minimality: Toselli & Widlund 2005, or Smith, Bjørstad & Gropp 1996.
  - Cite Chebyshev semi-iteration: Golub & Varga 1961; Saad 2003.
  - Cite two-grid theory: Hackbusch 1985; Trottenberg et al. 2001; Xu & Zikatanov 2002.
  - Cite compliance sensitivity: Bendsøe & Sigmund 2003; Haftka & Gürdal 1992.
  - Cite goal-oriented error: Becker & Rannacher 2001; Oden & Prudhomme 2001.
  - Cite the bounds: Rayleigh–Ritz dual bounds (e.g., Fraeijs de Veubeke 1965).
  - Rephrase contributions (iii) and (iv) accordingly.
- **Fix type:** Text only
- **Confidence:** High

### R1-02: The derivative of the surrogate objective is never evaluated, and the surrogate is not smooth in τ
- **Severity:** Major
- **Location:** Section 4.3, Eq. (14) ("The reported field-based estimates contain only the first term."); Section 7.2 ("it is about 2.7% of ‖û‖_K … Measuring this term within a complete design optimisation is left to subsequent work"); Appendix G.1 (weak-support indicator "less than 0.01 times its median"); Appendix F.2 (shift list 0, 10⁻¹², …, 10⁻⁴); Section 2.2 and Appendix A.1 (the retained cut band depends on which elements carry a "positive-area macro-cut patch").
- **Issue:** Every reported sensitivity is the field estimate s̃_c = −ûᵀK,cû. This is not the gradient of any objective that the learned model defines.
  - The complete derivative (14) adds −2(F_{I,c}q̂)ᵀr_I. Its bound (H.5) involves ‖F_{I,c}q̂‖_A, which is never measured.
  - The 2.7% estimate in Section 7.2 is relative to ‖û‖_K, not to |s_c|. For weakly participating cells |s_c| ≪ ‖û‖²_K, so the relative error of the complete derivative can be much larger.
  - Ĉ(τ) is only piecewise smooth, and it can be discontinuous where any of the following changes with τ: the binary weak-support node feature, the active, ghost-face and cut-band sets (and hence P itself), the subcell classification in the adaptive moment quadrature, the Cholesky shift chosen for the coarse solve, or the random-start power estimate of b.
- **Why it matters:** The title, abstract and Section 6.11 target thickness design. If the field-based estimate is used in MMA or OC, the gradient is inconsistent with the objective the optimiser actually evaluates. If the complete derivative is used, its accuracy is unknown and it may not exist at switching points. J.9 example 5 shows that the sign can even be wrong.
- **Suggested fix:**
  - For the pair and lattice cases, compute Ĉ,c by automatic differentiation through the geometry branch and the correction (the correction is already differentiated during training), or by central differences of Ĉ.
  - Report the residual-chain term relative to |s_c| and relative to ‖∇C‖, and compare both Ĉ,c and s̃_c with the exact C,c.
  - List which quantities switch discretely with τ, and state whether the weak-support feature can be smoothed.
  - Say explicitly which gradient Section 6.11 will use.
- **Fix type:** Re-analysis of existing data / new experiment (moderate)
- **Confidence:** High

### R1-03: The exact-equilibrium identities are tested below the algebraic error floor, and the solve residuals are not reported
- **Severity:** Major
- **Location:**
  - Section 6.9: "its recomputed residual stagnates at 3.6×10⁻⁴ … the errors above are measured at this solution"; "3.2×10⁻³" for the 3×3×1 layer; "exceeding the lattice compliance errors by less than 0.3% of their value".
  - Supplementary S1.2: "The saved result summaries retain the iteration count but omit the residual returned by the solver."
  - Table 4: A3 compliance error 7.3×10⁻⁵ %; Figure 9(d).
  - Appendix J.6: "Evaluating the terms in Eq. (18) additionally requires the signed residual work and ω, which are not stored."
- **Issue:** Eqs. (11)–(12) hold only at exact assembled equilibrium of one fixed linear operator. The lattice compliance errors are about 1×10⁻⁴ relative, while the recomputed relative residuals are 3.6×10⁻⁴ and 3.2×10⁻³. By Eq. (18), the error is then contaminated by Ūᵀρ, and by ω if the fp32 action is not exactly the energy operator. The load- and stiffness-dependent constant in the J.6 conversion is at least one.
  - The gap between the recursive residual (about 10⁻¹⁰) and the recomputed residual shows that the applied operator is not reproducible to the precision the CG solve assumes. That sits uneasily with the 10⁻⁷–10⁻⁸ consistency figures in Section 6.2, which are relative to the energy, not to the residual.
  - In the pair tests, compliance errors down to 10⁻⁶ relative are reported, but no recomputed residual was saved.
  - Consequently, neither the "0.3–0.4%" agreement with β nor the smallest compliance errors can yet be interpreted as properties of the operator.
- **Why it matters:** The lattice paragraph is presented as numerical confirmation of Eq. (12), and the abstract and conclusions quote these numbers.
- **Suggested fix:**
  - For every assembled accuracy result, report the recomputed residual, Ūᵀρ/C and ω/C.
  - Alternatively, report the residual-corrected J(Ū) of J.6, or re-solve with the network evaluated in fp64 (or with iterative refinement in fp64) for the verification runs.
  - Check GPU determinism: do repeated applications of Ŝ to the same vector agree bitwise?
  - State which compliance errors are resolved above the algebraic floor.
- **Fix type:** Re-analysis of existing data plus cheap re-runs
- **Confidence:** High

### R1-04: Verification of the CutFEM reference (Section 6.2) is insufficient for thin, heavily cut walls
- **Severity:** Major
- **Location:** Section 6.2 ("For heavily cut cells the discretisation error of the reference at n=32 is thus of order one percent"); Figure S06(a); Table 1 (γ = 10⁻⁴); Section 6.3 ("49–81% of the corresponding exact field energy resides in the ghost-penalty term"); Appendix A.1 ("These faces lie within each substructure").
- **Issue:**
  1. Only four cells are checked, the finest level is n = 48, and the behaviour is not monotone. For M1, n = 40 is further from n = 48 than n = 32 is (Figure S06a). The "difference from the finest level" is therefore not an error estimate, and no observed rate is given.
  2. The corner parameters of U1, M1, M2 and H1 are not reported. The population reaches τ ≈ 0.176. A rough estimate, 2τ/|∇φ| with |∇φ| ≈ 2π·O(1), gives a wall thickness of about 0.05, i.e. only about 1.5 background elements at n = 32. This is precisely where Q2 CutFEM is least reliable for bending-dominated response, and it is not verified.
  3. No independent reference is used, for example a body-fitted conforming mesh of the same geometry.
  4. γ = 10⁻⁴ is several orders of magnitude below the ghost-penalty coefficients usually used for elasticity. Compliance is insensitive to γ under consistent loads, but the nodal-force class shows that weakly cut DOFs are controlled almost entirely by the penalty. Robustness of conditioning with respect to cut position (the purpose of the ghost penalty) is not shown.
  5. Ghost faces never cross cell boundaries. The "reference" is therefore a modular CutFEM discretisation, different from a monolithic CutFEM of the lattice: small cuts adjacent to box faces are stabilised only from one side. This is stated only in Appendices A.1 and J.1.
  6. The figure axis reads "nodes per cell edge, n", whereas n is elements per axis.
- **Why it matters:** The claim that the reference "approximates the elastic response of the thin-walled cells" rests on this section. The discretisation error of the reference (about 1% for H1) is two orders of magnitude larger than the reported surrogate errors. That is legitimate only if the paper says clearly that it approximates the discrete operator, and if the discrete operator is shown to be adequate across the parameter domain.
- **Suggested fix:**
  - Add the thinnest-wall and most heavily cut validation cells.
  - Add n = 64, or estimate an observed order via Richardson extrapolation over at least three levels.
  - Compare at least one cell against a body-fitted tetrahedral P2 reference.
  - Report cond(D⁻¹A) or λ_min over cut positions for γ ∈ [10⁻⁵, 10⁻¹].
  - Move the statement that the reference is modular (no ghost penalty across cell faces) into Section 2 and justify it.
  - Give τ_c for all detailed cells.
- **Fix type:** New experiment (small) plus text
- **Confidence:** High

### R1-05: Worst-direction accuracy of the principal predictor is not reported
- **Severity:** Major
- **Location:** Abstract ("with at most 0.65% for any geometry"); Section 6.3; Table ST01 ("The maximum is not a worst individual direction"); Appendix B (ε_*, Eq. B.5); Appendix C (C.3); Appendix G.3 (block and bank Ritz searches); Appendix I (residual lower bound); Section 7.3 ("A uniform operator bound additionally requires quantitative coverage").
- **Issue:** All population statistics are means over directions within a geometry, then summarised across geometries. The paper defines ε_* and already has machinery that gives rigorous lower bounds on it: the generalised-eigenvalue bank search and the Cauchy–Schwarz bound (I.1). Yet no worst-found directional error, and no lower bound on ε_*, is reported for A3. In assembly and optimisation the direction is chosen by the coupled solve and changes with the design. The load-independent guarantee (C.3) is the one that matters there.
- **Why it matters:** The phrase "at most 0.65% for any geometry" will be read as a worst case, but it is a mean. A large ε_* on some cell would invalidate the transfer of the reported accuracy to new loads and neighbours.
- **Suggested fix:**
  - For A3 on the 80 validation cells and on the lattice cells, report the 99th percentile and maximum directional errors per class.
  - Report Ritz lower bounds on ε_* from a Lanczos/LOBPCG search on the pencil (Z^T Ŝ Z, S_*), for example 50–100 iterations, preconditioned by the exact Schur action that is available for validation cells.
  - Reword the abstract to say "largest geometry-mean".
- **Fix type:** Re-analysis of existing models (no retraining)
- **Confidence:** High

### R1-06: Weak non-learned baselines and an incomplete cost comparison
- **Severity:** Major
- **Location:** Section 6.5 and Table 3 (harmonic start); Table ST18; Sections 6.10 and 7.4 ("the learned route is cheaper for any number of queries"); Tables 5–6.
- **Issue:**
  1. The correction itself is a two-grid method: Chebyshev smoothing plus a Q1 Galerkin coarse space. The natural competitor is PCG on the full CutFEM lattice with a GMG or AMG preconditioner built from the same components (or a GPU sparse direct solver), not host PARDISO on 16 cores against one RTX 5090.
     - The learned route needs about 130–165 PCG iterations, each costing about 33 fine-level stiffness actions per cell, i.e. several thousand full stiffness actions per design iteration.
     - The retained system is itself large: 139k free retained DOFs out of 1.8M, so only about a 13× reduction.
  2. The "graph-harmonic" start is a scalar, volume-weighted graph Laplacian without elastic coupling. It is weak: in ST18 it ends worse than the zero start after 32/Q1/32 for U2 (1.25 vs 1.19) and H2 (0.0246 vs 0.0056) under nodal forces. The "7 to 290 times" claim in the abstract is measured against this baseline, on five development cells and one loading class.
  3. Offline costs (reference solves for the training directions on 591 geometries, 40k + 15k training steps) are excluded from the statement "cheaper for any number of queries".
- **Why it matters:** The computational case, and the claim that learning adds value beyond the correction, depend on the baseline.
- **Suggested fix:**
  - Add a monolithic PCG with GMG or AMG (same Chebyshev/Q1 components, GPU) on the Table 6 lattices at matched accuracy.
  - Add stronger starting fields: the coarse Galerkin solution itself (zero start with coarse first), a vector elastic harmonic extension, and, relevant to Section 6.11, the corrected field of the previous design iteration as a warm start.
  - Report offline cost and a break-even count.
  - Restrict the "7–290×" statement to its actual scope.
- **Fix type:** New experiment
- **Confidence:** Medium. This is partly outside my core area, but the baseline question follows directly from Section 5.

### R1-07: The benefit of training through the correction is confounded with the training population and training steps
- **Severity:** Minor (but it affects a stated conclusion)
- **Location:** Section 6.3 ("training through it reduces the remaining mean by a further factor of 1.3"); Section 6.6; Section 8 ("training through the correction reduces the largest remaining sensitivity errors by factors of 1.35 to 2.2"); Table 2.
- **Issue:** B+W uses B's weights: 305 geometries and 30k steps. A3 continues B on 591 geometries for another 15k steps. The difference therefore mixes the effect of the correction in the loop with the effect of more data and more training. The clean control is C+W, i.e. C's weights (591 geometries, same continuation, no correction in the loop) evaluated with A3's correction.
- **Why it matters:** Without C+W, the stated factors cannot be attributed to training through the correction.
- **Suggested fix:** Evaluate C+W on the population and the 14 pair configurations. If this is not done, rephrase the conclusion.
- **Fix type:** Re-analysis of existing data (evaluation only)
- **Confidence:** High

### R1-08: Per-cell relative sensitivity errors can overstate practical impact
- **Severity:** Minor
- **Location:** Section 2.3 (definition of e_s); Section 6.6 and Table 4 (U1/x N-z: target energy share 0.13%); Section 8 ("The principal mechanical finding …").
- **Issue:** e_s is normalised by each cell's own ‖s_m‖. For weakly participating cells this norm is tiny, so a 3–6% relative error may be negligible for the design update.
  - The paper acknowledges this in Section 7.2.
  - The M1/x cases (energy share 31%, C reaching 11.4%) do support a real effect.
  - Still, the main finding would be stronger if the error were also shown in a design-relevant norm.
- **Suggested fix:** Additionally report the error of the full structural gradient, ‖∇C̃ − ∇C‖/‖∇C‖ over all design variables and cells (with shared variables mapped as in Section 4.3), and the angle between the two gradients.
- **Fix type:** Re-analysis of existing data
- **Confidence:** Medium

### R1-09: Inconsistent δ/κ definitions and an over-interpreted identity
- **Severity:** Minor
- **Location:**
  - Section 6.4: "δ² = d_IᵀDd_I/u_IᵀDu_I … R(u) = uᵀKu/u_IᵀDu_I … Then the relative energy excess is exactly ε = δ²κ";
  - Appendix B.1: "We use the full-field Euclidean norm, W = I";
  - Appendix J.7: "uses the same full-field displacement norm for both Rayleigh quotients … it does not determine … a continuum bending-mode label";
  - contribution (iii): "quantifies why small displacement errors produce large energy errors in thin walls";
  - Section 7.3: "Relaxation removes the error components with large Rayleigh quotients efficiently".
- **Issue:**
  - The main text uses an interior Jacobi-weighted norm with a mixed denominator (the full energy divided by the interior D-norm), while the appendices define and use the full Euclidean norm. It is unclear which definition produced Table ST19.
  - ε = δ²κ holds by definition for any choice of norm, so it is a reparametrisation, not an explanation.
  - The bending interpretation contradicts the appendix's own caveat.
  - Section 7.3 equates a large κ (a ratio relative to R(u)) with high Jacobi-spectrum modes, which J.7 explicitly says are different concepts.
- **Suggested fix:** Use one definition throughout and state which one Table ST19 uses. Present the identity as a diagnostic decomposition. Either support the bending claim (for example with the energy split between membrane and bending in the walls, or with the Jacobi-spectrum location of the error) or drop it.
- **Fix type:** Text only (possibly re-tabulation)
- **Confidence:** High

### R1-10: The two-grid bound is vacuous, and the "controllable" claims should be stated precisely
- **Severity:** Minor
- **Location:** Eq. (17) and the following text; Appendix D ("which proves the ρ_k⁴ energy estimate"); Section 1 and the abstract ("can be reduced through … the correction budget").
- **Issue:** ρ_k is the maximum of |p_k| over the whole spectrum. Because λ₁ ≈ 4×10⁻⁴ ≪ a = b/30 (Table ST02b), ρ_k ≈ 1 and the ρ_k⁴ bound gives only nonexpansiveness, not a rate. A rate requires an approximation property of the restricted Q1 space for thin cut walls. This is not established, and it is doubtful for bending of walls that are about one element thick; M1 still shows 0.186% after 8/Q1/8. The operator ordering S ⪯ Ŝ_tg ⪯ Ŝ₀ is correct (and S ⪯ Ŝ holds for any admissible F).
- **Suggested fix:** State that only monotone non-increase is guaranteed. Report the measured per-geometry two-grid contraction factor, e.g. the largest generalised eigenvalue of (H_tgᵀAH_tg, H₀ᵀAH₀) on the rigid complement, estimated by Lanczos. Cite Xu & Zikatanov (2002) for the sharp two-grid identity.
- **Fix type:** Text plus small re-analysis
- **Confidence:** High

### R1-11: The Chebyshev interval and the coarse solve are verified inconsistently
- **Severity:** Minor
- **Location:** Section 5.1 ("on all 80 validation geometries this endpoint exceeds the converged largest eigenvalue by 2.1–5.0%"); Section 6.2 ("the 80-geometry check in Appendix D uses its own power-iteration start"); Appendix D ("The operational endpoint exceeds …"); Appendix F.2 (probing and shift selection).
- **Issue:**
  - Section 6.2 says the 80-geometry check did not use the deployed estimate of b, but Appendix D calls it "operational". The deployed b is verified on five cells only.
  - The minimum margin is 2.1%, and new geometries generated during optimisation have no guarantee. Eigenvalues above b are amplified quickly at k = 8.
  - Likewise, the colour-probed A_c is assumed exact ("Recovery assumes the resulting colour separation resolves every interacting coarse pair"), and the selected Cholesky shifts are not reported, although (F.1) and (J.9) rely on both.
- **Suggested fix:**
  - Make Section 5.1 and Appendix D consistent.
  - Use a certified upper bound, e.g. the Lanczos Ritz value plus its residual norm, at negligible cost.
  - Report ‖A_c,probed − V^TAV‖/‖V^TAV‖ and the chosen shifts on the validation set.
- **Fix type:** Re-analysis of existing data
- **Confidence:** Medium

### R1-12: Cut-surface loads are not reported for the principal predictor
- **Severity:** Minor
- **Location:** Figure 4 caption ("Cut targets also receive three macro-cut tractions, analysed separately"); Section 6.6 ("Table ST06 the cut-traction responses"); Table ST06 (B, C and S8 only); Figure 10 (B, C and S8 only).
- **Issue:** Loads applied directly to the cut surface exercise the cut band, where B's error concentrates (Section 6.4, 39–43% of the error energy near the cut). For B/M1/x they give a 9.9% sensitivity error. There are no A3 or B+W results for these loads, and the lattices load only box faces.
- **Suggested fix:** Add the A3 and B+W cut-traction rows to Table ST06 and Figure 10, and include a cut-surface load in one lattice.
- **Fix type:** Re-analysis of existing data / small new runs
- **Confidence:** High

### R1-13: How cell interfaces are matched with cut neighbours is unclear
- **Severity:** Minor
- **Location:** Section 2.3 ("Coincident box-face coordinates are shared"); Section 6.1 ("join a learned target to an exact neighbouring cell"); Appendix A.1 ("The retained box set is the union of the nine face nodes of every certified positive-area material patch").
- **Issue:** Face patches are "certified" per cell. When a cut target neighbours an uncut cell, or a cell cut by a differently placed plane, the active face-node sets on the shared face can differ. The paper does not say how unmatched face nodes are treated (free boundary? excluded?), nor whether the neighbour in the two-cell tests is cut by the same plane, which a physical specimen boundary would require.
- **Suggested fix:** State the matching rule and how certification tolerances are handled. Confirm that the reference and surrogate use identical maps B_m, and that the two-cell geometries are physically consistent.
- **Fix type:** Text only
- **Confidence:** Medium

### R1-14: Scope limits of the trained operator should be stated in the main text
- **Severity:** Minor
- **Location:** Appendix A.1 ("The fixed templates use E₀=1, ν₀=0.3"); Table ST11 domain (corner span ≤ 0.47, gradient norm ≤ 0.47, τ ∈ [0.175, 0.699]); Section 6.1 (single planar cut with normal in the xy-plane); Section 6.3 (orientation dependence evaluated only for B).
- **Issue:** The operator is tied to one background resolution, one Poisson ratio, one cut family and a bounded τ domain with span and gradient constraints. It is not equivariant: B's error rises from 6.62% to 7.49% in the transformed view, and A3 is not evaluated in any other view. In a lattice, and during optimisation, cells appear in arbitrary orientations and the design may leave the training box.
- **Suggested fix:** Summarise these limits in Section 2 or 7, and evaluate A3 in view 17. Section 6.11 must either enforce the training-domain constraints or show out-of-domain behaviour.
- **Fix type:** Text plus re-analysis
- **Confidence:** High

### R1-15: The wording "unseen" and the use of development cells
- **Severity:** Minor
- **Location:** Abstract ("on 80 unseen geometries"; "7 to 290 times more accurate"); Section 6.1 ("the first 40 geometries include 20 of the 80 reported validation geometries"; "the cells used in the assembly examples served repeatedly as development cases").
- **Issue:** 20 of the 80 validation geometries entered weight selection. The five cells behind Table 3 and the "7–290×" claim are development cells, and it is unclear whether the lattice cells were drawn from the validation stream. The disclosures in Section 6.1 are appreciated, but the abstract overstates independence.
- **Suggested fix:** Use "80 validation geometries (60 not used in selection)" in the abstract. State the provenance of the lattice cells, and scope the "7–290×" claim to five cells under consistent tractions.
- **Fix type:** Text only
- **Confidence:** High

### R1-16: Notation is heavily overloaded
- **Severity:** Minor
- **Location:** Throughout. Examples:
  - C: compliance, C_R, C_V, the coefficient map 𝒞(z), predictor C, and the coarse step "C" in ST04;
  - A: K_II, predictors A2b and A3, 𝔸, and a(·,·);
  - D: diag(A), D_c = K,c (J.4), predictor D, 𝔻, and D₁/D₂ in Figure S03;
  - H: the interior error, cells H1–H3, H_c, and H_tg;
  - P: the retained set, P_k, 𝒫, and P_fv;
  - ρ: the residual in (18), ‖T‖ in J.3, ρ_k, and ϱ_j;
  - T: T_α, T_k, T, T_*, 𝒯_k, and T_a;
  - M_I: a nodal mask in Eq. (4) and J_IᵀJ_I in Appendix E;
  - I: the index set and the identity;
  - γ: the ghost coefficient and γ_c (J.5);
  - b: the Chebyshev endpoint, b_r, b_cut, b_P, b_c, and b_tsh;
  - n: elements per axis, n_a, and "nodes per cell edge" in Figure S06.
- **Why it matters:** Sentences such as "A3's error in C" or "D-weighted …" become ambiguous, particularly in Sections 4–5.
- **Suggested fix:** Rename the predictors (e.g. 𝒫_A3 or NICE-full / NICE-B) and cells, and reserve A, C, D, H, P for mechanics. Add a nomenclature table.
- **Fix type:** Text only
- **Confidence:** High

### R1-17: Small inconsistencies
- **Severity:** Minor
- **Location:** Various
- **Issue:**
  - (a) Table ST13 footnote: "the nine configurations common to all predictors", but B has only seven.
  - (b) Abstract "9 to 29 times" versus 8.8–29 in Section 6.10.
  - (c) Section 6.2 reports the M2 and U1 finest levels as n = 48 and n = 40, but Figure S06(a) plots only up to n = 40 with no marker at the reference level. Say so in the caption.
  - (d) Section 7.2 compares a population-mean ε (0.074%) with a residual at an assembled trace. sqrt(mean) ≠ mean(sqrt), and the relevant ε is at q̂.
  - (e) Appendix H cites "Eq. (J.6) in Appendix J.5" for the PSD proof, but the nested-domain argument also needs fixed quadrature branches, which J.6 itself says are not guaranteed numerically. Soften "exact integration gives K,c ⪰ 0" in Section 4.3 to refer to the continuous integral.
  - (f) Section 2.2: the assumption that the rigid-mode restriction R_P has rank six (connectivity) should be checked per geometry and the check reported, since heavily cut cells could contain disconnected material islands.
- **Fix type:** Text only
- **Confidence:** High

---

## 4. What Section 6.11 must demonstrate

1. **Which gradient is used.** State whether the optimiser uses s̃ or the complete Ĉ,c. Verify it against finite differences of Ĉ and against the exact C,c along the optimisation path, and explain how switches in the active set, cut band and weak-support feature are handled (R1-02).
2. **Final design against exact condensation.** Re-evaluate the final learned-driven design with the exact reference, and run an optimisation driven by exact condensation from the same start. Compare objective histories, the final exact compliance, and the design distance.
3. **Training domain.** Enforce the τ range, corner span and gradient-norm constraints of Table ST11, or report the out-of-domain behaviour.
4. **Cost.** Report the wall time per iteration and in total, against exact condensation and a monolithic iterative solver (R1-06). Include a warm-start variant that reuses the previous iteration's field.
5. **Cut cells.** Include cut cells, ideally with a cut-surface load or support (R1-12).
6. **Algebraic accuracy.** Report the algebraic residual and Ūᵀρ at every iteration (R1-03).

## 5. Top three concerns

1. **The design derivative is untested (R1-02).** The paper targets thickness design but never evaluates the gradient of its own surrogate objective, which is also only piecewise smooth in τ. The field-based sensitivity is not that gradient.
2. **The accuracy claims sit on unverified foundations (R1-03, R1-04, R1-05).**
   - The equilibrium identities are compared with errors that lie below an unreported, mixed-precision algebraic residual.
   - The CutFEM reference is verified on four cells, without the thinnest walls, with non-monotone convergence and an unusually small ghost penalty.
   - Only direction-averaged operator errors are reported for the principal predictor.
3. **Novelty and baselines (R1-01, R1-06).** Classical Ritz, Galerkin, Chebyshev, two-grid and compliance-sensitivity results are presented as a new error analysis without attribution. The learned contribution is measured against weak baselines (a scalar graph-harmonic start, host PARDISO) rather than the monolithic multigrid solver that the correction's own components would give.
