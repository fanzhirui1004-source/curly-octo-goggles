# Review R7: "Learned static condensation for cut thin-walled TPMS cells with equilibrium correction" (CMAME)

Reviewer R7. Background: PIML-type learned substructures, meaning learned multiscale shape functions, condensed stiffness N^T K N, data-free energy training, corner-linear and Bézier boundary interpolation, oversampled/overlapping formulations, and large-scale topology optimisation. As instructed, I did not review Section 6.11 or the "[Placeholder ...]" sentences.

---

## 1. Summary

The manuscript approximates static condensation of cut thin-walled P-type TPMS cells (CutFEM, Q2, n = 32, about 67k–404k DOFs and 17k–26k retained DOFs per cell) on the complete retained space. It does not reduce the boundary representation. A geometry-conditioned, displacement-linear, matrix-free neural extension supplies an initial interior field. A fixed per-geometry two-grid correction (8 Chebyshev steps / trilinear Galerkin coarse solve / 8 Chebyshev steps) then reduces its interior residual. The condensed operator is F^T K F with the complete transpose. The network is trained through the correction with a log-energy plus eight-corner-sensitivity objective. The paper derives the classical Ritz identities and follows the interior error to assembled compliance (via energy participation) and to field-based thickness sensitivity. It shows that an uncorrected predictor can combine accurate compliance with sensitivity errors above 3% in weakly participating cells. The principal predictor reaches 0.074% mean directional energy excess on 80 validation geometries, below 0.06% compliance error and below 0.7% sensitivity error in 14 two-cell configurations, and at most 0.015% / 0.15% in two eight-cell lattices. The authors also report 9–29x faster condensation than host PARDISO, and they compare against a Bernstein-restricted boundary as a proxy for PIML.

## 2. Recommendation

**Major revision.**

**Strengths.** The work is careful and unusually candid about its limits: it separates training and selection sets, reports worst cases, verifies the reference, and admits development-case reuse. Several contributions are real:
- a displacement-linear, matrix-free learned extension on the complete retained trace of cut cells, where an explicit shape-function matrix is impractical;
- a fixed linear equilibrium correction placed inside the variational energy form, with a consistent complete transpose;
- differentiable training through that correction;
- a clean analysis that separates compliance error (energy participation) from local sensitivity error (derivative weighting).

**What stops acceptance in the present form.**
- **(a) Novelty is overstated.** Properties inherited from any admissible-extension Ritz construction are listed as contributions. The nearest classical prior art is not cited: component-based static condensation with reduced interior (bubble) spaces on full ports, and inexact substructuring.
- **(b) The Section 6.8 PIML comparison is a straw man.** A 24-coordinate, single-cell, corner-linear boundary under fine-scale face tractions is compared with a model holding more than 25,000 retained coordinates. It is presented in the Introduction as characterising PIML-type substructures.
- **(c) The headline cost claim mixes hardware.** The claim "cheaper for any number of queries" compares a GPU learned route with a CPU PARDISO route. The authors' own GPU data (Table ST14) indicate that an exact GPU factorisation applies the condensed operator 1.4–11x faster than the corrected learned operator.
- **(d) Offline costs are not reported.** The cost of training data and training, and the reliance on exact solutions, are missing, and scale is not demonstrated beyond eight cells. That is the regime in which learned substructures are actually used.

None of these is fatal. Most need rewriting, re-analysis of existing checkpoints and records, and two or three targeted new experiments.

---

## 3. Findings

### R7-01: Inherited variational properties presented as contributions
- **Severity:** Major
- **Location:** Abstract ("its energy ... defines a symmetric, positive semidefinite condensed stiffness, bounded below by the exact one"); Section 1, contribution (i) ("Its energy form defines a symmetric, positive semidefinite condensed stiffness with exact rigid-body behaviour, bounded below by the exact Schur complement"); Section 8, first paragraph; Section 7.1, second paragraph.
- **Issue:** For any admissible linear extension N with N_P = I and N R_P = R, the form N^T K N is symmetric positive semidefinite, has exactly the rigid kernel, and satisfies N^T K N − S = H^T A H ⪰ 0 (Eq. 9). This covers MsFEM, SCRBE-type bubble reduction and PIML's N^T K N with rigid-motion constraints. The Introduction does say that "the energy construction and rigid-motion constraints they employ are shared by the operator used here". However, the abstract, contribution (i) and the Conclusions still list these properties as features of NICE without attribution.
  - The "complete transpose" (Eq. 7, Appendix J.2) is simply N^T K N evaluated matrix-free with N = F. Any method that forms N^T K N with an explicit N already contains it.
  - Eqs. (9)–(12) and (C.2)–(C.3) are textbook Ritz/Schur results. The authors partly acknowledge this ("Classical Ritz orthogonality", "classical Ritz setting"), but contribution (iii) still reads as a new error analysis.
- **Why it matters:** CMAME readers from the substructuring, MsFEM and PIML communities will read contribution (i) as a claim of new variational structure. A contributions list should separate "inherited, retained" from "new".
- **Suggested fix:** Restructure the contributions into two lists.
  - *Inherited and retained (cite):* admissible extension, N^T K N energy form, exact rigid kernel, Ritz upper bound, energy training.
  - *New:*
    1. a displacement-linear, matrix-free learned extension on the complete retained trace of cut CutFEM cells;
    2. a fixed linear equilibrium correction composed inside the energy form, with its consistent transpose;
    3. training through that correction;
    4. the participation/derivative-weighting analysis relating compliance and local-sensitivity errors for learned substructures;
    5. the cut-band retained space for cut-surface virtual work.
  - Rephrase contribution (iii) as "application of classical Ritz and goal-oriented arguments to ...".
- **Fix type:** Text only
- **Confidence:** High

### R7-02: The closest classical prior art is missing, so the positioning of "complete retained space + approximate interior" is wrong
- **Severity:** Major
- **Location:** Section 1, second paragraph ("Two approximations reduce this cost ... This work takes the second route"); Section 7.1; reference list.
- **Issue:** Approximating the interior extension on the full port or interface space is not a new route. It is the defining idea of the static condensation reduced basis element method: interior bubble functions are reduced while port spaces remain full or are reduced separately, and a posteriori error bounds are provided (Huynh, Knezevic & Patera, ESAIM: M2AN 2013; Eftang & Patera, IJNME 2013 on port reduction; Ballani et al., CMAME 2018 for solid mechanics). Only Smetana & Patera (2016) is cited, and only for port reduction, which suggests SCRBE belongs to the "first route". Other omissions:
  - **Inexact substructuring / domain decomposition:** inexact subdomain solves in BDDC/FETI-DP (e.g. Li & Widlund 2007; Klawonn & Rheinbach 2007), and learned coarse spaces for FETI-DP (Heinlein, Klawonn, Lanser, Weber).
  - **Multiscale TO with coarse-basis correction:** spectral coarse-basis preconditioning for microstructural topology optimisation without scale separation (Alexandersen & Lazarov, CMAME 2015), which combines multiscale shape functions with fine-scale correction and addresses sensitivity accuracy.
  - **Multiscale FEM with oversampling:** GMsFEM / oversampled bases.
  - **Differentiable-solver training:** training through a numerical solver ("solver-in-the-loop", Um et al. 2020) and learned multigrid components (Greenfeld et al. 2019; Luz et al. 2020).
  - **Goal-oriented error estimation:** Becker & Rannacher 2001; Oden & Prudhomme 2001. This is exactly why energy-norm accuracy does not control a local functional (see R7-11).
  - **Inconsistent sensitivities from approximate models in TO:** e.g. Amir, Sigmund and co-workers on approximate reanalysis; Gogu 2015 on on-the-fly ROMs in TO.
- **Why it matters:** Without these, the novelty cannot be judged. Against SCRBE, the actual novelty is the learned, geometry-conditioned, displacement-linear interior extension together with the equilibrium correction, not the choice to keep the complete trace.
- **Suggested fix:** Add a related-work paragraph on component/substructure ROMs with full ports (SCRBE) and inexact substructuring, and state precisely what differs. SCRBE needs offline RB bubble spaces per parameter set and offers certified bounds; NICE conditions on geometry through a network and offers no certified bound (see R7-10). The authors should verify the exact bibliographic data of the works I list.
- **Fix type:** Text only
- **Confidence:** High for SCRBE and inexact DD relevance; medium for individual citations, which the authors should verify.

### R7-03: Section 6.8 is not a fair proxy for PIML-type substructures, and its result is used in the Introduction as a characterisation of that line
- **Severity:** Major
- **Location:**
  - Section 1, fourth paragraph: "restricting every box face to corner-linear displacements gives compliance errors of 78–85% even with exact cell operators".
  - Section 6.8: "degree one on the eight corners corresponds to the corner-linear boundary interpolation of the three-dimensional PIML examples".
  - Section 7.1: "the boundary restriction produces larger errors than the interior approximation of the corrected operator".
  - Table ST07; Figure 11.
- **Issue:** The experiment isolates boundary-restriction error, and it is useful as such. As a proxy for PIML it is biased in five ways:
  1. **Substructure scale is frozen at one TPMS cell.** PIML controls boundary-representation error mainly through how the domain is partitioned and how many super-nodes or boundary control points each substructure carries (substructure h-refinement), not only through polynomial degree on a fixed box. A thin-walled TPMS cell whose faces are cut by walls is close to the worst case for a single trilinear face field. PIML practice would subdivide the cell or use several nodes per face.
  2. **No oversampling or overlap.** The PIML-OFEM preprint (Guo et al. 2026b), which the authors cite, exists precisely to remove boundary-interpolation error through oversampled local bases in a partition-of-unity overlapping formulation. MsFEM oversampling is the classical remedy. Neither is tested.
  3. **The loads are adversarial to any reduced boundary.** The "neighbour-face" loads are fine-scale consistent tractions on thin-wall intersections of a restricted face, projected onto a degree-r space (Eq. I.3). A PIML user would apply loads at coarse nodes, or evaluate against the response to a load representable in the coarse space.
  4. **The sensitivity metric inflates errors on weakly participating cells.** Errors of 200–750% (Table ST07a, six-load set) are relative to the small eight-corner reference vector of a target carrying roughly 0.1–1% of the energy (see Table ST15 participations). For an optimiser, the absolute contribution of that cell's gradient is negligible (see R7-11).
  5. **Cost is unmatched.** At r = 1 on U1 the restricted model has 24 coordinates, against 28,206 for the full model. Stating that 24 coordinates are less accurate than 28,206 says nothing about accuracy per unit cost, which is the only axis on which PIML claims an advantage. With exact operators at r = 8 (about 14k coordinates on H1), errors are already 0.49–0.74%, comparable to B+W or A3.
- **Additional point.** PIML papers report percent-level or better accuracy on 3D lattice problems. The authors should reconcile "78–85%" with that record instead of leaving readers to infer that PIML-type models are 80% wrong. The likely explanation is the scale and loading choices above.
- **Why it matters:** This is the only quantitative evidence given for the central positioning claim, that the complete retained space is preferable to a reduced boundary. As designed, the comparison cannot support any statement about PIML-type substructures, and the Introduction's wording misrepresents that line.
- **Suggested fix.** Either (a) cut or move Section 6.8 to the Supplement, relabel it "effect of box-face polynomial restriction on single-cell substructures", and remove the PIML attribution and the 78–85% figure from the Introduction; or (b) make it fair:
  - (i) vary substructure size and super-node count (e.g. 1, 2^3 and 4^3 sub-substructures per TPMS cell, and several super-nodes per face);
  - (ii) include an oversampled or overlapping local basis (PIML-OFEM-type or MsFEM oversampling) and Bézier edge/face interpolation as published;
  - (iii) use coarse-representable loads, or report both load types;
  - (iv) report the error of the assembled gradient over all design variables and the absolute sensitivity error, alongside the per-cell relative error;
  - (v) present accuracy against coordinates and against wall-clock time on the same hardware, on a lattice with at least 27 cells.
- **Fix type:** Cut or move (minimum); new experiment (preferred)
- **Confidence:** High

### R7-04: Description of the PIML papers needs to be precise and checked against the sources
- **Severity:** Minor
- **Location:**
  - Section 1, fourth paragraph: "Its general boundary-coordinate formulation also permits an additional prescribed interpolation; the three-dimensional examples use a corner-based linear interpolation".
  - Section 3.1: "Unlike learned shape-function substructures that describe the boundary by a few coordinates".
  - Section 7.1: "its error is independent of how accurately the interior is learned".
  - Appendix I: "physics-informed local shape-function methods [Huang et al. (2023)]".
- **Issue:**
  - The statement about 3D examples should be checked and made specific: substructure size in fine elements, number of super-nodes per substructure, and whether boundary nodes between super-nodes are linearly interpolated.
  - Appendix I calls Huang et al. (2023) "physics-informed", but that paper is data-driven. The data-free, energy-trained variant is Huang et al. (2024).
  - "Independent of how accurately the interior is learned" is correct but one-sided. It should be followed by the levers PIML uses to reduce that error: partition refinement, Bézier enrichment (Guo 2026a) and oversampling or overlap (Guo 2026b). Otherwise the reader infers that the error is irreducible.
  - Section 7.1 should say that in PIML the boundary description is a deliberate trade for a coarse model small enough to handle 10^4–10^6 substructures (see R7-07). NICE does not make this trade, and does not offer it.
- **Why it matters:** Fair representation of prior work, and consistency between the text and what is actually compared.
- **Suggested fix:** Quote the specific configurations of the cited PIML examples with page or figure references. Correct "physics-informed". Add one sentence on the error-control levers of the PIML line.
- **Fix type:** Text only
- **Confidence:** Medium. My recollection of the 3D PIML configurations does not let me confirm or refute the "corner-based linear" wording; the authors must check it.

### R7-05: The cost claim rests on a GPU-versus-host comparison; the authors' own GPU data point the other way for application cost
- **Severity:** Major
- **Location:**
  - Abstract: "condenses a cell 9 to 29 times faster, stores 4 to 14 times less, and applies it 2 to 23 times faster".
  - Section 6.10: "Since both preparation and every application are cheaper, the learned route is cheaper for any number of queries per design iteration".
  - Section 7.4; Section 8, last paragraph; Table 5.
- **Issue.** The comparison mixes solvers and hardware:
  - The conventional route is PARDISO on 16 host cores; the learned route runs on one RTX 5090.
  - The shared front end runs on the host for one route (8.5–30 s) and on the GPU for the other (3.0–6.5 s). This inflates the "ready for queries" ratio of 3.1–8.1x with work that has nothing to do with condensation.
  - Table ST14 (same GPU, same four cells G1–G4) reports exact fp64 GPU factorisation times of 0.64–8.6 s and exact application times of 7.9/12.9/35.4 ms (G2) to 72.9/100/316 ms (G4) for batches of 1/16/64.
  - Table 5 gives the corrected learned application as 25/205/388 ms (G2) to 99/1,052/1,828 ms (G4).
  - On the same GPU, therefore, **the exact factor applies faster than A3 at every batch size**: about 1.4–3x at batch 1 and 6–11x at batch 64. The learned route is faster only in preparation, by about 4–8x (0.16–1.09 s against 0.64–8.6 s).
  - Break-even is roughly 90–320 vector applications per cell. The lattice solves of Section 6.9 need 183–234 block applications per cell for six loads, which is beyond break-even in every cell. On the authors' own numbers, a GPU exact-condensation route is plausibly several times faster per design iteration, although it uses more memory (fp64 factor 4.7–5.7 GiB for G1, G3, G4 against 0.7–1.4 GiB learned state; ST08a).
  - The ST14 records come from the D benchmark and may differ in implementation details. That is exactly why the comparison must be redone cleanly rather than omitted.
  - The single-vector speedups of 7.4–23x in Table 5 are largely a host-versus-GPU factor. Host PARDISO needs 826 ms for one vector on G1; the GPU factor needs 29 ms.
- **Why it matters:** "Cheaper for any number of queries" appears in the abstract, Section 6.10, Section 7.4 and the Conclusions. It is not supported under matched hardware, and the Supplement contains evidence against it. This is the claim a practitioner will act on.
- **Suggested fix.**
  - Report the conventional route on the same GPU: a GPU sparse direct solver such as cuDSS, or the fp64/fp32+IR factor path already used in ST14. Alternatively, run both routes on the host.
  - Run the front end on the same device for both routes, and time it separately.
  - Report break-even query counts and time per design iteration at matched compliance and sensitivity accuracy.
  - Remove "for any number of queries" unless it survives the matched comparison.
  - Keep the memory advantage, which appears genuine, as the main selling point if it holds.
- **Fix type:** Re-analysis of existing data (ST14 against Table 5) plus a new experiment (matched-hardware rerun)
- **Confidence:** Medium-high. The ST14/Table 5 comparison crosses benchmark campaigns, but the direction and size of the gap are unlikely to reverse.

### R7-06: Offline cost and dependence on exact solutions are not reported, while the method is contrasted with data-free PIML
- **Severity:** Major
- **Location:**
  - Section 3.3: "each retained direction is normalised to q_j^T S q_j = 1 using the reference substructure solution"; Eq. (8) sensitivity labels.
  - Appendix G.2–G.3: direction classes; adversarial search "Applying S^{-1}_⊥ uses an equilibrated Neumann solve".
  - Section 6.1: "A3's continuation took 2.7 h".
  - Section 1, fourth paragraph: data-free PIML is cited without contrast.
- **Issue.** Every training geometry (591, plus the 305 used for B) needs exact reference machinery:
  - interior factorisations for S-normalisation;
  - exact fields for the eight-corner sensitivity labels;
  - exact responses to generate the `force`, `force_c`, `face`, `face_c`, `support`, `support_k` and `glued` classes, and S^{-1} applications for `adv`. By the nominal class weights in G.3, these exact-solve-derived classes account for 77.5% of sampled directions; only `macro` and `grf` are solve-free.
  
  None of the following is reported: data-generation wall-clock or storage, B's 40,000-step training time, or total GPU-hours. The data-free PIML variant (Huang et al. 2024) needs none of this, and the manuscript does not acknowledge the contrast.
- **Why it matters:**
  1. The "any number of queries" argument ignores amortisation of offline cost. Break-even must be counted in cells times design iterations.
  2. Reliance on exact solutions limits extension to larger cells, finer n, other TPMS families or nonlinear materials. Each would need a new exact-data campaign, while a data-free energy method would not.
  3. The log-energy term does not in fact need exact S. For fixed q, minimising log(q^T Ŝ q) has the same minimiser, because Ŝ ⪰ S. The normalisation is a weighting and could be replaced by a cheap proxy such as q^T K_PP q. A data-free variant is therefore within reach for the energy term, which makes it more puzzling that the authors did not pursue or discuss it.
- **Suggested fix.**
  - Report offline cost in full: number of factorisations, number of labelled directions, data-generation GPU/CPU-hours, storage, and training GPU-hours for B and A3. Report break-even in cell-iterations against both the host and the GPU conventional routes.
  - Add a paragraph contrasting NICE with data-free PIML.
  - Ideally, add an ablation training A3-style with a data-free energy term (proxy normalisation, only `macro`/`grf`/random directions, w_s = 0), to show how much of the accuracy depends on exact data.
- **Fix type:** Text only (reporting); new experiment (data-free ablation)
- **Confidence:** High

### R7-07: Scale is demonstrated only up to eight cells, and per-cell costs imply the method is not in the regime of learned substructures
- **Severity:** Major
- **Location:** Sections 6.9–6.10 and 7.4; Tables 5, 6 and ST20.
- **Issue.** PIML-type methods are used for structures with 10^4–10^6 substructures on a workstation, because each substructure contributes a few dozen coarse DOFs. NICE's figures:
  - roughly 17–26k retained DOFs per cell (139k–144k free retained DOFs for eight cells);
  - 3–7.6 s front end and preparation per cell;
  - 0.25–1.4 GB operator state per cell;
  - about 200 block applications per cell per design iteration, at 25–100 ms per single-vector application;
  - an unpreconditioned-scale global PCG (114–165 iterations for 4–8 cells).
  
  Extrapolating linearly to 1,000 cells gives about 17–26 million global retained DOFs, 0.25–1.4 TB of operator state (streamed from host memory), about 1–2 h of front end and several hours of PCG per design iteration on one GPU. That rests on the unverified assumption that PCG iteration counts stay bounded. The learned solve's recomputed residual also stagnates at 3.6×10^-4 to 3.2×10^-3 because of the fp32 network (Section 6.9), and this floor may grow with lattice size.
  
  In effect NICE is an inexact, matrix-free substructuring solver for the full fine-scale model, not a reduced-order model. That framing is legitimate, but then its competitors are fine-scale iterative solvers and domain decomposition methods, not PIML.
- **Why it matters:** The paper places itself in the "learned substructures" line. Readers need to know it addresses a different point on the accuracy/cost curve, namely fine-scale-accurate analysis of tens of cells rather than coarse analysis of very many.
- **Suggested fix.**
  - Add a scaling study: lattices of 27, 64 and at least 125 cells, reporting global PCG iterations, time and memory per design iteration, and the residual floor.
  - State explicitly the number of cells beyond which the method is impractical on one GPU.
  - Reframe the positioning in Sections 1 and 7 accordingly.
  - If a coarse-level model is the eventual aim (e.g. Galerkin-projecting NICE cells onto a coarse space for the global solve), say so.
- **Fix type:** New experiment; text
- **Confidence:** High

### R7-08: Attribution of "training through the correction" is confounded with extra data and steps
- **Severity:** Major
- **Location:**
  - Section 6.3: "training through it reduces the remaining mean by a further factor of 1.3".
  - Section 6.6: "training through it reduces the remaining sensitivity error on the most demanding cells".
  - Section 7.3; Section 8 ("factors of 1.35 to 2.2"); Table 2.
- **Issue:** A3 is B continued for 15,000 steps on 591 geometries through the correction. B+W is B (305 geometries) evaluated with the correction. The A3/B+W difference therefore mixes three effects: training through the correction, a training population about twice as large, and 15,000 more steps. The clean control, C (B continued for 15,000 steps on the same 591 geometries without correction) evaluated with A3's correction ("C+W"), is not reported, although the C checkpoint exists. C's uncorrected improvement over B is small (6.89% → 6.33%), but after correction the remaining error depends on components that extra data may still improve.
- **Why it matters:** Training through the correction is one of the few genuinely new elements, and its measured effect is modest (a factor of 1.3 on mean energy; 1.35–2.2 on worst sensitivity). If most of that comes from data and steps, the contribution shrinks further.
- **Suggested fix:** Evaluate C+W on the 80-geometry population and on the 14 assembly configurations. Report A3 against C+W as the effect of training through the correction.
- **Fix type:** Re-analysis of existing data (existing checkpoint; evaluation only)
- **Confidence:** High

### R7-09: The sensitivity term of the objective is claimed as a contribution but is not ablated
- **Severity:** Major
- **Location:**
  - Contribution (ii): "trained through the correction with an objective that includes thickness sensitivity".
  - Section 3.3: "the reported configurations use w_s = 1".
  - Section 7.2: "The same separation motivates the sensitivity term of the training objective".
- **Issue:** Every reported predictor uses w_s = 1, so there is no evidence that the sensitivity term affects either energy or sensitivity accuracy. After correction, energy errors are around 0.1%. By Eq. (J.3), the sensitivity error is then dominated by the linear term, which the energy objective does not control. That is the one regime where the term could matter, and it is untested.
- **Why it matters:** The term is listed as a contribution and motivated at length by the analysis. It also requires exact sensitivity labels (see R7-06).
- **Suggested fix:** Retrain A3 with w_s = 0 (one continuation of about 2.7 h). Compare population energy, the 14-configuration sensitivity maxima and the lattice sensitivities. If the effect is negligible, drop the claim or downgrade it to an implementation detail.
- **Fix type:** New experiment
- **Confidence:** High

### R7-10: "Error reducible by the correction budget" is shown only thinly, has no estimator to select the budget, and is not assessed against cost
- **Severity:** Major
- **Location:**
  - Abstract: "Its error ... can be reduced through both the network and the correction budget".
  - Contribution (i); Section 6.5 (M1: 2/4/8 steps give 1.02/0.42/0.186%; Table 3); Section 7.3; Appendix I.
- **Issue.** Five problems:
  1. **Thin evidence.** Budget reduction is shown on one cell (M1) for k ∈ {2, 4, 8} and on five cells for k = 32. Multiple two-grid cycles, which would demonstrate actual convergence to the exact Schur complement, are never shown. The network is trained for one budget (8/Q1/8), and changing k at evaluation moves away from the trained regime.
  2. **Slow rate.** On M1, quadrupling the steps from 8 to 32 per stage gives only a 2.6x reduction (0.186% → 0.0727%), while the application cost, dominated by the stiffness actions, rises roughly fourfold.
  3. **No computable upper bound.** Appendix I gives only lower bounds. A user therefore cannot choose a budget to meet a target tolerance without the exact reference. "Controllable" in the usual numerical sense implies an error estimator. SCRBE-type methods (R7-02) provide one.
  4. **Precision floor.** The fp32 network sets a global residual floor (Section 6.9), so the budget cannot drive the error arbitrarily low.
  5. **The network's value shrinks as the budget grows.** On H2 at 32/Q1/32, the harmonic start already reaches 0.0006% with no network.
  
  The relevant question is therefore the error-versus-wall-clock Pareto front, not error versus budget.
- **Why it matters:** "Controllable" or "convergent" is a central selling point over PIML, whose error is fixed once trained. As shown, it is a qualitative property whose practical cost has not been measured.
- **Suggested fix.**
  - Provide error against wall-clock time (per cell and per design iteration) on the same GPU for: NICE with k = 2, 4, 8, 16, 32 and one or more two-grid cycles; harmonic or other non-learned starts with the same correction and with repeated cycles; a proper multilevel V-cycle start; and exact GPU factorisation.
  - Show whether any NICE configuration is Pareto-optimal.
  - Either provide an a posteriori upper estimator (e.g. a Rayleigh-quotient or residual-based estimator with a verified effectivity) or state clearly that the budget cannot be chosen without reference data.
- **Fix type:** New experiment; text
- **Confidence:** High

### R7-11: "Accurate compliance does not imply accurate local sensitivity" is known in principle and demonstrated on a case of limited practical weight
- **Severity:** Major
- **Location:**
  - Contribution (iv); Section 8, second paragraph ("The principal mechanical finding ...").
  - Section 6.6 (S8 on U1/x: compliance 0.00137%, target sensitivity 3.89%, target energy share 0.13%); Section 6.7.
- **Issue.**
  - That an energy-norm-accurate approximation need not approximate a local output functional to the same relative accuracy is the basic premise of goal-oriented error estimation. It is also well documented for approximate reanalysis and ROMs in topology optimisation. The contribution is therefore a quantification for learned substructures, not a new principle.
  - The flagship demonstration concerns a cell carrying 0.13% of the assembled energy, and the error is normalised by that cell's own small reference vector. In an optimiser, the relevant quantities are the error of the assembled gradient over all design variables (relative norm, angle, or constraint-weighted error), and ultimately the true performance of the optimised design.
  - A 4% relative error on a gradient block that is about 10^-3 of the total gradient norm may be irrelevant to the design. Conversely, the paper never shows that a PIML-level compliance accuracy produces worse designs.
- **Why it matters:** This is presented as the principal mechanical finding and used to argue against reduced-boundary methods (Sections 6.8, 7.1).
- **Suggested fix.**
  - Report the relative error and angle of the full assembled gradient (all cells, all corners) for the two-cell and lattice examples, next to the per-cell errors. This is computable from existing records.
  - Reword contribution (iv) as a quantification with appropriate citations.
  - Defer claims about design impact to the planned Section 6.11, where the exact-verified optimised design (under NICE and under a fair PIML-type baseline) is the real test.
- **Fix type:** Re-analysis of existing data; text
- **Confidence:** Medium-high

### R7-12: The non-learned starting-field baseline is weak
- **Severity:** Minor
- **Location:** Section 6.5, Table 3 ("a graph-harmonic interior extension"); Section 7.3 ("a harmonic starting field leaves 7 to 290 times the error").
- **Issue:** A material-volume-weighted graph Laplacian is a poor elastic extension. Its errors are 1.2–17% even after 8/Q1/8. Other cheap and linear non-learned starts exist:
  - a coarse Galerkin solve first (i.e. starting with the coarse correction rather than smoothing from zero);
  - the exact extension of a lower-resolution model (e.g. n = 16) interpolated to n = 32;
  - two or more two-grid cycles;
  - a three-level V-cycle using the 33/17/9 hierarchy the network already uses.
  
  The "7–290x" factor measures the network against a weak competitor. Also note that the abstract's "learned starting field" refers to B, not the principal predictor A3.
- **Why it matters:** This comparison is the main evidence that learning adds value beyond the numerical correction. B+W already meets all accuracy criteria, and the correction does most of the work (6.89% → 0.097%).
- **Suggested fix:** Add at least a coarse-first start and a multi-cycle or multilevel start at matched wall-clock time (see R7-10). Clarify in the abstract that the factor refers to B.
- **Fix type:** New experiment (cheap, per-cell); text
- **Confidence:** High

### R7-13: The whole-lattice direct comparison (Table 6) is dominated by host front-end time and omits the natural fine-scale iterative competitor
- **Severity:** Minor
- **Location:** Section 6.10, Table 6; ST20b; Supplementary Note S7 ("The host was shared with other jobs ... load average between 11 and 37").
- **Issue.**
  - Host cell setup and assembly take 140–152 s of the 242–256 s direct time, while the learned front end takes about 5 s on the GPU. The authors do give the phase-only comparison (91 s against 30–31 s), but the headline ratios include the front-end mismatch.
  - The direct runs were on a shared host, whereas Table 5's caption says timings were on "an otherwise idle job".
  - The eight-cell direct timings are lower bounds only.
  - The direct times exclude sensitivities.
  - Most important: the obvious competitor at this scale is not a direct solver. It is a matrix-free or AMG/GMG-preconditioned CG on the full fine CutFEM model on the same GPU, or a BDDC/FETI-DP solver. Such a solver would also not need 60–160 GB.
- **Why it matters:** The claim "A direct solution of the whole lattice ... already takes more time and memory than the learned route at four cells" is true but uses a strawman baseline for fine-scale analysis.
- **Suggested fix:**
  - Time the front end on the same device.
  - Rerun the direct solves on an idle host.
  - Add a GPU multigrid-preconditioned CG on the full fine model at matched accuracy.
  - Report sensitivities for both routes.
- **Fix type:** New experiment; text
- **Confidence:** High

### R7-14: Generality is narrow compared with "problem-independent" learned substructures, and simpler parametric surrogates are not considered
- **Severity:** Minor
- **Location:** Sections 2.1 and 6.1 ("All geometries carry a single planar cut with normal (cos ϑ, sin ϑ, 0)"); Table 1 (n = 32, E_Y = 1, ν = 0.3, γ = 10^-4).
- **Issue.**
  - The family is one TPMS type, eight corner parameters, one planar cut with a normal in a canonical half-quadrant, one resolution and one Poisson ratio. That is roughly 10 parameters, served by a network with 603k parameters.
  - PIML's claim is reuse across arbitrary material distributions inside a substructure. NICE's element-moment input suggests broader applicability, but this is not tested: no gyroid or diamond cells, no two cuts, no other cut orientations beyond the symmetry group, no other n or ν.
  - For a roughly 10-parameter family, a parametric reduced basis for the interior extension (SCRBE-style), or interpolation of exact operators, is a natural non-learned competitor and should at least be discussed.
- **Why it matters:** Generality is the main reason to learn rather than precompute.
- **Suggested fix:** Add at least one out-of-family test (e.g. a gyroid cell or a two-cut cell) with and without correction; the correction should partly rescue generalisation, which would be a strong result. Discuss parametric ROM alternatives.
- **Fix type:** New experiment (small); text
- **Confidence:** High

### R7-15: Surrogate accuracy is an order of magnitude below the reference discretisation error, which affects cost fairness
- **Severity:** Minor
- **Location:** Section 6.2: "For heavily cut cells the discretisation error of the reference at n = 32 is thus of order one percent, an order of magnitude above the energy errors of the corrected predictor".
- **Issue:** For heavily cut cells, NICE's error (about 0.01–0.1%) is well below the n = 32 discretisation error (about 1%). Two consequences follow:
  1. The correction budget could be reduced substantially without affecting accuracy relative to the continuum, which changes the cost comparison in NICE's favour.
  2. Equally, the conventional route could use a coarser n at similar continuum accuracy, which changes it the other way.
  
  In both cases, cost comparisons should be made at matched continuum-level accuracy, not at the discrete-operator level alone.
- **Why it matters:** It shows what accuracy is actually needed, and makes the cost comparison fair.
- **Suggested fix:** Discuss this, and ideally include one accuracy-matched cost point: NICE with a reduced budget, and the conventional route at n = 24.
- **Fix type:** Text; re-analysis of existing data
- **Confidence:** Medium

### R7-16: Retained cut-band coordinates inflate the global system when cut surfaces are traction-free
- **Severity:** Minor
- **Location:** Section 2.2 ("all displacement coefficients of active elements carrying a positive-area macro-cut patch"); Section 2.3 ("retained cut-band coordinates outside the box faces remain local to their substructure"); Table ST07 (M1: 15,423 controlled DOFs even at r = 1).
- **Issue:** Cut-band coordinates are private to each cell and carry only cut-surface loads. In the lattice and design settings emphasised here, the specimen-boundary cut is typically traction-free. The cut-band coordinates could then be condensed into the interior exactly, removing a large share of the global retained DOFs (on M1, about 15k of 40k) and of the PCG work. Keeping them is a modelling choice that increases cost. The paper also treats it as part of the "complete retained trace" novelty.
- **Why it matters:** It affects cost, scalability (R7-07) and the novelty claim about the cut band.
- **Suggested fix:** Offer an option in which traction-free cut bands are interior. Report its effect on cost and accuracy, or justify retaining them.
- **Fix type:** Text; new experiment (optional)
- **Confidence:** Medium-high

### R7-17: Field-based sensitivity is used as the accuracy measure, while an optimiser driven by the surrogate needs the consistent derivative
- **Severity:** Minor
- **Location:** Section 4.3, Eq. (14) ("The reported field-based estimates contain only the first term"); Section 7.2 ("Measuring this term within a complete design optimisation is left to subsequent work").
- **Issue.**
  - All reported sensitivity accuracies use the field-based estimate. An optimiser that minimises surrogate compliance with field-based gradients uses gradients inconsistent with its own objective. Line searches and MMA/OC convergence can suffer, and PIML-based TO uses the consistent coarse-model derivative.
  - The residual-term magnitude argument (about 2.7% of the field norm) is an order-of-magnitude estimate that has not been measured.
  - For the Section 6.8 comparison, the definition of sensitivity used for the restricted model should match.
- **Why it matters:** The design use case is the paper's motivation.
- **Suggested fix:** Measure the residual term of Eq. (14) for A3 by finite differences of the surrogate compliance on a few cells. State the gradient definition used in Section 6.8.
- **Fix type:** New experiment (small)
- **Confidence:** Medium-high

---

## 4. What I would accept as evidence of novelty

- **Clear delimitation:** a contributions list that explicitly marks the admissible extension, N^T K N, the rigid kernel, the Ritz lower bound and energy training as inherited (PIML, MsFEM, SCRBE). Claimed as new: the displacement-linear matrix-free learned extension on the full cut-cell trace, the fixed correction inside the energy form with its transpose, and training through it.
- **Evidence that learning adds value beyond the numerical correction,** at matched wall-clock time on the same hardware, against strong non-learned starts (coarse-first, multi-cycle or multilevel, lower-resolution exact extension). An error-versus-time Pareto front on which NICE is non-dominated.
- **Isolated effect of training through the correction:** A3 against C+W. Isolated effect of the sensitivity term: w_s = 0 against 1.
- **A fair comparison with the PIML line** at matched accuracy or matched cost, including substructure refinement and an oversampled or overlapping variant, evaluated by global gradient accuracy and, eventually, exact-verified optimised designs.
- **A matched-hardware cost comparison** (conventional condensation on the same GPU), including offline data and training cost amortised over cell-iterations, and a scaling study to at least 125 cells.

## 5. Top three concerns

1. **The cost claim is not established under matched hardware.** The headline "cheaper for any number of queries" compares a GPU learned route with a host PARDISO route. The authors' own GPU timings (ST14) indicate that exact GPU condensation applies 1.4–11x faster than the corrected operator, and a lattice design iteration exceeds the break-even query count. Offline data and training costs, which require exact solutions for about 78% of training directions and for all sensitivity labels, are not reported (R7-05, R7-06).
2. **Novelty relative to PIML and classical component-based condensation is overstated, and the PIML proxy is a straw man.** The variational properties are inherited, SCRBE and inexact substructuring are not cited, and Section 6.8 compares a 24-coordinate single-cell boundary under fine-scale tractions with a model of more than 25,000 coordinates. The resulting 78–85% error is then quoted in the Introduction as characterising PIML-type substructures (R7-01 to R7-04).
3. **Scale and practical controllability are not demonstrated.** Eight cells is the largest problem. Per-cell memory, preparation and query counts place the method in the regime of fine-scale substructuring solvers rather than reduced-order substructures. "Error reducible by the correction budget" comes with no a posteriori estimator, slow rates, an fp32 floor and no cost-accuracy Pareto analysis. The attribution of gains to training through the correction and to the sensitivity term is confounded or untested (R7-07 to R7-10).
