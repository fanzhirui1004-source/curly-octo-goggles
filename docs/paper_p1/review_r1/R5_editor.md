# R5 — Handling editor pre-review assessment (CMAME)

Manuscript: "Learned static condensation for cut thin-walled TPMS cells with equilibrium correction" (method: NICE)
Materials read: MANUSCRIPT_EN.md, APPENDICES_EN.md, SUPPLEMENTARY_EN.md, FIGURES.md, main-text figures F01, F02, F04, F05, F10.
Section 6.11 and the "[Placeholder …]" sentences were ignored except for planning space.

Line numbers (L…) refer to MANUSCRIPT_EN.md in its current state.

---

## 1. Editor's assessment

### Desk-reject risk: **medium**

- If the paper is submitted as it is now, with the placeholders in the abstract, Section 6.11 and Section 8, the risk is **high**. A manuscript with unfinished sections is returned without review.
- Once 6.11 is complete and the length plan below is applied, the risk falls to **low to medium**. The topic fits CMAME well: learned substructures, cut finite elements, variational error analysis and lattice design. The mathematics is correct in form, and the verification is unusually careful.
- The risks that remain come from presentation, not from scope:
  - **Length.** The main text is about 14,650 words, against roughly 8–12k for typical CMAME articles. The appendices add about 9,100 words, and the preprint runs to 86 pages.
  - **A diffuse story.** Three storylines compete: the method, the claim that compliance accuracy does not imply sensitivity accuracy, and a cost study. On top of that there are eight predictor labels and twelve cell labels.
  - **A benchmark reviewers will call unfair.** One GPU is compared against a 16-core host direct solver, and there is no GPU or iterative condensation baseline.

### Main strengths

1. **A clean variational construction.** A fixed linear correction is applied to a linear, admissible, rigid-preserving extension, and its energy is evaluated with the exact stiffness. This gives a symmetric positive semidefinite (SPSD) condensed operator that is bounded below by the Schur complement, with exact rigid kernel (Eqs. 6–9, 17). The guarantees do not depend on how accurate the network is. This is the central idea, and it is new relative to PIML-type and energy-element approaches.
2. **An error chain that is easy to follow and can be checked:**
   - Ritz identity (Eq. 9) → compliance as total reconstructed error energy (Eq. 11) → participation bound (Eq. 12) → sensitivity expansion (Eq. 13) → complete design derivative (Eq. 14).
   - Figure 10(a) confirms Eq. (12) almost exactly over 192 load cases.
   - The δ²κ decomposition in Section 6.4 is a memorable physical explanation: displacement errors of about 1% become energy errors of 2–35% in thin walls.
3. **Honest ablation.** Table 3 compares zero, harmonic and learned starts under the same correction; the learned start is 7–290 times more accurate than the harmonic one. B+W against A3 separates what the correction contributes from what training through it contributes. The limitations in Section 6.1 (weight-selection overlap, development cells) are disclosed.
4. **Verification of the discrete reference and of the implemented operator** (Section 6.2): mesh, penalty, integration, finite-difference step, symmetry, energy–action consistency and rigid energy.
5. **Tests where every cell is learned** (Section 6.9). The heterogeneous lattices show that the errors of individual cells combine through participation rather than accumulate.

### Main weaknesses

1. **Too long, and repetitive.** The Discussion (1,738 words) largely restates Sections 4 and 6. Section 6 (6,643 words) mixes principal evidence with diagnostics that belong in the supplement. Of the 80 subsections and notes in appendices and supplement, several are never cited from the main text (Appendices C, I, J.8, J.9).
2. **The headline mechanical finding is shown on the baselines, not on the method.** "Accurate compliance does not imply accurate sensitivity" is demonstrated with B, C and S8. For NICE both errors are small. The abstract, contribution (iv) and Section 8 currently present this as a result, when it reads better as the design criterion the method satisfies.
3. **Missing or unfair baselines for the cost claims.**
   - The hardware is unequal: one RTX 5090 against 16 host cores.
   - The front-end assembly runs on the GPU for one route and the host for the other, although it is the same algorithm.
   - There is no GPU sparse direct solver.
   - There is no matrix-free iterative interior solve (for example AMG-PCG) at matched accuracy, although Section 7.4 itself says this is the relevant comparison.
4. **Over-general wording in the abstract and title.**
   - The paper studies only the Schwarz-P level set, one planar cut per cell (normal in the xy-plane), a fixed n = 32 grid to which the network architecture is tied, and ν = 0.3. Yet it speaks of "TPMS cells" in general.
   - "80 unseen geometries" is not strictly true, because 20 of them entered checkpoint selection.
   - There is no Limitations section.
5. **Terminology overload.**
   - Predictor labels: P0, B, B+W, C, S8, A2b, A3, D.
   - Cell labels: U1, U2, M1, M2, H1, H2, H3, L1, G1–G4.
   - The method name NICE appears only in the abstract, the contributions and the conclusions; the results call it A3.
   - Colour and marker coding changes between figures: orange squares mean A3 in Figs. 5 and 9 but S8 in Fig. 10.

---

## 2. Findings

### R5-01 — Major — Incomplete design-optimisation example and its consistency with the rest of the paper
- **Location:**
  - Abstract, last sentence "[Placeholder: one sentence on the design-optimisation example…]"
  - §6.11
  - §8, L572
  - §7.2, L546: "Measuring this term within a complete design optimisation is left to subsequent work."
- **Issue:**
  - The placeholders must be removed before submission.
  - §7.2 explicitly defers measuring the complete-derivative term to future work. That conflicts with the planned §6.11, where an optimiser will be driven by the surrogate.
  - The reported sensitivities are field-based and contain only the first term of Eq. (14). An optimiser minimising the surrogate compliance with these gradients uses an inconsistent gradient. Reviewers will check this.
- **Suggested fix:**
  - In §6.11, report the magnitude of the residual-weighted term of Eq. (14) along the optimisation history, or a finite-difference check of the gradient used.
  - State which gradient the optimiser uses.
  - Rewrite the §7.2 sentence accordingly.
  - Keep §6.11 to about 600–900 words, one figure (history plus final design) and one table (final design verified against exact condensation: compliance, sensitivity error, wall-clock time against exact-condensation optimisation).
  - Move the shared-design-variable paragraph from §4.3 (L267) into §6.11, where it is used.
- **Fix type:** new experiment (already planned) + text only.

### R5-02 — Major — Length far above the CMAME norm
- **Location:** whole manuscript. The main text is about 14,650 words; the appendices about 9,090; 11 figures and 6 tables; 86 pages.
- **Issue:** Reviewers are likely to decline, or to ask for a shorter version. The Discussion repeats the Results, and several diagnostic studies (spectral modes, orientation, replacement decomposition, whole-lattice direct solve, reference convergence details) are secondary to the three main claims.
- **Suggested fix:** Apply the plan in Section 3 of this report. The target is about 9,500 words before §6.11 and about 10,300 with it. Move about 4,300 words of appendices to the supplementary material.
- **Fix type:** cut or move.

### R5-03 — Major — The central story is diffuse; abstract, contributions and conclusions do not have one headline
- **Location:**
  - Abstract: "Accurate compliance therefore need not imply accurate local sensitivity…"
  - Contribution (iv), L25: "The demonstration that accurate compliance does not imply accurate local sensitivity for learned substructures…"
  - §8, L568: "The principal mechanical finding is that accurate compliance does not imply accurate local thickness sensitivity…"
- **Issue:**
  - The paper presents NICE as the method, but it calls the compliance/sensitivity separation the "principal finding". That separation is shown only for predictors that are not the proposed method (B, C, S8; Table 4, Fig. 10).
  - The abstract spends about half its length on analysis statements and about 20 numbers.
  - Contributions (i) and (ii) overlap: both describe the operator.
  - Contribution (iv) mixes a negative finding about baselines with a positive result for NICE.
  - The conclusions follow a different order again: method, then finding, then complementarity, then cost.
- **Suggested fix:** Use one storyline everywhere:
  - The problem: reduced-boundary learned substructures incur a boundary error, which §6.8 shows is large for cut cells.
  - The idea: keep the full retained space, and learn only the interior initial field. A fixed equilibrium correction then turns it into a variational, bounded, controllable operator.
  - The analysis: the error chain explains why compliance accuracy is not enough for thickness sensitivity. This becomes the design criterion, not the headline.
  - The evidence: NICE meets both criteria for populations, pairs and lattices, at lower cost, and drives an optimisation (§6.11).

  Reduce the contributions to three:
  - (1) the NICE operator and its variational guarantees, including training through the correction;
  - (2) the error chain from interior error to compliance and sensitivity, including δ²κ;
  - (3) the numerical evidence, including the boundary-restriction comparison and the design example.

  Mirror the same three in §8.
- **Fix type:** text only.

### R5-04 — Major — "Unseen" geometries and development cells
- **Location:**
  - Abstract: "on 80 unseen geometries"
  - Fig. 5 title
  - §8, L568: "every one of 80 unseen geometries"
  - §6.1, L357: "the weights … were selected on a validation list whose first 40 geometries include 20 of the 80 reported validation geometries … the cells used in the assembly examples served repeatedly as development cases"
- **Issue:**
  - "Unseen" is contradicted by the disclosure in §6.1. Reviewers will read this as overclaiming, even though the effect is negligible (0.077% against 0.074%).
  - More seriously, every assembly cell (U1–L1) was a development case. So the "all fourteen configurations" claim, which appears in the abstract and the conclusions, rests on non-independent cells.
- **Suggested fix:**
  - Write "80 validation geometries not used in training", and quote the 60-geometry held-out figure alongside.
  - Add two-cell assembly tests on 4–6 cells drawn fresh from the validation strata (for example 60 held-out cells, one per stratum and x/y). Report them in Fig. 9 or in a sentence.
  - The lattices of §6.9 may already consist of fresh cells. If so, say so explicitly, because it partly answers this point.
- **Fix type:** text only + new experiment (small; the pipeline exists).

### R5-05 — Major — Fairness of the cost comparison
- **Location:**
  - §6.10, L494: "The conventional route runs on the 16 host cores … the learned route runs on one RTX 5090"
  - Table 5 "Front end (s): host / GPU"
  - Abstract: "condenses a cell 9 to 29 times faster, stores 4 to 14 times less, and applies it 2 to 23 times faster"
  - §6.10, L505: "the learned route is cheaper for any number of queries"
- **Issue:**
  - The speed-ups compare different hardware.
  - The front-end speed-up (3.1–8.1 times) comes from running the same assembly on the GPU. It says nothing about the method.
  - GPU sparse direct solvers (for example cuDSS) and GPU batched triangular solves exist. A CMAME reviewer will ask for at least one of them, or for normalisation.
  - "Cheaper for any number of queries" holds only for this hardware pairing.
- **Suggested fix:**
  - Either add a GPU direct-condensation baseline on the same card, or restate the claims as hardware-specific: "on one GPU against a 16-core host direct solver".
  - Remove the front-end ratio from the method claims.
  - Soften "for any number of queries" in the abstract and §8.
  - Report energy or cost-normalised figures if a GPU baseline is not possible.
- **Fix type:** new experiment (preferred), or text only (minimum).

### R5-06 — Major — No matched-accuracy iterative-condensation baseline
- **Location:**
  - §7.4, L558: "The relevant comparison is the total work required to attain the prescribed response accuracy"
  - Table 3, which uses fixed budgets only
- **Issue:**
  - The natural non-learned alternative is matrix-free condensation: apply S through an iterative interior solve (PCG with AMG, or the same two-level cycle iterated) to the accuracy NICE attains.
  - Table 3 shows the value of the learned start at a fixed budget. It does not show wall-clock time to reach, say, 0.1% energy excess from zero or harmonic starts with a good preconditioner.
  - Without this, a reviewer can argue that the network only replaces a stronger preconditioner.
- **Suggested fix:** On the five fixed-weight cells, add an accuracy-against-time (or against stiffness actions) curve for:
  - (a) NICE with k = 2, 4, 8, 16;
  - (b) zero or harmonic start with k = 8…64 plus Q₁(17);
  - (c) interior PCG with smoothed-aggregation AMG, to matched energy excess.

  This fits in one figure panel plus two sentences; the details can go in the supplement.
- **Fix type:** new experiment.

### R5-07 — Major — Scope stated too generally; no Limitations section
- **Location:**
  - Title and abstract: "cut thin-walled TPMS cells"
  - Eq. (1): P-type level set only
  - §6.1, L339 and L357: single planar cut with normal (cos ϑ, sin ϑ, 0)
  - Table 1: n = 32, ν = 0.3
  - §3.1, L135: grids of 65/33/17/9 fixed by the architecture
- **Issue:**
  - The architecture and training are tied to one TPMS family, one background resolution and one material.
  - There is no statement about retraining cost, or about transfer to gyroid or diamond cells, to other mesh resolutions, or to multiple or curved cuts.
  - The limitations appear only as scattered clauses.
- **Suggested fix:**
  - Say "Schwarz-P-type" (or "P-type TPMS") in the abstract; the title is covered in Section 4 of this report.
  - Add a short "Limitations and extensions" paragraph at the end of §7 (about 150–200 words) covering:
    - family and resolution dependence, and the retraining cost (A3 continuation 2.7 h; full training time);
    - the single planar cut;
    - evidence confined to the discrete reference (discretisation error of about 1% for heavily cut cells at n = 32, §6.2);
    - the hardware-specific cost comparison.
- **Fix type:** text only.

### R5-08 — Major — Too many labels; inconsistent naming and colour coding
- **Location:**
  - Table 2, and §6.1 L341: "The principal predictor A3 continues B … B+W applies A3's correction … S8 continues B with eight smoothing steps … only part of its training steps passed through the smoothing stage … P0 provides a reference … a separate uncorrected predictor D"
  - Fig. 5 and Fig. 9 legends: "B + correction (untrained)"
  - Table 3 column "B (learned)" (this is B with the correction, i.e. B+W)
  - Fig. 10 colour coding
- **Issue:**
  - There are eight predictor labels. Three are marginal:
    - P0 is used only in ST01 and S01.
    - D appears only in the supplementary legacy study.
    - S8 is described as "not interpreted as a trained smoothing variant", yet it carries two main-text results (Table 4, Fig. 10).
  - "W" in B+W is never explained as the correction operator 𝒲.
  - The method name NICE is not used in the Results, which say A3.
  - Twelve cell labels (U1, U2, M1, M2, H1, H2, H3, L1, G1–G4) are defined only in Supplementary R1.
  - Marker and colour meaning changes between figures:
    - Figs. 5 and 9: A3 is an orange square; S8 is a purple diamond; C is a blue circle.
    - Fig. 10: S8 is an orange square; C is a purple diamond; B is a blue circle.
- **Suggested fix:**
  - Keep five descriptive names in the main text, for example:

    | Proposed name | Current label |
    |---|---|
    | **NICE** | A3 |
    | **NICE-post** (correction added after training) | B+W |
    | **Smoothing-trained** | A2b |
    | **Uncorrected** | C |
    | **Base network** | B, used only for the fixed-weight study |

  - Drop P0 and D from the main text, and drop S8:
    - replace S8 in Table 4 by B or A2b, or remove Table 4 (see the plan);
    - in Fig. 10 use B, C and A2b, plus NICE (see R5-09).
  - Put a small cell key (label, stratum, retained volume, where used) as a panel or column next to Fig. 1 or Table 2.
  - Use one colour map across all figures.
  - Use "NICE" in every Results sentence where "A3" appears now.
- **Fix type:** text only + re-plot (re-analysis of existing data).

### R5-09 — Minor — The participation figure omits the proposed method; overlaps Fig. 9
- **Location:**
  - Fig. 10 caption: "192 observations from the 25 B, C and S8 model–configuration combinations"
  - Fig. 9(c,d), which shows load-level compliance against sensitivity
  - Table 4
- **Issue:**
  - The key mechanism figure contains no NICE or NICE-post points, so the reader cannot see how the correction moves the cloud in Fig. 10(b,d).
  - Fig. 9(c,d), Fig. 10(b) and Table 4 show the same load-level relation three times.
- **Suggested fix:**
  - Add NICE and NICE-post to Fig. 10; the data are in ST13.
  - Reduce Fig. 9 to panels (a,b).
  - Fold the three Table 4 rows into two sentences of §6.6, and move the full table to ST15/ST13.
- **Fix type:** re-analysis (re-plot) + cut or move.

### R5-10 — Minor — Self-containedness: key arguments rely on appendix equations
- **Location:**
  - §5.1, L283: "Eq. (B.8)"
  - §6.7, L464: "Eq. (J.4) relates local relative sensitivity error to participation…"
  - §6.7, L466 and §7.2, L544: "Eq. (H.2)"
  - §4.3, L265 and §7.2, L546: "Eq. (J.5)", "Eq. (H.5)"
  - More than 20 supplementary tables (ST01–ST20) are cited from the main text.
- **Issue:**
  - A reader cannot follow §6.7 and §7.2 without the appendices.
  - Conversely, Appendices C, I, J.8 and J.9 are never cited from the main text.
- **Suggested fix:**
  - State the one-line results inline (for example the J.4 bound as a sentence, and the J.5 condition "uniformly C¹-small H").
  - Remove the reliance on B.8.
  - Move uncited appendices to the supplement (see the plan).
  - Cap main-text citations of ST tables to one pointer per subsection.
- **Fix type:** text only + cut or move.

### R5-11 — Minor — Numerical inconsistencies to reconcile
- **Location:**
  - Table 6, learned compliance errors: 2×2×2 "0.014, 0.011, 0.0097"; 3×3×1 "0.015, 0.014, 0.015"
  - §6.9, L486: "0.014%, 0.011% and 0.0094%" and "0.015%, 0.013% and 0.010%"
- **Issue:** The same lattices and loads give different learned compliance errors. This is presumably the 10⁻⁶ conjugate-gradient tolerance in Table 6 against a tighter solve in §6.9, but the text does not say so. The Table 6 caption claims comparison against "the exact condensation of Section 6.9".
- **Suggested fix:**
  - Explain the difference in the Table 6 caption, or harmonise the numbers.
  - Two further checks:
    - The value 5.83% appears both as S8 on M1/x T-y (§6.6, Table 4) and as B on U1/x N-z (§6.7, L464). The Fig. 10(b) annotation suggests both are correct, but please confirm.
    - The §8 statement "to within 0.4%" applies to consistent loads only; random loads are 4–5%. Say so.
- **Fix type:** text only (or re-analysis if the numbers differ).

### R5-12 — Minor — Residual stagnation in lattice solves against the claimed operator consistency
- **Location:**
  - §6.9, L486: "its recomputed residual stagnates at 3.6×10⁻⁴ … 3.2×10⁻³"
  - §6.2, L373: "symmetric to a relative 3×10⁻⁸ … agrees … to 2×10⁻⁸"
- **Issue:**
  - A recomputed residual of 3×10⁻³ is large relative to the claimed operator consistency of 10⁻⁸. It will also matter for optimisation (§6.11) and for Eq. (18).
  - The attribution to single precision is asserted but not quantified.
- **Suggested fix:**
  - Explain the mechanism, for example fp32 action differences between the forward and the transpose, or non-deterministic reductions.
  - Report the action–energy consistency term of Appendix J.6 for these solves.
  - State whether fp64 accumulation removes the stagnation.
- **Fix type:** text only, or re-analysis.

### R5-13 — Minor — The abstract is too long and too dense
- **Location:** Abstract. It is about 370 words before the placeholder sentence, with about 20 numerical values.
- **Issue:** Most CMAME abstracts are about 150–250 words.
  - The method description (sentences 2–4) packs Chebyshev, Q₁, Galerkin, matrix-free, SPSD and rigid-motion details into three sentences.
  - The error-analysis sentences repeat the introduction.
- **Suggested fix:** Aim for about 220 words, in this order:
  - problem (1 sentence);
  - idea and guarantees (2);
  - analysis insight (1);
  - evidence, with four numbers only: population mean or maximum, assembly maximum, lattice maximum, and cost against conventional condensation on stated hardware (2);
  - design example (1).

  Also prepare the Highlights required by the journal.
- **Fix type:** text only.

### R5-14 — Minor — Section order: training is described before the correction it is trained through
- **Location:** §3.3, L188: "The correction of Section 5 can also be placed inside the training loop."
- **Issue:** The reader meets "trained through the correction" before the correction is defined, with a full analysis section (§4) in between.
- **Suggested fix:** Either:
  - move §3.3 to a new §5.4, "Training through the correction"; or
  - reorder to §2 Setting → §3 NICE operator (extension, correction, variational stiffness) → §4 Training → §5 Error analysis.

  The first option is the smaller change.
- **Fix type:** text only (reorder).

### R5-15 — Minor — The Discussion duplicates the Introduction and Results
- **Location:** §7.1 ¶3 (L536) repeats the Introduction ¶4 and §6.8.
  - §7.2 ¶1–3 (L540–544) repeat §4.2–4.3 and §6.7.
  - §7.3 ¶2 (L552) repeats §6.5 and Table 3 ("7 to 290 times", "13.5% becomes 0.19%").
  - §7.4 repeats §6.10.
  - The final ¶ of §7.4 (L562) repeats §5.3 and Eq. (18).
- **Issue:** Of the Discussion's 1,738 words, perhaps 700 are new.
- **Suggested fix:**
  - Condense §7 to about 850 words in three subsections:
    - (a) boundary restriction against interior approximation (the positioning against PIML);
    - (b) what learning and correction each contribute, including the practical message that the correction can be added to an existing network (NICE-post);
    - (c) limitations and extensions (R5-07).
  - Put the cost interpretation into the last paragraph of §6.10.
- **Fix type:** cut or move.

### R5-16 — Minor — The "3% accuracy reference" and the discretisation error of the reference
- **Location:**
  - §6.1, L361: "We use 3% as a common accuracy reference"
  - §6.2, L371: "the discretisation error of the reference at n = 32 is thus of order one percent, an order of magnitude above the energy errors of the corrected predictor"
- **Issue:**
  - The 3% threshold is not motivated.
  - The observation that NICE errors lie below the discretisation error of the model is an important argument, and should be made explicitly: the surrogate is not the limiting error.
- **Suggested fix:** Give one sentence justifying 3%, for example typical gradient-accuracy needs of OC/MMA, or state that it is merely a visual reference. Also add one sentence in §6.6 or §7 relating NICE errors to the discretisation error of about 1%.
- **Fix type:** text only.

### R5-17 — Minor — Positioning against hybrid neural solvers and energy-element models
- **Location:** Introduction ¶3 and ¶5 (L15, L19); citations to HINTS, DeepONet preconditioners, GMT, Jiang et al. (2026, arXiv), Guo et al. (2026b, arXiv).
- **Issue:**
  - A reader may ask what distinguishes NICE from "learned initial guess plus multigrid". The answer is:
    - a fixed-budget correction turned into an operator with variational guarantees;
    - the complete transpose;
    - training through the correction;
    - sensitivity analysis.

    This answer is spread over several paragraphs.
  - Two key comparators are recent preprints; their status should be checked at submission.
- **Suggested fix:**
  - Add one explicit "difference" sentence after the hybrid-solver citations.
  - Update the preprint references if they have since been published.
- **Fix type:** text only.

### R5-18 — Minor — The value of training through the correction is modest; frame it accordingly
- **Location:**
  - §6.3, L381: "training through it reduces the remaining mean by a further factor of 1.3"
  - §6.6, L444: "B+W … also meets both requirements in the same fourteen configurations"
  - Contribution (ii)
  - Abstract: "Trained through the correction … the predictor attains…"
- **Issue:**
  - NICE-post (B+W) already meets every assembly criterion.
  - Training through the correction gains a factor of 1.3 on the population mean and 1.35–2.2 on the worst sensitivities, at about 2.7 GPU-hours plus implementation complexity.
  - The abstract attributes all headline numbers to the trained-through variant, which slightly overstates its role.
- **Suggested fix:**
  - State in the abstract that the correction alone brings an existing network within tolerance.
  - Present training through the correction as a refinement for the hardest cells, as §7.3 already does well.
- **Fix type:** text only.

### R5-19 — Minor — Legacy predictor D and the historical deployment study
- **Location:**
  - §6.1, L341: "The historical computational examples use a separate uncorrected predictor D."
  - §6.10, L522
  - Supplementary Notes S1 and S4, Figs. S04–S05, Tables ST08–ST10 and ST14
- **Issue:** The study uses a different, uncorrected predictor run with TF32 arithmetic. It no longer supports any main-text claim, and it adds a label and roughly 1,300 supplementary words.
- **Suggested fix:**
  - Remove both main-text mentions.
  - Either move S1, S4 and the associated tables and figures to the data or code repository, or keep them in the supplement under a clearly separated heading such as "Archive: earlier uncorrected predictor".
- **Fix type:** cut or move.

### R5-20 — Minor — Data and code availability; wide tables
- **Location:**
  - The supplementary material refers to `evidence/*.json` and `figures_src/*.py`.
  - Tables 5 and 6 have 8 and 7 columns with long headers.
- **Issue:**
  - There is no Data availability statement.
  - Tables 5 and 6 will not fit a single-column elsarticle page without landscape layout or very small type.
- **Suggested fix:**
  - Add a data and code availability statement and an archive DOI.
  - Simplify Table 5 by splitting the application columns into ratios, or stacking the units.
  - Table 6 moves to the supplement (see the plan).
- **Fix type:** text only.

---

## 3. Length-reduction plan

**Assumptions:**
- Word counts are for the main text only (Sections 1–8, without references, tables and display equations), matching the measured 14,650.
- Caption savings are listed separately, because captions may not be in the measured count.
- "Supp." means SUPPLEMENTARY_EN.md; "App." means APPENDICES_EN.md.
- Moving is preferred to deleting throughout; nothing listed below loses evidence.

### 3.1 Main text

| # | Item (location) | Action | Destination | Est. words saved |
|---|---|---|---|---:|
| **Introduction (1,465 → ~915)** | | | | |
| 1 | ¶3 L15 "Multilevel iteration supplies…": the operator formulas F=𝒲Ê, Ŝ=FᵀKF, lower bound, complete transpose | Delete the formulas and keep one sentence of idea plus the hybrid-solver citations; the content is in §3.2 and §5.2 | — (already in §3.2 and §5.2) | 110 |
| 2 | ¶4 L17 PIML paragraph | Condense to about 150 words: PIML line of work → reduced-boundary error → one-line pointer to §6.8 | — | 110 |
| 3 | ¶6 L21 "We construct the initial extension…" | Merge into ¶3 and the contributions; it repeats §3.1 and contribution (ii) | §3.1 | 120 |
| 4 | ¶7 L23 "We analyse this construction…" | Condense to 2–3 sentences; it repeats §4 and contributions (iii) and (iv) | — | 80 |
| 5 | Contributions L25 | Reduce to 3 items (R5-03) | — | 60 |
| 6 | ¶1, ¶2, ¶5 (L11, L13, L19) | Light trimming | — | 70 |
| **Section 2 (732 → ~640)** | | | | |
| 7 | Paragraph after Table 1 (L105) | Reduce to one sentence | Table 1 footnote | 55 |
| 8 | §2.3 last ¶ (L85), aggregation sentences | Keep the metric definitions and move the aggregation rules | App. A.2 (already there) | 40 |
| **Section 3 (1,776 → ~1,325)** | | | | |
| 9 | §3.1 ¶1 L117, PIML contrast | Delete (repeats the Introduction) | — | 50 |
| 10 | §3.1 architecture ¶¶ L121, L123, L133, L135 | Condense from about 550 to about 300 words; Fig. 3 and App. G carry the details | App./Supp. G | 250 |
| 11 | §3.1 last ¶ L145, last two sentences | Delete | — | 30 |
| 12 | §3.3 ¶3 L190: difficult-direction search, 48-symmetry augmentation | One sentence plus pointer | Supp. G.3 | 90 |
| 13 | §3.3 | Relocate as §5.4 (R5-14); light trim | — | 30 |
| **Section 4 (993 → ~853)** | | | | |
| 14 | §4.3 ¶3 L252, Δq decomposition | Move together with the replacement study | App. H | 60 |
| 15 | §4.3 ¶5 L267, shared design variables | Relocate to §6.11, where it is used | §6.11 | 0 (relocated) |
| 16 | §4.1 last ¶, §4.2 last ¶, §4.3 ¶2: cross-reference sentences | Trim | — | 80 |
| **Section 5 (820 → ~700)** | | | | |
| 17 | §5.1 ¶2 L285 | Trim | — | 40 |
| 18 | §5.1 ¶3 L287: interval verification numbers "2.1–5.0%" (duplicated in §6.2) | Keep in one place | §6.2 / App. D | 40 |
| 19 | §5.3 L324–331 (Eq. 18 text) | Keep the equation; trim the explanation | App. J.6 | 40 |
| **Section 6 (6,643 → ~3,823 before §6.11)** | | | | |
| 20 | §6 intro L335 | Reduce to 2 sentences | — | 45 |
| 21 | §6.1 ¶1 L339: parameter ranges, cut-severity definitions | Keep strata; move ranges | Supp. ST11 | 40 |
| 22 | §6.1 ¶2 L341 and Table 2: predictor descriptions | Rewrite with the 5 descriptive names; drop S8, P0 and D; Table 2 goes from 7 rows to 5 | Supp. R1 | 100 |
| 23 | §6.1 ¶4 L359, aggregation rules | Move | App. A.2 | 90 |
| 24 | §6.1 ¶5 L361, assembly set-up | Trim; the Fig. 4 caption carries the geometry | — | 50 |
| 25 | §6.2 ¶¶ L369–373, verification | Keep a summary of about 180 words (reference error ≤0.06% for U and M and about 1% for H at n = 32; penalty and integration insensitivity; symmetry, energy–action consistency and rigid energy near 10⁻⁸); move the details | Supp. Note S5 + Fig. S06 + ST19 | 400 |
| 26 | §6.3 ¶1 L377, nodal-force stress-test rationale | Reduce to one sentence | Supp. R2 | 60 |
| 27 | §6.3 ¶3 L381, intermediate predictors | Condense | — | 80 |
| 28 | §6.3 ¶4 L383, orientation dependence (B only) | Move | Supp. ST01c / R2 | 70 |
| 29 | §6.4 ¶¶1–2 L391–393 and **Figure 6**, spectral distribution | Keep a summary of about 80 words (the M1 low modes below *a* motivate the coarse space); move Fig. 6 and the U1 nodal-force reversal | Supp. (new Fig. S07) + ST02b | 200 (+ 90 caption) |
| 30 | §6.4 L405 δ²κ | Keep; trim | — | 30 |
| 31 | §6.4 L407, linear against quadratic sensitivity term | Reduce to 2 sentences | Supp. ST02 / Fig. S03 | 70 |
| 32 | §6.5 ¶¶ L411–417: smoothing only, coarse, budget | Merge into one paragraph; the numbers stay in Fig. 8 | Supp. ST03, ST04 | 180 |
| 33 | §6.5 Table 3: the five 32/Q₁/32 rows | Keep the 8/Q₁/8 rows; state the 32-step result in one sentence | Supp. ST18 | 0 (table only) |
| 34 | §6.5 ¶ after Table 3 (L436), B+W against A3 locally | Move the B+W argument to §6.6; trim | — | 60 |
| 35 | §6.6 ¶1 L444: per-predictor listing, L1 host-memory detail | Condense; Fig. 9 and ST13 carry the numbers | Supp. ST13 | 230 |
| 36 | §6.6 **Table 4** | Remove; quote the 2–3 key numbers in the text | Supp. ST13 / ST15 | 40 (+ table) |
| 37 | §6.7, merged into §6.6: replacement study L466 | Keep the participation example (about 100 words); move the replacement decomposition | Supp. ST05 + App. H | 190 |
| 38 | §6.8, Bernstein restriction | Trim to about 270 words; keep Fig. 11, which is the positioning evidence | Supp. ST07 | 100 |
| 39 | §6.9 ¶1 set-up and ¶2 (3×3×1 repetition) | Condense; report both lattices in parallel form | — | 120 |
| 40 | §6.9 ¶3, bound comparison | Condense to 2 sentences | — | 80 |
| 41 | §6.10 ¶¶ L492, L494, L505, L507 | Trim and merge; remove the front-end ratio from the method claims (R5-05) | — | 140 |
| 42 | §6.10 L509–520 and **Table 6**, whole-lattice direct solution | Keep an 80-word summary with the key numbers (4-cell: 38 s against 242–333 s; 8-cell factor memory above 90 GB); move the table and details | Supp. Note S7 / ST20 | 420 (+ table) |
| 43 | §6.10 L522, legacy study | Delete the sentence (R5-19) | Supp. S4 or repository | 25 |
| **Section 7 (1,738 → ~890 including a new Limitations paragraph)** | | | | |
| 44 | §7.1 (427 → ~150): ¶3 repeats the Introduction and §6.8 | Condense into the "Boundary restriction against interior approximation" subsection | — | 277 |
| 45 | §7.2 (494 → ~200): ¶1–3 repeat §4.2–4.3 and §6.7 | Keep ¶4 (complete derivative), updated for §6.11 | — | 294 |
| 46 | §7.3 (460 → ~250) | Remove the numbers restated from §6.5; keep the complementarity argument and the NICE-post message | — | 210 |
| 47 | §7.4 (349 → ~100) | Merge reuse and amortisation into the end of §6.10; delete the Eq. (18) ¶ (it repeats §5.3) | §6.10 | 249 |
| 48 | New "Limitations and extensions" (R5-07) | Add | — | −180 |
| **Section 8 (482 → ~382)** | | | | |
| 49 | §8 ¶3 L570, which repeats abstract numbers | Condense; keep 3–4 numbers; leave room for the §6.11 sentence | — | 100 |
| | **Total main-text saving (net, including +180 for Limitations)** | | | **≈ 5,125** |

**Resulting size:**

| Stage | Main text (words) |
|---|---:|
| Current | 14,650 |
| After the cuts | ≈ 9,525 |
| After adding §6.11 (≈750) and its §8 sentence (≈50) | ≈ 10,325 |

This is inside the 9,500–11,000 target and leaves the requested margin.

**Additional caption savings** (not in the total above):

| Caption | Change | Words saved |
|---|---|---:|
| Fig. 3 | ≈190 → ≈90 | ≈100 |
| Fig. 8 | ≈150 → ≈80 | ≈70 |
| Fig. 9 | — | ≈60 |
| Table 5 | — | ≈60 |
| Fig. 6 | moved | ≈90 |
| **Total** | | **≈380** |

**Abstract:** ≈370 → ≈220 words (≈150 saved; not in the main-text count).

**Figures and tables after the plan:**

| | Now | After the plan | With §6.11 |
|---|---:|---:|---:|
| Figures | 11 | 10 (Fig. 6 moved; Fig. 9 reduced to panels a,b) | 11 |
| Tables | 6 | 4 (Tables 4 and 6 moved) | 5 |

Main-text figure and table count therefore stays at or below the current level.

### 3.2 Appendices → supplementary material (≈9,090 → ≈4,790 words)

| # | Item | Action | Destination | Est. words moved or saved |
|---|---|---|---|---:|
| A1 | App. A.2, metrics and aggregation | Keep only what §2.3 and §6.1 no longer contain; this absorbs items 8 and 23; drop the duplicated definitions | App. A | 130 (dedup) |
| A2 | App. F.1–F.4: coarse-space study, differentiable implementation, precision, moment integration | Move | Supp. (implementation note) | 809 |
| A3 | App. G.1–G.3: network coefficients, direction banks, optimisation | Move; the main-text §3.1 plus Fig. 3 suffice for the reader, and the supplement keeps reproducibility | Supp. | 1,587 |
| A4 | App. I.1–I.2: residual lower diagnostics (not cited from the main text) | Move | Supp. | 145 |
| A5 | App. I.3: Bernstein-restricted Galerkin system | Keep as 2 sentences in §6.8 or App. C | App. C | 80 (net) |
| A6 | App. J.1, assumptions | Merge into the head of App. B | App. B | 150 (dedup) |
| A7 | App. J.2, which restates App. B | Delete the duplicate; keep the half-energy remark in B | App. B | 150 (dedup) |
| A8 | App. J.3, compliance as error energy | Merge with App. C | App. C | 100 (dedup) |
| A9 | App. J.4–J.5, sensitivity bounds and the complete derivative | Merge with App. H into one sensitivity appendix; keep | App. H | 0 |
| A10 | App. J.6, finite-iteration residual | Keep, beside C | App. C | 0 |
| A11 | App. J.7, training coverage and rotations | Move | Supp. | 344 |
| A12 | App. J.8, corrections improve quantities differently (not cited) | Move | Supp. | 355 |
| A13 | App. J.9, illustrative matrix examples (not cited; definitions already in Supp. S2) | Move, next to S2 | Supp. S2 | 450 |
| | **Appendix reduction** | | | **≈ 4,300** |

**Remaining appendices (≈4.8k words):**
- A: discrete construction (condensed)
- B: variational identity and rigid kernel (with J.1 and J.2 merged in)
- C: assembly, compliance, finite-iteration residual and restricted trace (with J.3, J.6 and I.3 merged in)
- D: Chebyshev and coarse projection
- E: complete transpose
- H: sensitivity identities, bounds and the complete derivative (with J.4 and J.5 merged in)

These are exactly the appendices that support the main-text theorems.

**Estimated page count:** from 86 to about 55–62 pages in elsarticle 12 pt with appendices. Main text plus appendices shrink from about 23.7k words to about 15.1k.

### 3.3 Supplementary material

- **S1 and S4 with Figs. S04–S05 and Tables ST08–ST10, ST14** (legacy predictor D, about 1,300 words plus five figures and tables): move to the data or code repository, or place under a separate "Archive" heading (R5-19).
- **Order.** Reorder the supplement to follow the new main-text order:
  1. Reference verification (S5, S06)
  2. Population (ST01, S01)
  3. Diagnostics (ST02, Fig. S07 [ex-Fig. 6], S03)
  4. Correction (ST03, ST04, ST18, S02A/B)
  5. Assembly (ST05, ST06, ST13, ST15, ST16, former Table 4)
  6. Restriction (ST07)
  7. Cost (ST20 [former Table 6], S7)
  8. Implementation (former App. F and G)
  9. Further analysis (former App. I, J.7–J.9, S2)
- **Cross-references.** Put a one-line cross-reference table at the head of the supplement mapping each item to the main-text claim it supports.

---

## 4. Title recommendation

The current title, "Learned static condensation for cut thin-walled TPMS cells with equilibrium correction", has four problems:

- **It does not carry the method name.** The paper now calls the method NICE (neural-initialised condensation with equilibrium correction). "Learned static condensation" undersells the key distinction, that the network only *initialises* and a variational correction *completes*.
- **It says "TPMS cells" in general**, while only P-type cells are studied.
- **It omits thickness sensitivity and design,** which the error analysis, the training objective and the forthcoming §6.11 are built around.
- **"With equilibrium correction" dangles** at the end and reads as if it modified "cells".

**Recommended title** (once §6.11 is complete):

> **Neural-initialised static condensation with equilibrium correction for the analysis and thickness design of cut thin-walled TPMS lattices**

**Alternatives:**

1. *Neural-initialised condensation with equilibrium correction (NICE): accurate compliance and thickness sensitivity for cut thin-walled TPMS lattices.* Use this if the editor or authors prefer to foreground the sensitivity message.
2. *Neural-initialised static condensation with variational equilibrium correction for cut thin-walled lattice cells.* Use this if §6.11 turns out small and "design" would overclaim.

**Guidance:**

- Put the acronym NICE in the abstract and the keywords, not at the start of the title.
- If the authors keep "TPMS", state "Schwarz-P-type" in the first sentence of the abstract (R5-07).
- Add "neural-initialised condensation" and "thickness optimisation" to the keywords.
