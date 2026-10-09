# U05 notes: Sections 3.3 Chebyshev smoothing, 3.4 Coarse-grid correction and the two-grid cycle, 3.5 Training, 3.6 Applying the condensed stiffness

Source (old numbering): MANUSCRIPT_EN.md / MANUSCRIPT_CN.md lines 289-358 (old 4.3-4.6, Eqs. (15)-(18), Proposition 4, Remark 1, Algorithm 1).

New blocks (EN and CN line-aligned, 69 lines each): heading `### 3.3.`; P1 motivation (Eqs. (7), (14)); P2 generalised eigenvectors; Eq. (10); P3 fraction \(\chi_\ell\); P4 network prediction as initial approximation, Chebyshev semi-iteration, related methods; P5 error polynomial, interval, \(a=b/30\), 5%; heading `### 3.4.`; P6 coarse basis and residual; Eq. (11); P7 meaning of Eq. (11), \(Q_1(17)\); P8 two-grid cycle and \(\mathcal W\); Eq. (12); Proposition 2; P9 consequences; Remark 1; heading `### 3.5.`; P10 test displacements and normalisation; Eq. (13); P11 the two loss terms, gradients, search, augmentation, Tables 2 and ST01; P12 training through \(\mathcal W\); heading `### 3.6.`; P13 preparation, cost of one product, storage; Algorithm 1 title; three algorithm steps.

Length (EN words, display equations excluded, inline mathematics counted as one word): old source 1043, new 1451. The increase comes from splitting long sentences, from plain glosses at first use (what \(D\), \(C_V\), \(H_{\rm net}\) and the notation 8 / \(Q_1(17)\) / 8 denote; what Eqs. (7) and (14) express) and from writing out abbreviations such as "coarse factorisation" as "the factorisation of the coarse-grid matrix".

Self-check (Python, on the final files):
- (a) Every number of the old EN source is in the new EN text, except the old cross-reference and tag numbers, which are renumbered (list (b) below): old Eqs. (4), (5), (6), (9), (15), (16), (17), (18); old Sections 3.1, 4.1, 4.3-4.6; old Proposition 4. No measured number was removed or moved. All numbers kept: 5%, \(b/30\), \(17^3\), \(Q_1(17)\), 8 / \(Q_1(17)\) / 8, \(w_s=1\), \(q_j^TSq_j=1\), 48, 23,604, 4.2 GiB, \(\|\Phi_k\|_A\le1\), \(2t-t^2\), \(-1\), \(0<a<b\), \((0,b]\), \(a+b\), Tables 1, 2, ST01, ST13, Section 5.2, Section 5.5, Appendices C, D, D.1, E-F, F.1, G, Eqs. (3), (D.6).
- (b) The new EN text contains no number absent from the old source except renumbered references and tags (Sections 3.3-3.6, 3.1, 4.2; Eqs. (6), (7), (10)-(14), (17); Propositions 1 and 2). "Section 4.2" in 3.5 and "Proposition 1" in 3.5 are added cross-references (see (a) below). "4.2" also occurs as the old number 4.2 GiB.
- (c) EN and CN contain the same numbers on every line, the same inline mathematics, the same citations and identical display equations. The four display equations are byte-identical to the old ones apart from their tags (10)-(13).
- (d) No EN sentence above 35 words (longest 35 including the label "Proposition 2 (correction ordering)", mean 16.7). No CN sentence above 80 characters (longest 66, mean 28.4).
- (e) No banned word or old term in EN or CN. The only grep hits are the ordinal "first" in "the first \(\ell\) modes" (twice) and "the first term", which are not priority claims.

## (a) Sentences and numbers removed or moved

No sentence or number moved to another unit, the appendices or the supplement. Four labels were deleted under brief Section 3.1; everything else was rewritten in place.

### Old 4.3 "Error spectrum and Chebyshev smoothing" (old lines 289-305) → new 3.3 "Chebyshev smoothing"

| Old sentence | Where it is now |
|---|---|
| Heading "4.3. Error spectrum and Chebyshev smoothing" | heading "3.3. Chebyshev smoothing" (brief Section 4). The error-spectrum content (Eq. (10), \(\chi_\ell\)) stays in P2-P3. |
| "Equations (5) and (6) suggest reducing the interior imbalance of the learned field at fixed retained motion;" | P1 S1-S2. S1 is a plain gloss of what the two equations express, both stated in the source (old 3.1: \(\varepsilon=r_I^TA^{-1}r_I/q^TSq\); old 3.2: "Compliance underestimation is the total reconstructed error energy"). Old (5) → (7), old (6) → (14). |
| "polynomial smoothing [Adams et al. (2003)] and coarse energy minimisation address different components of it." | P1 S3 |
| "Set \(D=\operatorname{diag}(A)\) and order the generalised eigenvectors by increasing eigenvalue, \(Av_j=\lambda_jDv_j\), with \(v_j^TDv_k=\delta_{jk}\). For \(c_j=v_j^TDd_I\)," | P2 S1-S3. S1 adds "the diagonal of the interior stiffness matrix \(A\)" as a gloss; S3 names \(d_I\) as the interior error of Section 3.1 (defined there before Eq. (7)). |
| Eq. (15) | Eq. (10), unchanged |
| "The fraction \(\chi_\ell\) locates the error energy in the first \(\ell\) Jacobi-scaled interior modes (the exact-field fraction uses \(u_I^TAu_I\))." | P3 S1-S2; the parenthesis became its own sentence. |
| "The learned field is an initial approximation to \(Au_I=-K_{IP}q\), improved by Jacobi-preconditioned Chebyshev semi-iteration [Golub & Varga (1961)] with \(q\) fixed;" | P4 S1-S2 ("network prediction") |
| "a prescribed iteration count and geometry-dependent coefficients keep the corrected extension linear." | P4 S3 ("the coefficients depend on the geometry but not on \(q\), so the corrected recovery operator remains linear") |
| "Pairing a learned start with smoothing in this way mirrors the smoothing of a tentative prolongation in smoothed aggregation [Vaněk et al. (1996)] and hybrid solvers in which relaxation removes the high-frequency error that a learned prediction leaves [Zhang, E. et al. (2024)]." | P4 S4-S5, split at the two cited analogies. CN: 初始插值算子 for "tentative prolongation" (old CN 试探插值算子; 试探 avoided, brief 3.1). |
| "A degree-\(k\) error polynomial maps \(d_I\) to \(\Phi_kd_I\), \(\Phi_k=p_k(D^{-1}A)\), multiplying each coefficient in Eq. (15) by \(p_k(\lambda_j)\)." | P5 S1-S2 ("After \(k\) smoothing steps ... the error polynomial of degree \(k\)"; Appendix D: \(k\) steps give \(d_k=p_k(D^{-1}A)d_0\)). |
| "The Chebyshev polynomial targets \([a,b]\), \(0<a<b\), and is nonexpansive in the \(A\)-energy norm if the actual positive spectrum of \(D^{-1/2}AD^{-1/2}\) lies in \((0,b]\)." | P5 S3-S4; "nonexpansive" → "does not increase the error in the \(A\)-energy norm" (brief 3.1). |
| "Modes below \(a\) can decay slowly, which motivates a complementary coarse-grid correction." | P5 S5, with the pointer "of Section 3.4" (old 4.4). |
| "The correction sequences use \(a=b/30\), with \(b\) from a power estimate increased by 5%;" | P5 S6; "power estimate" written as "a power-iteration estimate of the largest eigenvalue of \(D^{-1}A\)", as in Table 1. The iteration count of Table 1 (40) is not repeated here. |
| "Section 5.2 verifies that \(b\) bounds the spectrum, and Appendix D gives the polynomial, recurrence, interval estimator and contraction bound." | P5 S7-S8 |

### Old 4.4 "Coarse-grid correction" (old lines 307-332) → new 3.4 "Coarse-grid correction and the two-grid cycle"

| Old sentence | Where it is now |
|---|---|
| "Let \(V\in\mathbb R^{i\times n_c}\) have full column rank (Appendix F.1 treats a basis without it), \(A_c=V^TAV\), and \(b_r=V^Tr_I\). Minimising the energy over \(\widehat u_I+\operatorname{range}V\), with \(\widehat u_I\) the current approximate interior field (the learned field or a smoothed iterate) and \(r_I=(K\widehat u)_I\) its interior residual, gives" | P6 S1-S3, reordered: \(\widehat u_I\) and \(r_I\) first, then \(V\), then \(A_c\) (named "coarse-grid matrix", brief 3.1) and \(b_r\). |
| Eq. (16) | Eq. (11), unchanged |
| "The coarse-grid (Galerkin) correction enforces equilibrium against the coarse variations," | P7 S1 ("enforces interior equilibrium for every virtual displacement in the coarse space"). The label "(Galerkin)" is deleted from the main text (brief 3.1: appendices only). |
| "and the last equality measures the energy removed by that projection [Xu (1992)]." | P7 S2 |
| "The principal coarse basis, denoted \(Q_1(17)\), consists of trilinear vector functions on a \(17^3\)-vertex grid restricted to interior DOFs (Appendix F.1);" | P7 S3 |
| "all coarse spaces act on interior DOFs only, so the retained values stay fixed." | P7 S4 ("the retained displacements stay fixed") |
| "Placing the coarse-grid correction between two \(k\)-step smoothing stages combines these corrections into one two-grid cycle, the correction \(\mathcal W\) of Section 4.1." | P8 S1-S2 (Section 3.1) |
| "With \(C_V=I_i-VA_c^{-1}V^TA\) and \(H_{\rm net}=J_I(\widehat E-E)\), the two-grid error operator \(\Phi_kC_V\Phi_k\) [Hackbusch (1985)], [Trottenberg et al. (2001)], [Xu & Zikatanov (2002)], [Falgout et al. (2005)] gives the interior error of \(F\) and, by Eq. (4), the operator error" | P8 S3-S5. S3 names \(C_V\) as the error operator of the coarse-grid correction (it is the matrix in Eq. (11)) and \(H_{\rm net}\) as the interior error of the network recovery operator. "operator error" → "the error of the condensed stiffness". Old Eq. (4) → Eq. (6). |
| Eq. (17) | Eq. (12), unchanged |
| Proposition 4 (correction ordering), S1: "Let the network extension \(\widehat E\) satisfy (A2), let the coarse solve be exact or use the nonnegatively shifted inverse of Appendix F.1 (Eq. (D.6)), and let \(\|\Phi_k\|_A\le1\)." | Proposition 2, S1 ("network recovery operator", "the coarse inverse with the nonnegative shift of Appendix F.1 (Eq. (D.6))") |
| "Then, under (A1), \(S\preceq\widehat S\preceq\widehat S_{\rm net}\), where \(\widehat S_{\rm net}=\widehat E^TK\widehat E\) is the uncorrected condensed stiffness (Appendix D.1)." | Proposition 2, S2 ("the condensed stiffness without correction") |
| "The same ordering holds for any geometry-fixed linear correction whose interior error map \(\Theta\) satisfies \(\Theta^TA\Theta\preceq A\)." | Proposition 2, S3 ("any linear correction that is fixed for a given geometry") |
| "The correction therefore cannot increase the error in the \(A\)-energy norm," | P9 S1 |
| "and by the order reversal of Appendix C it moves the assembled compliance towards the reference (Appendix D.1)." | P9 S2-S3. S2 states the order reversal in words, as in Appendix C ("matrix inversion reverses their order"; \(\widehat{\mathbb K}-\mathbb K\succeq0\) from the cell matrices by assembly, Eq. (C.1)). |
| "The ordering is guaranteed; the size of the reduction is not (Appendix D), and Section 5.5 measures it." | P9 S4-S5 |
| Remark 1, S1: "The guarantee rests on both conditions and covers the energy only." | Remark 1, S1, naming the two conditions ("on the smoothing and on the coarse solve"), which are the two that the remark then discusses (spectrum above \(a+b\); approximate coarse inverse \(G\)). |
| "If the spectrum of \(D^{-1}A\) extends above \(a+b\), the Chebyshev polynomial amplifies those modes and the cycle can increase the energy error (Appendix D)." | Remark 1, S2 |
| "Changing the smoothing degree changes the polynomial, so the ordering holds for each \(k\) separately and the reduction need not be monotone in \(k\)." | Remark 1, S3-S4 ("the number \(k\) of smoothing steps") |
| "For an approximate coarse inverse \(G\), the correction stays nonexpansive when \(2G-GA_cG\succeq0\) (Eq. (D.6)), which the nonnegative shift satisfies and an arbitrary approximation need not." | Remark 1, S5-S6 ("does not increase the energy error", brief 3.1) |
| "And the ordering does not reach the sensitivity: in the example of Appendix D.1, a projection that halves the energy of the interior error changes the sensitivity discrepancy of Eq. (9) from zero to \(2t-t^2\), against a reference sensitivity of \(-1\)." | Remark 1, S7-S9 ("does not apply to the sensitivity"; "a projection halves the energy of the interior error. The same projection changes the sensitivity difference \(\widetilde s_c-s_c\) of Eq. (17) ..."). Old Eq. (9) → Eq. (17). |

### Old 4.5 "Training directions and objective" (old lines 334-347) → new 3.5 "Training"

| Old sentence | Where it is now |
|---|---|
| Heading "4.5. Training directions and objective" | heading "3.5. Training" (brief Section 4) |
| "Training probes the extension through retained displacement directions:" | P10 S1 ("The network is trained on sets of test displacements"; "direction" → "test displacement", brief 3.1; "test displacement" is defined in Section 3.1, U03) |
| "prescribed polynomial and multiscale displacements supply broad spatial content," | P10 S2 |
| "and responses to equilibrated nodal loads, consistent tractions, spring supports and neighbouring cells supply mechanically generated directions." | P10 S3 ("traction loads"; "displacements imposed by a neighbouring cell", brief 3.1; class definitions in Appendix G.2) |
| "Rigid components are removed and each direction is normalised to \(q_j^TSq_j=1\) with the reference solution." | P10 S4 ("normalised with the exact condensed stiffness") |
| "For a batch of \(B\) directions, the objective is" | P10 S5 ("the loss function is") |
| Eq. (18) | Eq. (13), unchanged |
| "The energy term is the mean logarithm of the predicted-to-reference energy ratio." | P11 S1, written with the ratio \(q_j^T\widehat Sq_j/(q_j^TSq_j)\) and the reason (\(q_j^TSq_j=1\)) that the first term of Eq. (13) is this ratio. |
| "For an admissible extension, the variational identity of Section 3.1 makes its minimum correspond to the equilibrium field on each sampled direction, a Ritz principle." | P11 S2 ("By Proposition 1, for a recovery operator that reproduces the retained displacements, this term is smallest when the recovered field equals the exact field for every sampled test displacement"). "admissible" replaced (brief 3.1); "the variational identity of Section 3.1" → "Proposition 1" (now in Section 3.1, Eq. (6)); the label "a Ritz principle" deleted (brief 3.1). |
| "The sensitivity term compares the field-based thickness sensitivities (eight-component vectors) on the set \(\mathcal J_s\) of directions with reference sensitivity labels;" | P11 S3-S4 ("thickness sensitivities \(\widetilde{\boldsymbol s}_j\) computed from the recovered field (Section 4.2)"; "field-based" deleted, brief 3.1; the definition of \(\widetilde s_c\) is in Section 4.2, see open question 1) |
| "the reported configurations use \(w_s=1\)." | P11 S5 |
| "Gradients with respect to \(\theta\) pass through the reconstructed field to the geometry encoders, coefficient heads and linear displacement maps." | P11 S6 |
| "Directions with large energy ratios, found by a block search on the rigid complement, are added during training," | P11 S7-S8 ("the complement of the rigid-body modes", as in Appendix G.3) |
| "and geometry augmentation uses the 48 cube symmetries (Appendix G)." | P11 S9 |
| "Table 2 identifies the variants and Table ST01 records the training and model-selection settings." | P11 S10 |
| "NICE is trained on the operator that is deployed:" | P12 S1 |
| "Eq. (18) is evaluated on the corrected extension \(F=\mathcal W\widehat E\), with the two-grid cycle of Table 1, written 8 / \(Q_1(17)\) / 8 for its smoothing steps and coarse space." | P12 S2-S3; S3 spells out the notation (8 pre-smoothing steps, coarse-grid correction with \(Q_1(17)\), 8 post-smoothing steps), as in Table 1 ("8 Chebyshev steps, \(Q_1(17)\) coarse-grid ... correction, 8 Chebyshev steps"). |
| "Because \(\mathcal W\) is linear and fixed for a given geometry, the gradient passes through the smoothing recurrence and the coarse-grid correction to the network parameters, as in solver-in-the-loop training [Um et al. (2020)]." | P12 S4 |
| "The smoothing interval and coarse factorisation depend only on \(K\) and are recomputed per geometry, so no trainable parameters are added." | P12 S5-S6 ("the factorisation of the coarse-grid matrix") |

### Old 4.6 "Application of the condensed operator" (old lines 349-357) → new 3.6 "Applying the condensed stiffness"

| Old sentence | Where it is now |
|---|---|
| Heading "4.6. Application of the condensed operator" | heading "3.6. Applying the condensed stiffness" (brief Section 4) |
| "Geometry-dependent quantities are prepared once per geometry and reused across retained inputs: the element moments, the network's geometry encoding, coefficients and maps, the smoothing interval and the coarse factorisation (Table ST13 gives their cost)." | P13 S1-S3 |
| "Each application of \(\widehat S\) then requires one network pass, one two-grid cycle, one stiffness action and the transposes of the cycle and the network (Appendices E–F)." | P13 S4-S5 ("Each product of \(\widehat S\) with a vector of retained displacements"; "one product with \(K\)") |
| "Neither \(\widehat E\) nor \(\widehat S\) is formed:" | P13 S6 ("is formed as a matrix") |
| "a cell stores its geometry-dependent coefficients and coarse factorisation," | P13 S7 |
| "whereas a dense \(\widehat S\) of a cell of median retained size, 23,604 DOFs, would take 4.2 GiB in double precision." | P13 S8 ("a cell with the median number of retained DOFs, 23,604") |
| "**Algorithm 1. Corrected condensed stiffness action.**" | "**Algorithm 1. Action of the corrected condensed stiffness.**" CN: "**算法 1. 修正后凝聚刚度矩阵的作用。**" |
| Step 1 "Extend \(q\) with \(\widehat E\) and apply the two-grid cycle \(\mathcal W\) at fixed retained values to obtain \(\widehat u=Fq\)." | Step 1 ("Compute \(\widehat Eq\) with the network") |
| Step 2 "Form \(y=K\widehat u\) and return \(F^Ty\): apply \(\mathcal W^T\), then the transposed learned extension." | Step 2 ("the transpose of the network recovery operator") |
| Step 3 "Assemble these actions through Eq. (3); after the global solve, recover \(F_mB_m\widehat U\) for field and sensitivity evaluation." | Step 3, split into two sentences |

### Labels deleted (brief Section 3.1)

1. "(Galerkin)" in "coarse-grid (Galerkin) correction" (old 4.4); the main text now says "coarse-grid matrix \(A_c=V^TAV\)".
2. "a Ritz principle" (old 4.5).
3. "field-based" (old 4.5).
4. "Error spectrum" in the heading of old 4.3 (new heading from brief Section 4; the content is kept).

## (b) Cross-references renumbered (each occurrence mapped once)

| Old | New | Place |
|---|---|---|
| Section 4.3 (heading) | Section 3.3 | heading |
| Section 4.4 (heading) | Section 3.4 | heading |
| Section 4.5 (heading) | Section 3.5 | heading |
| Section 4.6 (heading) | Section 3.6 | heading |
| Eq. (5) | Eq. (7) | 3.3 P1 |
| Eq. (6) | Eq. (14) | 3.3 P1 (forward reference to Section 4.1) |
| Eq. (15) | Eq. (10) | tag; 3.3 P5 |
| (none; "a complementary coarse-grid correction") | Section 3.4 | 3.3 P5 (added pointer to old 4.4) |
| Eq. (16) | Eq. (11) | tag; 3.4 P7 ("the last relation in Eq. (11)", old "the last equality") |
| Section 4.1 | Section 3.1 | 3.4 P8 |
| Eq. (4) | Eq. (6) | 3.4 P8 |
| Eq. (17) | Eq. (12) | tag |
| Proposition 4 | Proposition 2 | label; P9 S4 and Remark 1 S1 also name Proposition 2 |
| Eq. (9) | Eq. (17) | Remark 1 |
| (none; "the interior error \(d_I\)") | Section 3.1 | 3.3 P2 (added pointer to where \(d_I\) is defined) |
| Section 3.1 ("the variational identity of Section 3.1") | Proposition 1 | 3.5 P11 S2 |
| (none; "field-based thickness sensitivities") | Section 4.2 | 3.5 P11 S3 (added pointer to the definition of \(\widetilde s_c\), old 3.3) |
| Eq. (18) | Eq. (13) | tag; 3.5 P12 |
| (A1), (A2) | (A1), (A2) of Section 2.4 | Proposition 2 |
| Table 1, Table 2 | unchanged (Table 2 stays in Section 5.1) | 3.5 |
| Tables ST01, ST13 | unchanged | 3.5, 3.6 |
| Section 5.2, Section 5.5 | unchanged | 3.3 P5, 3.4 P9 |
| Eq. (3) | unchanged | Algorithm 1, step 3 |
| Appendices C, D, D.1, E–F, F.1, G; Eq. (D.6) | unchanged | 3.3-3.6 |

## SUPPLEMENT ADDITIONS

None. The source of this unit does not mention the Uncorrected continuation or the Smoothing-trained variant.

## (d) Open questions

1. The sensitivity \(\widetilde s_c=-\widehat u^TK_{,c}\widehat u\) is defined in Section 4.2 (old 3.3), which now follows Section 3.5, where the loss function first uses \(\widetilde{\boldsymbol s}_j\). This unit uses the forward pointer "(Section 4.2)", as U02 does in Section 2.3. If the assembler wants the definition at first use, one sentence could be added to 3.5 P11. That sentence would come from old 3.3 and would also need \(K_{,c}\), so this unit did not add it.
2. 3.3 P1 refers forward to Eq. (14) in Section 4.1. The forward reference is required by the new order (method before error analysis).
3. The names "geometry encoders", "coefficient heads", "linear displacement maps", "element moments" and "geometry encoding" in 3.5 and 3.6 follow old 4.2. They were checked against the current U04 draft of Section 3.2, which uses "encoders", "coefficient heads" (defined there as small output networks), "element moments" and linear maps to and from the displacement features. They agree. If U04 changes these names, the same names should be used here. U04 also points to "Sections 3.3 and 3.4" for the two-grid correction and to "Section 3.5" for training, which matches this unit.
4. Old 5.2 says "The nonexpansiveness of Section 4.3 is guaranteed by \(b\ge\lambda_{\max}(D^{-1}A)\)." The Section 5.2 writer should point to Section 3.3 and use "does not increase the energy error". 3.3 P5 keeps the matching statement, with the condition that the actual positive spectrum lies in \((0,b]\).
5. The title of Algorithm 1 changed in wording only. The brief lists only figure and table blocks as parsed structures. If a build script matches the old title "Corrected condensed stiffness action", it needs updating.
6. CN terms: 粗网格校正 is used for the coarse-grid correction (field-standard), and 修正 / 两重网格修正 for \(\mathcal W\), following the brief and U03. 本文的修正序列 translates "the correction sequences".

## VERIFIER

Checked against MANUSCRIPT_EN.md / MANUSCRIPT_CN.md lines 289-358, the brief, U03 (Section 3.1), U04 (Section 3.2), U06 (Sections 4.1-4.2) and Appendices C, D, D.1, F.1, G.2-G.4 of APPENDICES_EN.md.

Python checks (after the changes below):
- Numbers, old EN vs new EN: every difference is a renumbered reference or tag (old Sections 4.1, 4.3-4.6 → 3.1, 3.3-3.6; old Eqs. (4), (5), (6), (9), (15)-(18) → (6), (7), (14), (17), (10)-(13); Proposition 4 → 2) or an added pointer (Sections 3.1, 3.4, 4.2; Proposition 1), plus the repeated "8" and "Q_1(17)" in the spelled-out notation 8 / \(Q_1(17)\) / 8. No measured number missing or added.
- Numbers, new EN vs new CN: identical multisets, also paragraph by paragraph (25 blocks each, 69 lines each).
- Citations: identical lists, in order, in old EN, new EN and new CN. Inline mathematics: identical multisets in EN and CN; every old inline expression is present in the new EN. Display equations: byte-identical to the old ones apart from the tags (15)-(18) → (10)-(13); EN and CN identical.
- Cross-references checked against the other units: Eq. (6) and (7) in U03, Eq. (14) and (17) in U06, \(d_I\) and \(\mathcal W\) defined in Section 3.1 (U03), \(\widetilde s_c\) defined in Section 4.2 (U06).
- Claims checked against the appendices: \(k\) steps give a degree-\(k\) polynomial \(d_k=p_k(D^{-1}A)d_0\) (Appendix D); \(b\) is a power-iteration estimate for \(D^{-1}A\) multiplied by 1.05 (Appendix D); \(\mathbb K\preceq\widehat{\mathbb K}\) by assembly and order reversal of inverses (Appendix C, Eq. (C.1)); the D.1 example halves the energy and moves the discrepancy from zero to \(2t-t^2\); the training classes (equilibrated nodal loads, consistent tractions, springs, neighbour-induced traces) and the block search on the retained rigid complement (Appendix G.2-G.3).
- Sections A-F of the verifier task: no claim strengthened or weakened, no PIML comparison, priority claim, training-cost or machine statement; no figure or table blocks in this unit; the Algorithm 1 title line is not parsed by the build scripts (latex/build_*.py contain no "Algorithm" pattern).

Changes made:
1. EN 3.5 P11 S2: "the principle of minimum potential energy (Proposition 1) makes this term smallest" → "By Proposition 1, ... this term is smallest". Proposition 1 is the Ritz identity, a consequence of the minimum potential energy principle; the parenthesis equated the two. CN: "由最小势能原理（命题 1）" → "由命题 1"; "精确再现主自由度位移的" → "在主自由度上等于给定位移的" (brief 3.1 term for "admissible"; 精确再现 is reserved for rigid-body motion in U03).
2. EN 3.4 P9 S2: "inherit this ordering" → "inherit the ordering of Proposition 2"; the antecedent of "this ordering" was the energy-norm sentence, not the matrix ordering. CN: "保持这一序关系，其逆矩阵的序关系则反向" → "继承命题 2 的序关系，求逆后序关系反向".
3. Remark 1, last sentence (36 words) split into two sentences in EN and CN, content unchanged.
4. CN 3.3 P1: "二者均表明，应……" → "二者均表明，宜……"; "应" read as a requirement, while the source says "suggest".
5. CN 3.3 P2: "所占的比例" for \(\chi_\ell\) instead of "占比", which the brief reserves for the energy fraction \(w_m\) (应变能占比); P3 already uses "比例".
6. CN 3.5 P11: "能量比较大的测试位移" → "能量比取值较大的测试位移"; the old wording can be read as "能量 比较大" (relatively large energy) instead of "large energy ratio".
7. CN, translationese and 其: "其总能量为" → "总能量取"; "其递推关系" → "及其递推关系"; "为其内部残差" → "为相应的内部残差"; "存储其依赖于几何的系数" → "存储依赖于几何的系数"; "网络采用若干组测试位移进行训练" → "网络以若干组测试位移作为训练数据"; "NICE 采用……同一算子进行训练" → "NICE 的训练采用……同一算子".
8. NOTES table rows for P11 S2 and Remark 1 updated to the new wording.

Not changed, but checked:
- Remark 1 S1 names "the two conditions on the smoothing and on the coarse solve" where the source says "both conditions". The two discussed in the remark are \(\|\Phi_k\|_A\le1\) (spectrum up to \(a+b\)) and the coarse inverse (\(2G-GA_cG\succeq0\)), so the gloss is the only consistent reading.
- 3.3 P1 S1 glosses what Eqs. (7) and (14) express; both statements are in U03 and U06. 3.4 P9 S2 states in words the assembled ordering behind "the order reversal of Appendix C"; Appendix C contains it.
- 3.3 P5 "a power-iteration estimate of the largest eigenvalue of \(D^{-1}A\)" matches Appendix D (40 iterations of \(D^{-1}A\), factor 1.05); the iteration count stays in Appendix D and Table 1.

Unresolved (for the assembler):
- Open questions 1-6 above stand. Open question 4 (old Section 5.2 "nonexpansiveness of Section 4.3") belongs to the Section 5.2 unit.
- U04 says "encoders" and "linear maps"; this unit keeps the source names "geometry encoders" and "linear displacement maps". The meaning agrees; the assembler may unify the wording.

