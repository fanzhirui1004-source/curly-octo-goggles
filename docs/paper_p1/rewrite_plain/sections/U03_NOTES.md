# U03 notes: Section 3 introduction, Figure 2, Section 3.1

Source (old numbering): MANUSCRIPT_EN.md / MANUSCRIPT_CN.md lines 218-246 (old Section 4 introduction, Figure 2, old 4.1) and lines 128-148 (old 3.1, Proposition 1, Eqs. (4), (5)).

New blocks (EN and CN aligned, 17 blocks each): heading `## 3.`; S0 Section 3 introduction; Figure 2 image; Figure 2 caption; heading `### 3.1.`; P1 definition of the recovery operator \(F\) and its two conditions; Eq. (4); P2 meaning of Eq. (4), virtual work, need for \(F^T\); Eq. (5); P3 the second term of Eq. (5), symmetry, null space, Appendices E and B.2; P4 minimum potential energy, interior error \(H\), cross terms; Proposition 1; Eq. (6); P5 consequences, definition of the energy error and of "test displacement", residual; Eq. (7); P6 computability and coverage; P7 the NICE operator \(F=\mathcal W\widehat E\).

Order inside 3.1 follows the unit instruction: Eq. (4) (old 11), Eq. (5) (old 12), Proposition 1 with Eq. (6) (old 4), Eq. (7) (old 5), then the NICE operator. Eq. (4) must stay as displayed and it shows \(F=\mathcal W\widehat E\). P1 therefore names \(\widehat E\) and \(\mathcal W\) in one sentence before Eq. (4), and P2 states that every relation of Eq. (4) except \(F=\mathcal W\widehat E\) holds for any recovery operator with the two properties. P7 then gives the full NICE definition.

Length (EN words, display equations excluded): old source 796, new 1158. The increase comes from splitting long sentences, from merging two sources that each restated the other (kept once where possible), from the roadmap of Section 3 now naming Sections 3.1-3.6, and from plain explanations at first use (the interior residual as interior force imbalance, "test displacement", the uncorrected \(\widehat S_{\rm net}\) in the caption).

Self-check (Python, on the final files): (a) every number of the old EN source is in the new EN or is a renumbered reference (old Eqs. (11), (12), (14) and Sections 4.1-4.5 only; see (b)); (b) the new EN contains no number absent from the old source except renumbered references and tags (Sections 2.4, 3, 3.2-3.6; Eqs. (6), (7), (9)); (c) EN and CN contain the same numbers in every block, the same inline mathematics and identical display equations; the four display equations are byte-identical to the old ones apart from their tags; (d) no EN sentence above 35 words (longest 31, mean 16), no CN sentence above 80 characters (longest 61, mean 29); (e) no banned word or old term in EN or CN (the only remaining grep hit is a false positive: `\boxed` in Eq. (6) for "box").

## (a) Sentences and numbers removed or moved

### Old Section 4 introduction (old line 220) → new Section 3 introduction

| Old sentence | Where it is now |
|---|---|
| "NICE is defined by one object, the corrected extension \(F\), which maps a retained displacement to a displacement of the whole cell" | S0 S1 |
| "its energy \(\widehat S=F^TKF\) is the condensed stiffness" | S0 S2 |
| "the Galerkin operator of \(F\)" | removed from the main text (brief 3.1: appendices only). The same relation is stated in Section 1.1 P7 (U01): "\(\widehat S=F^TKF\) ... the coarse-grid matrix of this interpolation". |
| "exact static condensation is the case \(F=E\), the ideal interpolation" | S0 S2 ("exact static condensation is the case \(F=E\)"); "the ideal interpolation" is a duplicate of Section 1.1 P7 (U01), where \(E\) is introduced as the ideal interpolation of algebraic multigrid. |
| "The construction divides the work on \(F\)." | deleted (banned "division of tasks"). |
| "The energy form, applied with the transpose of the complete extension, supplies the mechanical structure for any admissible extension (Section 4.1)." | S0 S3-S4, without the "supplies" cadence: Section 3.1 computes the condensed stiffness from the strain energy and applies it with \(F^T\); for any recovery operator that reproduces the retained displacements and rigid-body motions it is symmetric positive semidefinite and never softer than the exact one. The properties named are those of old 4.1 and Proposition 1 (now Eq. (6)). |
| "A geometry-conditioned network, exactly linear in the retained displacement and admissible by construction, supplies the trial interior field (Section 4.2)." | S0 S5-S6; "trial interior field" deleted (brief 3.1); "admissible" written as "reproduces them [the retained displacements]". |
| "A fixed correction \(\mathcal W\) inside the same energy reduces the interior imbalance that the network leaves (Sections 4.3 and 4.4)" | S0 S7, written as "applied to the predicted displacements before the strain energy is computed". |
| "and the network is trained through that correction (Section 4.5)" | S0 S8 |
| "Figure 2 shows how the parts define both the condensed action and the displacement recovered after assembly." | S0 last sentence |
| (none) | S0 S9 added as a roadmap item: "Section 3.6 describes how the condensed stiffness is applied without forming a matrix." Content from old 4.6 ("Neither \(\widehat E\) nor \(\widehat S\) is formed"). |

### Figure 2 caption

| Old | New |
|---|---|
| Title "Learned displacement extension, equilibrium correction and variational assembly." | "Overview of NICE: condensed stiffness of one cell, assembly and design." The new title follows the two panel titles of the relabelled image. |
| (a) "Rigid-body motion is separated from the retained displacement before the deformation is extended by the network." | (a) S1 |
| "The rigid field is reconstructed and the prescribed retained values are restored;" | (a) S2, with the image label \(\widehat Eq\) |
| "the correction \(\mathcal W\) then reduces interior imbalance at fixed retained displacement," | (a) S3, with the image label \(\widehat u=Fq\) |
| "and applying the local stiffness and the transpose \(F^T\) of the complete extension gives the work-conjugate retained force." | (a) S4: "the corresponding nodal forces on the retained DOFs, \(\widehat Sq\)" |
| "The ordering in (a) is that of Proposition 4; it concerns the energy error, not the sensitivity error." | (a) S5-S6; Proposition 4 → Proposition 2; the ordering is written out as in the image, and \(\widehat S_{\rm net}\) is explained as the condensed stiffness without correction (definition from old Proposition 4). |
| (b) "The cell operators are assembled on the shared box-face DOFs and the assembled system is solved;" | (b) S1, "cell-face DOFs", as in the relabelled image |
| "the assembled solution supplies the inputs for local field recovery, the compliance and the field-based thickness sensitivity." | (b) S2; "field-based" deleted (brief 3.1) |
| "A design iteration updates the corner thickness parameters and so regenerates every cell's geometry; the network parameters stay the same and the coefficients of \(\mathcal W\) are recomputed." | (b) S3-S4 |

The caption quotes none of the labels removed from the image ("learning: trial field", "correction: improvability", "variational form: structure", "learned extension", "equilibrium correction", "box-face"). It uses the image's labels "neural network", "two-grid correction", "transpose of \(F\)", "shared cell-face DOFs".

### Old 4.1 (old lines 228-246) → new 3.1

| Old sentence | Where it is now |
|---|---|
| "Let \(\widehat E\) be an admissible extension, \(J_P\widehat E=I_p\), that reproduces rigid-body motion, \(\widehat ER_P=R\) for the rigid-body modes \(R\) of Section 2.2; Section 4.2 constructs it with a geometry-conditioned network (Eq. (14))." | split: the two conditions are stated for a general \(F\) in P1 S3 (with "rigid-body modes of Section 2.2") and P1 S4 (assumption (A2), Section 2.4); the statement for \(\widehat E\) is P7 S2-S3 (Section 3.2, Eq. (9)). |
| "The extension becomes a mechanical operator through the energy, under the original stiffness, of the field it produces." | P1 S1 |
| "Let the interior correction \(\mathcal W\) be one symmetric two-grid cycle at fixed retained displacement, specified in Sections 4.3 and 4.4;" | P7 S4 (Sections 3.3 and 3.4); also named in P1 S5 |
| "it is the equilibrium correction of the method's name." | removed: duplicate of Introduction paragraph 4 (U00): "This cycle is the equilibrium correction in the name of the method." The brief says to state this once. |
| "It is linear and preserves retained values, and the identity corresponds to no correction." | P7 S5-S6 ("Taking \(\mathcal W\) as the identity gives the uncorrected recovery \(F=\widehat E\)") |
| "For each geometry, the correction schedule and its coefficients are fixed independently of the retained displacement." | P7 S7 ("the steps of the cycle and their coefficients") |
| "The final extension and its condensed stiffness are" + Eq. (11) | P1 last sentence + Eq. (4) |
| "For a virtual retained displacement \(\delta q\), ... \(\delta\widehat{\mathcal U}=\ldots\)." | P2 S3 |
| "Applying the condensed operator therefore requires, after the stiffness action, the transpose \(F^T\) of the complete extension (rigid reconstruction, retained-value restoration and correction included)." | P2 S4-S5 (the parenthesis became its own sentence) |
| "With \(F_I=J_IF\), the condensed action has the block form" + Eq. (12) | P2 last sentence + Eq. (5) |
| "The second term transfers the work of the interior residual to the retained DOFs." | P3 S1, with "the interior force imbalance of the recovered field" added as a plain gloss |
| "It vanishes for an equilibrated field; otherwise it is needed for the returned force to be the derivative of the stated energy." | P3 S2-S3 |
| "Symmetry and positive semidefiniteness follow from \(K=K^T\succeq0\), without requiring symmetry of the learned gather, scatter or grid-transfer maps." | P3 S4-S5 ("the gather, scatter and grid-transfer maps of the network", Section 3.2) |
| "Since the null space of \(K\) consists of the rigid-body modes (Section 2.2) and \(FR_P=R\), the only null modes of \(\widehat S\) are the retained rigid-body modes (Appendix B);" | P3 S6-S7 ("zero-energy modes") |
| "Appendix E derives the transpose \(F^T\), and Appendix B.2 shows why it makes the operator error quadratic." | P3 S8-S9 |
| "By Eq. (4), \(\widehat S-S=H^TAH\) with \(H=J_I(F-E)\): the operator error is the energy of the interior error of \(F\), which Sections 4.3 and 4.4 reduce." | P7 S9 ("By Eq. (6), the error of the condensed stiffness is the energy of the interior error of \(F\), which Sections 3.3 and 3.4 reduce"). The restated formula \(\widehat S-S=H^TAH\), \(H=J_I(F-E)\) is dropped as a duplicate of Proposition 1 and P4 in the same section. |
| "After an assembled solve, \(F_mB_m\widehat U\) supplies the local reconstructed displacement for response and sensitivity evaluation." | P7 S10 |
| (none) | P7 S8 added: "The corrected operator \(F\) thus also reproduces the retained displacements and the rigid-body motions, and Proposition 1 applies to it (Appendix B.2)." Supported by old 4.1 (which uses \(FR_P=R\) for the corrected \(F\)) and Appendix B.2 ("Corrections driven by the interior residual leave this field unchanged, so \(FR_P=R\)"). |
| (none) | P2 S2 added: "All relations in Eq. (4) other than \(F=\mathcal W\widehat E\) hold for any recovery operator with the properties above." Needed because Eq. (4) now precedes the NICE definition. |

### Old 3.1 (old lines 128-148) → new 3.1

| Old sentence | Where it is now |
|---|---|
| Heading "3.1. Condensed stiffness: a one-sided error quadratic in the interior error" | replaced by the new heading of brief Section 4 |
| "Under (A1), \(Eq\) minimises the energy at prescribed \(q\)." | P4 S1, with "by the principle of minimum potential energy" (brief 3.1, replacement for "Ritz") |
| "Any admissible linear extension \(F\), with \(J_PF=J_PE=I_p\), differs from it only in the interior DOFs." | P4 S2 |
| "Writing \(H=J_I(F-E)\), interior equilibrium \(J_IKE=0\) eliminates the cross terms in the condensed stiffness \(\widehat S=F^TKF\) of \(F\) (Appendix B.2)." | P4 S3-S4 |
| Proposition 1 statement, "exceeds the exact Schur complement by the energy of the interior error" | Proposition 1, "exceeds the exact condensed stiffness \(S\)". "Schur complement" is said once in Section 1.1 (U01) and possibly in Section 2.2 (see (d)). Label "(Ritz identity)" kept, its only occurrence in the main text of this unit. |
| Eq. (4) | Eq. (6), unchanged |
| "An approximate extension therefore adds stiffness in proportion to the energy of its interior error; the absence of a first-order term follows from equilibrium of the reference field." | P5 S1-S2 |
| "Exact static condensation is the case \(F=E\), \(H=0\)." | P5 S3 |
| "For \(u=Eq\), \(\widehat u=Fq\) and \(d_I=Hq\), the interior residual \(r_I=(K\widehat u)_I\) satisfies \(r_I=Ad_I\), so the relative directional energy error \(\varepsilon(q)=\ldots\), nonnegative by Eq. (4), is" | P5 S4-S6, reordered: the energy error is defined first (with "test displacement" for a retained displacement used to measure accuracy), then \(u\), \(\widehat u\), \(d_I\) and the residual. "directional energy error" → "relative energy error". |
| Eq. (5) | Eq. (7), unchanged |
| "The residual is available without the reference extension; evaluating its \(A^{-1}\)-norm needs one interior solve per direction." | P6 S1-S2 ("exact recovery \(E\)", "per test displacement") |
| "A bound over all directions follows only from the coverage of the sampled directions (Appendices B and G.4)." | P6 S3 ("how well the sampled test displacements cover the retained displacements") |

No number was removed. The source contains no citation, no table and no statement about the Uncorrected continuation or the Smoothing-trained variant.

## (b) Cross-references renumbered (each occurrence mapped once)

| Old | New | Place |
|---|---|---|
| Section 4 (heading) | Section 3 | heading |
| Section 4.1 | Section 3.1 | S0 S3 |
| Section 4.2 | Section 3.2 | S0 S6; P7 S2 |
| Sections 4.3 and 4.4 | Sections 3.3 and 3.4 | S0 S7; P7 S4; P7 S9 |
| Section 4.5 | Section 3.5 | S0 S8 |
| (none) | Section 3.6 | S0 S9 (added roadmap item) |
| Proposition 4 | Proposition 2 | Figure 2 caption (a) |
| Section 3.1 (heading) | Section 3.1 | heading (new content) |
| Eq. (11) | Eq. (4) | tag and P2 S2 |
| Eq. (12) | Eq. (5) | tag |
| Eq. (4) | Eq. (6) | tag; P5 ("nonnegative by Eq. (6)"); P7 S9 |
| Eq. (5) | Eq. (7) | tag |
| Eq. (14) | Eq. (9) | P7 S3 |
| (A1), (A2) of the old Section 3 introduction | (A1), (A2) of Section 2.4 | P1 S4 ("assumption (A2) of Section 2.4"); P4; Proposition 1 |
| Section 2.2 | Section 2.2 (unchanged) | P1 S3; P3 S6 |
| Appendices B, B.2, E, G.4 | unchanged | P3, P4, P6, P7 |
| (none) | Section 3.2 | P3 S5 (gather, scatter and grid-transfer maps of the network) |

## SUPPLEMENT ADDITIONS

None. The source of this unit does not mention the Uncorrected continuation or the Smoothing-trained variant.

## (d) Open questions

1. FIGURES.md carries a copy of the Figure 2 caption. The assembler should replace it with the new caption (EN) so that both agree.
2. "Test displacement" is defined in P5 of 3.1. The current U02 draft (Section 2.3) says "The local accuracy of the condensed stiffness is measured by the relative energy error of Section 3.1" and does not define "test displacement", so the definition here is the first one. If U02 changes, the phrase "called a test displacement" may be moved.
3. "Schur complement" is used once in Section 1.1 (U01); the current U02 draft does not use it, and this unit does not use it either.
4. The classical status of Proposition 1 and of the residual form is cited in the old Section 3 introduction ("The Ritz identity, the residual form ... are classical [Fraeijs de Veubeke (1965)], [Toselli & Widlund (2005)], [Becker & Rannacher (2001)] ..."), which now becomes the Section 4 introduction, after Proposition 1. This unit adds no citations. The assembler may move the part of that sentence that concerns the Ritz identity and the residual form into 3.1, directly after Eq. (7), so that the attribution sits next to Proposition 1; Section 1.1 P2 (U01) already attributes the minimum-energy property to the literature.
5. The statement that the network recovery operator "reproduces them [the retained displacements] by construction" in S0, and P7 S8 that the corrected \(F\) also reproduces the rigid-body motions, rely on Eq. (9) and Appendix B.2. The Section 3.2 writer should keep Eq. (9) and the sentence "\(J_P\widehat E=I_p\) and \(\widehat ER_P=R\) ... hold by construction".
6. The introduction (U00, paragraph 4) states that the cycle is the equilibrium correction of the method's name. This unit does not repeat it. If the assembler prefers the statement at the formal definition, P7 S4 can take the clause "which is the equilibrium correction in the name of the method" and the introduction sentence can be shortened.
