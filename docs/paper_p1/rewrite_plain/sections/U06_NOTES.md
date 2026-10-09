# U06 notes: Section 4 (Effect of interior displacement errors on compliance and sensitivities), 4.1, 4.2

Source (old numbering): MANUSCRIPT_EN.md / MANUSCRIPT_CN.md lines 117-127 (old Section 3 introduction; the assumption list of lines 121-126 now lives in Section 2.4, U02) and lines 149-217 (old 3.2, old 3.3 with the closing list of three conditions). Old 3.1 (lines 128-148) is in Section 3.1 (U03).

New blocks (EN and CN aligned, 24 blocks each): heading `## 4.`; P0a (what the section does, the three results); P0b (scope, classical results, assumptions); heading `### 4.1.`; P1 (two assembled systems, notation); Eq. (14); P2 (two parts of the error energy, energy fractions, \(\beta\)); Proposition 3; Eq. (15); P3 (small energy fraction); P4 (incomplete solve); Eq. (16); P5 (residual work); heading `### 4.2.`; P6 (exact sensitivity, definition of \(\widetilde s_c\)); Proposition 4 lead-in; (a); Eq. (17); (b); P7 (linear cross term, sign); P8 (\(\widetilde s_c\) is not \(\widehat C_{,c}\)); Eq. (18); P9 (residual term, reported sensitivities, design intervals, switches); P10 (closing paragraph: compliance and sensitivity are checked separately).

Length (EN words, display equations excluded): old source 1013 (including the assumption list and the list of three conditions), new 1160. The increase comes from splitting long sentences and from the definition of \(\widetilde s_c\) required by brief 3.1.

Self-check (Python, on the final files): (a) every number of the old EN source is in the new EN text or is a renumbered or moved reference, listed in (a) and (b) below; (b) the new EN contains no number absent from the old source except renumbered references and tags (Section 2.4, Sections 4.1 and 4.2, Propositions 3 and 4, Eqs. (7), (14)-(18)); (c) EN and CN contain the same numbers in every block, the same citations, the same inline mathematics, and identical display equations; the five display equations are byte-identical to the old ones apart from their tags; (d) no EN sentence above 35 words (longest 32, mean 15.4); one CN sentence above 80 characters, and only because of its five-item citation list (the text without the citations has 37 characters); (e) no banned word or old term in EN or CN. The remaining EN grep hits are false positives: `\boxed` (for "box"), "The first comes from" (an enumeration, not a priority claim) and the en dash of "(A1)–(A3)" (a range, not an em dash).

## (a) Sentences and numbers removed or moved

### Old Section 3 introduction (old lines 117-127) → new Section 4 introduction

| Old sentence | Where it is now |
|---|---|
| Heading "3. Error of an approximate extension in analysis and design" | heading "4. Effect of interior displacement errors on compliance and sensitivities" (brief Section 4) |
| "With the retained DOFs fixed, the local approximation lies in the interior field." | P0a S1 ("Because all retained DOFs are kept, a cell is approximated only through its interior displacements.") |
| "This section follows the error of any admissible linear extension through the condensed stiffness to the two quantities that design uses, and arrives at three results." | P0a S2. "any admissible linear extension" → "any recovery operator that satisfies assumption (A2) of Section 2.4". "and arrives at three results" removed: the first of the three results (Proposition 1) is now in Section 3.1, so this section derives two; the three are still listed in P0a S3-S8. |
| "First, the condensed stiffness exceeds the exact Schur complement by the energy of the interior error: the error is one-sided, quadratic and computable from the interior residual (Section 3.1)." | P0a S3-S5. Now refers to Proposition 1 in Section 3.1 (unit instruction). "exact Schur complement" → "exact condensed stiffness" (brief 3.1; "Schur complement" is said once elsewhere). "one-sided" → "never softer than the exact one" (wording of the approved abstract). "computable from the interior residual" now points to Eq. (7), the more specific reference, because Section 3.1 is already named in S3. |
| "Second, this error reaches the assembled compliance weighted by the share of the energy that each cell carries (Section 3.2)." | P0a S6 (Section 4.1; "fraction of the strain energy") |
| "Third, it reaches the thickness sensitivity through the stiffness derivative, with a term linear in the interior error, so that an accurate compliance does not imply an accurate local sensitivity (Section 3.3)." | P0a S7-S8 (Section 4.2) |
| "The relations are stated for a single substructure and apply to every cell of an assembly." | P0b S1 |
| "The Ritz identity, the residual form and the self-adjoint compliance sensitivity are classical [Fraeijs de Veubeke (1965)], [Toselli & Widlund (2005)], [Becker & Rannacher (2001)], [Haftka & Gürdal (1992)], [Bendsøe & Sigmund (2004)];" | P0b S2. "The Ritz identity" → "Proposition 1" (brief 3.1 allows "Ritz identity" once, as the label of Proposition 1 in Section 3.1); "the residual form" → "the residual form of the energy error in Eq. (7)". All five citations kept unchanged. |
| "the share-weighted bound and the sensitivity relations are derived here for approximate extensions." | P0b S3 ("The bound weighted by the energy fractions ... derived in this paper for approximate recovery operators") |
| "The results are stated as Propositions 1–3 under the following assumptions, which Appendix B.1 states in full." | P0b S4: "Propositions 3 and 4 use the assumptions of Section 2.4." The clause "which Appendix B.1 states in full" is a duplicate of Section 2.4 (U02: "Propositions 1–4 use the following assumptions, which Appendix B.1 states in full"). |
| Assumption list (A1), (A2), (A3), (D), with \(A=K_{II}\succ0\), \(J_PF=I_p\), \(FR_P=R\), \(\mathbb K\) positive definite, "(Section 2.2)", "(Section 2.3)", "Appendix B.1" | moved to Section 2.4 (U02), per brief Section 4. |

### Old 3.2 (old lines 149-179) → new 4.1

| Old sentence | Where it is now |
|---|---|
| Heading "3.2. Compliance: the error weighted by the energy share" | "4.1. Compliance" (brief Section 4) |
| "Consider the exact equilibria of the two supported systems in Eq. (3), with the same local matrices, assembly maps, homogeneous supports and nonzero retained load." | P1 S1-S2 ("cell stiffness matrices"; "the same nonzero load on the retained DOFs") |
| "Writing \(u_m=E_mB_mU\), \(\widehat u_m=F_mB_m\widehat U\) and \(a(v,v)=\ldots\), global equilibrium and local energy orthogonality give (Appendix C)" | P1 S3-S4 ("the energy orthogonality within each cell") |
| Eq. (6) | Eq. (14), unchanged |
| "Compliance underestimation is the total reconstructed error energy, with orthogonal contributions from the changed retained solution and from the interior departure from equilibrium at that solution." | P2 S1-S3 |
| "How much of a cell's error reaches it depends on how much energy the cell carries." | P2 S4 |
| "At the exact assembled traces \(q_m=B_mU\), define the energy shares \(w_m=\ldots\), which sum to one, and \(\beta=\ldots\), with zero-energy rigid-body responses contributing zero." | P2 S5-S7. "traces" → "retained displacements"; "energy shares" → "energy fraction" (brief 3.1). Added in S6: "where \(\varepsilon_m\) is the energy error of cell \(m\) as defined in Eq. (7)" (back-reference only; the definition is that of Section 3.1 and Appendix C.1). S7: "its contribution to \(\beta\) is taken as zero" (Appendix C.1: "its contribution is defined directly as zero"). |
| "**Proposition 2 (share-weighted compliance error).** Under (A1)–(A3), the relative compliance error is bounded by the share-weighted energy error (Appendix C.1):" | "**Proposition 3 (energy-weighted compliance error).** ... bounded by the weighted energy error \(\beta\) (Appendix C.1):" |
| Eq. (7) | Eq. (15), unchanged |
| "A cell with a small energy share can thus leave the compliance accurate even when its own field is not." | P3 ("energy fraction"; "its own interior displacements are inaccurate") |
| "For the numerical checks of Section 5.8, Eq. (6) also separates the operator error from that of an incomplete assembled solve." | P4 S1 ("Eq. (14) also separates the error of the condensed stiffness from the error of an incomplete assembled solve, and the numerical checks of Section 5.8 use this separation.") |
| "For an approximate solution \(\bar U\), with the recomputed residual \(\rho=\ldots\) and the recovered fields \(\bar u_m=\ldots\)," | P4 S2-S3 |
| Eq. (8) | Eq. (16), unchanged |
| "so the signed residual work \(\bar U^T\rho\) measures the effect of stopping the solve early (Appendix C.2)." | P5 |

### Old 3.3 (old lines 181-210) → new 4.2

| Old sentence | Where it is now |
|---|---|
| Heading "3.3. Sensitivity: a linear term through the stiffness derivative" | "4.2. Sensitivities" (brief Section 4) |
| "On a differentiable design interval with fixed active and retained DOFs, assembly maps, homogeneous supports and a design-independent load, the exact compliance sensitivity is \(s_c=-u^TK_{,c}u\), where \(K_{,c}=\partial K/\partial\tau_c\), summed over the affected substructures; interior equilibrium removes the design derivative of the exact extension [Giles & Pierce (2000)], [Haftka & Gürdal (1992)]." | P6 S1-S4 (conditions; formula; "For a parameter that affects several substructures, their contributions are summed."; interior equilibrium with both citations) |
| "The field-based sensitivity estimate \(\widetilde s_c=-\widehat u^TK_{,c}\widehat u\) uses the same stiffness derivative with the reconstructed field; over the eight corners these form the vectors \(\boldsymbol s\) and \(\widetilde{\boldsymbol s}\)." | P6 S5 and S7. "field-based sensitivity estimate" → "the sensitivity computed from the recovered field" (brief 3.1). Added S6, required by brief 3.1 ("define once that it ... ignores the design dependence of the network"): "It ignores the dependence of the recovery operator \(F\) on the design, and hence the design dependence of the network." Source support: old line 199 ("which also contains the design dependence of the extension") and Eq. (18). |
| "**Proposition 3 (sensitivity error).** Under (A1), (A2) and (D), at the same retained displacement:" | "**Proposition 4 (sensitivity error).** Under (A1), (A2) and (D), with both sensitivities evaluated at the same retained displacements:" |
| "(a) the field-based estimate differs from the exact sensitivity by a term linear and a term quadratic in the interior error," | (a), "the sensitivity \(\widetilde s_c\)", "interior displacement error" |
| Eq. (9) | Eq. (17), unchanged (the closing semicolon kept) |
| "(b) with \(\mathcal E=d^TKd\) ..., where the constants \(L\) and \(Q\) of Eq. (H.4) depend on the cell, the direction and the stiffness derivatives, so that the energy error controls the relative sensitivity error only at the order \(\sqrt\varepsilon\) (Appendix H.1)." | (b) S1-S2; "direction" → "test displacement" (brief 3.1) |
| "Interior equilibrium sets \((Ku)_I=0\) but generally leaves \((K_{,c}u)_I\ne0\), so the linear cross term survives, and a decrease of the energy error need not decrease the sensitivity error monotonically." | P7 S1-S2 |
| "Because thickening a corner enlarges the material domain, \(K_{,c}\succeq0\) under exact integration (Eq. (H.7)); the quadratic term of Eq. (9) is then nonpositive, while the linear term can have either sign (Appendix H.2)." | P7 S3-S4 |
| "The field-based estimate is not the derivative of the surrogate compliance, which also contains the design dependence of the extension." | P8 S1 ("the approximate compliance \(\widehat C\)", the name used in Section 2.3, U02) |
| "For a parameter affecting one substructure, differentiation at fixed retained DOFs gives" | P8 S2 |
| Eq. (10) | Eq. (18), unchanged |
| "where \(\widehat q\) is the assembled retained solution obtained with \(\widehat S\), \(r_I=\ldots\) and \(F_{I,c}=\ldots\)." | P9 S1 |
| "The second term vanishes for an equilibrated extension, but energy accuracy alone does not control it:" | P9 S2 |
| "whether Eq. (10) or the field-based estimate is the more accurate gradient depends on whether \(\|H_{,c}q\|_A\) is small compared with \(\|E_{I,c}q\|_A\), where \(H_{,c}\) and \(E_{I,c}=J_IE_{,c}\) are the design derivatives of the interior extension error and of the exact interior extension (Appendix H.2, Eq. (H.6))." | P9 S3 (with "(Eq. (H.6))") and S4 (definitions, with "(Appendix H.2)"); the chained reference was split, one per sentence. |
| "All sensitivities reported in this paper are the field-based estimates \(\widetilde s_c\), which estimate the exact sensitivity." | P9 S5-S6, kept as the unit instruction asks: "All sensitivities reported in this paper are \(\widetilde s_c=-\widehat u^TK_{,c}\widehat u\). They estimate the exact sensitivity and ignore the design dependence of the network." |
| "Both derivatives hold on intervals where the discrete choices of Appendix B.1 are fixed." | P9 S7 ("Both \(\widetilde s_c\) and the derivative of Eq. (18)") |
| "Across a switch of the discrete model the reference compliance itself can jump, and any jump that the surrogate adds is bounded by its own error level, since \(0\le C-\widehat C\le\beta C\) at every design (Eq. (7); Supplementary Note S4.3)." | P9 S8-S9. Eq. (7) → Eq. (15), now inline ("holds at every design by Eq. (15)"), so that the sentence has one parenthetical reference, "(Supplementary Note S4.3)". |
| "Appendix H specifies the numerical stiffness derivatives." | P9 S10 |

### Closing list of three conditions (old lines 212-216) → closing paragraph P10 (unit instruction)

| Old | Where it is now |
|---|---|
| "For the construction of Section 4, these results become three conditions on the extension:" | removed. The construction (old Section 4) is now Section 3 and precedes this section, so the list no longer leads into it. Replaced by the closing paragraph P10, which states what the relations imply for checking a learned condensation. |
| "1. its interior error energy must be small in the directions that the assembled solution selects, which Eq. (5) expresses through the interior residual (Section 4.2);" | removed from Section 4; the reasoning now sits where the method uses it. Section 3.3 P1 (U05) states that Eq. (7) [old (5)] expresses the energy error through the interior residual and that this motivates reducing the interior force imbalance; Section 3.5 (U05) trains on test displacements that arise from mechanical loading. Old references Eq. (5) → (7) and Section 4.2 → 3.2 are therefore not used here. |
| "2. a correction applied at deployment must not increase that energy, so that Eq. (4) places the corrected condensed stiffness between the uncorrected one and the exact Schur complement (Proposition 4);" | removed: duplicate of Proposition 2 and the paragraph after it in Section 3.4 (U05; \(S\preceq\widehat S\preceq\widehat S_{\rm net}\)) and of the Figure 2 caption (U03). Old references Eq. (4) → (6) and Proposition 4 → 2 are therefore not used here. |
| "3. the local sensitivity must be verified independently of the compliance (Sections 5.6 and 5.10)." | P10 S1 and S4 ("a learned condensation must be checked separately for the compliance and for the sensitivities"; "Section 5.6 checks both quantities in two-cell assemblies, and Section 5.10 checks them in thickness designs"). P10 S2-S3 give the two reasons, both restated from this section: Proposition 3 (a cell with a small energy fraction contributes little to the compliance error) and Proposition 4(b) (control only at the order \(\sqrt\varepsilon\)). |

No number was removed. Every number of the source that does not appear in the new text is a renumbered reference (listed in (b)), part of the assumption list moved to Section 2.4, or part of list items 1 and 2 removed above (old Eq. (5), Eq. (4), Section 4.2, Proposition 4, list labels "1." and "2."). The number words "three results" and "three conditions" disappear with the restructuring described above. The source contains no table, figure or statement about the Uncorrected continuation or the Smoothing-trained variant.

## (b) Cross-references renumbered (each occurrence mapped once)

| Old | New | Place |
|---|---|---|
| Section 3 (heading) | Section 4 | heading |
| Section 3.1 | Section 3.1 (old 3.1 → 3.1) | P0a S3 ("Proposition 1 in Section 3.1") |
| Section 3.2 | Section 4.1 | P0a S6; heading 4.1 |
| Section 3.3 | Section 4.2 | P0a S7; heading 4.2 |
| Propositions 1–3 | Propositions 3 and 4 (Proposition 1 is cited separately in P0a and P0b) | P0b S4 |
| Proposition 2 | Proposition 3 | label in 4.1; P10 S2 |
| Proposition 3 | Proposition 4 | label in 4.2; P10 S3 |
| Eq. (6) (tag and reference in old line 172) | Eq. (14) | tag; P4 S1 |
| Eq. (7) (tag and reference in old line 210) | Eq. (15) | tag; P9 S9 |
| Eq. (8) | Eq. (16) | tag |
| Eq. (9) (tag and reference in old line 197) | Eq. (17) | tag; P7 S4 |
| Eq. (10) (tag and reference in old line 210) | Eq. (18) | tag; P9 S3, S7 |
| Section 4 (old line 212) | removed with the list lead-in | see (a) |
| Eq. (5), Section 4.2 (item 1) | not used (would be Eq. (7), Section 3.2) | see (a) |
| Eq. (4), Proposition 4 (item 2) | not used (would be Eq. (6), Proposition 2) | see (a) |
| Sections 5.6 and 5.10 | unchanged | P10 S4 |
| Section 5.8 | unchanged | P4 S1 |
| Eq. (3) | unchanged | P1 S1 |
| Appendices B.1, C, C.1, C.2, H, H.1, H.2; Eqs. (H.4), (H.6), (H.7); Supplementary Note S4.3 | unchanged | 4.1, 4.2 |
| (none) | Section 2.4, assumption (A2) | P0a S2 (assumptions now in Section 2.4) |
| (none) | Section 2.4 | P0b S4 |
| (none) | Eq. (7) (old Eq. (5)) | P0a S5 (was "(Section 3.1)"); P0b S2 (residual form); P2 S6 (definition of \(\varepsilon_m\)) |

## SUPPLEMENT ADDITIONS

None. The source of this unit does not mention the Uncorrected continuation or the Smoothing-trained variant.

## (d) Open questions

1. Citations for the classical results. The sentence "Proposition 1, the residual form of the energy error in Eq. (7) and the self-adjoint compliance sensitivity are classical results [...]" now stands in the Section 4 introduction, one section after Proposition 1. This matches U03 open question 4. The assembler may move the part on Proposition 1 and Eq. (7) (Fraeijs de Veubeke, Toselli & Widlund, Becker & Rannacher) to the end of the paragraph after Eq. (7) in Section 3.1, and keep the self-adjoint compliance sensitivity (Haftka & Gürdal, Bendsøe & Sigmund) here.
2. Name of \(\widetilde s_c\). The main text calls it "the sensitivity computed from the recovered field" at its definition and "the sensitivity \(\widetilde s_c\)" afterwards, with "exact sensitivity" for \(s_c\). Appendix H (A3 unit, its open question 5) writes "sensitivity estimate". Both are compatible; the assembler may unify them if wanted.
3. "Approximate compliance" vs "surrogate compliance". This unit follows Section 2.3 (U02) and writes "the approximate compliance \(\widehat C\)". Supplementary Note S4.3 and Table ST14 keep "surrogate compliance" in their titles. The assembler may align the supplement wording.
4. Proposition 3 is now labelled "(energy-weighted compliance error)", replacing "share-weighted". The units for Sections 5.6 and 5.8 should cite it as Proposition 3 with Eq. (15) and use "energy fraction" for \(w_m\), including the Figure 11 caption ("energy shares \(w_m\)" in the old caption) and the old Section 5.6 text "Proposition 3(b)", which becomes "Proposition 4(b)".
5. Old condition 1 (interior error small in the test displacements selected by the assembled solution, expressed through the interior residual) is no longer stated in Section 4. It relies on Section 3.3 P1 (U05) keeping the sentence that Eq. (7) and Eq. (14) motivate reducing the interior force imbalance. If that sentence is cut in assembly, the reason for the correction would need a short restatement there.
6. "Learned condensation" in P10 (CN 学习型静力凝聚) is used once, for a static condensation whose interior displacements come from a network. If the assembler prefers, "a condensed stiffness built from network-predicted interior displacements" says the same in more words.
