# U02 notes: Section 2 (2, 2.1, 2.2, 2.3) and new Section 2.4 Assumptions

Source (old numbering): MANUSCRIPT_EN.md / MANUSCRIPT_CN.md lines 52–116 (old Section 2) and lines 121–126 (the sentence that introduces the assumption list and the list (A1)–(A3), (D), from the old Section 3 introduction).
Output: `rewrite_plain/sections/U02_EN.md` and `rewrite_plain/sections/U02_CN.md`, 74 lines each.

## 0. Format and alignment

- EN and CN are line-aligned: the same 74 lines hold the same heading, paragraph, display block or list item. Every EN paragraph and its CN paragraph contain the same inline mathematics (same multiset), the same numbers and the same citation links in the same order (checked by script).
- The four display blocks (Eqs. (1)–(3) and the unnumbered design problem) are byte-identical to the source in each language. The CN design problem keeps `\text{s.t.}` and the EN one keeps `\text{subject to}`, as in the source.
- Equation tags (1), (2), (3) are unchanged (brief section 5 maps them to themselves).
- Citations: [Burman et al. (2015)] twice and [Burman (2010)] once, as in the source.
- Length (EN words, display mathematics and headings excluded, inline mathematics counted as one word): old 1,127 (old Section 2: 1,034; assumption sentence and list: 93); new 1,467 (Section 2.4: 145). CN Chinese characters: old 1,682, new 2,141. The text is longer because long sentences were split, TPMS and the cut-plane elements are defined in plain words, the two parts of the retained set are stated separately, the operator \(F\) had to be introduced in Section 2.4, and (D) now names the main fixed discrete choices.
- Self-check results (after the verifier pass): no EN sentence above 30 words; no CN sentence above 80 characters; no banned word or old term of brief section 3.1 in either file. "Schur complement" / "Schur 补" appears once, at the definition of \(S\), as the term map requires.

## (a) Sentences and numbers removed, moved or restated

### Old Section 2 introduction (old line 54) → new Section 2 introduction

| Old text | New text / where it is now |
|---|---|
| "The cells are discretised by a stabilised cut finite element method (CutFEM) [Burman et al. (2015)] on a fixed Cartesian background mesh." | kept, sentence 1 |
| "The walls and the cut are not meshed: they decide which background elements are active and enter the stiffness through integrals over the material part of each element." | split into sentences 2–3 ("The walls and the cut are not meshed." without a generalising "In CutFEM,") |
| "A change of wall thickness or of the cut therefore needs no remeshing, the thickness sensitivity follows from the derivatives of these integrals (Appendix H), and every cell lives on the same background grid, on which the network of Section 4.2 places its fixed latent grids." | sentences 4–5. "latent grids" → "grid hierarchy" (brief 3.1); Section 4.2 → Section 3.2 |
| "The price is that the active elements, and with them the retained set, change with the geometry, which Sections 3 and 4 must accommodate; their relations use only ..." | sentences 6–8. "The price is that" removed (banned flourish) and replaced by "However,"; "retained set" → "set of retained degrees of freedom". "Sections 3 and 4" kept (old 3 → 4 and old 4 → 3, so the pair is unchanged). |

### Old 2.1 (old lines 58–71) → new 2.1 "Geometry and finite element model"

| Old text | New text |
|---|---|
| "We consider three-dimensional thin-walled TPMS cells in the reference box \(\mathcal B=[0,1]^3\)." | "This paper considers three-dimensional thin-walled cells based on triply periodic minimal surfaces (TPMS). Each cell occupies the reference box \(\mathcal B=[0,1]^3\), whose six faces are called the cell faces." TPMS defined at first use in the body (brief 3.2); "cell faces" introduced because the term replaces "box faces". CN keeps 参考立方体, as in the old CN text and the Appendix F rewrite. |
| "Schwarz-P-type trigonometric level-set field, a spatially varying band parameter" | "level-set function"; CN 带参数 → 带宽参数 |
| "... set the width of the implicit band and with it ...; we call them corner thickness parameters, and derivatives with respect to them thickness sensitivities." | split into three sentences; "we call" → "They are called" (CN 本文称之为) |
| "has unit normal \(\boldsymbol n\), pointing away from the retained material" | "pointing away from the material that remains after the cut" (avoids a clash with "retained DOFs") |
| "which is certified by interval enclosures of the level-set and band functions over the element (Appendix A.1)" | "This is certified by interval enclosures ..." ("certified" kept from the source; CN 严格确认) |
| "Integration over the material domain supplies the bulk stiffness" | "gives the elastic stiffness of the walls" |
| "ghost-penalty stabilisation [Burman (2010)] couples neighbouring active elements to control small-cut effects [Burman et al. (2015)]" | two sentences: the ghost penalty "adds terms that couple neighbouring active elements"; these terms "control the effect of small cuts, that is, of active elements that contain very little material". "adds terms" follows Eq. (A.1). |
| "the reference in this work is the equilibrium of this discrete system: all errors are measured against it, not against the continuum problem" | two sentences, same content |
| "Figure 1 ...; Table 1 (Section 5.1) ..., and Appendix A gives the integration and assembly details." | kept, split into two sentences |

### Old 2.2 (old lines 75–89) → new 2.2 "Retained degrees of freedom and static condensation"

| Old text | New text |
|---|---|
| "the retained DOFs, collected in the set \(P\), are the DOFs of the box-face nodes of every positive-area material patch on the faces of the cell box and all displacement DOFs of active elements carrying a positive-area cut-surface patch (Appendix A.1)" | split: the set has two parts (Appendix A.1); one part (cell-face nodes of every material patch of positive area on the cell faces); the other part (all displacement DOFs of active elements carrying a cut-surface patch of positive area) |
| "These elements, the layer of active elements intersected by the cut plane, form the cut band: although some of its nodes lie away from the cut plane, their basis functions contribute ..." | "cut band" → "cut-plane elements" (brief 3.1), two sentences |
| "This full retained space, with both intercell and exterior-boundary DOFs, is used throughout prediction and assembly." | "The complete set of retained DOFs is used in both the network prediction and the assembly. It includes the DOFs coupled to neighbouring cells and the DOFs on the exterior boundary." |
| "Let \(I\) denote the remaining interior DOFs, \(p=|P|\), \(i=|I|\), and \(n_a=p+i\)." | two sentences, same symbols |
| "The selectors \(J_P\) and \(J_I\) extract the two sets from the active displacement vector" | "selection matrices"; the vector is named \(u\) (it is the \(u\) of \(q=J_Pu\)) |
| "In the displayed \((P,I)\) ordering, static condensation gives" | "With the DOFs ordered as \((P,I)\), static condensation gives the exact displacement recovery \(E\) and the exact condensed stiffness \(S\), that is, the Schur complement:" Names added per brief 3.1 (exact recovery \(E\); exact condensed stiffness, "the Schur complement" said once). |
| "... whose restriction \(R_P=J_PR\) has rank six: the geometry generator accepts only ..." | colon replaced by "Two features of the discrete model ensure these properties (Appendix A.1)." followed by the two features (face-connected active elements; coupling of material connected only through partially filled elements through shared basis functions and the ghost penalty). Same strength as the source; the qualification of Appendix A.1 for numerically integrated element matrices stays in the appendix and in (A1). |
| "The interior block \(A\) is then positive definite, since an interior field of zero energy would be ..., so prescribing \(q\) removes every interior zero-energy motion." | "The interior block \(A\) is then positive definite, because an interior field with zero energy would be a rigid-body motion that vanishes on the retained DOFs. Prescribing \(q\) therefore removes every interior motion with zero energy." (causal link kept) |
| "The equilibrium displacement extension \(u=Eq\) uniquely minimises the discrete energy over \(J_Pu=q\), and \(Sq\) is its work-conjugate retained force." | "The exact displacement recovery \(u=Eq\) is the unique minimiser of the discrete energy subject to \(J_Pu=q\), and \(Sq\) is the vector of the corresponding nodal forces on the retained DOFs." (brief 3.1: extension → recovery; work-conjugate force → corresponding nodal forces) |

### Old 2.3 (old lines 93–115) → new 2.3 "Assembly, compliance and the design problem"

| Old text | New text |
|---|---|
| "Coincident box-face DOFs, identified by their background-grid position, are shared; retained cut-band DOFs outside the box faces remain local to their substructure." | two sentences; "cell-face DOFs"; "Retained DOFs of the cut-plane elements that do not lie on the cell faces belong only to their own substructure." |
| "Because the thickness field on a shared face depends only on its four corner values and \(\phi\) is periodic, the material patches ... coincide on every face not intersected by a cut; where a cut removes material from one side only, ..." | three sentences, same content; "four" kept |
| "The reference and approximate assembled systems, which use the same maps \(B_m\), are" | "The reference and the approximate assembled systems use the same maps \(B_m\):" ("Boolean assembly map" kept, as in the Appendix B.1 rewrite) |
| "Here \(\widehat S_m\) is the approximate condensed stiffness of Section 4 ..." | Section 4 → Section 3 |
| "Local condensation commutes with assembly because the interior sets are disjoint" | "Because the interior DOF sets of different cells are disjoint, condensing each cell before assembly gives the same matrix as condensing the assembled fine-scale system." (plain statement of the same fact) |
| "and the supported reference matrix \(\mathbb K\) is assumed positive definite" | separate sentence |
| "Compliance is \(C=f_g^TU\), with \(\widehat C=f_g^T\widehat U\) and relative error \(e_C=...\)." | same content |
| "measured by the relative directional energy error of Section 3.1" | "relative energy error of Section 3.1" (brief 3.1; old 3.1 → new 3.1) |
| "Thickness sensitivities use the eight-component vector of Section 3.3, with relative Euclidean error \(e_s=...\)" | two sentences; Section 3.3 → Section 4.2; "With \(\widetilde{\boldsymbol s}\) computed from the approximate displacements, their relative error in the Euclidean norm is ...". The meaning of \(\widetilde{\boldsymbol s}\) is taken from old 3.3 (new 4.2), where it is defined. |
| "In design, the corner thickness parameters at the lattice vertices, collected in \(\boldsymbol\tau_g\) and mapped ..., solve" | "In design, the corner thickness parameters at the lattice vertices are the design variables. They are collected in ... The design problem is" |
| "together with limits on each cell's corner span and gradient norm that keep the cells in the trained domain (Section 5.10)" | "In addition, the corner span and the gradient norm of each cell are bounded to keep every cell within the range of geometries used in training (Section 5.10)." |
| "Its gradient collects ... (Section 3.3)" | "The compliance gradient collects ... (Section 4.2)" |
| "Of the four properties named in Section 1, two can now be made precise." | **removed** (unit instruction: the new introduction no longer lists four properties). The reference to "Section 1" is therefore gone. |
| "\(\widehat S_m\) must be symmetric and positive semidefinite with the retained rigid-body modes as its only null space, and the supported \(\widehat{\mathbb K}\) of Eq. (3) must be positive definite, which follows from \(\widehat S_m\succeq S_m\) and \(\mathbb K\succ0\) (Appendix B.2); and its error must be traceable to \(e_C\) and \(e_s\), which weight it differently." | restated as three requirements in four sentences: symmetric positive semidefinite with the rigid-body modes \(R_P\) of the retained DOFs as only null space; supported \(\widehat{\mathbb K}\) positive definite, following from \(\widehat S_m\succeq S_m\) and \(\mathbb K\succ0\) (Appendix B.2); error traceable to the compliance error \(e_C\) and the sensitivity error \(e_s\), which weight it differently |
| "Section 3 derives what any admissible extension gives towards these two; Section 4 constructs a condensed stiffness that has all four." | "Section 3 constructs a condensed stiffness that meets these requirements and needs no interior factorisation for a new geometry, even when the cut changes the discrete space. This condensed stiffness can also be improved at deployment without retraining. Sections 3.1 and 4 derive what any recovery operator that satisfies the assumptions of Section 2.4 provides towards these requirements." The two old properties that are not among the three requirements are named with their old qualifiers ("even when the cut changes the discrete space", "at deployment"; old Section 1), so the content of "has all four" is kept. Old 4 → 3; old Section 3 → Sections 3.1 and 4, because its content now lives there (old 3.1 → 3.1, old 3 intro/3.2/3.3 → 4). |

### Old Section 3 introduction (old lines 119–126) → new Section 2.4 "Assumptions"

| Old text | New text / where it is now |
|---|---|
| Old line 119 (whole paragraph): "With the retained DOFs fixed, the local approximation lies in the interior field. This section follows the error ... three results. First, ... Second, ... Third, ... The relations are stated for a single substructure and apply to every cell of an assembly. The Ritz identity, the residual form and the self-adjoint compliance sensitivity are classical [Fraeijs de Veubeke (1965)], [Toselli & Widlund (2005)], [Becker & Rannacher (2001)], [Haftka & Gürdal (1992)], [Bendsøe & Sigmund (2004)]; the share-weighted bound and the sensitivity relations are derived here for approximate extensions." | **not in this unit.** It belongs to unit U06 (new Section 4 introduction), as the unit instructions state. Nothing from line 119 was used here. |
| Old line 121: "The results are stated as Propositions 1–3 under the following assumptions, which Appendix B.1 states in full." | "Propositions 1, 3 and 4 are stated under the following assumptions, which Appendix B.1 states in full. Proposition 2 also uses (A1) and (A2)." Strict mapping old 1–3 → new 1, 3, 4; the second sentence records that new Proposition 2 (old 4: "Let the network extension \(\widehat E\) satisfy (A2) ... Then, under (A1) ...") also uses (A1) and (A2). |
| (new) | "Sections 3 and 4 consider an approximate displacement recovery operator \(F\), which maps the retained displacements \(q\) of a cell to the displacements \(Fq\) of all its active DOFs." Added because Section 2.4 now precedes new Section 3.1, where \(F\) was introduced in the old order (old 3.1: "Any admissible linear extension \(F\), with \(J_PF=J_PE=I_p\) ..."). No new claim. |
| (A1) | unchanged in content; "cell stiffness" → "cell stiffness matrix" |
| (A2) "The extension \(F\) is linear and admissible, \(J_PF=I_p\), and reproduces rigid-body motion, \(FR_P=R\)." | "The recovery operator \(F\) is linear and reproduces the retained displacements, \(J_PF=I_p\). It also reproduces rigid-body motions, \(FR_P=R\)." (brief 3.1: extension → recovery operator; admissible → reproduces the retained displacements) |
| (A3) "Cells share only retained degrees of freedom, and the supported assembled stiffness \(\mathbb K\) is positive definite (Section 2.3)." | same content ("retained DOFs", "stiffness matrix") |
| (D) "On a design interval, the discrete choices listed in Appendix B.1 are fixed." | kept, plus one sentence that names the main choices from the Appendix B.1 list: "They include the active elements, the ghost-penalty faces, the retained and interior DOF sets, the assembly maps with the supports, and the load." The remaining items of that list (network node indicators, weak-region stencils, coarse-column screen, coarse-factor shift) stay in Appendix B.1 only. |

### Numbers

Old EN prose numbers (outside citations and mathematics): section and appendix labels 2, 2.1, 2.2 (×2), 2.3 (×2), 3, 4, 3.1, 3.3 (×2), 4.2, 5.1, 5.10, A.1 (×3), B.1 (×2), B.2, Figure 1, Table 1, Eq. (3), Section 1, Propositions 1–3, (A1)–(A3). The words "eight", "six", "four", "two" are kept as words.

- All old numbers appear in the new EN and CN text, except: "3.3" (both occurrences renumbered to 4.2), "4.2" (renumbered to 3.2), "Section 4" / "Section 3" in the sense of old numbering (swapped, see (b)), and "Section 1" (removed with the sentence on the four properties).
- New numbers not in the source: "2.4" (new heading; cross-reference "assumptions of Section 2.4"), "3.2" (renumbered from 4.2), "3.1" in "Sections 3.1 and 4" (old Section 3 split), "Propositions 1, 3 and 4" and "Proposition 2" (renumbered old 1–3 and old 4), and one more "Sections 3 and 4" (in the sentence that introduces \(F\)).
- The unit contains no measured data. Eq. (1)–(3) tags are unchanged.
- EN and CN contain the same numbers, line by line.

### Inline mathematics added (no new symbols)

\(u\) (to name the active displacement vector), \(E\) and \(S\) (named before Eq. (2)), \(R_P\) (in the requirement on the null space; it is defined in 2.2), \(S_m\) (in "replaces \(S_m\)"), \(\widetilde{\boldsymbol s}\) (named in the definition of \(e_s\), already in the source formula), \(Fq\) (in the definition of \(F\) in 2.4). All symbols are defined in the source.

## (b) Cross-references renumbered

| Old | New | Place |
|---|---|---|
| Section 4.2 (network, fixed latent grids) | Section 3.2 | Section 2 introduction |
| Sections 3 and 4 (must accommodate the changing retained set) | Sections 3 and 4 (old 3 → 4, old 4 → 3; the pair is unchanged) | Section 2 introduction |
| Section 4 (approximate condensed stiffness \(\widehat S_m\)) | Section 3 | 2.3, after Eq. (3) |
| Section 3.1 (relative energy error) | Section 3.1 (old 3.1 → new 3.1) | 2.3 |
| Section 3.3 (eight-component sensitivity vector), twice | Section 4.2, twice | 2.3 |
| Section 1 ("four properties named in Section 1") | removed | 2.3, last paragraph |
| Section 3 ("derives what any admissible extension gives") | Sections 3.1 and 4 (old 3.1 → 3.1; old 3 intro, 3.2, 3.3 → 4) | 2.3, last paragraph |
| Section 4 ("constructs a condensed stiffness") | Section 3 | 2.3, last paragraph |
| (none) | Section 2.4 (added) | 2.3, last paragraph |
| Propositions 1–3 | Propositions 1, 3 and 4 | 2.4. Old Propositions 1, 2, 3 map to new 1, 3, 4. A following sentence states that new Proposition 2 (old Proposition 4) also uses (A1) and (A2). |
| Section 2.2, Section 2.3 (in (A1), (A3)) | unchanged | 2.4 |
| Section 5.1, Section 5.10, Figure 1, Table 1, Eq. (3) | unchanged | 2.1, 2.3 |
| Appendix A, A.1, B.1, B.2, H | unchanged (appendix labels keep their names) | throughout |

## SUPPLEMENT ADDITIONS

None. The source of this unit does not mention the Uncorrected continuation or the Smoothing-trained variant.

## (d) Open questions

1. **Range of propositions in 2.4.** Resolved by the verifier: the text now reads "Propositions 1, 3 and 4 are stated under the following assumptions ... Proposition 2 also uses (A1) and (A2)."
2. **Old Section 3 introduction.** The rest of old line 119 (the three results and the sentence on the classical literature with its five citations) belongs to U06, the new Section 4 introduction. U06 should also replace any "the following assumptions" phrasing by a pointer to Section 2.4.
3. **Definition of \(F\).** Section 2.4 now introduces \(F\) in one sentence. The unit that writes new Section 3.1 (old 4.1 plus old 3.1) may refer back to Section 2.4 instead of introducing \(F\) again, or keep its own definition; both are consistent.
4. **"Schur complement" said once.** It is said here, at the definition of \(S\) before Eq. (2). U01 (Section 1.1, paragraph 4) also writes "the exact condensed stiffness, that is, the Schur complement" (its open question 1). Keeping the parenthetical at the definition in Section 2.2 is the natural choice; the assembler should drop one of the two.
5. **"retained (master)"** is not repeated here, because U00 (introduction, paragraph 2) already defines "retained (master) degrees of freedom". "DOFs" is introduced as an abbreviation in 2.1, as in the source.
6. **TPMS** is spelled out here ("triply periodic minimal surfaces (TPMS)"); the title and keywords (U00) use the abbreviation only. If another unit spells it out earlier in the body, keep only the first expansion.
7. **"corner span" and "gradient norm"** are not defined in the main text before Section 5.10 and Table ST02, as in the source. A one-line definition could be added by the unit that rewrites Section 5.10 if the advisor asks.
8. **"Reference box"** is kept in EN (consistent with the Appendix F rewrite and Table 1, "Unit box"); CN uses 参考立方体, as the old CN text and the Appendix F rewrite do.

## VERIFIER

Checks run: sentence-by-sentence comparison of old EN/CN lines 52–127 with the new files; Appendix A.1, B.1 and B.2 opened to confirm the destinations named here; old Section 1 (line 17) opened to confirm the four properties; old Proposition 4 (old 4.4) opened to confirm that it uses (A1) and (A2); Python number diff old EN vs new EN (only differences: renumbered cross-references 3.3 → 4.2 twice, 4.2 → 3.2, and the added 2.4, 3.1, Propositions 1, 3, 4 and 2) and new EN vs new CN (identical, line by line, together with inline mathematics and citations); display blocks of Eqs. (1)–(3) and the design problem byte-identical to the source in both languages; 74 aligned lines in both languages; banned-word and old-term scan; EN sentence length (none above 30 words). No figures or tables in this unit. No PIML, priority, machine-load or training-cost statement.

Changes made (EN and CN in parallel):

1. Section 2 introduction: removed "In CutFEM," (it generalised a statement about this discretisation to CutFEM in general); split the 31-word last sentence into two.
2. 2.1: "This is decided with interval enclosures" → "This is certified by interval enclosures" (the source says "certified"; "decided" weakened the claim). CN 严格确认.
3. 2.2: the source justifies the rigid-body null space and the rank of \(R_P\) by two facts (face-connected active elements; coupling of material connected only through partially filled elements). The writer's version attached "because" to the first fact only and left the second as an unconnected statement. Now: "Two features of the discrete model ensure these properties (Appendix A.1)." followed by both features.
4. 2.2: restored the causal link "The interior block \(A\) is then positive definite, because an interior field with zero energy would be a rigid-body motion ..." (the writer had dropped "since"); "\(Sq\) contains the corresponding nodal forces" → "\(Sq\) is the vector of the corresponding nodal forces".
5. 2.3: 32-word sentence on the corner span and gradient norm shortened ("bounded to keep every cell within the range of geometries used in training").
6. 2.3, last paragraph: restored the qualifiers of the old four properties that the writer had dropped, "even when the cut changes the discrete space" and "at deployment" (old Section 1, line 17). Null-space requirement written as "its null space must consist only of the rigid-body modes of the retained DOFs, the columns of \(R_P\)" (the writer's "rigid-body modes \(R_P\)" named a matrix as a set of modes). The pointer to the general analysis was narrowed by the writer to "how the error ... reaches \(e_C\) and \(e_s\)" in Section 4; the source says Section 3 derives what any admissible extension gives towards both precise properties, and Proposition 1 (\(\widehat S\succeq S\), now Section 3.1) is part of that. Now: "Sections 3.1 and 4 derive what any recovery operator that satisfies the assumptions of Section 2.4 provides towards these requirements."
7. 2.4: "Propositions 1–4 use the following assumptions" → "Propositions 1, 3 and 4 are stated under the following assumptions, which Appendix B.1 states in full. Proposition 2 also uses (A1) and (A2)." This is the strict map of old 1–3 (brief section 5) and keeps the correct information that new Proposition 2 (old 4) uses (A1) and (A2).
8. CN style: 对……进行凝聚 → 先装配细尺度系统再凝聚所得的矩阵; 汇集几何、材料与离散参数 → 表示几何、材料与离散化参数的集合; 背离切割后剩余的材料 → 由切割后剩余的材料指向外侧; 按 (P,I) 排列自由度 → 将自由度按 (P,I) 排序后; 并称对它们的导数 → 并将关于这些参数的导数称为; 网络训练所覆盖的 → 训练所用的; vector parenthetical in 2.1 made a single clause; (A2) "所恢复的位移在主自由度上等于给定位移"; last sentence of 2.3 rewritten to avoid a long pre-nominal chain.
9. NOTES rows (a), (b), Numbers and open question 1 updated to the final text.

Unresolved / for the assembler:

- "Schur complement" is said here (2.2) and in U01 (1.1); keep only one (open question 4).
- U06 (new Section 4 introduction) points to "assumption (A2) of Section 2.4"; consistent with this unit.
- The statement in 2.2 that the null space consists of the rigid-body modes is unconditional, as in the source; Appendix A.1 adds that with numerically integrated element matrices this is an assumption. Not changed (the source has the same strength), but the advisor may ask.
