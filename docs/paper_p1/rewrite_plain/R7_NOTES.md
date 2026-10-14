# R7 notes: section-by-section revision from the author's annotations

## Working rules (author, 11 October 2026)

- The English text is the one submitted. It must carry everything the Chinese text says; never drop content from the
  English to meet the 250-word limit. Tighten the English wording first; if that is not enough, agree a cut with the
  author and make it in both languages.
- Do not over-disambiguate. Fix real errors, but do not rewrite every clause so that it admits only one reading.
- The author annotates the Chinese text section by section, starting with the abstract. Each revised passage is shown
  in Chinese and English before it enters the paper.
- During this revision the PDFs, the advisor set and the Word files are not rebuilt after each change. Edit the unit
  files in `sections/`, run `python3 assemble_r5.py --write`, then commit and push at once, so that nothing is lost
  when the container is reset. Rebuild the outputs once, when the author asks.
- The full texts of the PIML papers (Huang et al. 2023, 2024; Jiang et al. 2026; Guo et al. 2026a, 2026b; Zhang et al.
  2026) are kept in the private repository `fanzhirui1004-source/piml-papers`. This repository is public, so they are
  never committed here.
- The authors are not in the group that developed the learned substructure (PIML) methods. The abstract does not
  comment on the limits of those methods: such a sentence would read as provocative. The neutral comparison in
  Introduction ¶3 and Section 1.1 stays as it is.

## Abstract (committed; wording settled with the author)

1. Opening: thickness optimisation of cut lattices needs both fine-scale accuracy and a fast analysis at every design
   iteration. Do not open with cut cells carrying supports and loads (in a fine-scale model they are ordinary cells).
2. Fine-scale models: one short sentence on their cost; no degree-of-freedom count.
3. Homogenisation: periodic boundary conditions on a unit cell; the assumption fails for two parallel reasons, lack of
   scale separation and cells cut by the trimming plane. Say that the assumption no longer holds, not merely that the
   accuracy is insufficient ("这一假设在不满足尺度分离或胞元被切割时不再成立").
4. Static condensation in one sentence: reduces the global system without loss of accuracy, but every interior
   stiffness matrix is refactorised at every design iteration.
5. "本文提出 NICE 方法以避免上述分解", then one sentence on how it works (all face and cut-plane-element degrees of
   freedom retained; network with the cell geometry as input; two-grid cycle on the force imbalance; condensed
   stiffness from the strain energy of the corrected field).
6. Properties: upper bound on the exact condensed stiffness, error of second order in the interior displacement
   error, independent of the network accuracy. The conditions (linear in the retained displacements, reproduces them
   and rigid-body motions) and the guarantee that the correction does not increase the error (Proposition 2) stay in
   the body: the author found the conditional statement unclear in an abstract.
7. Sensitivities: their error is of first order in the interior displacement error, so the accuracy of the compliance
   does not ensure that of the sensitivities; both are trained and checked. No "然而" before this sentence.
8. Formal register throughout (no "这样得到的", "只会", "这两点"). EN at most 250 words; numbers as before.
9. Terminology: the fully resolved model is "细尺度有限元模型" (EN fine-scale), as in multiscale literature; the
   supplement's "细观尺度/细观模型" (S6) were unified to "细尺度/细尺度模型". The homogenised side stays "宏观".
10. The current abstract is the author's own revision of 11 October. Suggested edits kept: "该假设" before
   "在不满足尺度分离……时不再成立" and "通过两重网格修正减小内部不平衡力". Rejected: inserting "上述性质" before
   "不依赖于网络的训练精度" (the author: making every clause fully unambiguous reads as machine-written).

## Introduction (proposal, awaiting the author)

The introductions of the PIML papers (Huang et al. 2023, 2024; Jiang et al. 2026; Guo et al. 2026a; Zhang et al. 2026)
give their own results only in the "this paper" paragraph and do not expand the theory there. Proposed changes:

1. Move the comparison in ¶1 (homogenisation 27%, NICE 0.011%) to the end of the NICE paragraph.
2. In ¶2–¶3 (static condensation, learned component models) tighten the wording only.
3. Merge ¶4–¶6 (method, properties, compliance and sensitivity; about 1,300 Chinese characters) into one paragraph of
   about 500 characters, and leave the details to the contribution list and Sections 3–4, which repeat them now.

## Introduction, second round (author's annotations of 11 October, applied in 36bc53c)

1. ¶1 speaks of graded lattices only; trimming to a part's shape is introduced in ¶2, where the periodic-cell assumption
   of homogenisation fails. "几何细节" replaced: the analysis must compute the response of every cell accurately, and
   accuracy calls for resolving the deformation at the cell scale.
2. ¶3 states the cost of the interior factorisations: interior DOFs several times the retained ones (Figure 1b),
   a sparse Cholesky factorisation per cell and a dense condensed matrix (4.2 GiB for the median cell, factors of
   0.60–9.4 GiB for the deployed cells of Section 5.9), repeated for every cell at every iteration; on the lattices
   of Section 5.9 conventional exact condensation is slower than the direct solve. Open: the author may keep only
   the comparison with the direct solve and drop the two memory figures.
3. ¶4: no unmatched "一类模型". First sentence covers elements and substructures; Capuano & Rimoli (2019) and learned
   substructures are given as two examples. Last sentence: for cut cells, supports and loads act on all nodes of the
   cut-plane elements, so all their DOFs are retained and a reduced boundary description is unsuitable; the number
   of retained DOFs is large and varies with the geometry, so one network output per retained DOF is impractical.
4. ¶5 last sentence: the assembled system is solved by preconditioned conjugate gradients, which need only products
   of the condensed stiffness with vectors, so neither the recovery map nor the condensed stiffness is formed.
5. ¶6 starts from the conditions (linear, reproduces retained displacements and rigid-body motions) and then states
   the properties; "退化为精确静力凝聚" removed ("内部位移精确时二者相同").
6. ¶7: the statement that compliance accuracy does not ensure sensitivity accuracy is no longer restricted to
   approximate static condensation used in design; it holds whenever both are computed from an approximate
   displacement field.
7. Figure 1 caption: "壁面半透明" → "材料表面以半透明绘出".

## Sections 1.1–7: formal register (applied, commits daa652b to 78d56e2)

- No 前者/后者, 前一项/后一项 or 它; explicit nouns or 其一/其二 instead. Explanatory "，即" asides removed.
- Chains of short sentences merged with semicolons or colons in Chinese; in English, sentences above about 50 words
  were split again at the semicolons (the English is the submitted text).
- Terminology: 二阶小量 for the order of the stiffness error (Section 3.1, Section 4, conclusions); 上界 instead of
  "在矩阵意义上不小于" in the conclusions; 一阶项 for the sensitivity error; the two terms of Eq. (17) are 线性项 and
  二次项, as in Section 5.4 and the English "linear" and "quadratic".
- 壁面, 壁截面, 最厚的壁 and 胞元壁 replaced by 材料表面, 材料截面, 厚度最大的部位 and "全部胞元" (Figures 1, 4 and
  13, Sections 5.6, 5.10 and 7); 薄壁 kept.
- The conclusions state the failure of homogenisation as the abstract does (periodic boundary conditions fail without
  scale separation or when cells are cut).
- Numbers, links and equations are unchanged except where a merged sentence drops a repeated symbol; every unit was
  checked with `CHECK_BASE=HEAD python3 check_r6.py Uxx` and the assembler.
- Reading copies: R7_intro_applied.md (introduction, CN and EN) and R7_body_CN.md (Sections 1.1–7, CN).

## Word annotations on the abstract and Section 1 (12 October), applied after discussion

- Abstract: the fine-scale sentence is the author's ("细尺度有限元模型须离散全部胞元，每次设计迭代都需重新组装和求解
  大规模方程组"); the assumption is the subject of the homogenisation sentence ("所采用的周期性边界条件假设…不再成立");
  "利用" before the network. The English stays at 250 words by small trims elsewhere, with no content dropped.
- Introduction ¶2 states the two premises of homogenisation (scale separation; complete, periodic cells) and how each
  fails, instead of repeating the abstract.
- ¶3: interior DOFs "约为其 6 倍（按中位数计）" (median ratio 6.25 over the 80 validation geometries, 6.5 for the
  cell of Figure 1b); times stated as 16–20 min for conventional exact condensation against 14–17 min for the direct
  solve on the eight-cell lattices. Avoid "甚至…还" and lists of raw numbers.
- ¶4 ends with what cut cells require (all DOFs of the cut-plane elements retained; size and composition of the
  retained set change with the geometry). Do not explain why other routes are not taken; say why ours is taken.
- ¶5: "利用"; "随后保持主自由度位移不变，施加一次……两重网格循环".
- ¶6 states the network properties directly (no "若…则…"). The guarantee of the correction is explained in words:
  Chebyshev smoothing is built for an eigenvalue interval of the diagonally scaled interior stiffness matrix and
  amplifies no error component when the upper end is not below the actual largest eigenvalue; the exact coarse solve
  is an energy projection. No reference to "命题 2" in the introduction.
- ¶7: the sentence "这一现象并不限于静力凝聚…" is deleted; "柔度的精度不能保证灵敏度的精度" is folded into the
  preceding consequence so that "类似现象" has a clear referent.
- Contributions: (1) retained DOFs: loads and supports act directly on them, and the boundary-reduction error is
  avoided (Section 5.7); (2) network for varying retained sets, one network for the cuts within the training range,
  energy properties carry over (the classical property is not claimed as new); (3) correction: what it does, the
  guarantee under the conditions above, and 6.89% → 0.0965% for the base network.
- Figure captions (Figures 1–16): only what is shown and how to read it; no reconstruction or rendering details, no
  definitions or conclusions repeated from the text.
- Section 1.1 rewritten by research line: condensation and component reduction (with the shared energy property),
  lattices and unfitted discretisations, learned component models (no evaluation such as "最为系统", no lineage in
  detail, no item-by-item comparison), solvers and operator learning, and one closing paragraph that positions NICE
  (the only comparison). Not "与 NICE 最接近的是…": prior work is described in its own terms.

## Section 2: separate subsection on the cut finite element discretisation (author's request, 12 October)

- 2.1 Geometry; 2.2 Cut finite element discretisation (new); 2.3 Retained DOFs and static condensation; 2.4 Assembly,
  compliance and design problem; 2.5 Assumptions. Every cross-reference to the old 2.2–2.4 in the main text,
  appendices and supplement was shifted by one; the reference to the discrete reference model in Section 5.2 now
  points to 2.2.
- 2.2 states the background mesh and active elements, the stiffness as an unnumbered display (elastic form on the
  material part plus the ghost penalty \(g_h\), moved from Appendix A.1, which now points to 2.2), the role of the
  ghost penalty for small cuts, the positive-weight quadrature, and three consequences (no remeshing; varying active
  elements, ghost faces and retained DOFs; discrete reference solution). The display is unnumbered so that the
  equation numbers of the paper do not change.
- The Section 2 opening is a one-sentence outline; its former content is in 2.2.

## Section 2: Word annotations on Sections 2.1–2.5 (12 October, second file)

Twenty items were discussed and all are applied (CN and EN).

- 2.1: the cell geometry is "defined in" the reference box (not "occupies"); the material domain is introduced with
  one lead-in sentence (no "共同"). \(\tau(x)\) is the "thickness parameter field" (no "带宽", "band"); "a larger
  \(\tau\) gives a thicker material layer". The corner values determine \(\tau(x)\) and the material domain
  uniquely; the continuity of the thickness across shared faces and the coincidence of the material cross-sections
  on uncut shared faces are stated here (moved from 2.4). Thickness sensitivities are the derivatives of the
  compliance. The corner numbering \(c=1,\ldots,8\) (used nowhere else) and the sentence on uncut cells are deleted;
  the cut surface is defined as the material surface created by the cut plane.
- 2.2: \(Q_2\) is explained as 27-node quadratic Lagrange hexahedral elements; \(h=1/n\) is defined where the ghost
  penalty uses it; "nonzero volume" replaces "positive measure" (also in A.1); no "strictly". The positive
  quadrature weights are explained by their role (every quadrature point contributes a positive semidefinite
  matrix). The closing paragraph no longer lists "three consequences": no remeshing; one grid hierarchy for every
  geometry, with geometry-dependent template coefficients for the varying sets; the discrete solution is the
  reference for the errors of NICE, and its own discretisation error is assessed in Section 5.2 and Supplementary
  Note S1.
- 2.3: cut-plane elements are defined directly (active elements crossed by the cut plane in which the cut surface
  has nonzero area); the reason for retaining all their DOFs is stated (the cut plane passes through element
  interiors, so supports and loads on the cut surface act on retained DOFs only).
- Terminology elsewhere: "隐式带"/"implicit band" → "材料层"/"material layer" (A.1, Table 1, S1); "带参数"/"band
  parameter" → "角点厚度参数"/"corner thickness parameter" (Appendix H); "正面积材料片"/"positive-area patch" →
  element faces that contain material (2.3, A.1, Table 1).
- New EN sentences above 40 words were split at their semicolons.
- Item 2: the six faces of the reference cube are "胞元边界面" in CN (EN unchanged: cell faces), as in the usual
  "单胞边界" of homogenisation; "胞元表面" was read as the surface of the material. All 65 occurrences were replaced,
  together with the remaining box-face uses of "表面" (A3, S2, S3, S4, Figure 11 caption). The material part on a
  boundary face is not given a new name ("边界面上的材料截面"); the cut surface is "切割面". This reverses the R5
  replacement "胞元边界面 → 胞元表面", which had followed the EN change "box face → cell face"; the assembler check now
  flags "胞元表面" instead.
- Items 19–20: 2.4 starts with its main point and Eq. (3); the face-matching argument is in 2.1 and the one-sided-cut
  rule in Appendix A.1; "assume \(\mathbb K\succ0\)" is stated once, in 2.5. The three requirements moved to 2.5,
  now "Requirements and standing conditions" ("近似凝聚的要求与前提条件"); the third requirement reads "the effect of
  the error on compliance and sensitivities must be quantifiable". Each condition states where it comes from
  (discretisation, construction of NICE, construction of the assembly, or assumption), and one sentence lists the
  four genuine assumptions. "Assumption (A1)" etc. became "condition (A1)" (Sections 3.1 and 4, S4); Appendix B.1 is
  "Standing conditions"; Appendix C speaks of the coupling conditions of Section 2.4.

## Section 2.5 dissolved; interior loads in Appendix C.3 (12 October; CN applied, EN pending the author's reading)

- The former 2.5 mixed design goals with the premises of the propositions, listed proven properties, the class of
  recovery operators and genuine assumptions as one list of "conditions", and needed a commentary on each item. It is
  removed; every item now stands where it belongs:
  - the three requirements on the approximate condensed stiffness open Section 3, and the outline of Section 3 says
    which section meets which requirement;
  - the properties of \(K\) stay in 2.3, with one sentence on exact versus numerical integration (Appendix A.1,
    Table ST04);
  - the support assumption (\(\mathbb K\succ0\)) is stated in 2.4;
  - the two properties of the recovery operator are stated in 3.1, with the remark that the propositions hold for any
    linear recovery operator having them and that NICE has them by construction;
  - the design interval stays in 4.2, pointing to Appendix B.1 ("基本设定").
- The labels (A1)–(A3) and (D) are gone; Propositions 1–4 state their premises directly.
- Interior loads, as in the PIML papers (Huang et al. 2024, Guo et al. 2026a: one sentence in the main text, the
  derivation in an appendix), but without "without loss of generality", because the error relations of Section 4 use
  this premise: 2.3 says "为叙述简洁，设载荷只作用于主自由度…处理方法见附录 C.3". The new Appendix C.3 gives the exact
  condensation with interior loads (stiffness unchanged), the NICE variant (energy minimisation over
  \(Fq+J_I^T\widehat p_I\), right-hand side \(F^T(f-KJ_I^T\widehat p_I)\) by the existing transpose, no factorisation),
  the identity "load error \(=H^TA(p_I-\widehat p_I)\)" (product of the two interior errors; checked numerically on a
  random SPD example), the first-order error \(H^Tf_I\) and the missing \(f_I^TA^{-1}f_I\) without the particular
  solution, and that Section 4 and the examples assume \(f_I=0\). No limitation sentence in 6.3.

## astra second-round review: implementation (13 Oct)

All decisions in r7_drafts/astra_r2_decisions.md (batches 1-7, notation, 2/3-2) are applied to the CN and EN units;
the discussion with the exact wordings is in r7_drafts/astra_r2_discussion.md, the verified batch-6 edit pairs in
r7_drafts/astra_r2_batch6_edits.json.
- EN now mirrors the CN dissolution of Section 2.5 (requirements open Section 3, premises stated in Propositions 1-4,
  Appendix B.1 "Basic setting", new Appendix C.3 "Interior loads").
- Notation: E -> \Psi, F -> \widehat\Psi, \widehat E -> \widehat\Psi_{\rm net}, \psi_k -> \bar p_k, in all units;
  Figures 2, 3, 13 and the graphical abstract regenerated (matplotlib 3.11.2, the version in the old figure metadata).
- Rigid-body kernel under numerical integration is now derived in Appendix A.1 (author checked that every cell
  contains a fully occupied subcell); the face-connectivity sentence is restricted to Schwarz-P cells.
- D1: the global PCG with the balanced two-level preconditioner is described in Section 3.6, now titled
  "凝聚刚度矩阵的施加与装配求解" / "Applying the condensed stiffness and solving the assembled system".
- Own errors fixed: Table 3 cut-surface row 7 -> 10 configurations; gyroid continuation "6 to 23" -> "10 to 23" times
  in 6.3 and the S7 reading paragraph (ST22: 9.6-23.2 over the six load and support classes).
- Supplement: GPU software in S3 (PyTorch 2.8.0, CUDA 12.8, NVIDIA cuDSS 0.8); GPU model once in 5.9 and S3.
- CN-only style clean-up of leftover 它/后者/explanatory "，即" in U04 and the supplement.
- Offline cost stays unreported (author: the dataset was generated before the solver was sped up).
- Pipeline: assembler joins units with a blank line (Appendix D/G and supplement headings restored); the Word build
  writes aligned blocks as arrays, so no literal '&' appears in other readers.

## GPT Pro narrative review: implementation (14 Oct)

All decisions in r7_drafts/gptpro_decisions.md (groups A-D; the review itself is r7_drafts/gptpro_review_CN.md) are
applied to the CN and EN units; the EN abstract stays at 249 words.
- Introduction: the method paragraph names the problem (retained sets that change with the cut) and ends with "no
  interior factorisation"; the properties paragraph drops the smoothing-interval and coarse-projection explanation and
  points to the conditions of Section 3.4; the complementarity numbers (6.89% -> 0.0965%, 5.4-265 times, Table 4)
  sit in contribution (3) only; Capuano & Rimoli is cited once, in Section 1.1.
- Training vs checks: training covers the energy and the sensitivities, assembled checks the compliance and the
  sensitivities (abstract, introduction, conclusion); Section 2.4 says that NICE uses the estimate of the reference
  sensitivities; Section 4.2 separates the error at given q from the assembled error.
- Claims narrowed: "any network parameters" (6.3), recovery operators that are linear and reproduce the retained
  displacements and rigid-body motions (conclusion), "generally dominates" (5.4), errors below the largest cell error
  (5.8), "the compliance errors above" (5.8), speed with its deployment (abstract, 5.9).
- Fixes: 4.69% -> 4.68% (5.5; ST05a now 4.6850), "its refinement" instead of n=64 (6.3).
- Structure: Section 5 opens with a route map; 5.10 announces the scale timings; 6.1 paragraph 3 is shorter and ends
  with the coarse-level outlook (moved from the conclusion); 6.2 says where the cost goes; 6.3 paragraph 3 is split
  in three and refers to Tables 3 and 5 for the exact checks.
- Conclusion adds the homogenisation design (0.18% less material, 2.44% higher compliance) and states that the 27%
  concerns the uniform starting design.

## GPT Pro narrative review, second round: implementation (14 Oct)

Decisions R2-1 to R2-8 in r7_drafts/gptpro_decisions.md (review: r7_drafts/gptpro_review2_CN.md) are applied to the CN
and EN units; the EN abstract stays at 249 words.
- 5.9 and 6.2 separate the one-time cell preparation (geometry encoding, correction setup) from the per-application
  network evaluation and correction.
- Abstract: 0.074% is stated for traction loads over 80 Schwarz-P validation geometries (20 of them uncut; the old EN
  "80 cut" was wrong), and the plate sentence separates the initial design (27%) from the final design (0.03%,
  0.3%); the conclusion no longer calls the 80 geometries cut. EN compensation: "Fine-scale models", "Periodic
  homogenisation fails ...", "per iteration" ("geometry-conditioned" is a retired term and was not used).
- Introduction: dense condensed matrix and Cholesky factor are attributed to the conventional implementation.
- Conclusion: the properties "follow from the construction and hold for any network parameters".
- 5.8: the comparison with the two-cell maxima and its causal "because Eq. (15)" are gone; Eq. (15) is stated as the
  bound on the lattice compliance error only.
- Appendix H and 5.10: the reverse-mode moment derivative is the exact derivative at fixed clipping topology and
  agrees with the central difference (H.3) to 2e-8 (Supplementary Note S1; r1_psd.py ad_vs_fd_rel, moments_ad).
- 6.1: the outlook on merging the coarse-grid correction into the network is removed.

## Section 2.2: integration over cut elements and element moments (14 Oct)

- Prompted by the author's question whether the element moments are a CutFEM standard. They are not prescribed by
  CutFEM; the moment form K_e = sum M_ea T_a is the subdivision quadrature regrouped (same result as direct
  integration with the same quadrature rule), possible because all elements of the Cartesian background mesh share
  the same Q2 templates. The paper explained the formula in Appendix A.1 but cited no cut-cell quadrature work and
  did not say why moments are used.
- 2.2 (CN/EN): one sentence on the usual options (adaptive subdivision: Parvizian et al. 2007, Düster et al. 2008;
  moment fitting: Müller et al. 2013; high-order implicit quadrature: Saye 2015), then the moment form and its
  equivalence to direct integration; the next paragraph now says that the material distribution within an element
  enters the elastic stiffness, the thickness sensitivity and the network's element features only through the
  moments. 3.2 (CN/EN): the parenthetical definition of the element moments now points to Section 2.2.
- References: the four entries (metadata checked against Crossref) are in `sections/REFS_ADD.md`; `assemble_r5.py`
  now merges that file into the reference list at alphabetical positions (the list itself still comes from the R5
  base). Same entries appended to `references_verified.bib`.
- Checks: number/link/maths parity EN = CN for U02 and U04; EN sentences over 40 words unchanged (150); cited works
  73 -> 77. PDF and Word not rebuilt (author's instruction).
