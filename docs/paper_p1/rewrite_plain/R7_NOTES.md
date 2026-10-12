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
