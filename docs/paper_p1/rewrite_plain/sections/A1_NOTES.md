# A1 notes: Appendices A, B and C (APPENDICES_EN.md / APPENDICES_CN.md, lines 1-203)

Output files: `rewrite_plain/sections/A1_EN.md`, `rewrite_plain/sections/A1_CN.md`. Both have 203 lines, the same as the source chunk. Blank lines, headings and all 17 display-math blocks are unchanged and sit on the same lines in EN, CN and the source. No paragraph was merged, moved or deleted. No supplement additions.

## 1. Changes by category

### 1.1 Terminology (brief section 3.1)

EN (29 replacements):

| Line | Old | New |
|---|---|---|
| 5 | retained box set | retained cell-face node set |
| 5 | cut set | cut node set |
| 5 | off-plane cut-band DOFs | DOFs of the cut-plane elements that lie off the cut plane |
| 28 | the band (the wall band \|φ\| ≤ τ) | the implicit band (clarifies that this is not the cut band) |
| 30 | retained box-face DOFs | retained cell-face DOFs |
| 32 | geometry-level energy score averages over the evaluated directions in one class | the energy error of a geometry is the mean over the evaluated test displacements of one load class |
| 34 | (heading) directional norms | energy-error measures |
| 40 | exact extension; linear extension; entire extension and correction | exact recovery \(E\); linear recovery operator \(F\); complete recovery operator, including the correction |
| 40 | load functional factors through the retained DOFs | load functional depends only on the retained DOFs |
| 40 | Consistent boundary loads | Boundary loads integrated consistently |
| 44 | weak-region element and face stencils | element and face stencils of weakly connected regions |
| 46 | (heading) transpose of the complete extension | transpose of the recovery operator |
| 55 | linear admissible extension | admissible linear recovery operator |
| 57 | has that retained trace | takes these retained displacements |
| 64 | approximate extension; condensed operator | approximate recovery operator; condensed stiffness |
| 73 | corrected extension | corrected recovery operator |
| 75 | retained rigid complement; all-direction energy error | orthogonal complement of the retained rigid-body modes; largest energy error |
| 112 | (heading) directional stiffness | stiffness ratio |
| 114 | energy metric assigns different significance; fix the retained rigid gauge | energy error weights ... differently; remove the rigid-body part of the retained displacements |
| 130 | the rigid gauge is then not needed | the condition \(q\perp R_P\) is then not needed |
| 132 | chosen trace direction; with the trace fixed; unscaled direction | chosen test displacement; with the retained displacements fixed; unscaled mode |
| 136 | elimination from the modular assembled system | eliminating the interior DOFs from the assembled system of all substructures |
| 155 | (heading) reconstructed error energy | error energy of the recovered field |
| 159 | free assembled retained trace | free assembled retained displacements |
| 169 | reconstructed field; retained-solution error | recovered field; error of the retained solution |
| 185 | exact assembled trace | exact assembled retained displacements |
| 202 | pair configurations | two-cell configurations |

CN (69 replacements):

- 保留 (22 occurrences) → 主自由度 / 主自由度位移 / 主自由度分量 / 主自由度分块 / 主节点集 (line 5). Line 198 "仅保留残差的二次误差" (verb, not a term) was rephrased as "的误差仍为残差的二次量" so that 保留 no longer appears.
- 延拓 (7) → 精确位移恢复 \(E\) / 位移恢复算子 \(F\).
- 胞元边界面 (2) → 胞元表面.
- 切割带 (2) → 切割平面单元 (line 5); line 28 was a mistranslation (see 3.5) → 隐式带区.
- 方向 (6 of 7): 全方向 → 最大; 方向范数 → 能量误差度量; 方向刚度 → 刚度比; 评估方向 → 评估测试位移; 迹方向 → 测试位移; 未缩放方向 → 模态. Kept: 每个坐标方向四点 (quadrature, line 28).
- 迹 (5) → 主自由度位移 (and related wording); 刚体规范 (2) → 除去刚体分量 / \(q\perp R_P\) 这一条件; 重构 (2) → 恢复位移场.
- 组装 (13) → 装配, to match the main text and the supplement, which use 装配 only. All 13 occurrences of 组装 in APPENDICES_CN.md were in this chunk, so the appendices now use 装配 throughout.
- 一致的边界载荷 → 按一致积分施加的边界载荷; 弱区域 → 弱连接区域; 邻胞元 → 相邻胞元; 凝聚算子 → 凝聚刚度矩阵; 胞元对配置 → 两胞元配置.
- 校正 → 修正 where it denotes the correction in the recovery operator (3: lines 40, 73 twice). 校正泛函 (line 198) is kept because it names the residual-corrected functional \(J\), not the two-grid correction.

### 1.2 Variant names (brief section 3.3)

None occur in this chunk. No code identifiers or archive record names occur.

### 1.3 Style

- No banned slogans, flourishes or em dashes occur in the source chunk. "supplies both reciprocity and cancellation" (line 108) was rephrased as "ensures reciprocity and cancels".
- EN sentences of about 40 words or more were split at existing semicolons or clause boundaries: lines 5, 28 (four splits), 30 (two splits), 55, 64 (two splits), 130, 132 (two splits). Line 202's final clause now reads "; the second inequality holds because ...". Line 136's closing sentence was reworded from "This proves the equivalence of ..." to "... is therefore equivalent to ...".
- CN: shorter sentences and less translationese in lines 5, 17, 19, 26, 28, 30, 32, 42, 64, 108, 114, 130, 132, 136, 153, 159, 169, 179, 185, 202. Examples: "由这样的内部背景面组成" → "由满足以下条件的内部背景面组成"; "是在单元材料部分上对各坐标指数取 0 至 4 的局部单项式的积分" → "为局部单项式在单元材料部分上的积分，单项式在各坐标上的指数取 0 至 4"; "几何生成仅接受这样的胞元：" → "几何生成时，只有……的胞元才被接受". Every CN sentence is now at most 100 characters, with inline math counted as one character.
- Added inline symbols, with no numbers: \(E\) in line 40 ("The exact recovery \(E\)"), and \(q\perp R_P\) in line 130 (replacing "rigid gauge").

## 2. Cross-references renumbered (old → new; same line in EN and CN)

| Line | Old | New | Context |
|---|---|---|---|
| 55 | Eq. (4) | Eq. (6) | Ritz identity \(\widehat S-S=H^TAH\) |
| 55 | Eq. (5) | Eq. (7) | residual form of the energy error |
| 64 | Table 6 | Table 5 | lattice cost table (Cholesky of supported whole-lattice systems) |
| 73 | Eq. (14) | Eq. (9) | network construction \(\widehat Eq=\ldots\) |
| 132 | Eq. (15) | Eq. (10) | cumulative Jacobi-scaled error fraction \(\chi_\ell\) |
| 138 | Eq. (4) | Eq. (6) | Ritz identity |
| 155 | Proposition 2 | Proposition 3 | heading "(proof of Proposition 3)" |
| 185 | Eq. (7) | Eq. (15) | share-weighted compliance bound |
| 202 | Eq. (8) | Eq. (16) | inexact-solve identity |

That is 9 changes per language and 18 in total. Each old number was checked against the \tag in MANUSCRIPT_EN.md before mapping.

References checked and left unchanged: Eq. (2) (line 48; (2)→(2)); Proposition 1 (heading line 46; 1→1); Section 2.2 (line 30); Section 2.3 (line 136); Section 5 (line 32); Section 5.4 (line 130); Appendices A.1, F.1, G.1; Eqs. (A.1), (B.1)–(B.8), (C.1), (C.2); Table ST04; Supplementary Note S4.3.

## 3. Self-check (Python script, old chunk vs new chunk)

(a) Numbers: the multiset of digit strings is unchanged in EN and in CN except for the nine renumbered references above. Per-line differences occur only on lines 55, 64, 73, 132, 138, 155, 185 and 202. All 17 display-math blocks are byte-identical to the source. Every inline-math expression of the source is still present.

(b) Old terms left in place, with the reason:
- EN "admissible" and CN "运动容许" (line 55): appendix proof term, allowed by the task.
- EN "per coordinate direction" and CN "每个坐标方向" (line 28): a quadrature coordinate direction, not a test displacement. The EN now says "coordinate direction" to avoid ambiguity.
- EN "share" (line 30, "share the nine nodes"): a verb, not "energy share".
- CN 校正泛函 (line 198): see 1.1.
- EN "retained" is kept throughout, as instructed.

(c) Line counts: EN 203, CN 203, source chunk 203. Blank lines are on identical line numbers in all three. One structural difference already existed in the source and is unchanged: CN line 132 begins with inline math while EN line 132 begins with a word. One number difference already existed: line 17 CN "0 至 4" versus EN "zero to four".

## 4. Uncertain points for the orchestrator

1. Line 136, "coupling assumptions of Section 2.3": per the map, old 2.3 → 2.3, and the coupling (disjoint interiors, shared retained DOFs) is still described in new Section 2.3. However, assumption (A3), which restates it, now lives in Section 2.4. If the main-text unit prefers it, this could read "Section 2.4" or "assumption (A3), Section 2.4". The same applies to line 30, "rigid-body kernel assumed in Section 2.2": it is stated in 2.2 and formalised as (A1) in 2.4. Both were left as the map gives them.
2. Line 32: "load class" / "载荷类别" was chosen to match the main text's "loading class" (old Section 5.1/5.3, which cites Appendix A.1 for this averaging rule). Some of the nine validation classes are imposed displacements rather than loads. If the main-text unit uses "test-displacement class" / "测试位移类别", line 32 should follow it.
3. Headings were renamed (B, B.2, B.3, C.1); their labels are unchanged. The main text and supplement cite these appendices only by label ("Appendix B.3", "Appendix C.1"), so no other file needs to change.
4. CN line 28 correction: the source CN read "依次由定义切割带和切割平面的三个插值不等式进行裁剪". The EN "band" there is the implicit wall band \(|\phi|\le\tau\) of Eq. (1), which gives two of the three inequalities; the cut plane gives the third. It is not the cut band. CN now reads 隐式带区, consistent with the main-text CN 隐式带宽 / 带函数, and EN now reads "implicit band". This fixes a translation error and does not change content.
5. Coined non-map words were replaced: "trace" became "retained displacements", and "rigid gauge" became "rigid-body part" or the condition \(q\perp R_P\). Neither is in the brief's map. If other appendix units keep "trace" / 迹 (for example Appendices D–H), the usage will differ between units. Both forms are mathematically correct.
6. CN 组装 → 装配 (13 occurrences) also goes beyond the map. It was done because the main text and supplement use only 装配, and the rest of APPENDICES_CN.md already used 装配 (7 occurrences).
7. CN line 5 uses 胞元表面主节点集 for "retained cell-face node set". 主节点 is the standard CN counterpart of master nodes. The alternative is 胞元表面主自由度节点集.
8. Line 44 keeps two parenthetical references (Appendix G.1, Appendix F.1) in one list sentence because they belong to different list items. It could be split if the one-reference-per-sentence rule is applied strictly.
9. Commit 9ee4036 ("Add in-progress R5 section drafts") holds an intermediate snapshot of A1_EN.md and A1_CN.md taken while this unit was still being edited. Use the files on disk, which are final; they differ from that commit.
