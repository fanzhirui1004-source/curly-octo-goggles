# S2 notes: Supplement Tables ST08–ST11, Notes S1–S2 (SUPPLEMENTARY_EN.md / SUPPLEMENTARY_CN.md, lines 243–478)

Output files: `rewrite_plain/sections/S2_EN.md` and `rewrite_plain/sections/S2_CN.md`. Each holds the full replacement text for source lines 243–478 (236 lines, the last one blank). Line N of the source is line N−242 of each output file.

## 0. Structure

- Line count unchanged: 236 lines in the old EN chunk, the old CN chunk, the new EN chunk and the new CN chunk.
- No paragraph, list item, table row, heading or equation was added, removed, merged or reordered. EN and CN are aligned line by line: same blank lines, same headings and heading levels, same list items, same table rows with the same number of columns. Each prose line contains the same numbers and the same inline mathematics in EN and CN.
- Display equation (S2.1) (lines 410–414) and every table data row are byte-identical to the source, except for the variant names in the ST08 and ST09 rows (Section 2 below).
- Several long sentences were split into two or three sentences on the same line (EN 308, 377, 381, 390, 400, 404, 408, 416, 418, 430; CN the same lines plus 245, 312, 329, 375). No content was dropped.

## 1. Terminology (brief section 3.1)

Counts are occurrences in the source chunk; line numbers are source line numbers. EN total 55, CN total 65 (variant names are counted separately in Section 2). Counts are approximate where one phrase touches two rows (for example "box trace" at 408 is counted under box face and under trace).

| Old EN → new EN | Old CN → new CN | Lines | Count |
|---|---|---|---|
| target cell / target / target-cell / target-face → test cell / test cell / test-cell / test-face (columns of ST11: "Test-face compliance (3 loads)", "Test-face sensitivity (3 loads)"; defined in the ST11 caption as the loaded face of the test cell) | 目标胞元 / 目标 / 目标面 → 被测胞元 / 被测面 | 245, 247, 308 (2), 312, 329 (2), 422, 424, 430 (5), 434 (2), 459 (2) | 17 / 17 |
| box face(s), box-face, non-box, box trace → cell face(s), cell-face, "not on the cell faces" | 胞元边界面 → 胞元表面 | 381 (2), 402, 404, 408 (2), 416 (2), 425, 428, 430, 432 | 11 / 12 |
| cut band, cut-band DOFs → (DOFs of the) cut-plane elements | 切割带 → 切割平面单元 | 408, 425, 430 (2) | 4 / 4 |
| field-based sensitivity-vector errors → errors of the sensitivity vector | 场基灵敏度 → 灵敏度 | 245 | 1 / 1 |
| unit consistent tractions; consistent face tractions; unit-resultant consistent body force → unit traction loads; face traction loads; body force with unit resultant | 一致面力；一致体力 → 面力载荷；体力载荷 | 381 (2), 423, 430 | 4 / 4 |
| stratum → cut-severity group | 分层 → 切割程度分组 | 312 | 1 / 1 |
| trial spaces → displacement spaces | 试探空间 → 位移空间 | 418 | 1 / 1 |
| "its ordering comes from the contraction of \(H\)" → "Its ordering holds because the two-grid cycle that defines \(H\) does not increase the energy error" | 其有序性来自 \(H\) 的压缩性 → 修正的有序性源于定义 \(H\) 的两重网格循环不增大能量误差 | 418 | 1 / 1 |
| restricted trace space; box trace → restricted space of cell-face displacements; cell faces | 受限迹空间；胞元边界面迹 → 受限的胞元表面位移空间；胞元表面 | 408, 416 | 2 / 2 |
| exact Schur complement; exact local Schur operators; exact cell operators → exact condensed stiffness matrix (the Schur complement), said once at 404; exact condensed stiffness matrices of the cells | 精确的 Schur 补；精确局部 Schur 算子；精确胞元算子 → 精确凝聚刚度矩阵（即 Schur 补）；各胞元的精确凝聚刚度矩阵 | 404 (2), 416, 430 | 4 / 4 |
| retained space / full retained-space solution → space of retained displacements / solution with all retained DOFs | 保留空间 / 完整保留空间解 → 主自由度位移空间 / 采用全部主自由度的解 | 418, 430, 477 | 3 / 3 |
| (EN keeps "retained") | 保留自由度、保留胞元边界面位移 → 主自由度、胞元表面主自由度位移 | 404, 418 | 0 / 2 |
| Retained volume (ST08b column), retained part (H1) → unchanged in EN | 保留体积 → 剩余体积 (as in S1, Table ST02); 保留部分 → 该胞元切割后的剩余部分 | 314, 377 | 0 / 2 |
| identity columns / identity representation (verb "retain") → "keep" | 保留单位列 / 保留单位表示 → 仍取单位列 / 仍取单位表示 | 408, 430 | 0 / 2 |
| Production step / production references / production central difference → step used in the computations / "Elsewhere in this work, the reference solutions are computed at …" / central difference used in the computations | 生产步长 / 生产计算 → 计算所用步长 / 本文其他部分 / 计算所用 | 381, 400 (2) | 3 / 3 |
| "Complete continuous-neighbour assembly results" (title) → "Complete results of the two-cell assemblies with a continuous-thickness neighbour" | 完整的连续邻胞元装配结果 → 两胞元装配的完整结果（相邻胞元厚度连续） | 243 | 1 / 1 |
| "Two variants are examined", "the interface-only variant" (restriction cases) → "Two cases", "the interface-only case" (avoids confusion with the five network variants) | 两种变体 / 仅界面受限变体 → 两种情形 / 仅界面受限情形 | 416, 422 | 2 / 2 |
| Note S2 title "Ablation of the retained representation: Bernstein-restricted box faces" → "Ablation: Bernstein-restricted cell faces" | 保留表示的消融：Bernstein 限制的胞元边界面 → 消融：胞元表面的 Bernstein 限制 | 402 | (counted under box face) |
| — | 受限 Galerkin 系统 / Galerkin 系统 / 施加支撑后的系统 → 受限 Galerkin 方程组 / Galerkin 方程组 / 方程组 | 406, 408, 477 | 0 / 3 |

## 2. Variant names (brief section 3.3)

The supplement keeps all five variants.

| Old EN → new EN | Old CN → new CN | Lines | Count |
|---|---|---|---|
| Uncorrected → Uncorrected continuation | 未校正 → 未修正延续 | ST08 rows 256–266 (11), ST09 rows 337–343 (7) | 18 / 18 |
| Base network, corrected → Base network + correction | 基础网络（加校正） → 基础网络加修正 | ST08 rows 279–292 (14), ST09 rows 352–361 (10) | 24 / 24 |
| Base network, Smoothing-trained, NICE | 基础网络、平滑训练、NICE | unchanged | 0 |

Term-replacement total: EN 55 + 42 = 97; CN 65 + 42 = 107.

## 3. Cross-references

Renumbered (one occurrence per language):

| Line | Old EN | New EN | Old CN | New CN | Reason |
|---|---|---|---|---|---|
| 418 | (Section 4.4) | (Section 3.4) | （第 4.4 节） | （第 3.4 节） | old Section 4.4 (coarse-grid correction, old Proposition 4) is new Section 3.4 |

Checked and unchanged (main text): Section 5.1 (line 308), Section 5.6 (329), Section 5.7 (404), Figure 9 (422). Sections 5.k keep their numbers. No main-text equation, proposition or table number occurs in this chunk (old Tables 5–7 and old Propositions 1–4 are not cited here).

Unchanged (supplement and appendix labels): Supplementary Section R1, Tables ST08, ST08b, ST09, ST10, ST10b, ST11, ST11a, ST11b, Figures S01(a), S01(c), S01(d), S04(a–c), Supplementary Note S4, Eq. (H.7), Eq. (S2.1).

## 4. Style (brief section 1)

- EN: no em dash in prose (the "—" in Table ST10, row H2, column \(n=56\) is an empty-cell marker and was kept). No banned flourishes occurred in the source EN chunk.
- CN: "这正是中心差分的二阶截断误差" (400) → "对应中心差分的二阶截断误差". Translationese fixed: "其邻胞元为精确的" / "目标胞元为学习的" (245, 312, 329) → "相邻胞元采用精确凝聚" / "被测胞元采用学习子结构"; "灵敏度最大值涵盖两个胞元" → "灵敏度最大值在两个胞元上取"; "自由的自由度" (477) → "未约束自由度"; "三种载荷中幅值最大的、相对于 \(n=64\) 的带符号柔度偏差" (381) restructured as "斜线前为相对 \(n=64\) 的带符号柔度偏差，取三种载荷中绝对值最大者". "至多" → "不超过" in 377 and 400. Sentences over 100 characters were split; the longest CN sentence is now under 85 characters (formulas counted as one character).
- EN 426 "The comparison concerns accuracy, not accuracy at equal cost." → "The comparison concerns accuracy only and does not compare accuracy at equal cost." (removes the "not X but Y" form, same content).

## 5. Self-check (Python, scratchpad script)

(a) Number multiset (regex `\d+(?:[.,]\d+)*`), old chunk vs new chunk: EN removed {4.4: 1}, added {3.4: 1}; CN removed {4.4: 1}, added {3.4: 1}. Both differences are the cross-reference of Section 3 above. No other number changed, and every prose line has the same numbers in EN and CN.

(b) Old terms remaining after the rewrite:

- EN "directions" (329 "three cut-surface traction directions"; 381 "three coordinate directions") and CN 方向 (same lines): Cartesian directions of physical traction loads, not the coined "direction" = test displacement. Kept.
- EN "consistent with" (400) and CN "保持一致" (408): ordinary words, not "consistent tractions".
- Galerkin (406, 408), Ritz ("nondecreasing Ritz compliance", 418), Schur ("(the Schur complement)", 404): mathematical terms that the task allows in the appendices and supplement. Schur is now said once.
- EN "retained" (Retained volume, retained part, retained DOFs, retained displacements, retained cell-face displacements): the brief keeps "retained" in EN. "Retained volume" (ST08b column) and "retained part" (377) mean the material left after the cut, as in the main text and in S1.
- "—" in Table ST10: table empty-cell marker.
- None of the following remains: box face, cut band, energy share, extension, field-based, consistent traction(s), nodal forces, weakly supported, slot, latent, target, stratum/strata, equilibrium correction, work-conjugate, nonexpansive, trial, improvability, division of tasks, trace, contraction, "Uncorrected" without "continuation", "Base network, corrected"; CN 保留, 延拓, 场基, 一致面力, 一致体力, 切割带, 边界面, 分层, 目标, 未校正, 加校正, 试探, 压缩, 迹, 这正是, 生产.

(c) Line counts: EN 236, CN 236 (source chunks 236 each). Blank lines, headings, list items and table column counts match line by line.

## 6. Uncertain points and decisions for the assembler

1. **ST08b label collision (needs a decision, same as reported by S1).** Brief section 5 maps old main-text Table 5 to "Supplementary Table ST08b". This chunk already contains `#### ST08b. Held-out two-cell configurations (NICE)` (line 310), which the main text (old Section 5.6) and Supplementary Section R1 cite as "Supplementary Table ST08b". Following "supplementary labels unchanged", the existing label was kept. When old Table 5 is placed in Note S8, one of the two tables needs another label (for example ST08c for old Table 5), and every citation must follow.
2. **Section 4.4 → 3.4 at line 418.** The old text cites Section 4.4 for "the contraction of \(H\)". The sentence now says the two-grid cycle that defines \(H\) does not increase the energy error, which is the content of new Proposition 2 in new Section 3.4. If the assembler prefers the proposition, "(Proposition 2)" would be the more specific reference.
3. **"consistent body force" (381).** "consistent" was dropped, as for traction loads ("一致积分"是施加方式, per the approved term table). If the appendices define how body forces are integrated, no information is lost; otherwise the reader no longer learns that the body force is integrated consistently.
4. **"Production" (381, 400).** Replaced by "used in the computations" / "Elsewhere in this work". Meaning: the step size and the \(n=32\) reference used for all results reported in the paper.
5. **Titles changed in wording only:** Table ST08 ("Complete results of the two-cell assemblies with a continuous-thickness neighbour"), Supplementary Note S2 ("Ablation: Bernstein-restricted cell faces"), Table ST11 ("Bernstein restriction of the cell-face displacements"), ST11a ("Every cell face restricted"). Labels unchanged. If a table of contents or the LaTeX build lists these titles elsewhere, it should follow.
6. **ST08b caption (312).** "Maximum relative errors (%) over the six face loads (compliance) and over both cells (thickness sensitivity)" was kept with the same scope: compliance maximised over the six face loads, sensitivity over both cells. Whether the sensitivity maximum is also over the six loads is not stated in the source and was not added.
7. **"cut-band DOFs retain their identity representation" (430).** Rendered as "the DOFs of the cut-plane elements keep their identity representation", with the same scope as the source; line 408 and 425 restrict this to DOFs not on the cell faces.
