# S3 notes: Supplementary Notes S3–S5 (SUPPLEMENTARY_EN.md / SUPPLEMENTARY_CN.md, lines 479–626)

Output files: `rewrite_plain/sections/S3_EN.md` and `rewrite_plain/sections/S3_CN.md`. Each one replaces lines 479–626 of its source file in full. Line numbers below are file line numbers.

## 0. Line count and alignment

- Both files have 148 lines, the same as the source chunk, so no later line moves. Line 627 of the source (`## Supplementary Note S6. ...`) follows directly. A splice of each file into its source was checked: the result has the same 888 lines, and only the changed lines listed below differ.
- EN and CN are still line-aligned. Blank lines, headings, the display equation (573–575) and every table row are on the same line numbers in both languages. Each table row has the same number of columns in EN and CN.
- No paragraph, table row or equation was merged, moved, deleted or reordered. Some long sentences were split inside their paragraph.
- Changed lines:
  - EN: 481, 483, 485, 487, 489, 493, 497, 504, 519, 530, 541, 550, 554, 577, 579, 585, 591, 593, 595, 597, 606, 608, 610, 621, 622.
  - CN: the same lines, plus 556, 567, 568, 570, 581 and 587, where only the Chinese wording changed (保留 → 主自由度 and plain-style fixes).
- All other lines are byte-identical to the source. This includes every data row of Tables ST12a–ST12e, ST13, ST14 and ST15 (except the two ST15 rows that named the slot tables).
- No supplement additions. This chunk holds no material on the Uncorrected continuation or the Smoothing-trained variant that would move to Note S8.

## 1. Changes by category

### 1.1 Terminology (brief section 3.1)

EN:

| Old | New | Lines |
|---|---|---|
| unit consistent tractions | unit traction loads | 481 |
| consistent loads / consistent tractions / consistent face loads | traction loads / face traction loads | 487, 550, 585 (2), 597 |
| the consistent-load implementation integrates the Q2 surface shape functions | the traction loads are obtained by integrating the Q2 surface shape functions | 585 |
| three Cartesian directions / Cartesian traction directions / the same three directions | the three Cartesian axes / traction loads along the three Cartesian axes / the same three loads | 481, 585 |
| target face / neighbour face | loaded face of the test cell / loaded face of the neighbouring cell | 585 |
| target cut surface / the target's cut plane / the neighbour | cut surface of the test cell / cut plane of the test cell / the neighbouring cell | 585 |
| target cell | test cell | 610 |
| Cut-band DOFs off the box faces (share of free retained) | Cut-plane-element DOFs off the cell faces (fraction of free retained DOFs) | 497 (ST12a header) |
| field-based sensitivities \(\widetilde s_c\) | sensitivities \(\widetilde s_c\) | 487 |
| compare the field-based sensitivities with the exact ones | compare the sensitivities from NICE with the exact ones | 593 |
| field-based gradient (Table ST14 title) | gradient | 595 |
| field-based thickness sensitivity errors | thickness-sensitivity errors | 606 |
| used to correct local extensions | used to correct the network's interior displacements in each cell | 577 |
| at the fixed recovered fields | at the fixed recovered displacement fields | 487 |
| weakly supported node | weakly connected node | 593 |
| at the exact traces | at the exact retained displacements | 608 |
| slot tables | position-embedding tables | 621, 622 (ST15) |
| Eigenvectors ... are retained | Eigenvectors ... are kept | 579 (verb; avoids confusion with retained DOFs) |

CN:

| Old | New | Lines |
|---|---|---|
| 保留自由度 / 自由保留自由度 / 私有保留块 / 保留位移 / 保留解 | 主自由度 / 自由主自由度 / 私有主自由度块 / 主自由度位移 / 主自由度解 | 483, 485, 497, 504, 556, 568, 570, 577, 593, 608, 610 |
| 保留体积分数 | 切割后剩余的体积分数 | 481 (as S1: 剩余体积) |
| 保留……的特征向量 (verb) | 选取……的特征向量 | 579 |
| 单位一致面力 / 一致载荷 / 一致面力 / 一致面载荷 | 单位面力载荷 / 面力载荷 | 481, 487, 550, 585 (2), 597 |
| 沿三个笛卡尔方向 / 笛卡尔面力方向 / 相同三个方向 | 沿三个笛卡尔坐标轴 / 同样三个载荷 | 481, 585 |
| 目标面 / 邻胞元面 / 目标切割表面 / 目标胞元 / 邻胞元 | 被测胞元受载表面 / 相邻胞元受载表面 / 被测胞元的切割表面 / 被测胞元 / 相邻胞元 | 585, 610 |
| 胞元边界面以外的切割带自由度（占自由保留自由度的比例） | 胞元表面以外的切割平面单元自由度（占自由主自由度的比例） | 497 |
| 场基灵敏度 / 场基梯度 / 场基厚度灵敏度误差 | 灵敏度 / 梯度 / 厚度灵敏度误差 | 487, 593, 595, 606 |
| 用于校正局部延拓的内部基 | 用于修正各胞元内网络预测内部位移的内部基 | 577 |
| 含弱支撑节点的模板 | 含弱连接节点的模板 | 593 |
| 精确迹处 | 精确主自由度位移处 | 608 |
| 槽表 | 位置嵌入表 | 621, 622 |
| generic 校正 (为校正做准备, 校正中的光滑, 校正设置, NICE 的网络状态与校正, 双精度校正算术, 将校正由双精度改为单精度) | 修正 | 487 (2), 554 (2), 577, 591 (2) |
| 粗求解 / 粗分解 | 粗网格求解 / 粗网格分解 | 487, 554, 593 |
| 各继续训练 | 各延续训练 | 621 (matches 未修正延续) |
| 带支撑装配算子 / 带支撑的保留自由度 | 施加支撑后的装配算子 / 施加支撑后的主自由度 | 567, 577 |

As in A2 and S1, generic 校正 became 修正. No standard term such as 粗网格校正 occurs in this chunk.

The pattern follows the earlier units: "loaded face of the test cell" also appears in S2. "position-embedding table" follows A3 (Appendix G, file line 359). "weakly connected" follows A3 (346, 373). "at the exact retained displacements" follows A3's "At the same retained displacement" (565).

### 1.2 Variant names (brief section 3.3)

None of the variant labels (Uncorrected, Base network (corrected), Smoothing-trained, NICE as a variant label) occurs in this chunk. "The base network" (621) is unchanged. EN "the continuations" (621) is unchanged; CN 各继续训练 became 各延续训练, the word used in 未修正延续. Record identifiers and code names are unchanged (mtype, iparm, GAMG, BoomerAMG, HMIS, extended+i, MUMPS, cuDSS, M1/x, U1/y, G1–G4).

### 1.3 Banned patterns and plain style (brief section 1)

- No em dash occurs in the chunk, before or after the edit. En dashes in ranges (0.2–0.5, 235–274, 0.07–0.67%, ...) and in "action–energy" are kept.
- 585: "a test configuration rather than a physically cut specimen" ("not X but Y" form) became "a test configuration and not a physically cut specimen" [而不是].
- CN 593: 这正是采用共享厚度变量的优化器所接收的梯度 became 聚合后的向量即采用共享厚度变量的优化器所接收的梯度 (正是 removed).
- EN long sentences were split at existing clause boundaries, with content unchanged: 481, 483, 485, 487, 489, 493, 541, 550, 554, 585, 591, 593, 606, 608, 610.
  - 487: the single list sentence of route (c) phases became one sentence per phase, in the same order (cell preparation; lattice and \(\mathbb K_{PP}\) assembly and preconditioner setup; conjugate gradients; sensitivities). The parenthesis "(reached at ...)" became "the residuals reached are ...". "Its arithmetic is that of the timed route of Table 1" became "Route (c) uses the arithmetic of the timed route of Table 1".
  - 489: "For BDDC with exact local solvers and one subdomain per cell, ..." became "The BDDC configuration considered has exact local solvers and one subdomain per cell." This wording does not suggest that BDDC was run to completion (the source says only its factorisations were timed and its MUMPS setup exceeded 90 GiB).
  - 591: the colon-list of recorded quantities became "Three quantities are recorded at the final iterate \(\bar U\). The first is ... The second is ... Here ... The third is ...". This adds the number words three/first/second/third (no digits); the count is that of the source list.
  - 593: the switch criterion is now a list after "any of the following:". It is one sentence of 51 words, the only EN sentence above 40 words; it is a list of four conditions.
  - 608: the run-on accuracy sentence became one sentence per quantity, introduced by "The values below are given for the \(2\times2\times2\) block / the \(3\times3\times1\) layer."
  - 610: the sweep set-up sentence was split into four; "one corner parameter ..., the largest- or the median-sensitivity corner, was varied" became "...; this was the corner with the largest or the median sensitivity".
- CN translationese fixes and splits (every CN prose sentence is now at most 100 characters, counting inline math as one character):
  - 481: 后者的保留体积分数 → 后者切割后剩余的体积分数; the long first paragraph split into five sentences.
  - 483: 将每个胞元的完整切割胞元刚度（保留自由度与内部自由度）装配 → 将每个胞元在主自由度与内部自由度上的完整切割胞元刚度装配; the PARDISO sentence split.
  - 485: 用稠密核对其私有保留块进行分解与消元 (稠密核对 could be misread as 核对) → 以稠密计算核分解并消去其私有主自由度块; 凝聚点阵系统 → 凝聚后的点阵方程组.
  - 487: one 150-character list sentence became five sentences; 其算术精度 → 路线 (c) 的运算精度.
  - 489: 在残差首次低于……处记录迭代解，并给出其柔度和能量范数与最终迭代解之差 → 残差首次低于……时记录当时的迭代解，以及该迭代解与最终迭代解在柔度和能量范数上的差别; BDDC 在设置内存超过 90 GiB (ungrammatical) → BDDC 在设置阶段的内存超过 90 GiB; 不含……、粗问题和迭代 split into its own sentence.
  - 519: 这是因为……并未加速 → 原因是……并未随之加速.
  - 541, 550: parenthetical definitions became sentences.
  - 554: 两者均不包括……，该刚度两条路线都需保存 → 两者均不包括……，两条路线都需保存该矩阵.
  - 581: 并从那里流式传输 → 使用时从 CPU 内存流式传输; 表 ST20 的标题说明 → 表 ST20 的表题.
  - 587: 胞元对求解采用以……作为预条件的共轭梯度法 (long pre-nominal modifier) → 胞元对的求解采用共轭梯度法，以……作为预条件子.
  - 591: split as in EN; 双精度校正算术 → 修正以双精度运算.
  - 593: the comparison sentence now states the order (先聚合，再比较); the switch criterion became 若……改变了以下任一项，则……：…….
  - 606: 相同的记录给出 → 对……作相同记录，结果为.
  - 610: split as in EN; 按梯形法则 → 按梯形公式; 与之相当 → 与之相同 (EN "by the same amount").

## 2. Cross-references renumbered (old → new; same line in EN and CN; each occurrence mapped exactly once, no chaining)

| Line | Old | New | Checked against |
|---|---|---|---|
| 481 | Table 6 / 表 6 | Table 5 / 表 5 | old Table 6 is the cost table "Cost of one lattice analysis with sensitivities"; new Table 5 |
| 487 | Eq. (9) / 式 (9) | Eq. (17) / 式 (17) | old \tag{9}: sensitivity error \(\widetilde s_c-s_c\), which defines \(\widetilde s_c\) |
| 493 | Table 6 / 表 6 | Table 5 / 表 5 | as 481 |
| 504 | Table 6 / 表 6 | Table 5 / 表 5 | as 481 (its "DOFs: total / free retained" column) |
| 519 | Table 6 / 表 6 | Table 5 / 表 5 | as 481 |
| 530 | Table 6 / 表 6 | Table 5 / 表 5 | as 481 |
| 541 | Table 6 / 表 6 | Table 5 / 表 5 | as 481 |
| 608 | Eq. (7) / 式 (7) | Eq. (15) / 式 (15) | old \tag{7}: compliance bound \(0\le(C-\widehat C)/C\le\beta/(1+\beta)\le\beta\) |

Eight changes per language, sixteen in total.

Left unchanged because the map leaves them the same or because they are appendix or supplementary labels:
- main text: Table 1 (487), Section 5.6 (606), Section 5.8 (481, 577, 591, 608), Section 5.9 (577), Section 5.10 (577), Figure 9 (585);
- appendices: Appendix C.2 (591, twice), Appendix F.1 (593), Appendix F.2 (487), Appendix G (614, 620), Appendix G.1 (593);
- supplement: Tables ST01, ST12, ST12a–ST12e, ST13, ST14, ST15, ST16, ST20, Supplementary Notes S4.1, Supplementary Section R1.

The chunk contains no Proposition reference, no main-text section reference that the map changes (only 5.6, 5.8, 5.9, 5.10), and no reference to old Table 5 or old Table 7.

## 3. Self-check (Python, old chunk against new chunk)

(a) Number multiset (digit strings such as 1,041.6, 0.9999998, 2026.1, the digits inside \(10^{-6}\) and inside labels such as ST12e). EN and CN give the same result:
- removed: 6 ×6 (Table 6), 9 (Eq. 9), 7 (Eq. 7);
- added: 5 ×6 (Table 5), 17 (Eq. 17), 15 (Eq. 15).

These are exactly the eight cross-references of section 2, on lines 481, 487, 493, 504, 519, 530, 541 and 608. No other number changed. The multiset of inline math `\( ... \)` is identical to the source on every line in both languages, and the display equation (573–575) is byte-identical. In the new chunk, EN and CN have identical number multisets and identical inline-math multisets on every line.

Number words in EN (no digits): "three" 19 → 20, "first" 3 → 4, "second" 0 → 1, "third" 0 → 1, all from the enumeration on line 591. CN adds the matching 三个量、第一个、第二个、第三个.

(b) Old terms that remain, and why:
- EN "retained" (17 occurrences: 481, 483, 485, 497, 504, 556, 568, 570, 577, 593, 608, 610): the brief keeps "retained" in EN. This includes 481 "retained volume fractions" (the volume left after the cut, as in S1 and the main text).
- EN "Schur-complement option" (485) and "Cells: Schur complement" (523) / CN Schur 补: the PARDISO option and a table column; Schur complement is allowed outside the main text.
- EN "weak-region" (593, 620, 622, 623) / CN 弱区域: not on the map; it names the extra layers on stencils with weakly connected nodes, as kept in A3.
- EN "Cartesian axes" / CN 笛卡尔坐标轴: physical coordinate axes; the word "direction" no longer occurs.
- CN 一致 (591): only in 作用–能量不一致量 (action–energy inconsistency), not 一致面力.
- CN 目标 (610): only in 代理目标 (surrogate objective), not 目标胞元.
- "shared" / "shares" (EN 485, 489, 585, 593, 597): ordinary English, not "energy share".

None of the following occurs in the new chunk:
- EN: box face, box, cut band, cut-band, energy share, share of, extension, direction, field-based, consistent, nodal force, weakly supported, slot, latent, target, stratum/strata, equilibrium correction, work-conjugate, nonexpansive, trial field, improvability, division of tasks, trace, admissible, Ritz, Galerkin, Uncorrected, "Base network, corrected", precisely, notably, in effect, rather than, em dash.
- CN: 保留, 延拓, 场基, 非扩张, 方向, 正是, 平衡校正, 一致面力, 一致载荷, 切割带, 边界面, 份额, 潜空间, 槽, 弱支撑, 迹, 目标胞元, 分层, 功共轭, 试探场, 可改进性, 分工, 未校正, 加校正, 校正, em dash.

(c) Line counts: EN 148, CN 148, source chunk 148.

## 4. Uncertain points for the orchestrator

1. **"consistent" dropped in the load description (585).** The source said "The consistent-load implementation integrates the Q2 surface shape functions". The new text says "The traction loads are obtained by integrating the Q2 surface shape functions" [面力载荷由 Q2 表面形函数积分得到], which describes consistent integration without the word. A3 kept "consistently integrated" in Appendix G. If the orchestrator wants the same phrase here, write "The traction loads are consistently integrated with the Q2 surface shape functions" / 面力载荷按 Q2 表面形函数一致积分得到.
2. **"loaded face of the test cell / of the neighbouring cell" (585).** The source said "the target face" and "the neighbour face", which Figure 9 calls the T and N faces, the faces carrying the tractions. "loaded face" is my reading; S2 uses the same phrase.
3. **Line 610, which triple is which.** "0.230, 0.252 and 0.228% against 0.228, 0.249 and 0.225% on M1/x" does not say which triple belongs to the surrogate and which to the exact model. The context (surrogate "up to 0.25%" on M1/x) suggests the first triple is the surrogate, but I kept the source's ambiguity in both languages (CN 偏离值为……，对比……) and did not assign them.
4. **ST14 title (595).** "field-based gradient" became "gradient" (CN 梯度). If the main text names \(\widetilde s\) differently (A3 used "sensitivity estimate" in H.2), the title could read "Residual work, dual-norm bound, gradient error and difference quotients ...".
5. **Line 593, "the sensitivities from NICE".** The brief replaces "field-based sensitivity" with "sensitivity". Here the computed sensitivities must be told apart from the exact ones, so I wrote "the sensitivities from NICE" [NICE 所得灵敏度]. These runs use NICE (591), so the qualifier adds no claim.
6. **Line 489, BDDC sentence.** "For BDDC with exact local solvers and one subdomain per cell, the Dirichlet matrix ..." became "The BDDC configuration considered has exact local solvers and one subdomain per cell. The Dirichlet matrix ..." [所考察的 BDDC 配置……]. I avoided "BDDC is used", which would suggest a completed BDDC solve.
7. **Main-text dependency on Table 5.** Seven lines now cite Table 5 (old Table 6). This holds only if the main-text unit for Section 5.9 numbers the cost table as Table 5, as the brief requires.
8. **CN 各延续训练 (621).** EN "the continuations" covers every variant trained on from the base network's selected parameters (Base network + correction, Uncorrected continuation, Smoothing-trained, NICE). CN 延续 matches 未修正延续; revert to 继续训练 if the orchestrator prefers.
