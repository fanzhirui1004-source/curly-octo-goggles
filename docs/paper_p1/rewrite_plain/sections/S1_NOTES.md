# S1 notes: Supplement R1 variant key and Tables ST01–ST07 (SUPPLEMENTARY_EN.md / SUPPLEMENTARY_CN.md, lines 1–242)

Output files: `rewrite_plain/sections/S1_EN.md` and `rewrite_plain/sections/S1_CN.md`. Each is a full replacement for lines 1–242 of its source file.

## 0. Line count and alignment

- Both files have 242 lines, the same as the source chunk, so the replacement does not shift any later line (line 243 of the source, `## Table ST08. ...`, follows directly).
- EN and CN stay line-aligned. Blank lines, headings, list items, the display equation (lines 68–71) and every table row fall on the same line numbers in both languages. Each table row has the same number of columns in EN and CN.
- No paragraph, table row or list item was merged, moved, deleted or reordered. Some long sentences were split inside their paragraph.
- No supplement additions.
- Changed lines:
  - EN: 3, 10, 12, 15, 35, 37, 49, 51, 54, 56, 58, 62, 64, 66, 76, 77, 84, 86, 88, 91–93, 95, 97, 99, 101, 113, 115, 117, 120–122, 126, 140, 142, 150, 154, 156, 170–179, 183, 199, 203, 216, 218.
  - CN: the same lines except 91–93 and 120–122 (the CN group names 轻度/中度/重度切割 are unchanged), plus 9, 11, 13, 46, 59, 73, 75, 80, 138, 181, 185, 214 and 241, where only the Chinese wording changed.
- All other lines are byte-identical to the source.

## 1. Changes by category

### 1.1 Terminology (brief section 3.1)

EN, 88 replacements:

| Old | New | Lines |
|---|---|---|
| consistent tractions / consistent-traction (14) | traction loads / traction-load | 99 (2), 113, 115, 140 (3), 142 (3), 154, 183, 203, 216 |
| nodal forces / nodal-force (5) | nodal point loads / nodal-point-load | 115, 140 (2), 142, 183 |
| direction(s), retained directions, directional, direction sets (20) | test displacement(s); "energy error" for "directional energy error"; "test-displacement sets" | 54, 58, 64, 66, 99 (4), 126, 140 (4), 154 (2), 183 (2), 216 (2) |
| direction class (title) | load class | 97 |
| stratum / strata / geometry stratum (8) | cut-severity group(s) | 15, 35, 37, 86, 88, 113, 115, 117 |
| Light cut / Moderate cut / Heavy cut, (light) / (moderate) / (heavy) (9) | Lightly / Moderately / Heavily cut, (lightly cut) etc. | 91–93, 113 (3), 120–122 |
| ghost-penalty share, first-order share (4) | ghost-penalty fraction, first-order fraction | 140, 142, 154, 156 |
| graph-harmonic extension; extended; extension (4) | recovers the interior displacements ... by graph-harmonic interpolation; recovered; this recovery | 218 |
| neighbour-induced class | class with displacements imposed by a neighbouring cell | 126 |

Other EN wording:
- Line 54: "geometry rotation" became "geometry replacement". It means the cycling of the three GPU-resident geometries, and the new word avoids confusion with the orientations.
- Line 76: "were retained" became "were kept", to avoid confusion with retained DOFs.
- Line 95: "the box" became "the cell box", and "the retained sets ... have a median of 23,604 DOFs" became "the numbers of retained DOFs ... have a median of 23,604".
- Line 66: "classes with labels" became "classes with sensitivity labels". This states what the labels are and adds no new content.

CN, 109 replacements:

| Old | New | Count |
|---|---|---|
| 未校正 | 未修正延续 | 17 |
| 基础网络（加校正）/ 加校正基础网络 | 基础网络加修正 | 7 |
| 一致面力 | 面力载荷 | 14 |
| 节点力 | 集中节点载荷 | 5 |
| 方向 (保留方向, 方向能量误差, 方向集, 方向类别, 方向均值, 采样方向, 验证方向, 方向百分位数) | 测试位移; 载荷类别 (line 97) | 21 |
| 保留 (保留位移, 保留集, 保留值, 保留下来的列, 予以保留, 保留体积, 保留胞元立方体体积) | 主自由度位移, 主自由度数, 主自由度值/分量; 剩余的列; 选用; 剩余体积; 切割后剩余的胞元立方体体积 | 15 |
| 几何分层 / 分层 (as strata) / 角度–体积分层 | 切割程度分组 / 角度–体积组合 | 8 |
| 份额 (ghost 罚项份额, 一阶份额) | 占比 | 4 |
| 能量分数 / 误差分数 / 精确场分数 / 每个分数 (ST05b) | 能量占比 / 误差能量占比 / 精确场能量占比 / 每个占比 | 4 |
| 延拓 | 位移恢复 / 图调和插值 / 求得 | 4 |
| 邻胞元构型, 邻胞元诱导类别 | 相邻胞元构型, 相邻胞元施加边界位移的类别 | 2 |
| generic 校正 (NICE 的校正, 校正实验, 评估时校正, 完整校正, 校正 \(\mathcal W\), 相同校正, 校正后) | 修正 | 8 |

As in A2, generic 校正 became 修正. The standard terms 粗网格校正 (lines 203, 216, 241) and 偏差校正 (EMA bias correction, lines 61 and 66) were kept.

### 1.2 Variant names (brief section 3.3)

All five variants are kept. They are renamed in the R1 key, ST01, ST03, ST03b and ST05a, and in the prose of lines 54, 66, 76, 77 and 99:
- "Uncorrected" became "Uncorrected continuation" (17 occurrences); CN 未校正 became 未修正延续.
- "Base network, corrected" and "corrected base network" became "Base network + correction" (6 occurrences); CN 基础网络（加校正） became 基础网络加修正.

In prose, "Smoothing-trained" now reads "the Smoothing-trained variant". Its table label is unchanged. "Base network" and "NICE" are unchanged.

Record identifiers are unchanged: v2L1, A0_ctrl, A2b_tail8, B2grid, A3_2grid, fresh_val_*, fresh_train_*, the class names `force`, `force_c`, `support`, `support_k`, `face`, `face_c`, `macro`, `grf`, `glued`, `adv`, and Q1(17).

### 1.3 Consistency edits made necessary by the three-variant main text

- Line 3: "Variant labels follow the main text (Table 2); the key below also gives ..." now reads "Variant labels follow the main text (Table 2). The two variants not listed there, the Uncorrected continuation and the Smoothing-trained variant, are evaluated in this supplement. The key below gives ...". The new Table 2 no longer lists these two variants (brief section 3.3), so the old sentence would have implied that it does.
- Line 56: "Training settings, common to the variants of Table 2:" now reads "Training settings, common to the variants of Table 2, the Uncorrected continuation and the Smoothing-trained variant:". This keeps the old claim that the settings apply to all four trained variants. Without the edit, the claim would have been weakened once the two variants left Table 2.

Neither edit adds a number.

### 1.4 Style (brief section 1)

- There were no banned slogans or flourishes in the source chunk.
- The only "—" characters are table placeholders (ST01 line 51; ST05a lines 159–179) and the sentence on line 154 that explains this placeholder ("— marks entries not tabulated"). None is used as punctuation in prose, so all were kept. En dashes in ranges and compounds (G1–G4, 1.20–1.40, work–energy, angle–volume) were kept.
- EN sentences of more than 40 words were split. No EN prose sentence now exceeds 40 words.
  - Line 64: the training times sentence was split. The 60 GPU-hour offline-cost sentence was reordered as "The offline cost of NICE, about 60 GPU-hours, comprises ... This generation covered ... and took about 42 GPU-hours".
  - Line 66: the geometry family is now defined in its own sentence, and the definitions are split in two.
  - Line 84: "In blocks of eight, ... is stratified" became "In each block of eight geometries, ... is drawn by stratified sampling". "is a sampling coordinate, not the material fraction" became "is a sampling coordinate and is not the material fraction".
  - Line 99: the first sentence was rewritten so that "mean / maximum" is explained plainly. The percentile sentence and the resampling sentence were split.
  - Line 140: the column list is now one sentence per column group ("The next three columns ...", "The ghost-penalty column ...", "The last two columns ...").
  - Line 154: the first-order fraction definition was split in two.
  - Lines 199, 203 and 216 were split at existing clause boundaries.
  - Line 95: "rather than a random sample" became "and do not form a random sample".
- CN translationese was fixed and long sentences were split (lines 11, 13, 35, 54, 58, 59, 62, 64, 66, 73, 75, 76, 77, 80, 84, 86, 95, 99, 126, 140, 154, 199, 203, 216, 218, 241). Examples:
  - 并通过八步光滑进行训练 became 训练时施加八步光滑.
  - d0 和 d1 表示切割角所在的 (0,π/4) 的半区间 became 表示切割角位于 (0,π/4) 的哪一半区间.
  - 能量项对数内部取 10^-12 的下限截断 became 能量项对数的自变量以 10^-12 为下限截断.
  - 两者均为最终参数得分较低，因而予以保留 became ……因而选用最终参数.
  - 部署算术下的算子验证 (ST04 title) became 部署运算精度下的算子验证.
  - 其 was reduced in lines 77 and 199.
- Every CN prose sentence is at most 100 characters, with inline math counted as one character. The one exception is line 59, at 103 characters, of which about 80 are the backticked list of ten class identifiers.

## 2. Cross-references renumbered (old → new; same line in EN and CN; each mapped once)

| Line | Old | New | Checked against |
|---|---|---|---|
| 62 | Eq. (18) / 式 (18) | Eq. (13) / 式 (13) | old \tag{18}: training objective with the log energy term |
| 183 | Eq. (15) / 式 (15) | Eq. (10) / 式 (10) | old \tag{15}: cumulative fraction \(\chi_\ell\) with its interior-energy denominator |

That is two changes per language and four in total.

The following references were checked and left unchanged, because the map leaves them the same or they are supplementary or appendix labels:
- main text: Table 2 (lines 3, 56), Table 4 (216), Section 5.1 (64, twice), Section 5.4 (140), Section 5.5 (9, 154, 199), Figure 6 (183), Figure 8a,b (154), Figure 9 (35);
- appendices: Appendix G and Eq. (G.2) (54), Appendix B.3 (140), Appendix F.1 (203), Appendix F.2 (199);
- supplement: Tables ST01, ST02, ST03, ST08b, ST10 and ST13, and Supplementary Section R1.

The chunk contains no Proposition reference, no old Table 5/6/7 reference and no main-text section reference that the map changes.

## 3. Self-check (Python, old chunk compared with new chunk)

(a) Number multiset (digit strings such as 40,000, 0.0965, 1e+03).
- EN: removed {18, 15}, added {13, 10}.
- CN: removed {18, 15}, added {13, 10}.
- These are exactly the two renumbered cross-references. No other number changed, and the per-line multisets differ only on lines 62 and 183.
- The inline-math multiset `\( ... \)` is identical to the source in both languages, and the display equation is byte-identical.
- EN and CN have the same per-line number multisets on every line except four, all present in the source: line 60 (EN "one", CN 1), line 62 (EN "unit weight", CN 权重为 1), line 80 (EN "Twenty", CN 20) and line 113 (EN "these geometries", CN 这 60 个几何).

(b) Old terms that remain, and why:
- EN "retained" throughout: the brief keeps "retained" in EN. This includes line 35 "retained volume" and line 86 "retained cell-box volume", which refer to the volume left after the cut, as in the main text.
- EN "cell box" (86, 95): the main text uses "cell box" (old Section 5.1). Only "box face" is on the map.
- EN "share a pair" (218): a verb, not "energy share".
- EN "stratified sampling" (84, 86) and CN 分层抽样 (84, 86 twice): the standard statistical sampling term, not the coined "cut strata" label. The strata themselves are now "cut-severity groups" / "angle–volume combinations".
- EN/CN "Galerkin projections" / Galerkin 投影 (203): a mathematical term allowed outside the main text.
- CN 粗网格校正 (203, 216, 241) and 偏差校正 (61, 66): standard terms (coarse-grid correction, EMA bias correction).
- "—" as a table placeholder (see 1.4).

None of the following occurs in the new chunk:
- EN: consistent traction, nodal force, direction, stratum, strata, Light/Moderate/Heavy cut, energy share, extension, Uncorrected (without "continuation"), "Base network, corrected", neighbour-induced, box face, cut band, latent, slot, target cell, work-conjugate, nonexpansive, field-based, weakly supported, trial field, improvability, division of tasks.
- CN: 保留, 方向, 一致面力, 节点力, 份额, 延拓, 未校正, 加校正, 邻胞元诱导, 切割带, 胞元边界面, 正是.

(c) Line counts: EN 242, CN 242, source chunk 242.

## 4. Uncertain points for the orchestrator

1. **ST08b label collision (needs a decision).** Brief section 5 maps old main-text Table 5 to "Supplementary Table ST08b". The supplement already has a table ST08b: `#### ST08b. Held-out two-cell configurations (NICE)`, at SUPPLEMENTARY_EN.md line 310. R1 rows 25–33 of this chunk ("W1 ... (Tables ST08b, ST10)", "W2 (Table ST08b)", ..., "RH (Table ST08b)") refer to that existing table, so they were left unchanged.
   - If old Table 5 is placed in Note S8 as "ST08b", one of the two tables needs a new label.
   - If the existing held-out table is renamed, lines 25–33 here must follow.
   - If old Table 5 gets another label (for example ST08c), nothing here changes.
2. **Lines 3 and 56 (section 1.3).** These two edits go slightly beyond term replacement. They keep the supplement consistent with a Table 2 that lists only the main-text variants. If the main-text unit keeps all variants in Table 2 after all, revert both lines to the source wording. In CN, line 3 says 未列入该表 so that no new digit is introduced.
3. **"load class" / 载荷类别** (ST03 title, line 97) matches A1 (line 32) and A3 (line 104). If the main text chooses "test-displacement class" / 测试位移类别, change line 97.
4. **"Cut-severity group" for the G1–G4 table (line 37).** Its values are only Cut / Uncut, so the header reads slightly loosely there. "Cut status" / 切割状态 would be the alternative if the orchestrator prefers it.
5. **Column renames beyond the map.** "First-order share" → "First-order fraction" and "Ghost-penalty share" → "Ghost-penalty fraction". CN: 一阶份额 → 一阶占比, ghost 罚项份额 → ghost 罚项能量占比, ST05b 能量分数 → 能量占比.
   - The main text does not use "first-order share" (checked by grep), so no cross-unit dependency arises.
   - The archive key `first_order_share` only appears in `figures_src`, not in the manuscript.
6. **CN line 203: 支撑筛选 became 列筛选.** This avoids reading 支撑 as a structural support, as A2 did for Appendix F.1, where the CN text describes the screen without naming it. EN keeps "support screen", the name Appendix F.1 (EN) uses.
7. **Line 218: "graph-harmonic extension" became "graph-harmonic interpolation" and "recovered".** Harmonic extension is itself a standard mathematical term. It was replaced because the task lists "extension" for removal. Revert if the orchestrator prefers to keep the mathematical term in the supplement.
8. **Line 154, smoothed values.** EN says "Smoothed values are tabulated for the base network under traction loads". CN keeps 仅 (only), as in the source CN. The table confirms that only the base network's force_c rows carry smoothed values, so both readings are correct.
9. **Line 64, GPU-hours and offline cost.** This line keeps its statements (4.6 h, 3.5 h, 5.0 h, 4.8 h, 29.6 GiB, 7.5 GPU-hours, about 42 GPU-hours, about 60 GPU-hours) because they are in the source and the task requires every number to be kept. Brief section 2 forbids adding such statements, not keeping existing ones in the supplement. Flagged in case the author wants this paragraph reviewed.
10. **Line 126, plural to singular.** "The stiffness-scaled support and neighbour-induced classes" became "The class with stiffness-scaled supports and the class with displacements imposed by a neighbouring cell". ST03c lists seven classes and omits exactly `support_k` and `glued`, so each phrase names one class.
11. **Intermediate snapshot in git.** Commit 12c0b42 ("Update in-progress R5 section drafts") contains an intermediate snapshot of S1_EN.md and S1_CN.md, taken while this unit was still being edited. Use the files on disk, which are final.
