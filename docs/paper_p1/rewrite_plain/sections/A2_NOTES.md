# A2 notes: Appendices D, E, F (APPENDICES_EN.md / APPENDICES_CN.md, lines 204–339)

Files: `rewrite_plain/sections/A2_EN.md`, `rewrite_plain/sections/A2_CN.md`. Each is a full replacement for lines 204–339 of its source file.

## 0. Line count and alignment

- Both files have 136 lines, the same as the source chunk, so the replacement does not shift any later line.
- EN and CN stay line-aligned. Blank lines, headings and display-math lines fall on the same line numbers, and every paragraph sits on the same line in both languages.
- Paragraphs, equations and table rows were not merged, split across lines or reordered. Display-math blocks (D.1)–(D.6), (E.1)–(E.4) and the unnumbered ordering display in D.1 are byte-identical to the source.
- Changed lines (chunk numbering; file line = chunk line + 203):
  - EN: 19, 43, 45, 55, 57, 59, 65, 67, 75, 77, 79, 87, 105, 116, 129, 131, 135.
  - CN: the same lines plus 32, 89 and 123, where only the Chinese wording changed.

## 1. Removed or moved material

Nothing was removed or moved. No sentence, number, citation, equation or table row was deleted. Some long sentences were split at existing semicolons (EN 45, 55, 129, 131, 135; CN 45, 55, 129, 131, 135) without changing their content.

## 2. Cross-references renumbered (each occurrence mapped exactly once, no chaining)

| Chunk line (file line) | Old | New | Checked against |
|---|---|---|---|
| 55 (258) | Eq. (16) / 式 (16) | Eq. (11) / 式 (11) | old \tag{16}: coarse-grid correction energy identity |
| 55 (258) | Section 4.4 / 第 4.4 节 | Section 3.4 / 第 3.4 节 | old 4.4 Coarse-grid correction is now 3.4 |
| 57 (260) | (proof of Proposition 4) / （命题 4 的证明） | (proof of Proposition 2) / （命题 2 的证明） | old Proposition 4 (correction ordering) is now Proposition 2 |
| 65 (268) | Eq. (6) / 式 (6) | Eq. (14) / 式 (14) | old \tag{6}: compliance error identity |
| 65 (268) | Eq. (9) / 式 (9) | Eq. (17) / 式 (17) | old \tag{9}: sensitivity error |
| 79 (282) | Eq. (14) / 式 (14) | Eq. (9) / 式 (9) | old \tag{14}: definition of \(\widehat E q\) |
| 135 (338) | Table 6 / 表 6 | Table 5 / 表 5 | old Table 6 is the cost table and is now Table 5 |

These were left unchanged because the map leaves them the same: Section 5.5 (lines 55, 135), Section 5.10 (131, 135), Sections 5.2–5.8 (135), Section 5.4 (135), Table 4 (135), Appendix D.1 and F.1, appendix equations (D.1)–(D.6) and (E.1)–(E.4), and Supplementary Tables ST05a, ST05b and ST06–ST07.

## 3. Terminology (brief section 3.1)

EN:
- Line 77 heading: "Transpose of the complete extension" became "Transpose of the recovery operator".
- Line 135: "The learned displacement extension" became "The network's interior displacements".
- Line 65: "The field-based sensitivity error" became "The sensitivity error".
- "nonexpansive" was rephrased in three places:
  - line 43: "the polynomial is nonexpansive in the A-energy norm" became "the polynomial ... does not increase the error in the A-energy norm";
  - line 55: "guarantees little more than non-expansion" became "guarantees little more than that the cycle does not increase the energy error";
  - line 75: "suffices for nonexpansiveness" became "suffices for the correction not to increase the energy error".
- Line 59: "A geometry-fixed linear correction" became "A linear correction that is fixed for a given geometry".
- Line 75: "every retained column carries positive energy" became "every column kept in \(V\) carries positive energy". This avoids confusing the columns of \(V\) left after the support screen with retained DOFs.
- Line 105: "preserves retained displacement ... contributes a retained force" became "leaves the retained displacements unchanged ... contributes a force on the retained DOFs".

CN:
- Line 77 heading: 完整延拓的转置 became 位移恢复算子的转置.
- Line 135: 学习的位移延拓 became 网络预测的内部位移.
- Line 65: 场基灵敏度误差 became 灵敏度误差.
- 非扩张 (lines 43, 55, 75) became 不增大……误差 / 不增大能量误差.
- 保留 as the retained-DOF term became 主自由度:
  - line 19: 保留分量 became 主自由度分量;
  - line 87: 直接的保留贡献 became 主自由度上的直接贡献;
  - line 105: 保留位移 / 保留力 became 主自由度位移 / 主自由度上的力;
  - line 135: 给定保留分量 became 给定的主自由度分量.
- Line 89: 在零点处按多项式连续延拓 (an analytic continuation, not displacement recovery) became 在零点处按多项式连续取值. This removes the word 延拓.
- Generic "correction" 校正 became 修正, to match the brief's CN names 基础网络加修正 and 固定网络参数下的修正:
  - line 57 heading: 修正、序关系与近似粗逆;
  - line 59: 线性修正;
  - line 131: 当修正以单精度执行时;
  - line 135: 修正（包括光滑与粗网格代数运算）, 固定网络修正研究, 同一修正.
- The standard terms 粗网格校正 (lines 67, 107, 131) and 子空间校正 (line 55) were kept.

Variant names (brief section 3.3): the chunk mentions only "the base network" / 基础网络 (line 135), which is unchanged. No Uncorrected, "Base network (corrected)", Smoothing-trained or NICE variant label occurs here.

## 4. Banned patterns and plain style

- No em dashes occur in the chunk, before or after the edit. The en dashes in number ranges (78–150, 2.1–5.0%, 0.97–0.99, 4.7–19.4, 5.2–5.8, \(4\)–\(9\times10^{-4}\)) are kept.
- EN line 105: the opening "Indeed," was deleted. The sentence now reads "The final interior field equals ...".
- EN line 67: "the precise condition is" became "the energy removed by the correction is". This describes the left-hand side of (D.6). CN 则精确条件为 became 则修正所减少的能量为.
- EN line 45:
  - "is an estimate of the endpoint, not the containment used above" became "only estimates the endpoint and does not establish the containment used above";
  - "conditional on its actual spectral values" became "conditional on the actual spectrum".
- Lines 19 and 45: "target interval" / "targeted interval" (CN 目标区间) became "smoothing interval" / 光滑区间. This is the same interval that F.2 already calls the smoothing interval, and the change avoids confusion with "target cell".
- EN line 87: "The overwrite" became "The final overwrite of the retained values", the wording the old Section 4.2 uses.
- EN line 116: "The complete reaction is" became "The resulting reaction is" (CN 完整的反力为 became 所得反力为).
- EN line 135: "The correction, smoothing and coarse algebra alike, runs ..." became "The correction, including the smoothing and the coarse algebra, runs ...". The sentence was also split in two.
- CN 正是 was removed:
  - line 131: 这正是附录 D.1 中的平移 became 该平移即附录 D.1 中的平移;
  - line 135: 也正是 became 也是.
- CN translationese fixes:
  - Line 19: 生成式 (D.1) 中多项式的递推格式为 became 式 (D.1) 中的多项式可由如下递推得到.
  - Line 32: 注意到……并应用…… became 利用……及……可验证该多项式.
  - Line 43: 此处 (where) became 该区间上.
  - Line 45:
    - the long 将……与……进行了比较 sentence with its pre-nominal Lanczos modifier was split into two sentences;
    - Euclid 归一化 became 按欧几里得范数归一化;
    - 认证界 became 经过证明的上界.
  - Line 79: Euclid 配对 became 欧几里得内积.
  - Line 105: 事实上 was removed.
  - Line 123: 两种作用 became 前向与转置两种作用 (the EN "both actions" means these two).
  - Line 129:
    - 内部支撑上绝对值之和 became 在内部自由度上绝对值之和, so that 支撑 is not read as a structural support;
    - the sentence was split.
  - Line 131:
    - 支撑筛选 became 上述筛选;
    - （即粗因子平移） became a plain sentence;
    - 使用相同的求解 became 使用相同的粗求解;
    - sentences were split at semicolons.
  - Line 135:
    - 卷积采用真正的 float32 算术 became 卷积按真实的 float32 精度计算;
    - 灵敏度缩并 became 灵敏度计算中的缩并;
    - 实线性映射 became 实数线性映射.

## 5. Self-check (Python, run on the final files)

(a) Number multiset, old chunk compared with new chunk. EN and CN give the same result. The only differences are the renumbered cross-references:
- removed: 16, 4.4, 4 (Proposition), 6 (Eq.), 6 (Table);
- added: 11, 3.4, 2, 17, 5;
- Eq. (9) and Eq. (14) swap places (old 14 became 9, old 9 became 17, old 6 became 14), so 9 and 14 cancel in the multiset.

No other number changed. Within each line, EN and CN also have identical number multisets and identical inline-math multisets.

(b) Old terms that remain, and why:
- "retained" (EN, many lines): the brief keeps "retained" in EN.
- "nodal-force vectors" (EN 79) / 节点力向量 (CN 79): this is the generic nodal-force vector, not the stress-test load class "nodal forces".
- "Ritz residual", "Ritz value" (EN/CN 45): these are standard Lanczos terms, not the Ritz-approximation label of the map.
- "Galerkin matrix" (EN/CN 131): allowed in the appendices.
- 保留 (CN 75, \(V\) 中保留的每一列): this means the columns of \(V\) kept after the screen, not the retained DOFs.
- 粗网格校正 (CN 67, 107, 131) and 子空间校正 (CN 55): these are standard terms.
- "first" (EN 32 "the first step", EN 105 "the first expression"): ordinal use, not a priority claim.

None of the following occurs in the new chunk: box face, cut band, energy share, extension, direction, field-based, consistent traction, weakly supported, slot, latent, target cell, stratum/strata, equilibrium correction, work-conjugate, nonexpansive, trial field, improvability, division of tasks, Uncorrected, Indeed, precisely, or an em dash. In CN, none of 延拓, 场基, 非扩张, 目标, 方向, 正是, 平衡校正 or 一致面力 occurs.

(c) Line count: EN 136, CN 136, source chunk 136.

## 6. Open questions and uncertain points

1. **CN 校正 → 修正 for generic "correction".** I applied this so the text matches the brief's CN names (基础网络加修正, 固定网络参数下的修正). 粗网格校正 and 子空间校正 are kept. If the main-text agents keep 校正 for generic correction, lines 57, 59, 131 and 135 of A2_CN should follow them.
2. **主粗基 (CN 129, "principal coarse basis").** I kept it because the main text uses it (MANUSCRIPT_CN line 318). Next to 主自由度 it may be misread as "coarse basis of the retained DOFs". 默认粗基 or 主要粗基 would be clearer if the main text changes.
3. **"reference box" / 参考立方体 (line 129).** Kept. The old Section 2.1 defines it as \(\mathcal B=[0,1]^3\), and it is not on the map ("box face" is). If the main-text rewrite renames it, update this occurrence.
4. **"operational (upper) endpoint" (EN 45) / 实际使用的端点.** Kept as plain wording.
5. **Line 65 claim, kept as in the source.** "the exact assembled compliance increases monotonically ... the total reconstructed energy error of Eq. (14) decreases". The hypothesis \(\Theta^TA\Theta\preceq A\) gives only non-strict monotonicity. The wording was kept because the brief forbids weakening or strengthening claims. The author may want "does not decrease / does not increase".
6. **Line 55, "the operator ordering stated in Section 3.4".** This ordering is Proposition 2 in the new numbering. The section reference was kept and only renumbered.
