# S4 notes: Supplementary Notes S6–S7 and supplementary figure captions (SUPPLEMENTARY_EN.md / SUPPLEMENTARY_CN.md, lines 627–887)

Output files: `rewrite_plain/sections/S4_EN.md` and `rewrite_plain/sections/S4_CN.md`. Each is a full replacement for lines 627–887 of its source file. Line N of the source is line N−626 of each output file.

## 0. Line count and alignment

- Line count unchanged: 261 lines in the old EN chunk, the old CN chunk, the new EN chunk and the new CN chunk. The replacement does not shift any line, and line 887 is still the last line of each file.
- EN and CN stay line-aligned. Blank lines, headings and heading levels, table rows and table column counts, figure lines (`![Figure S0N](figures/...)`) and caption lines (`**Figure S0N. Title.** ...`, `**Table ST2N. Title.** ...`) are on the same line numbers in both languages.
- No paragraph, table row, heading or figure was added, removed, merged or reordered. Some long sentences were split inside their own line.
- Every table data row is byte-identical to the source except the first cell of the ST22 rows 845–848 (class names) and the ST19b and ST20 header rows 795 and 808 (CN 808 only).
- All inline mathematics `\( ... \)` is identical to the source in both languages (checked as a multiset).
- Changed lines:
  - EN (39): 629, 633, 640, 641, 642, 652, 653, 657, 661, 716, 720, 722, 724, 742, 778, 795, 800, 804, 817, 829, 831, 833, 835, 837, 839, 841, 843, 845, 846, 847, 848, 854, 856, 867, 869, 875, 879, 883, 887.
  - CN (45): the same lines except 843 (CN header 载荷类别 was already correct), plus 651, 655, 694, 791, 806, 808 and 815, where only the Chinese wording changed.
- No supplement additions (no material in this chunk moves to Note S8).

## 1. Terminology (brief section 3.1)

Counts are occurrences in the source chunk; all were replaced. EN 63 + 7 variant names = 70; CN 86 + 7 variant names = 93.

### 1.1 EN

| Old | New | Lines | Count |
|---|---|---|---|
| consistent traction(s), consistent face traction, consistent-traction, unit consistent tractions; "consistent nodal load" | traction load(s), face traction load, traction-load classes; "the nodal load obtained by consistent integration of the traction load" (wording of Appendix H in A3, line 153) | 640, 641, 661, 804, 837 (2), 839 (3), 845, 846, 875, 879 (2), 883 (3) | 17 |
| nodal forces, nodal-force, single-face-force | nodal point loads, single-face nodal point loads | 839, 847, 848, 879 (2), 883 | 6 |
| directional energy error; validation directions; over the directions; evaluated directions; spring-support directions | energy error; validation test displacements; over the test displacements; evaluated test displacements; spring-support test displacements | 837 (3), 841, 854, 879 (2), 883 (2) | 9 |
| loading class / Loading class | load class / Load class (as S1, ST03) | 837, 843 | 2 |
| cut band, cut-band DOF, cut-band flag, Cut-band nodes, cut-band membership | DOFs of the (clamped) cut-plane elements, the private flag of the cut-plane elements, Cut-plane-element nodes, nodes of the cut-plane elements | 653, 720, 724, 795, 800, 817 | 6 |
| Weak-support nodes; weak support | Weakly connected nodes; weakly connected nodes | 795, 800 | 2 |
| field-based gradient estimate; Field-based estimate; field-based NICE estimate; Field-based sensitivity-error diagnostics | estimated gradient; Sensitivity estimate; NICE sensitivity estimate; Sensitivity-error diagnostics | 633, 642, 817, 883 | 4 |
| transpose of the complete extension (2); network extension is admissible | transpose of the complete recovery operator; the network's displacement recovery reproduces the retained displacements | 829, 833, 867 | 3 |
| admissible | reproduces the retained displacements | 833 | 1 |
| stratum | cut-severity group | 831, 869 | 2 |
| box face; the box | cell face; the cell box | 875, 831 | 2 |
| energy share; norm share; Shares of | energy fraction; norm fraction; Fractions of | 887, 883 (2) | 3 |
| target-cell; the target's | test-cell; of the test cell | 887 (2) | 2 |
| retained space | retained DOFs | 831 | 1 |
| the network's initial field | the network's interior displacements | 837 | 1 |
| neighbour-induced displacement class; neighbour-face z traction | the class with displacements imposed by a neighbouring cell; the z traction load on the face of the neighbouring cell | 879, 887 | 2 |
| "the complete surrogate derivative of Eq. (10)" | "the complete derivative of the surrogate compliance, Eq. (18)" (wording of A1/S3) | 642 | (counted under cross-references) |

EN keeps "retained" (retained DOFs, retained displacements, retained volume fraction), as the brief requires.

### 1.2 CN

| Old | New | Lines | Count |
|---|---|---|---|
| 一致面力 / 单面一致面力 / 一致节点载荷 | 面力载荷 / 单面面力载荷 / 由面力载荷经一致积分得到的节点载荷 | 640, 641, 661, 804, 837 (2), 839 (3), 845, 846, 875, 879 (2), 883 (3) | 17 |
| 节点力 / 单面节点力 / 单面力 | 集中节点载荷 / 单面集中节点载荷 | 839, 847, 848, 879 (2), 883 | 6 |
| 方向能量误差 / 验证方向 / 在每个胞元的方向上 / 所评估方向 / 和方向上 / 弹簧支撑方向 | 能量误差 / 验证测试位移 / 测试位移 | 837 (3), 841, 854, 879 (2), 883 (2) | 9 |
| 切割带 | 切割平面单元 | 653, 720, 724, 795, 800, 817 | 6 |
| 弱支撑 | 弱连接 | 795, 800 | 2 |
| 场基 | 灵敏度估计值 / 删除 | 633, 642, 817, 883 | 4 |
| 延拓 | 位移恢复算子 / 位移恢复 | 829, 833, 867 | 3 |
| 可容许 | 在主自由度上等于给定位移 | 833 | 1 |
| 分层 | 切割程度分组 | 831, 869 | 2 |
| 胞元边界面 | 胞元表面 | 875 | 1 |
| 份额 / 能量份额 | 占比 / 应变能占比 | 883 (2), 887 | 3 |
| 目标胞元 | 被测胞元 | 887 (2) | 2 |
| 保留 (DOF sense: 保留自由度, 自由保留自由度, 保留空间, 保留系统) | 主自由度, 未约束主自由度, 主自由度方程组 | 795, 800 (2), 804, 806 (2), 808, 831 (2), 856 | 10 |
| 保留体积分数 (volume left after the cut) | 剩余体积分数 (as S1, S2) | 720, 791, 804 | 3 |
| 保留 (verb "keep") | 沿用 / 取 | 831 (3) | 3 |
| generic 校正 (校正 \(\mathcal W\), 校正 8 / \(Q_1(17)\) / 8, 原样的校正 …) | 修正 (as A2, S1) | 829, 831, 835 (2), 837, 867 (2) | 7 |
| 邻胞元诱导位移类别 / 邻胞元面 z 向面力 | 相邻胞元施加边界位移的类别 / 相邻胞元表面 z 向面力载荷 | 879, 887 | 2 |
| 网络初始场 | 网络预测的内部位移 | 837 | 1 |
| 胞元盒; 活动单元 | 胞元立方体 (as S1); 激活单元 (as A1) | 831 | 2 |
| 正是; 恰恰 | 删除 | 837; 633 | 2 |

## 2. Variant names (brief section 3.3)

The supplement keeps all five variants.

| Old EN → new EN | Old CN → new CN | Lines | Count |
|---|---|---|---|
| Uncorrected → the Uncorrected continuation | 未校正 → 未修正延续 | 879, 883 (3), 887 | 5 / 5 |
| base network with the correction applied at deployment → Base network + correction (the deployment-only meaning is kept in one clause: "applies the correction to the base network at deployment", "which applies the correction at deployment") | 部署时施加校正的基础网络 → 基础网络加修正（另以一句说明其含义） | 879, 887 | 2 / 2 |
| Smoothing-trained → "the Smoothing-trained variant" in running prose (887); the label in the 879 list is unchanged | 平滑训练 → 平滑训练变体 (887) | 887 | wording only |
| Base network, NICE | 基础网络、NICE | unchanged | 0 |

The S7 labels *NICE, zero-shot* and *NICE, continued* (record A3G_ft) are not among the five variants and are unchanged. Record identifiers and file names are unchanged: A3G_ft, fresh_val_*, gval_*, glat222, force_c, face_c, force, face, support, support_k, macro, grf, `evidence/gcell/*.json`, `evidence/newval_A3_2grid.json`, and the figure files `S06_reference_verification.png`, `S01_distributions.png`, `S03_sensitivity_diagnostics.png`, `S07_energy_share_variants.png`.

## 3. Cross-references to the main text

Renumbered (old → new). Each change occurs once on the given line in EN and once in CN, so 13 changes per language, 26 in total. Each occurrence was mapped once from the source text; no replacement was applied to an already replaced number.

| Line | Old EN | New EN | Old CN | New CN | Basis |
|---|---|---|---|---|---|
| 633 | Section 3.3 | Section 4.2 | 第 3.3 节 | 第 4.2 节 | old 3.3 (sensitivity error) → 4.2 |
| 633 | Table 6 (2×) | Table 5 (2×) | 表 6 (2×) | 表 5 (2×) | old Table 6 (cost of one lattice analysis) → Table 5 |
| 642 | Eq. (10) | Eq. (18) | 式 (10) | 式 (18) | old \tag{10}: complete derivative of the surrogate compliance |
| 652 | Table 6 | Table 5 | 表 6 | 表 5 | timed route of the cost table |
| 716 | Table 7 | Table 6 | 表 7 | 表 6 | old Table 7 (thickness optimisation cases) → Table 6 |
| 742 | Table 7 | Table 6 | 表 7 | 表 6 | same |
| 778 | Section 3.3 | Section 4.2 | 第 3.3 节 | 第 4.2 节 | derivatives on a fixed discrete model |
| 833 | Propositions 1–3 | Propositions 1, 3 and 4 | 命题 1–3 | 命题 1、3 和 4 | old 1 → 1, old 2 → 3, old 3 → 4 |
| 833 | Section 4.1 | Section 3.1 | 第 4.1 节 | 第 3.1 节 | old 4.1 (variational condensed stiffness: symmetry, semidefiniteness, null modes) → 3.1 |
| 833 | Proposition 4 | Proposition 2 | 命题 4 | 命题 2 | old Proposition 4 (correction ordering, spectral condition) → 2 |
| 835 | Section 4.5 | Section 3.5 | 第 4.5 节 | 第 3.5 节 | old 4.5 (training) → 3.5 |
| 837 | Eq. (5) | Eq. (7) | 式 (5) | 式 (7) | old \tag{5}: energy error \(\varepsilon(q)\) |

Mapped, but the written numbers are unchanged:
- 829 "Sections 3 and 4" / 第 3 节和第 4 节: old 3 → 4 and old 4 → 3, so the pair maps onto itself. It is written in ascending order. In the new order Section 3 holds the construction and Propositions 1–2 and Section 4 holds Propositions 3–4, so the reference still covers the construction and all four propositions.
- 829 "Propositions 1–4" / 命题 1–4: the set {1, 2, 3, 4} maps onto itself.

Checked and unchanged:
- main text: Section 2.3 (831), Section 5 (829), Sections 5.1 (629, 720, 837, 841), 5.8 (661, 831, 856 twice, 858), 5.8–5.10 (833), 5.9 (661), 5.10 (629, 642, 720), 6.1 and 6.3 (829), 6.3 (720, 867); Tables 1 (831) and 2 (829, 831, 835); Eq. (1) (829, 831); Figure 11 (887, twice); Figure 15a,b (742);
- appendices: Appendix A.1 (778), F.1 (652, 800), G.1 (800), G.2 (837), H (640), Eq. (H.3) (642, 643, 817);
- supplement: Notes S4.1, S6.1–S6.5, Section R1, Tables ST01, ST02, ST03, ST05a, ST08, ST09, ST10, ST16–ST23.

The chunk contains no reference to old Table 5 (→ ST08b) and no reference to old Sections 3.2 or 4.2–4.4, old Eqs. (11)–(18) or old Propositions in any other form.

## 4. Style (brief section 1)

- EN, "not X but Y" and aphoristic forms rephrased without change of content:
  - 661 "The twin is a verification run, not a cost baseline" → "The twin serves as a verification run and is not used as a cost baseline".
  - 724 "narrows the macroscale underestimation rather than widening it" → "narrows the macroscale underestimation". "Narrows" already states the direction.
  - 829 "..., not the level-set function of Eq. (1)" → "and it does not use the level-set function of Eq. (1)".
  - 837 "what does not carry over is the accuracy of the network's initial field, which belongs to the family …" → "The accuracy of the network's interior displacements does not carry over, since it is tied to the family …".
  - 867 "needs new training data (Section 6.3), but not a new construction" → "needs new training data (Section 6.3); the construction itself needs no change".
- EN sentences split at existing clause boundaries: 629, 633, 657, 720, 722, 724, 778, 800, 804, 831, 835, 837, 839, 856, 879, 883. The remaining EN sentences above 40 words are lists of settings or numbers (657, 800, 806, 815, 858, 875) and a definition (722).
- EN 829 "that is, …" apposition became "comprises …". EN 841 and 879 titles lose "Directional".
- No em dash is used as punctuation in prose. The "—" characters that remain are empty-cell markers in Tables ST17a, ST17c, ST18a and ST18c. En dashes in ranges and compounds (0–3, 27.0–31.8%, primal–dual, normal–shear, H1–H3) are kept.
- CN: 恰恰 (633) and 正是 (837) removed. Translationese fixed:
  - 633 梯度范数和 KKT 残差均不用于停止判据 → 停止判据不采用梯度范数或 KKT 残差; 设计迭代与表 6 中的分析不同之处在于 → 与表 5 中的分析相比.
  - 641 objective row reordered: 柔度 \(\widehat C\)：在……作用下，取最终共轭梯度迭代解对应的功.
  - 694, 724 体积界被违反 → 体积约束未得到满足.
  - 716 相对于孪生设计的精确柔度为 → 与孪生设计精确柔度的相对差为.
  - 722 节点被等同 → 视为同一节点; 它 → 宏观模型; 被切割面相交的单元 → 与切割面相交的单元.
  - 817 两个满足体积约束 \(V^*\) 的均匀设计 → 两个体积等于 \(V^*\) 的均匀设计. "At the volume bound" means that the volume equals \(V^*\).
  - 815 精确核查 → 精确检验 (same term as Table ST21).
  - 其 reduced in 835 and 856.
- CN sentences over about 100 characters were split (657, 724, 800, 804, 831, 835 and others). Inline math counts as one character. The remaining long sentences are:
  - 815 (132 characters): a list of exact-check numbers;
  - 831 (103): the first sentence, which contains the gyroid level-set formula and the Table 1 settings.

## 5. Self-check (Python, scratchpad script)

(a) Number multiset (regex `\d+(?:[.,]\d+)*`), old chunk against new chunk:
- EN removed {6: 1, 3.3: 2, 10: 1, 7: 1, 4.1: 1, 4.5: 1}, added {2: 1, 4.2: 2, 5: 2, 18: 1, 3.5: 1, 3.1: 1}.
- CN removed and added the same sets.
- Per line, every difference is one of the cross-references of Section 3:
  - 633: 3.3 → 4.2, 6 → 5 twice;
  - 642: 10 → 18;
  - 652: 6 → 5;
  - 716, 742: 7 → 6;
  - 778: 3.3 → 4.2;
  - 833: "1–3" → "1, 3 and 4" adds 4, 4.1 → 3.1, Proposition 4 → 2;
  - 835: 4.5 → 3.5;
  - 837: 5 → 7.
- No other number was added, removed or changed. EN and CN have the same number multiset on every line except 724, where the source already differs (EN "six-element mesh", CN "6 个单元"). An intermediate CN draft had added a second "0" on line 629; this was corrected.
- The inline-math multiset is identical to the source in both languages.

(b) Old terms that remain, and why:
- EN "retained" (retained DOFs, retained displacements, retained volume fraction, free retained, retained system): the brief keeps "retained" in EN. "Retained volume fraction" (720, 791, 804) is the volume left after the cut, as in the main text and in S1/S2.
- EN "search direction" (633), "three Cartesian directions" (875) and CN 搜索方向, \(y\) 方向, 沿长边方向, 笛卡尔坐标方向 (633, 661, 804, 875): geometric directions, not the coined "direction" (test displacement).
- EN "consistent integration" (640) and CN 一致积分 (640): the integration method, worded as in A3 line 153. The term "consistent tractions" itself is gone.
- EN "cell box" (831): the main-text and S1 term; only "box face" is on the map.
- EN "shared faces", "share the … axis" (804, 831, 879, 887): verbs and adjectives, not "energy share".
- EN/CN "Schur-complement option" / Schur 补选项 (817): the name of the PARDISO option.
- EN "Weak-region" / CN 弱区 (795, 800): the stencil-layer name used in A1, A3 and S3; only "weakly supported nodes" is on the map.
- CN 目标函数 (641, 650, 778): "objective function", not 目标胞元.
- CN 相邻胞元 (831, 879, 887): ordinary "neighbouring cell", the map's own wording, not 邻胞元诱导.
- "—" in tables: empty-cell markers.
- None of the following occurs in the new chunk:
  - EN: consistent traction(s), nodal force(s), directional, validation directions, loading class, cut band / cut-band, weak support, field-based, extension, admissible, stratum, box face, energy share, norm share, target, retained space, initial field, neighbour-induced, latent, slot, trial, work-conjugate, nonexpansive, improvability, division of tasks, equilibrium correction, "Uncorrected" without "continuation", "base network with the correction applied at deployment".
  - CN: 一致面力, (non-集中) 节点力, 方向能量误差, 切割带, 弱支撑, 场基, 延拓, 可容许, 分层, 胞元边界面, 份额, 目标胞元, 保留, 未校正, generic 校正, 邻胞元诱导, 初始场, 正是, 恰恰, 胞元盒, 活动单元.

(c) Line counts: EN 261 and CN 261, the same as each source chunk (`wc -l`, final newline included). Blank lines, headings, table pipes per row and figure lines match line by line.

## 6. Uncertain points for the assembler

1. **Line 831, last sentence (scope kept, wording extended).** "No gyroid cell entered the training or checkpoint selection of any variant of Table 2" became "… of any variant of Table 2 or of the two variants evaluated only in this supplement". The new Table 2 lists three variants. Without the addition, the claim would silently drop the Uncorrected continuation and the Smoothing-trained variant, which the old Table 2 covered. This follows S1's handling of line 56. Revert if Table 2 keeps all five variants.
2. **Figure S02 caption (879): "use the markers and colours of the main-text figures".** The main-text figures now show three variants (task "relabel figures with new terms and three variants"). The markers and colours of the Uncorrected continuation and the Smoothing-trained variant may no longer appear in any main-text figure. The sentence was left as it is. If those two variants are no longer drawn in the main text, it should read "… of Figures S02–S04" or name the markers.
3. **Figure images.** The captions of S02, S03 and S04 now say "Uncorrected continuation", "Base network + correction", "energy fraction" and "test cell". If the panel labels or legends inside these PNGs still read "Uncorrected", "energy share" or "target", the images need the same relabelling as the main-text figures. The file name `S07_energy_share_variants.png` was not changed, because build scripts reference it.
4. **Line 856, clarification.** "The learned operators add 20 and 22 iterations" became "on the gyroid lattice, the learned operators add 20 (zero-shot) and 22 (continued) iterations". The source does not say which is which. The assignment follows from Table ST23 (434 − 414 = 20 for zero-shot, 436 − 414 = 22 for continued; the P block adds 3 and 4). No number was added.
5. **Line 829, "Sections 3 and 4".** See Section 3: the pair maps onto itself under the new order. If the assembler prefers the specific new subsections, "Sections 3.1 and 3.4 and Propositions 1–4" would be more specific. It was not done, because the source cites whole sections.
6. **Line 833, "the properties of Section 4.1" → "Section 3.1".** Old 4.1 holds the symmetry, semidefiniteness and rigid-body null-space statements. New 3.1 merges old 4.1 with old 3.1 (Proposition 1), so 3.1 is the correct target.
7. **Lines 633 and 642, "field-based".** "From the field-based gradient estimate" became "from the estimated gradient". The ST16 "Gradient" row became "Sensitivity estimate of Section 5.10". The definition of the estimate (\(-\widehat u^TK_{,c}\widehat u\), ignoring the design dependence of the network) is given once in the main text (brief section 3.1), so it is not repeated here.
8. **Line 640, "consistent nodal load".** This became "the nodal load obtained by consistent integration of the traction load", worded as in Appendix H (A3 line 153), so that "consistent" survives only as the integration method.
9. **"Reading." (867).** The bold paragraph lead was kept, CN 解读. "Summary." / 小结 would be the plainer alternative.
10. **CN "降低 5.8 至 10.2 倍" (839) and "降低 1.6 和 1.2 倍".** These were kept as in the source. Some editors prefer "降为原来的 1/5.8 至 1/10.2". That form was not used, because it adds digits to the line.
11. **Intermediate snapshot in git.** Commit f9b3c48 ("Scale figure draft: …"), made by another process while this unit was in progress, contains an intermediate version of S4_EN.md and S4_CN.md. It differs from the final files on about ten lines. Use the files on disk, which are final.
