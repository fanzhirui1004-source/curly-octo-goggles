# A3 notes: Appendices G, H, I (APPENDICES_EN.md / APPENDICES_CN.md, lines 340–596)

Files: `rewrite_plain/sections/A3_EN.md` and `rewrite_plain/sections/A3_CN.md`. Each one replaces lines 340–596 of its source file in full. Line numbers below are file line numbers. The chunk line number is the file line number minus 339.

## 0. Line count and alignment

- Both files have 257 lines, the same as the source chunk, so later lines do not move.
- EN and CN are still line-aligned. Blank lines, headings, display-math lines and table rows sit on the same line numbers in both languages. Each paragraph is on the same line in both.
- No paragraph, equation or table row was merged, moved or deleted. Display-math blocks (G.1)–(G.4), (H.1)–(H.7), (I.1), (I.2) and the unnumbered displays are byte-identical to the source. Inline math is unchanged on every line (checked per line, old vs new).
- Changed lines:
  - EN: 340, 344, 346, 348, 350, 359, 361, 363, 373, 375, 383, 391, 406, 408, 419, 421, 432, 434, 441, 443, 448–451, 454, 455, 457, 459, 461, 463, 467, 477, 481, 492, 494, 503, 514, 516, 518, 541, 553, 555, 563, 565, 567, 576, 588, 596.
  - CN: the same lines, plus 385, 397, 404, 423, 430 and 496, where only the Chinese wording changed.

## 1. Removed or moved material

Nothing was removed or moved. No sentence, number, citation, equation or table row was deleted. Some long sentences were split, without changing content:
- EN: 344, 359, 419, 461, 503, 553, 565, 576.
- CN: 344, 348, 359, 419, 461, 503, 553, 563, 565, 576.

## 2. Cross-references renumbered (each occurrence mapped once, no chaining)

| File line | Old | New | Checked against |
|---|---|---|---|
| 344 | Section 4.2 / 第 4.2 节 | Section 3.2 / 第 3.2 节 | old 4.2 (network) is now 3.2 |
| 348 | Section 4.2 / 第 4.2 节 | Section 3.2 / 第 3.2 节 | same |
| 359 | Equation (13) / 式 (13) | Equation (8) / 式 (8) | old \tag{13}: local interaction update \(X^{\ell+1}=\dots\) |
| 383 | Eq. (12) / 式 (12) | Eq. (5) / 式 (5) | old \tag{12}: block form of the condensed action \(\widehat Sq\) |
| 421 | Eq. (14) / 式 (14) | Eq. (9) / 式 (9) | old \tag{14}: definition of the network output \(\widehat Eq\) |
| 494 | Eq. (9) / 式 (9) | Eq. (17) / 式 (17) | old \tag{9}: sensitivity error \(\widetilde s_c-s_c\) |
| 503 | Eq. (10) / 式 (10) | Eq. (18) / 式 (18) | old \tag{10}: complete derivative \(\widehat C_{,c}\) |
| 516 | (proof of Proposition 3) / （命题 3 的证明） | (proof of Proposition 4) / （命题 4 的证明） | old Proposition 3 (sensitivity error) is now Proposition 4 |
| 518 | Eq. (9) / 式 (9) | Eq. (17) / 式 (17) | old \tag{9} |
| 555 | Eqs. (9) and (H.5) / 式 (9) 与式 (H.5) | Eqs. (17) and (H.5) / 式 (17) 与式 (H.5) | old \tag{9} |
| 563 | Eq. (10) (twice) / 式 (10)（两处） | Eq. (18) (twice) / 式 (18)（两处） | old \tag{10} |
| 567 | Eq. (9) / 式 (9) | Eq. (17) / 式 (17) | old \tag{9} |

The following were left unchanged because the map leaves them the same:
- Figure 3 and Figure 3a (344), Figure S01(c) (576).
- Table 2 (419), Table ST15 (419), Supplementary Table ST01 (461).
- Supplementary Notes S5 (419) and S1 (576).
- Sections 5.9 and 5.10 (514).
- Appendices D, H.1 and B.2.
- Appendix equations (E.1), (B.4), (B.5), (G.1)–(G.4), (H.1)–(H.7), (I.1) and (I.2).

## 3. Terminology (brief section 3.1)

EN (CN in brackets):
- direction(s) → test displacement(s) [方向 → 测试位移]:
  - 340, heading: "Network coefficients and directional training" → "Network coefficients and training test displacements" [网络系数与训练测试位移];
  - 432, G.2 heading: "Direction sets" → "Test-displacement sets" [测试位移集];
  - 459, G.3 heading: "Difficult-direction search" → "Search for difficult test displacements" [困难测试位移的搜索];
  - 463, G.4 heading: "Direction coverage" → "Coverage by the test displacements" [测试位移的覆盖];
  - in the text of lines 344, 434, 441, 455, 457, 461, 467, 477, 541 and 596.
- "direction class" (443) → "load class" [载荷类别]. This matches A1 (line 32) and the main text ("loading class"). The table header "Class" [类别] is unchanged.
- latent → feature / grid hierarchy / coarse grid [潜空间 → 特征通道 / 多层网格 / 粗网格]:
  - 344: "latent channels" → "feature channels"; "coarse latent levels" → "coarse grids of levels".
  - 375: "latent hierarchy" → "grid hierarchy". The CN sub-heading 多层级传播 became 多层网格传播.
  - 391: "latent nodes" → "coarse-grid nodes".
  - 406: "hierarchy" → "grid hierarchy".
- slot → local node position [槽位 → 局部节点位置 / 单元局部节点位置]:
  - 350, 359, 408: "slot", "slot embedding", "local slot" → "local node position", "position embedding".
  - 363, an element-only context: 单元局部节点位置. See open question 2.
- box → cell face [胞元边界面 → 胞元表面]: 346 (node indicator), 448, 449, 450 (table rows).
- cut band → cut-plane elements [切割带 → 切割平面单元]: 346 ("cut-band indicator" → indicator of cut-plane-element nodes).
- weakly supported / weak-support / weak node → weakly connected (node) [弱支撑 → 弱连接节点]: 346, 373.
- consistent tractions / consistent loads → traction loads with consistent integration [一致面力 / 一致载荷 → 按一致积分施加的面力载荷 / 载荷]: 448, 449, 451, 492. The phrase "consistently integrated" stays, because it describes how the loads are applied.
- target cell → test cell [目标胞元 → 被测胞元]: 457.
- neighbour-induced traces → displacements imposed by a neighbouring cell [邻胞元诱导的迹 → 相邻胞元施加的边界位移]: 454 (`glued` row). On 565, "At the same trace" → "At the same retained displacement" [在相同迹下 → 在相同的主自由度位移下].
- extension → recovery [延拓 → 位移恢复]:
  - 503: "the two extension terms" → "the two terms that contain the derivative of the recovery operator";
  - 553: "extension error" → "recovery error";
  - 563: "design derivative of the extension error" → "design derivative of the recovery error".
- field-based sensitivity / field estimate → sensitivity / sensitivity estimate [场基灵敏度 / 场估计 → 灵敏度 / 灵敏度估计]:
  - 494: "field-based quadratic estimate" → "quadratic sensitivity estimate";
  - 514: "field-based sensitivities" → "sensitivities";
  - 565, 567: "field estimate", "field-estimate term" → "sensitivity estimate".
- Other wording changes on 563–567:
  - "residual-chain term/contribution" → "residual term" (565, 567), the same term that 563 calls "the residual term of Eq. (18)";
  - "linear field-error term" → "the term linear in the interior displacement error" (563).
- retained → 主自由度 / 主节点 (CN only; EN keeps "retained"):
  - 344, 397: 保留节点 → 主节点;
  - 346: 保留 indicator → 主节点 indicator;
  - 350: 保留位移分量 → 主自由度位移分量;
  - 359, 404: 保留特征 → 主节点特征;
  - 406, 421: 保留值 → 主自由度值;
  - 430: 保留自由度 → 主自由度;
  - 518: 保留方向 → 主自由度位移;
  - 555: 公共保留向量 → 相同的主自由度位移;
  - 477: "在 P 中保留全部自由度" → "P 中全部自由度均作为主自由度".
  - 主节点 follows A1_CN (胞元表面主节点集).
- EN "retained rigid complement" (461) → "the complement of the rigid-body modes in the space of retained displacements" [主自由度位移空间中刚体模态的补空间].
- "stiffness share" (373) → "stiffness fraction" [刚度份额 → 刚度占比]. This keeps it consistent with "energy fraction / 应变能占比". It is not on the map.
- 481: "exact condensed derivative" → "derivative of the exact condensed stiffness" [精确凝聚导数 → 精确凝聚刚度矩阵的导数]. CN 496: 凝聚刚度 → 凝聚刚度矩阵.
- 383: "condensed action of Eq. (12)" → "action of the condensed stiffness in Eq. (5)" [凝聚作用 → 凝聚刚度矩阵的作用].
- 361: "ordered consistently by face direction" → "by face orientation" [按面方向 → 按面的朝向]. Here "direction" is geometric, but changing it avoids confusion with the test-displacement term.

Variant names (brief section 3.3): no variant name occurs in this chunk (Uncorrected, Base network, corrected, Smoothing-trained, NICE as variant). Nothing changed.

## 4. Banned patterns and plain style

- "precisely when" (541) → "if and only if" [当且仅当, unchanged in CN].
- Em dashes: there are none in prose. CN 359 "“收集—混合—散布”" used em dashes as connectors and became 收集、混合与散布. The EN "gather--mix--scatter" (an en-dash compound) is kept. En dashes in ranges (0.5–24) and in "Cauchy–Schwarz" are kept.
- 383: "is obtained ..., not by identifying one transfer map with the transpose of the other" was a "not X but Y" construction. It became "...; one transfer map is not identified with the transpose of the other" [并不将一个传递映射视为另一个的转置].
- 461: "Candidate energy ratios remain directional observations; their maximisation does not supply an all-direction upper bound" → "The candidate energy ratios are observations for individual test displacements; maximising them does not give an upper bound over all test displacements".
- 541: "The constants distinguish derivative coupling, interior coercivity, and reference sensitivity scale" → "The constants separate the coupling through the stiffness derivative, the interior coercivity and the scale of the reference sensitivity".
- 567: "Pointwise variational stiffness dominance does not itself enforce monotonicity of the surrogate compliance" → "That the variational condensed stiffness dominates the exact one at every design does not by itself make the surrogate compliance monotone with respect to the design".
- 576:
  - "interrupt that discrete nesting relation" → "break this discrete nesting relation";
  - "fixed active topology alone does not verify its numerical preservation" → "a fixed active topology alone does not establish that it holds numerically" [并不能保证其在数值上成立];
  - the long "show that ... but that ..." sentence was split in two.
- 588: "Rigid reproduction" → "Reproduction of rigid-body motions" [由刚体运动的再现得].
- 596:
  - "A large value detects an inaccurate direction" → "A large value identifies a test displacement on which the condensed stiffness is inaccurate";
  - "These are real-arithmetic inequalities, with numerical evaluation requiring the stated original quadratic forms" → "These inequalities hold in real arithmetic; their numerical evaluation requires the original quadratic forms stated above".
- CN translationese fixes:
  - 344: 架构 → 网络结构; 相分离 → 分开; the long sentence with a pre-nominal modifier was split.
  - 348: "均值聚合与……拼接后，作为 75 输入节点编码器的输入" → "均值与 11 个节点特征拼接，构成节点编码器的 75 个输入"; the message-passing sentence was split with a colon.
  - 361: 属主侧 → 本侧; 学习通信映射 → 可学习的信息传递映射.
  - 383: the double 因此 was reduced (故 … 因此).
  - 391: 被嵌入 → 放入; 采样回来 → 取回.
  - 419: the 一次标定遍历中所记录的该系数组最大幅值的两倍 chain was split into two clauses.
  - 423: 对……进行采样 → 从……中采样.
  - 430: 活跃自由度 → 激活自由度, as used elsewhere in the chunk.
  - 467: Euclid 范数 → 欧几里得范数, as in A2.
  - 477: 不声称……存在任何上界 → 本文也不给出……的任何上界.
  - 514: 而……则是 → ……则不同，是…….
  - 553:
    - 仅有值误差……并不意味着导数误差为二次 → 仅凭函数值误差……并不能推出导数误差为二次量;
    - the 例如 sentence was split at its semicolon.
  - 563: the double 因此 was removed (因而).
  - 565: the colon-introduced example became its own sentence.
  - 567: the sentence 尽管 \(C'=-1\)，而……仍为负 was rewritten as 而 \(C'=-1\)，灵敏度估计也仍为负.
  - 576: 证得厚度导数的半正定性 → 证得厚度导数半正定; the long 表明 sentence was split.
- Every CN sentence in the chunk is now at most 100 characters, counting each inline formula as one character. Two EN sentences are just above 35 words: 344 (now split) and 383 (one sentence with a semicolon; left as is).

## 5. Self-check (Python, run on the final files)

(a) Number multiset, old chunk compared with new chunk. EN and CN give the same result:
- removed: 4.2 ×2, 13, 12, 14, 9 ×4, 10 ×3, 3 (Proposition);
- added: 3.2 ×2, 8, 5, 9 (old Eq. 14), 17 ×4, 18 ×3, 4 (Proposition);
- every difference is a cross-reference in the table of section 2. No other number changed.

Per line, inline math is identical, old vs new, in both languages. EN and CN of the new chunk have identical inline math on every line. Their number multisets also agree on every line except 467. That difference is already in the source: EN "unit Euclidean norm", CN 范数为 1.

(b) Old terms that remain, and why:
- "retained" (EN, many lines): the brief keeps "retained" in EN.
- "reference box" (EN 453; CN 参考立方体): the cell domain \(\mathcal B=[0,1]^3\) defined in old Section 2.1. It is not "box face". A2 also kept it.
- "approximation target" (EN 477; CN 逼近目标): not "target cell".
- "shared" (EN 350, 359, 454): ordinary English, not "energy share".
- `\boxed` (EN/CN 534, 548): LaTeX.
- "consistently integrated" (EN 448, 449, 451, 492; CN 一致积分): describes how the loads are applied. The term map says this belongs in the appendix.
- "weak-region layers" (EN 373, 404, 408; CN 弱区域层): not on the map. It names the extra layers on stencils with weakly connected nodes.
- "band parameter" (EN 567; CN 带参数): the thickness band parameter \(\tau_c\) defined in Section 2.1. It is not "cut band".
- "variational stiffness error" (545) and "variational condensed stiffness" (567): mathematical terms needed by the proof.
- Code identifiers and record names (`force`, `force_c`, `face`, `face_c`, `support`, `support_k`, `macro`, `grf`, `glued`, `adv`) are unchanged.

None of the following occurs in the new chunk:
- EN: box face, cut band, energy share, extension, direction, field-based, field estimate, consistent traction, weakly supported, slot, latent, target cell, stratum/strata, equilibrium correction, work-conjugate, nonexpansive, trial field, improvability, division of tasks, trace, precisely, Uncorrected, an em dash.
- CN: 保留, 延拓, 场基, 场估计, 非扩张, 方向, 正是, 平衡校正, 一致面力, 一致载荷, 切割带, 边界面, 份额, 潜空间, 槽位, 弱支撑, 迹, 目标胞元.

(c) Line count: EN 257, CN 257, source chunk 257.

## 6. Open questions and uncertain points

1. **"All variants of Table 2" (419).** In the source, Table 2 listed five variants. The new Table 2 lists three. The statement \(k_b=0.5\) is true for all five, but it now reads as covering only the three in Table 2. I kept the reference unchanged, because the map sends Table 2 to Table 2 and I may not add numbers. If the author wants the wider scope, one option is "All network variants (Table 2 and Supplementary Note S8) set \(k_b=0.5\)".
2. **slot → 局部节点位置 vs 单元局部节点位置.** The index \(s\) runs over both element stencils and ghost-face stencils, which have 18 owner-side and nine neighbour-side nodes. 单元局部节点位置 would be wrong for a face stencil. I therefore used 局部节点位置 where both stencil types are meant (350, 359, 408) and 单元局部节点位置 only in the element-only sentence (363).
3. **Class name.** I used "load class / 载荷类别" (443) to match A1 and the main text. Two classes, `macro` and `grf`, are imposed displacement fields rather than loads. If the main-text agents choose "test-displacement class / 测试位移类别", this line should follow them.
4. **`glued` row (454).** "far-face clamping or springs" is read as the far face of the neighbouring cell, as in the old CN. The row now reads "a neighbouring cell that shares the interface degrees of freedom and is clamped or supported by springs on its far face".
5. **"sensitivity estimate" (494, 565, 567).** The brief replaces "field-based sensitivity" with "sensitivity". In H.2 the estimate \(\widetilde s_c\) has to be told apart from the exact \(s_c\) and from the complete derivative \(\widehat C_{,c}\). I therefore wrote "sensitivity estimate" there. If the main text (new Section 4.2, Proposition 4) uses another name for \(\widetilde s_c\), these lines should follow it.
6. **"coarse approximation" / 粗略近似 (588).** Kept as in the source. It may mean a coarse-grid approximation, but I did not narrow it.
7. **"Neumann solve" and "displacement pins" (461; CN 位移约束点).** I kept these as plain technical wording.
8. **CN 修正 vs 校正.** A2 changed generic "correction" to 修正. This chunk contains no generic "correction", so nothing depends on that choice here.
