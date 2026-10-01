# 通读审查 R3（2026-10-01）：过度防御、正文残留、术语、叙事

作者要求：通读全文，找出过度防御、不该放在正文叙事里的内容（像当年的 H2/y、像 NICE-post 这类）、不合适的术语（要用计算力学的专业术语，不要自造词）、行文不畅，以及我自己拓展的其他问题，并想好怎么改。

做法：我自己通读了主文（摘要到结论，16,657 词）、附录目录与 I、J 两个附录、补充材料 R1/R2；之前启动的多视角子任务里"过度防御"和"正文残留"两份清单已经完成（各 56、58 条），按作者指示并入本文，其余视角（术语、叙事、精度、图表、附录）是我自己审的。引文一律英文原文，改法给英文替换文本；改法后标"[机械]"的不需要作者决定，标"[决定]"的需要作者拍板。

---

## 0. 总评

1. **主线已经立住。** 摘要、引言、§2–§4 基本是干净的论证：问题—两条路线—三要素—误差关系—构造。问题集中在 **§5.1–5.2、§5.9–5.11、§6 和几条长图注**，那里混进了三层不该有的东西。

2. **第一层：对想象中审稿人的辩护。** 同一条保留意见被说了三到五遍：
   - "误差是对离散模型而不是对连续问题"：2.1、§5 路线图、5.2（两次）、5.11、6.4，共 6 次；
   - "报告的灵敏度是场基估计，不是代理目标的梯度"：3.3（两次）、5.11、6.2、6.4、7，共 6 次；
   - "保证的是排序不是速率 / 精度是经验的"：4.3（三句）、6.1、6.2、7；
   - "没有和整体迭代求解器比 / 不声称有优势"：5.10、6.3、6.4；
   - "不含数据生成与训练的一次性成本"：5.10、6.3、6.4、7；
   - "一个训练种子"：5.3、6.4；"大规模未验证、优化是局部的"：5.11（两次）、表 6、6.4、7；
   - Guo 等 (2026a) 表 1 的数字：引言、6.1 各一遍；"逐胞验证而非先验保证"：5.2、6.2。
   另有几段整段是在和审稿人辩论：5.2 的参考解验证（逐胞的网格收敛叙述、罚参数扫描、差分步长研究，约 250 词）、5.9 的残差停滞段（同一件事用三种方法证明，约 150 词）、6.2 的中心差分失败记录（约 150 词）、表 6 图注解释 141 s 为什么不等于 81 s（约 110 词）、5.8 的"这不是对边界缩减方法的建模"（60 词三处引用）、5.5 的"弱但便宜的比较对象"。
   **政策（建议采纳）：每条保留意见只在它的定义处说一次**（离散参考→2.1；场基估计→3.3 式 (9) 处；排序→4.3；谱包含逐胞验证→5.2），6.4 统一收集一次；结果节只写测了什么、对照什么；结论不再重复。估计可去掉 1,300–1,600 词（正文 8–10%）。

3. **第二层：开发过程残留。** 训练后勤（中断/恢复/日志、检查点打分时刻、"四个检查点里选"、"指定为主预测器"、池替换周期、1.05 TB、"三个更早的训练阶段"、"去掉了一个近机构胞"）、评估覆盖簿记（"Uncorrected 评了 11 个配置，其中 9 个……基础网络评了 7 个"、图 9 图注的 H3/L1 覆盖说明、图 10 的"52 个组合，不含 L1"）、求解器取证（"instrumented re-solves"、对偶范数界、单精度重跑）、实现琐事（TF32、主机内存 90 GiB、"host evaluation whose seeded generator"、SuperLU 失败尝试、32 线程、几何生成回退 1+1e-4）、**内部变体名 NICE-post**（它不是一个训练出来的预测器，就是"基础网络 + 部署时加校正"）、"Two predictors from earlier comparisons appear in the supplement only"（P0、S8）、图文件名里的运行号（F02_validation_A3、F05_assembly_A3、F10_energy_participation_r1）、文件末尾的 `<!-- restructured r2 -->`。还有一处**悬空引用："Sections 5.1 and 6.5"**（6.5 已并入 6.4）。这一层约 600–900 词。

4. **第三层：术语。** 自造词或机器学习味的词和标准词并用：energy excess 与 energy error 并存（同一个量 ε）；"read out"；"complete transpose"；"energy-consistent operator application"；"polynomial relaxation" 与 "smoothing" 并存；"interior coarse correction / coarse update / coarse solve / Galerkin solve" 四个名字；"retained coordinates / coefficients / degrees of freedom" 三个名字；"macro-box / macro-cut / macro-domain"；"direction banks"；"predictor"；"weights / updates / pool / checkpoint / weight selection"；"host"、"front end"、"instrumented"；"development cells"、"diagnostic assembly"。符号重载：W 三个意思、V 两个、C 三个、R 两个、D 两个。对照表见 §3。

5. **叙事。** 结构（1–7）是对的，但：引言第 4 段 400 词 25 条引用一气呵成；贡献 (iii) 一句 130 词把 §5 全列了一遍；§3 开头一句 120 词、9 条引用的"哪些是经典结果"；4.2 里插了一段相关工作；式 (18) 放在 4.6 的算法节里显得游离；5.7 只有一段，和 5.6 是同一件事；§6.1 第一段重复引言；§7 第三段是摘要数字的再版。见 §4。

6. **数量估计。** 以上全做之后正文约 14,300–14,800 词，表 2 少一行、表 5 少一列、§5 由 11 节变 10 节、§6.2 缩一半。

---

## 1. 过度防御

### 1.1 重复的保留意见：保留哪一次、删哪几次

| 保留意见 | 现在出现 | 保留 | 删去或改写 |
| --- | --- | --- | --- |
| 误差对离散模型而非连续问题 | 2.1；§5 路线图末句；5.2 两句；5.11 "All design comparisons are made within this discrete model…"；6.4 | 2.1 定义；6.4 一句（含重切胞参考解的量级） | 5.2 的两句 "The comparisons therefore measure the approximation of the discrete operator, not of the continuum. This discrete model also defines the design problem, so a coarser reference would change the model rather than the cost of reaching it; the surrogate's energy error lies well below … under refinement." → "All comparisons that follow are made against this discrete reference (Section 2.1)."；5.11 的整句删；§5 路线图末句 "All errors are measured against equilibrium of the discrete problem of Section 2." 可留（一句话）[机械] |
| 场基估计不是代理梯度 | 3.3 "All sensitivities reported in this paper are the field-based estimates … are not the derivative of the surrogate compliance."；5.11 "not the complete surrogate derivative of Eq. (9)"；6.2 "It therefore estimates the exact sensitivity and is not the gradient of the surrogate objective…"；6.4 "The reported sensitivities are field-based estimates that omit…"；7 "The field-based sensitivity estimates the exact gradient and is not the gradient of the surrogate objective, a distinction that design use must respect." | 3.3（定义处）；6.4 一句 | 5.11 句尾删 "not the complete surrogate derivative of Eq. (9)"；6.2 改为 "For optimisers with a line search this inconsistency between objective and gradient matters (Section 3.3)."；7 整句删 [机械] |
| 排序而非速率 / 精度是经验的 | 4.3 末三句；6.1 "the guarantee is an ordering, not a rate, since the reductions … are empirical, and it imposes no monotonicity…"；6.2 开头 "…which Section 5.2 verifies per geometry rather than a priori; the accuracy is empirical."；7 "in every example a better initial field or a larger correction budget reduced it" | 4.3 一次 | 4.3 末三句改为 "The ordering is guaranteed; the size of the reduction is not (Appendix D), and Section 5.5 measures it. It does not extend to the sensitivity error of Eq. (8) (Appendices J.8 and J.9), and changing the smoothing degree changes the polynomial, so the reduction is not monotone in k."；6.1 改为 "An internal update that does not increase the A-energy error for any retained input yields a condensed stiffness between the learned stiffness and the exact Schur complement, independently of the network (Section 4.3)."；6.2 开头改为 "The variational properties hold for every input, and the ordering for every geometry whose spectrum the smoothing interval contains (Section 5.2)."；7 改为 "…and in every example a better initial field or a larger correction budget reduced the energy error."（5.5 里灵敏度误差是非单调的，"in every example…reduced it" 对灵敏度不成立）[机械] |
| 未与整体迭代求解器比较 | 5.10 末句 "NICE's lattice solve is itself an inexact substructuring iteration; fine-scale iterative solvers of the whole lattice were not compared (Section 6.4)."；6.3 "not a faster fine-scale solve, and no advantage over iterative solvers of the whole lattice is claimed"；6.4 | 6.4 | 5.10 末句删；6.3 改为 "Its output is a condensed operator per cell with a quantified and reducible error; whole-lattice iterative solvers are a different route (Section 6.4)." [机械] |
| 不含数据生成与训练成本 | 5.10 "none includes the one-off cost of data generation and training (Section 5.1)"；6.3；6.4；7 "excluding the one-off cost of data generation and training" | 5.10（成本比较的定义处）；6.3 的回本估算本身就是在算这件事 | 6.4、7 删 [机械] |
| 一个训练种子 | 5.3 "(95% bootstrap interval over geometries 1.20–1.40; one training seed)"；6.4 | 6.4 | 5.3 括号改为 "(95% bootstrap interval over geometries 1.20–1.40)" [机械] |
| 大规模未验证 / 优化是局部的 | 5.11 "NICE's accuracy at these sizes was not verified against the exact reference."；5.11 "(an extrapolation)"；表 6 "Not checked"；6.4；7 "its accuracy at that size was not verified, and the optimisations are local" | 表 6 的 "Not checked"；6.4 | 5.11 两处删；7 改为 "A design iteration of a 110-cell plate with 32.7 million degrees of freedom takes 43 min on one GPU." [机械] |
| Guo 等 (2026a) 表 1 的数字 | 引言第 4 段三句；6.1 "For orientation only, since error definitions, problems and baselines differ: Guo et al. (2026a, Table 1) report displacement errors of 7.76–13.64% …" | 引言，压成一句（见 §4.1） | 6.1 整句删 [机械] |
| 谱包含逐胞验证而非先验 | 5.2 "The containment is thus verified per geometry, not guaranteed a priori; symmetry, positive semidefiniteness and Ŝ ⪰ S do not depend on it."；6.2 开头 | 5.2 | 6.2 按上一行改 [机械] |

### 1.2 整段的辩论（改写文本）

**(a) 5.2 参考解验证**（现约 420 词，改为约 150 词）。替换 "To verify the discrete reference, …" 到 "…no change of integration branch at the fixed active set." 的整段为：

> The n = 32 reference was verified by refinement to n = 48–64 on six cells (Supplementary Note S2, Table ST14): the compliance of uncut and moderately cut cells changes by at most 0.06% and their eight-corner sensitivities by at most 0.11%, those of heavily cut cells by up to 1% and 3.6%. Varying the ghost-penalty coefficient between 10⁻⁵ and 10⁻³ and refining the volume integration change compliance and sensitivities by at most 0.12%, and the finite-difference stiffness derivative and its step are verified in Appendix H. All comparisons that follow are made against this discrete reference (Section 2.1).

逐胞的 H1 非单调、H2 到 n=64 未收敛、最薄胞、最大误差胞的数字留在补充材料 Note S2 / 表 ST14。[机械；数字已在 ST14]

**(b) 5.2 算子验证**。"To verify the operator: the accuracy runs evaluate the network in single precision and the stiffness actions, smoothing and coarse solve in double precision (Table 1)." → "To verify the operator (arithmetic as in Table 1):"；删 "Four of the per-cell margins in Table ST04 come from a host evaluation whose seeded generator gives a different power-iteration start."（这句还碰到了"多台机器"的红线）；"the power-iteration endpoint used by all GPU runs exceeds" → "the power-iteration endpoint exceeds"；末句改为 "so the spectrum containment holds on every geometry tested; symmetry, positive semidefiniteness and Ŝ ⪰ S do not depend on it." [机械]

**(c) 5.9 残差停滞段**（约 190 词 → 90 词）。替换 "The learned solves reach the prescribed recursive residual, …" 到 "…with the compliance error unchanged from a recursive residual of 10⁻³ onwards." 为：

> The learned solves reach the prescribed recursive residual, while the recomputed residual ‖f_g − K̂Ū‖/‖f_g‖ stagnates at 3.6×10⁻⁴ (block) and 3.2×10⁻³ (layer), consistent with rounding in the single-precision network (the correction in single precision changes it by less than 1%). By Eq. (18), the residual work Ūᵀρ at the final iterate is at most 2.5×10⁻⁸ of the compliance, so the errors above are those of the operator (Supplementary Note S6.3, Table ST19).

"Instrumented re-solves confirm this directly" 这种实验室口吻一并去掉。[机械]

**(d) 6.2 中心差分记录**（约 150 词 → 40 词）。替换 "A fixed-q̂ central-difference check on the fourteen pair configurations …" 到 "…add switches of their own." 为：

> A central-difference check of Ĉ_{,c} could not resolve it: the discrete model of Section 3.3 switches at almost every perturbation and the difference quotient grows as the step decreases (Supplementary Note S6.3, Table ST19).

[机械]

**(e) 表 6 图注**（约 230 词 → 90 词）。替换 "The optimisation runs integrate the moments and their reverse-mode derivatives …" 到 "…(memory in Table ST26)." 为：

> Times are not comparable with Table 5: the optimisation runs regenerate every cell's geometry, integrate the moment derivatives without the fused kernels of the timed route and solve for one load (Supplementary Note S9, Table ST22c).

[机械]

**(f) 5.8 的免责声明**。替换 "This ablates our own retained representation at a fixed substructure size of one cell, … (Supplementary Note S4)." 为：

> The ablation restricts only the retained representation of the present cells at fixed cell size; port-reduced methods control this error by partition refinement, boundary enrichment or oversampling [Huang et al. (2024); Guo et al. (2026a); Guo et al. (2026b)] and are not reproduced here (Supplementary Note S4).

[机械]

**(g) 5.5 调和起点**。替换 "With 64 steps per stage, eight times the smoothing work, … so it is a weak but inexpensive comparator (Tables ST08 and ST08b)." 为：

> With 64 steps per stage, eight times the smoothing work, the harmonic start still lies 2.6 to 50 times above the base network's eight-step result in five of the six cells and reaches it only in H2 (Tables ST08 and ST08b).

并删 "Locally, NICE-post is about as accurate as NICE."（表 3 本身就说明了）。[机械]

**(h) §3 开头的"哪些是经典结果"**（120 词、9 条引用）。替换 "Several relations are classical and are restated in the present notation: …" 到 "…the identity ε = δ²κ used in Section 5.4." 为：

> The Ritz identity (4), the residual form (5), the energy orthogonality behind Eq. (6) and the self-adjoint compliance sensitivity are classical [Fraeijs de Veubeke (1965); Toselli & Widlund (2005); Becker & Rannacher (2001); Haftka & Gürdal (1992)]; the participation bound (7), the sensitivity relations (8)–(9) with their bounds (Appendices J.4 and J.5) and the identity ε = δ²κ of Section 5.4 are derived here for approximate extensions.

Chebyshev 与两重网格的四条引用（Golub & Varga、Hackbusch、Trottenberg、Xu & Zikatanov、Falgout）移到 4.2/4.3 各自出现处。[机械]

**(i) 3.3 的"切换"段**（约 170 词 → 80 词）。替换 "These derivatives hold on intervals where …" 到 "…the departure is below 10⁻⁷." 为：

> These derivatives hold on intervals where the active elements, ghost faces, retained degrees of freedom and the network's binary node indicators are fixed. Across such a switch the discrete reference compliance itself can jump, as it does for exact condensation on the same background mesh; because 0 ≤ C − Ĉ ≤ βC at every design (Eq. 7), any jump that the surrogate adds is bounded by its own error level (Supplementary Note S6.3).

扫描的数字（0.25%、4×10⁻⁵、10⁻⁷）留在 S6.3。[机械]

### 1.3 小的辩解式措辞（删或改）

| 位置 | 原文 | 改法 |
| --- | --- | --- |
| 2.1 | "they are not pointwise wall thicknesses, and we call them corner thickness parameters" | "we call them corner thickness parameters (band parameters, not pointwise wall thicknesses)" [机械] |
| 2.2 | "(10 of the 1,109 cells generated for the training expansion were rejected)" 和 "the deployed operator reproduces the six rigid-body modes to a relative energy of 2×10⁻¹¹ (Section 5.2)" | 两处删；拒绝率若要留，放附录 A.1 [机械] |
| 2.3 | "All relative measures use a nonzero reference denominator." | 删 [机械] |
| 4.3 | "(Appendix J.8; Appendix J.9 gives a matrix example in which the energy error falls while the sensitivity error rises)" | "(Appendices J.8 and J.9)" [机械] |
| 4.5 | "the reported configurations use w_s = 1, and the effect of this term was not isolated" | 句到 "w_s = 1." 为止；"not isolated" 留 6.4 [机械] |
| 5.1 | "the threshold is not derived from an optimiser tolerance" | 整句改为 "A 3% line on compliance and on each cell's sensitivity vector is drawn as a common reference, with maxima over the six loads and, for sensitivity, over both cells." [机械] |
| 5.1 | "where the target's cut plane meets the shared face, the pair is a diagnostic assembly rather than a physically cut specimen" | 删分号后的半句（或改 "a test configuration"）[机械] |
| 5.1 | "The directions of the sensitivity vectors are reproduced more closely than their magnitudes: in the fourteen development configurations, NICE's …" | 这是结果，搬到 5.6 第一段末尾 [机械] |
| 5.3 | "Individual sampled directions are not bounded by these geometry means:" | 去掉否定式引子，直接 "Over the 5,120 sampled consistent-traction directions of the 80 geometries, …"；并删 "and over the 3,840 directions of the 60 geometries outside weight selection 0.35%, 0.73% and 1.24%" → "(0.35%, 0.73% and 1.24% on the 60 geometries outside selection)" [机械] |
| 5.4 | "with the weighting W = diag(0, D) of Appendix B.1, admissible because d vanishes on the retained coordinates; the identity is a diagnostic decomposition, not an independent explanation" | "Then ε = δ²κ exactly (Appendix B.1)."；可容许性说明进附录 B.1 [机械] |
| 5.7 | "In assembly the changed retained displacement also interacts with the extension error, so field-only and solution-only replacements do not add to the full sensitivity error (Table ST11, Appendix H)." | 删（"field-only / solution-only replacements" 正文从未定义）[机械] |
| 5.11 | "(largest values reached: 0.450 and, at an intermediate design of case A, 0.452)" | 删 [机械] |
| 5.11 | "Hom-y and X-y were not checked, and the 0.060% between X-y and B1 is comparable to the surrogate errors of 0.013–0.038% found on the checked plate designs, so that ranking is not resolved." | "Hom-y and X-y were not checked exactly; the 0.060% between X-y and B1 lies within the surrogate error of the checked plate designs, so that ranking is open." [机械] |
| 6.1 | "where a compact coarse model matters more, the balance can differ" | 删 [机械] |
| 6.2 | "Moreover, assembly selects its directions through the coupled solution, whereas training controls the average error over sampled directions; a uniform operator bound would require quantitative coverage of the nonrigid retained space (Appendix J.7)." | 删；6.4 "no certified error bound is provided" 后加 "(Appendix J.7)" [机械] |
| 6.3 | "if GPU and host time are counted alike; the trained operator applies only to the cell family and discretisation of its training (Section 6.4)" | 句到 "counted alike." 为止；回本数字 "230 to 270" → "a few hundred design iterations, roughly ten optimisations of the size of case A" [机械] |
| 引言 | "Homogenisation provides such a model only under a separation of scales that a single layer of cut cells does not offer (Section 5.11)." | 一个算例撑不起一般论断："…only under a separation of scales, which a single layer of cut cells lacks; in the example of Section 5.11 it underestimates the compliance by 27–37%." [机械] |

---

## 2. 不该放在正文的内容

### 2.1 逐条

| 内容 | 位置 | 去向 | 正文保留 |
| --- | --- | --- | --- |
| 训练后勤："304 of the base network's 305 and 287 produced later (one near-mechanism cell was removed); with one pool replacement every 100 updates, each visits about 150 distinct geometries, in the same order for all three. Measured to the last update on one NVIDIA GeForce RTX 5090, … (5.0 h and 4.8 h including all selection evaluations; NICE's run was interrupted at update 5,600 and resumed from update 4,000, and both segments are logged; peak device memory 29.6 GiB). … and occupies 1.05 TB. With the three earlier training stages from which the base network descends…" | 5.1 第 3 段 | 表 ST01 注 | "All variants share the architecture of Section 4 (about 6×10⁵ trainable parameters; Table ST20). The three continuations draw from the same set of 591 geometries, which contains the base network's training set, and see the same geometries in the same order (Table ST01). Training the base network took 4.6 h and NICE's continuation 3.5 h on one GPU; generating the directions and exact sensitivities of the 691 training and validation geometries took about 42 GPU-hours, so the offline cost of NICE is about 60 GPU-hours." [机械] |
| 选择机制："each continuation was scored at 7,500 and 15,000 updates and the final weights were retained in every recorded case, the base network's weights were selected among four checkpoints, and NICE was designated the principal predictor after all predictors had been compared on the 80 geometries and the pairs" | 5.1 第 4 段 | 表 ST01 注 | "Twenty of the 80 validation geometries (6 uncut, 14 cut) were used for model selection (Table ST01); statistics are therefore also given for the 60 geometries outside selection. The cells of the two-cell examples belong to the 20 selection geometries; nine further cells outside selection are assembled in Section 5.6 as an independent check." [机械] |
| "Two predictors from earlier comparisons appear in the supplement only (Supplementary R1)." | 5.1 | 删 | —（P0、S8 的处理见 §7）[机械] |
| **NICE-post**（表 2 一行；5.3；表 3 列名 "Base network (NICE-post)"；5.6 三句；图 9 图注；图 10 图注；6.1） | 多处 | 取消这个名字：它就是"基础网络的权重 + 部署时加 NICE 的校正"，不是一个训练出来的预测器 | 表 2 删此行，表下加一句 "The base network with the correction W applied at deployment, without retraining, is also evaluated (Sections 5.3, 5.5 and 5.6)."；表 3 列名 "Base network, corrected"；5.3 改为 "Applying the correction to the base network without retraining already gives 0.0965% (largest 0.91%); training through it lowers the mean by a further factor of 1.31 (95% bootstrap interval 1.20–1.40)."，删 "and by 1.5–2.0 under nodal forces, supports and single-face loads, on 96–99% of the geometries"；5.6 删 "The largest sensitivity errors of NICE and NICE-post are … NICE has the lower sensitivity maximum in eleven of the fourteen configurations (Table ST09)."，保留一句 "The base network with the correction applied at deployment also meets both references (largest sensitivity error 0.95%, U1/x), so the correction is what brings the assembled responses within the reference."；6.1 "NICE-post already meets the assembled accuracy reference" → "the base network with the correction applied at deployment already meets the assembled accuracy reference"；图 9 的 NICE-post 柱改标 "Base network + correction"（需重画 F05）或去掉；图 10 图注改为 "all loads of the predictor–configuration combinations of Table ST09"。[决定：建议取消命名；是否从图 9 中去掉由作者定] |
| 评估覆盖簿记："The Uncorrected continuation was evaluated on eleven configurations, the nine common to Uncorrected, Smoothing-trained, NICE-post and NICE (the base network was evaluated on seven of them) and L1/x and L1/y;" | 5.6 | 表 ST09 注 | "The Uncorrected continuation fails the sensitivity reference in four of the eleven configurations on which it was evaluated (U1/x, U1/y, M1/x, M1/y, up to 11.4%) and, on M1, also the compliance reference (up to 4.1%)." [机械] |
| 图 9 图注 "H3 is a further heavily cut cell evaluated for Smoothing-trained, NICE-post and NICE only; L1 is a lightly cut cell evaluated for Uncorrected, NICE-post and NICE." | 图 9 | 表 ST09 注 | "Not every variant was evaluated on every configuration (Table ST09)." [机械] |
| 图 10 图注 "411 observations from 52 combinations of the base network, Uncorrected, Smoothing-trained, NICE-post and NICE in Table ST09, excluding L1" | 图 10 | — | "all loads of the variant–configuration combinations of Table ST09" [机械] |
| "Four of the per-cell margins in Table ST04 come from a host evaluation whose seeded generator gives a different power-iteration start." | 5.2 | 删（ST04 注可留一句） | — [机械] |
| 求解器取证（单精度重跑、对偶范数界、"Instrumented re-solves"、式 (18) 各项平衡到 9×10⁻⁹） | 5.9 | Note S6.3 | 见 §1.2(c) [机械] |
| 中心差分记录（重建次数、步长序列） | 6.2 | Note S6.3 | 见 §1.2(d) [机械] |
| "SciPy's default sparse direct solver (SuperLU), tried as a further serial …"（失败的尝试） | 5.10 | 删 | — [机械] |
| 单线程直接解：表 5 一列、"2,131 to 2,425 s with one thread"、"5,864 and 8,131 s for the serial direct solution"、"72 to 74 times longer"、"and with one thread for comparison with serial solvers" | 5.10、表 5 | 表 ST17b | 主线只用 16 线程的数字（摘要、结论都是）[决定：建议删] |
| "With 32 threads, the factorisation of route (a) is 1.2 to 1.8 times faster but the whole direct solution only 2 to 22% faster, since cell setup and assembly do not speed up (Supplementary Note S5)." | 5.10 | 已在 Note S5 | 删 [机械] |
| 几何生成回退："Where the geometry generator left the exact certification of an element's local material support unresolved (Appendix A.1), the free parameters of the failing cells were scaled by 1+10⁻⁴ or 1+10⁻³, at most twice per run (Supplementary Note S9.1, Table ST24)." 与图 12 图注 "at iterations 16 and 19 the NICE design had been perturbed by the geometry-generation fallback (Table ST24), and at iteration 19 its volume lay 0.036% above V*" | 5.11、图 12 | Note S9.1、表 ST24 | 正文一句 "In five design iterations over all runs the geometry generator required a perturbation of at most 5.8×10⁻⁴ in a corner parameter (Supplementary Note S9.1)."（数字请按 ST24 核对）；图注的半句删 [机械] |
| 切换计数段："The discrete model switched between all consecutive design iterations (Section 3.3): active elements, ghost faces, retained coordinates or binary node features changed in at least 3 of 8 cells (case A) and 9 of 24 (plates), the coarse-factor shift in at most four cells (Table ST24). The recomputed residual stagnated between …" | 5.11 | 表 ST24 | "The discrete model switched (Section 3.3) between every pair of consecutive design iterations, and the residual work of Eq. (18) stayed below 3.1×10⁻⁷ of the compliance (Table ST24)." [机械] |
| "(two for 51 cells)"、"On one RTX 5090 with 90 GiB of host memory"、"At 110 cells the GPU memory was fully in use" | 5.11 | 表 ST26 | "On one GPU, …"；"4 timed" [机械] |
| 表 6 图注的内存句 "The NICE runs of A, B1, B2 and X needed at most 24.3 GiB of GPU memory and a peak of 5.0 GiB of host memory (main process; geometry generation excluded); in the scale runs, cell operators beyond 4 GiB of GPU memory were streamed from host memory (memory in Table ST26)." | 表 6 | 表 ST26 | 删 [机械] |
| 图 13 图注 "(layer z=1 differs by at most 7.0×10⁻⁴ in B2 and 2.0×10⁻⁴ in X-z)" 和 "at 135 cells the factorisation of K_PP on the GPU ran out of device memory"（正文与 6.4 已各说一次） | 图 13 | — | 删两处 [机械] |
| 表 1 "Arithmetic" 行（60 词，TF32、节号交叉引用） | 表 1 | 附录 F.3 | "Network in single precision; stiffness actions, energies and correction in double precision (single precision in the timed route of Table 5; Appendix F.3)" [机械] |
| 5.7 中复述附录 J.4 的界 "Appendix J.4 gives ‖s̃−s‖₂ ≤ 2L√𝓔 + Q𝓔, with 𝓔 … L … Q …" | 5.7 | 附录 J.4 | "Appendix J.4 bounds the sensitivity error by the interior error energy with a linear and a quadratic term; the relative error is further divided by the small local reference of a weakly participating cell." [机械] |
| "development cells / development cases / development configurations"（6 处：引言贡献 (iii)、5.1 两处、5.6、6.4、7） | 多处 | — | "cells used in model selection" / "selection cells"；"served repeatedly as development cases" 删 [机械] |
| 图文件名 F02_validation_A3、F05_assembly_A3、F10_energy_participation_r1（运行号） | figures/ | — | 打包时按图号重命名（已在清理清单），这三个尤其要改 [机械] |
| 文件末尾 `<!-- restructured r2 -->` | MANUSCRIPT_EN.md 末行 | 删 | — [机械] |
| "Sections 5.1 and 6.5" | 5.11 第 3 段 | — | "Sections 5.1 and 6.4"（悬空引用，重排残留）[机械] |

### 2.2 表 2 的五个"预测器"各自承担什么

| 预测器 | 承担的论点 | 处理 |
| --- | --- | --- |
| NICE | 方法 | 保留 |
| Base network（无校正、40k 步） | "学习给试探场"的起点；表 3 的"学习起点"列；图 5b、图 6、图 7、图 8 的对象 | 保留，首次定义为 "the network trained without correction (base network)" |
| Uncorrected（基础网络再训 15k 步、无校正） | 排除"多训了 15k 步"这一解释（6.89% → 6.33%）；表 4、图 9 的对照 | 保留，但正文只需一句（5.3）加表 4 的一列 |
| Smoothing-trained（只经 8 步光滑训练） | "只有光滑没有粗解，柔度准而灵敏度不准"——表 4 U1/x 那一行是全文的关键例子 | 保留；名字可改为 "Smoothing only"（更直白），非必须 |
| NICE-post | 不是训练出来的预测器；它承担的两点（校正不重训就有效；通过校正训练再降 1.31 倍）各一句话即可 | 取消命名（见 2.1）[决定] |

### 2.3 §5 与 §6 各小节的判决

| 小节 | 判决 |
| --- | --- |
| 5.1 设定 | 保留，压缩约 40%（后勤、选择机制、"两个早期预测器"、诊断装配） |
| 5.2 参考解与算子验证 | 保留，压缩约 60%（两段，见 §1.2(a)(b)） |
| 5.3 几何与载荷依赖 | 保留；中间预测器一段压成三句 |
| 5.4 误差的组成 | 保留（谱、空间分布、ε=δ²κ、线性项——这是误差分析的实证核心） |
| 5.5 固定权重下的校正 | 保留；调和起点一句、删 "Locally, NICE-post…" |
| 5.6 装配后的柔度与灵敏度 | 保留，吸收 5.7；删覆盖簿记与 NICE/NICE-post 逐配置比较 |
| 5.7 能量参与 | **并入 5.6** 作末段（β=wε 的核对与图 10）；5.8–5.11 顺延编号 [决定] |
| 5.8 保留表示的消融 | 保留（支撑 6.1 的"保留什么"论点），压缩约 30% |
| 5.9 全学习格架 | 保留；残差段压缩 60% |
| 5.10 成本 | 保留；去掉单线程列、32 线程句、SuperLU 句 |
| 5.11 厚度优化 | 保留；去掉回退、切换计数、规模簿记；X-y/X-z 的一段压成两句："Continuing from the homogenisation designs with NICE lowers their compliance by a further 1.0% (X-y) and 0.53% (X-z) in 12 design iterations, so the optimisations are local; X-z ends 0.87% below B2 and X-y 0.060% above B1, the latter within the surrogate error of the checked designs (Figure 12b, Figure 13a,b; Table ST23)." [决定] |
| 6.1 | 第一段删引言已有的边界缩减方法解释与 Guo 数字（约 −40%）；第二段按 §1.1 改 |
| 6.2 | 压一半（§1.2(d)） |
| 6.3 | 按 §1.1、§1.3 改 |
| 6.4 | 作为唯一的限制收集处保留；其他地方的重复删 |

---

## 3. 术语对照表

原则：用有限元 / 子结构 / 区域分解 / 多重网格 / 结构优化的既有词；自造词只在首次定义后用一个固定名字；机器学习词只留在 4.4–4.5 描述网络本身的地方。

| 现用 | 建议 | 出现（正文次数） | 处理 |
| --- | --- | --- | --- |
| energy excess / directional energy excess 与 energy error 并用 | **relative energy error** ε(q) = qᵀ(Ŝ−S)q / qᵀSq，定义处注明由式 (4) 非负（"the excess of the condensed strain energy over the exact one"） | excess 13、error 24 | 统一为 energy error；图 5、图 8、表 3、表 ST03 的标题同步 [决定：建议统一为 error] |
| "how it is read out" / "read out" | "how its condensed stiffness is formed" / "evaluated through its energy" | 3（摘要、引言、结论） | 替换 [机械] |
| complete transpose | "the transpose Fᵀ of the complete extension (rigid reconstruction, retained-value restoration and correction included)"，定义一次后写 Fᵀ | 6 | 替换 [机械] |
| 4.6 标题 "Energy-consistent operator application" | "Application of the condensed operator" | 1 | 改标题 [机械] |
| polynomial relaxation / relaxation | Chebyshev smoothing / smoothing | relaxation 9、smoothing 44 | 4.2 标题 "Error spectrum and Chebyshev smoothing"；正文统一 smoothing（引 Zhang 2024 处除外）[机械] |
| interior coarse correction / coarse update / coarse solve / Galerkin solve | coarse-grid (Galerkin) correction | 14 | 统一 [机械] |
| equilibrium correction（方法名内）/ internal correction / multilevel correction / fixed correction / two-grid | 定义一次："the interior correction W: one two-grid cycle (Chebyshev smoothing, coarse-grid Galerkin correction, smoothing) at fixed retained displacement, which we call the equilibrium correction"；此后 "the correction W" | 多处 | 4.1 定义，4.2–4.3 分述 [机械] |
| retained coordinates / retained coefficients / retained degrees of freedom；box-face coefficients；internal coordinates | **retained degrees of freedom (DOFs)**；"coefficients" 只用于基函数系数；interior DOFs | coordinates 19、space 13 | 全文统一（含附录、补充材料的对应处）[决定：量大但机械] |
| internal / interior | interior | 混用 | 统一 interior [机械] |
| macro-box / macro-cut / macro-domain / macro-cut patch / macro-cut tractions | cell box / cut plane, cut surface / box volume / cut-surface tractions | 6 | 替换 [机械] |
| cut band / cut-band coordinates | 保留，2.2 首次定义 "the layer of active elements intersected by the cut plane (the cut band)" | 11 | 保留 |
| direction banks / validation directions | direction sets / sampled directions | 2（正文）；补充材料多 | 替换正文 [机械] |
| predictor | **variant**（表 2 标题 "Variants compared in the numerical examples"） | 13 | 替换（附录 7、补充材料 11 处同步）[机械] |
| weights / updates / pool / checkpoint / weight selection | parameters θ / training steps / training set / —— / model selection | 12 / 8 / 3 / 1 / 8 | 替换；"weights" 在 4.4 指系数时保留 [机械] |
| base network | 保留，首次定义 "the network trained without correction (base network)" | 32 | 保留 |
| Smoothing-trained | 可改 "Smoothing only" | 9 | 可选 |
| energy participation / participation-weighted / participation bound | energy share w_m / energy-weighted / share-weighted bound | 12 | [决定：图 10 标题与 5.7 并入后一起改] |
| host (memory) / host evaluation / host time | CPU memory / 删 / CPU time | 27 | 替换 [机械] |
| front end | cell preparation (geometry, moments, network encoding, correction setup) | 表 5、表 6 | 替换 [机械] |
| instrumented re-solves | 删 | 1 | [机械] |
| diagnostic assembly / development cells / development cases | test configuration / selection cells | 2 + 6 | 替换 [机械] |
| learned operator / learned substructure / learned reduced model / surrogate / component model | "learned substructure"（对象）、"condensed stiffness Ŝ"（矩阵）；"surrogate" 只用于 surrogate compliance / objective | 4 / — / 14 | 统一 [机械] |
| rigid modes / rigid motion / rigid kernel / rigid-body modes | rigid-body modes / rigid-body motion / rigid-body kernel | 多 | 统一 [机械] |
| thickness sensitivity / eight-corner sensitivity / local sensitivity / design sensitivity / corner sensitivity / eight-parameter sensitivity vector | thickness sensitivity（定义：柔度对胞的八个角参数的导数，八分量向量）；"local" 只在与格架梯度对比时用 | 多 | 统一 [机械] |
| coarse-factor shift | 正文删（附录 F.2 定义处保留） | 3 | [机械] |
| binary node features | binary node indicators (of the network input) | 3 | [机械] |
| equal nodal forces、stiffness-scaled spring supports、neighbour-induced displacements、polynomial / multiscale displacements | 保留（4.5 与附录 G 已定义） | — | 保留 |
| consistent tractions | 保留（有限元标准用法：面力的协调节点力） | — | 保留 |
| trace / interface / port | trace 保留（区域分解标准）；"port" 只在引用 port reduction 时用 | 6 / 5 / 1 | 保留 |

**符号重载（建议改掉便宜的几个）**

| 符号 | 现在的用法 | 建议 |
| --- | --- | --- |
| W | 𝒲 校正（6）；W = diag(0, D) 加权（5.4）；W_{ℓh} 通道混合（式 15） | 加权改 M；通道混合改 Θ_{ℓh} |
| V | 粗空间基 V（4.3）；体积 V ≤ V*（5.11） | 体积改 𝒱 |
| C | 柔度 C；C_R 刚体系数提取；C_V 粗投影 | C_R → Γ_R，C_V → Π_V |
| R | 刚体模态 R、R_P；Rayleigh 商 R(d)、R(u)（5.4） | Rayleigh 商写 λ_A(d) = dᵀAd/dᵀDd |
| D | D = diag(A)；D_c = K_{,c}（附录 J.4） | 附录直接用 K_{,c} |
| H | 误差映射 H；胞标签 H1/H2/H3 | 不改（上下文清楚），但 5.4 "in H2 only 66 of the 200 modes" 这类句子里读者要停一下；可把胞标签改为 C1/C2…（作者定） |

---

## 4. 叙事与结构

**论文应当讲的故事（一段）：** 构件模型要能装配、要回内部场、误差要可追踪、要可改进；均匀化、精确凝聚、学习各缺一角。把近似只放在内部延拓、在完整保留空间上做，并用能量形式 FᵀKF 读出，学习的模型就成为 Ritz 近似：对称半正定、刚体核、被精确 Schur 补从下方界住、误差二阶；一个固定的两重网格校正在该能量内降低内部残差而不破坏任何性质。误差进入柔度时按能量份额加权、进入灵敏度时经刚度导数带线性项，所以柔度准不等于灵敏度准，设计用途要两者同时核验。在切割 TPMS 薄壁胞上实现（NICE），单胞、装配、格架、设计逐级验证，成本与整体直接解和常规凝聚相比。

**偏离最大的三处：** (1) §5.1–5.2 用近 900 词讲训练后勤和参考解辩护，读者在进入第一个结果前先读了一页"交代"；(2) §6.1–6.2 重讲 §1、§3、§4 已说过的内容，再加保留意见；(3) §7 第三段把摘要的数字再抄一遍。

| 位置 | 问题 | 改法 |
| --- | --- | --- |
| 摘要第 4 句（约 95 词） | 三个并列从句太长 | 拆成三句："Approximating only the interior extension, on the complete retained space, leaves the displacement patterns exchanged between cells unchanged. Forming the condensed stiffness from the energy of that extension, Ŝ = FᵀKF, makes it a Ritz approximation bounded below by the exact Schur complement, with an error quadratic in the interior error. A fixed, geometry-specific multilevel correction applied inside that energy reduces the interior residual without losing any of these properties and cannot increase the error when its smoothing interval contains the spectrum." [机械] |
| 引言第 4 段（约 400 词、25 条引用） | 一段讲完 PIML 一系、Guo 的全节点选项、能量型模型、组件 RB、学习求解器、多尺度基 | 拆成两段：第一段 PIML 一系 + 本文定位（Guo 的三句压成："Guo et al. (2026a, Table 1) also tried a learned interior with every boundary node retained and set it aside because its error grew with the size of the network output. The present construction retains the complete trace of cut thin-walled cells, 1.7–2.6×10⁴ degrees of freedom per cell, and restores the accuracy of the learned interior with the correction, at the price of tens of thousands of coarse degrees of freedom per cell."）；第二段从 "Related energy-based models include…" 起 [机械] |
| 贡献 (iii)（一句 130 词） | 把 §5 全列了一遍 | "(iii) A realisation for cut thin-walled Schwarz-P cells in a stabilised cut finite element model, with a geometry-conditioned network that handles the cut-dependent discrete space and the cut band, verified on 80 validation geometries, on two-cell assemblies of selection and held-out cells and on eight-cell lattices, with an ablation of the retained representation, a cost comparison with the whole-lattice direct solution and conventional exact condensation, and thickness optimisations driven by the field-based sensitivities, checked against exact condensation and compared with a homogenised model (Section 5.11)." [机械] |
| §3 开头 | 120 词的经典结果清单挡在分析前面 | 见 §1.2(h) [机械] |
| 4.2 第 3 段 "Polynomial relaxation supplies a parallelisable component of multigrid methods [Adams…], the combination of a trial extension with smoothing is the principle of smoothed-aggregation prolongation [Vaněk…], and pairing relaxation with a learned initial field exploits… [Zhang…]." | 方法节里的相关工作 | 压成一句放 4.2 第 2 段末："The pairing is that of smoothed-aggregation prolongation [Vaněk et al. (1996)] and of relaxation after a learned initial guess [Zhang et al. (2024)]."；Adams 的引用移到 4.2 首句 [机械] |
| 式 (18) 及其段落在 4.6 | 一个误差关系放在算法节末尾，§3 开头却已前引它 | 可选：把 "At the assembled level, the energy relation also separates…" 一段移到 3.2 末尾（成为式 (8)，后续编号 +1，由脚本改），4.6 只剩算法 [决定] |
| §5 路线图段 | 可以，略长 | 删末句以外的 "Sections 5.1 and 5.2 define … verify the discrete reference and the deployed operator." 中的 "and verify …"（5.2 压缩后只是一段） |
| 5.2 | 见 §1.2 | 压成两段后可并入 5.1 作末两段，5.3–5.11 编号前移一位 [决定：若并入，编号再动一次；建议保留 5.2 独立但只剩两段] |
| 5.6 ↔ 5.7 | 5.7 一段重复 5.6 的 U1/x 例子 | 并入（见 §2.3）[决定] |
| 5.8 位置 | 在装配与格架之间插入"保留什么"的消融 | 可以不动；也可移到 5.9 之后作为"装配"组的收尾——保持现状即可 |
| 6.1 第一段 | 复述引言的边界缩减方法与 Guo 数字 | 删 "Substructure methods that describe the boundary by a few coordinates … in exchange for coarse models small enough for very large structures. For orientation only, … against exact condensation."，接 "For the local sensitivity of the cut cells examined, …" [机械] |
| 6.1 ↔ 7 ↔ 5.5 | 表 3 的 7–290 倍、2,400–12,000 倍、13.5%→0.19% 三处各说一遍 | 5.5 报告；6.1 解释（留）；7 改为 "under the same correction a deterministic start leaves one to three orders of magnitude more error than the learned field, and the correction alone reduces the base network's error on the hardest cell by two orders of magnitude without retraining" [机械] |
| §7 第三段 | 摘要数字的再版（230 词） | 压成约 120 词："Realised for cut thin-walled Schwarz-P cells, with tens of thousands of retained degrees of freedom and a discrete space that changes with the cut, NICE keeps the mean energy error under consistent tractions at 0.074% on 80 validation geometries and the compliance and sensitivity errors below 0.3% and 1.5% in every two-cell configuration and below 0.02% and 0.15% in lattices of eight learned cells. An eight-cell design iteration takes 81–110 s on one GPU against 868–1,042 s for the whole-lattice Cholesky solution on 16 CPU threads. Driven by the field-based sensitivities, the method of moving asymptotes reproduces the design obtained with exact condensation to within 0.0051 in every corner parameter; on cut 24-cell plates the exact checks give compliance errors below 0.04% and gradient errors below 0.33%, whereas a homogenised model underestimates the compliance by 27–37%; a design iteration of a 110-cell plate with 32.7 million degrees of freedom takes 43 min. The construction transfers to other component families with new training data; the structure and the error relations do not depend on the family." [机械] |
| 4.6 缺"每几何准备 / 每次作用"的成本划分 | 冷读者找不到离线/在线的界限 | 4.6 首句后加："Per geometry: element moments, the network's geometry encoding and coefficients, the smoothing interval and the coarse factorisation (Table ST17 gives their cost). Per application of Ŝ: one network pass, 2k smoothing steps, one coarse solve and the transposed sequence." [机械] |

---

## 5. 科学表述与数字核对

核对过的（无误）：式 (4)–(7)、(10)–(14) 的陈述与假设；Chebyshev 多项式在 (0, b] 上的非扩张性（4.2 的条件正确）；S ⪯ Ŝ_tg ⪯ Ŝ_0 的条件（精确 Galerkin 更新 + ‖P_k‖_A ≤ 1）；ε = δ²κ 的定义与恒等（5.4）；式 (17) 的 Ritz 解释；摘要与结论中的数字对 5.3、5.6、5.9、5.10、5.11 和表 5、表 6（0.074%、0.27%/1.49%、81–110 s 对 868–1,042 s、0.0051、43 min、32.7 M、27%/37%）；图表引用顺序（图 1–13、表 1–6 均按序首次引用）；式号引用（(5) 残差、(7) 参与界、(8)/(9) 灵敏度、(16) 网络延拓、(17) 目标、(18) 残差功）与重排后的编号一致。

需要改的：

| 位置 | 问题 | 改法 |
| --- | --- | --- |
| 5.11 "Sections 5.1 and 6.5" | 6.5 不存在 | "Sections 5.1 and 6.4" [机械] |
| 引言 "Homogenisation provides such a model only under a separation of scales that a single layer of cut cells does not offer" | 一个算例支撑的一般论断 | 见 §1.3 末行 [机械] |
| 7 "in every example a better initial field or a larger correction budget reduced it" | 对灵敏度误差不成立（5.5：M1 的灵敏度误差非单调） | "…reduced the energy error" [机械] |
| 3.3 "the discrete moments used here violate it in a few elements" | 正文里的实现细节，且 J.5 已有 | 删此半句，保留 "For corner thickening with fixed basis and ghost contribution, K_{,c} ⪰ 0 (Eq. J.6), yet the cross term can have either sign (Appendix J.4)." [机械] |
| 5.4 "and all lie below the lower endpoint a = 0.173 of the smoothing interval; in H2 only 66 of the 200 modes lie below a = 0.138" | 可以，但 a 的数值随胞变，读者不知 a = b/30 已在表 1 | 加 "(a = b/30, Table 1)" [机械] |
| 5.9 "the assembled retained solution differs from the exact one by 0.13%" | 用什么范数？ | 写明 "in the 𝕂-norm" 或所用范数（按补充材料核对）[机械] |
| 2.2 "the retained set P contains the active box-face displacement coefficients and all displacement coefficients of active elements carrying a positive-area macro-cut patch" | "coefficients" 指 DOF | 按术语表改为 degrees of freedom [机械] |
| 表 1 "65 Q2 node positions per axis" | "node positions" 可写 "nodes" | 小 |

未发现数值与表格矛盾（抽查 20 余处）。

---

## 6. 图表

| 图/表 | 处理 |
| --- | --- |
| 表 1 | Arithmetic 行缩短（§2.1） |
| 表 2 | 删 NICE-post 行；标题 "Variants compared…"；表下加一句说明"基础网络 + 部署时校正"也被评估 [决定] |
| 表 3 | 列名 "Base network (NICE-post)" → "Base network, corrected"；首列 "Base network, uncorrected" 保留 |
| 表 4 | 保留；表注 "Configuration-level sensitivity maxima include both cells." 可删 |
| 表 5 | 删单线程列；表头 "(c) NICE (one RTX 5090)" → "(c) NICE (one GPU)"（GPU 型号在正文说一次）；"Front end" → "cell preparation" [决定：单线程列] |
| 表 6 | 图注按 §1.2(e)；"4 timed (51 cells: 2)" → "4 timed"；"Not checked" 保留 |
| 图 5 | 图注 "Predictors as in Table 2." → "Variants as in Table 2."；若统一 energy error，标题同改 |
| 图 9 | NICE-post 柱：改标 "Base network + correction" 或去掉（需重画 F05_assembly_A3；脚本在 figures_src）[决定]；图注末两句按 §2.1 |
| 图 10 | 图注按 §2.1；若 participation → energy share，标题 "Energy-share-weighted compliance error and local sensitivity" |
| 图 12 | 图注删回退半句；"(line; the designs of the two runs differ)" 可留 |
| 图 13 | 图注删层差与 135 胞两处 |
| 文件名 | F08=图 1、F01=图 2、F12=图 7、F13=图 12、F14=图 13；F02/F05 带 A3、F10 带 r1 → 打包时统一重命名并改路径（已在清理清单） |

---

## 7. 附录与补充材料

正文实际引用：附录 A、A.1、A.2、B、B.1、C、D（3）、E、F、F.3、G（2）、H（4）、I、J.1–J.9（各 1–3 次）；Note S1–S6、S9；表 ST01–ST27 中的大部分。

| 附录/Note | 正文引用 | 处理 | 理由 |
| --- | --- | --- | --- |
| A 离散构造与度量 | 2.1、5.1 | 保留 | — |
| B 变分恒等式、刚体核、方向范数 | 3.1、4.1、5.4 | 保留；吸收 J.1（假设）和 J.2（"为什么完整转置给出二阶误差"——J.2 开头就是 "Appendix B derives…"，是对 B 的评注） | 去掉 J 的碎片 |
| C 装配与柔度排序 | 3.2 | 保留；吸收 J.3（"柔度作为完整重构误差能量"是 C 的同一件事） | 同上 |
| D 多项式光滑与粗投影 | 4.2、5.2 | 保留；吸收 J.8（校正保持目标、对不同量的改进不同） | 同上 |
| E 完整延拓的转置 | 4.1 | 保留；吸收 J.6（有限迭代残差与能量/作用一致性）或 J.6 独立成 "Inexact assembled solves" | — |
| F 粗空间、分解与算术 | 4.3、表 1 | 保留；F.1 "Explicit coarse-space study" 若只被 Note S3 引用，移补充材料 | — |
| G 网络系数与方向训练 | 4.4、4.5 | 保留；吸收 J.7（方向覆盖、旋转、谱）——若 6.2 那句删了，J.7 在正文无引用，可移补充材料 | — |
| H 灵敏度恒等式与设计区间 | 3.3、5.7、5.9、5.11 | 保留；吸收 J.4（能量到八角灵敏度的界）和 J.5（完整设计导数及其阶）；5.2 的差分步长研究移入 | 任务 #101 做过 H/J.5 去重，但 J.5 仍独立存在 |
| I "Residual lower diagnostics" | 6.4 | 保留，改题 "A computable lower bound on the energy error"；末段关于 5.8 消融的一句移到 Note S4 | 标题不说明内容 |
| J "Further variational and mechanical analysis" | 多处 | **解散**：J.1→B，J.2→B，J.3→C，J.4+J.5→H，J.6→E，J.7→G 或补充材料，J.8→D，J.9→补充材料 Note S8（S8 已是"illustrative matrix example 的定义"，两处重合） | 标题不说明内容；九个碎片各自依附一个主题附录 [决定；改后正文引用由脚本改] |
| 补充材料 R1 | 5.1 | 删 "Former label" 列（B、C、S8、A2b、B+W、A3 是内部名）；"Archived run identifier" 列仅在 Zenodo 快照确实以这些名字存放结果时保留；P0、S8 两行：S8 在补充材料出现 36 次（ST01、ST03、ST08 等），P0 14 次——若保留它们，R1 中改为 "additional variants reported in the supplement only"，并去掉正文的那句；若删除，需清理这些表的列 | [决定] |
| 补充材料 R2 "Further diagnostic observations" | 无 | 删（四段零散观察，"These values are ratios, not percentages" 之类是笔记口吻），能留的并入相应 Note | [决定] |
| Note S6.3 | 3.3、5.9、6.2 | 保留；接收 §1.2(c)(d)(i) 移出的数字 | — |
| Note S9.1、表 ST24 | 5.11 | 保留；接收回退与切换计数 | — |

---

## 8. 句子层面（高、中优先级）

| 位置 | 原文（节选） | 改法 |
| --- | --- | --- |
| 摘要第 1 句 | "…but a learned model is seldom a mechanical model: it rarely assembles, carries no bound, cannot be improved once trained, and its error in the quantities that analysis and design require is unknown." | 可以；"rarely assembles" → "rarely admits assembly" |
| 摘要第 4 句 | 95 词 | 拆三句（§4） |
| 摘要末句 | 数字堆叠 90 词 | 拆成两句：精度一句、成本与设计一句 |
| 引言第 1 段第 1 句 | 70 词 | "Structures assembled from repeated thin-walled cells, such as graded TPMS lattices with supports, external loads and cells cut by the specimen boundary [Yu et al. (2019)], call for a reusable model of one cell. The model must return the forces conjugate to the displacement imposed by the surroundings, so that the structure can be assembled, and the interior field, from which local design quantities such as thickness sensitivities are evaluated." |
| 引言第 2 段 "The two kinds of error enter the structural response differently. An interior error is weighted by the energy the cell carries, so a weakly participating cell can leave the compliance accurate while its thickness sensitivity, weighted by the stiffness derivative, remains inaccurate: the distinction on which goal-oriented error estimation rests [Becker…], observed for approximate reanalysis and inexact solves in topology optimisation [Amir…]." | 冒号后挂了 5 条引用的名词短语 | 拆句："This is the distinction on which goal-oriented error estimation rests [Becker & Rannacher (2001); Oden & Prudhomme (2001)]; it has been observed for approximate reanalysis and inexact solves in topology optimisation [Amir et al. (2009, 2010); Gogu (2015)]." |
| 2.3 "Because the thickness field on a shared face depends only on its four corner values and φ is periodic, the material patches of two neighbours coincide on every face not intersected by a cut; where a cut removes material from one side only, the unmatched face coordinates remain coordinates of the cell that carries them." | 可以 | "coordinates" → "degrees of freedom" |
| 4.1 "Applying the condensed operator therefore requires the transpose of the complete extension after the stiffness action, including the rigid reconstruction, retained-value restoration and any internal corrections." | 可以 | 这就是 "complete transpose" 的定义处：加 "(the transpose Fᵀ)" |
| 4.4 第 3 段 "Encoders form element and node embeddings, which exchange information in two rounds; coefficient heads then assign weights to prescribed element and ghost-face incidences, to transfers between latent grids and to coarse-grid convolutions." | ML 术语密集但在网络节可接受 | "embeddings" → "feature vectors"；"heads" 保留并在首次出现处加 "(small output networks)" |
| 5.1 第 1 段 "U, L, M and H identify the uncut, lightly, moderately and heavily cut cells used for detailed comparisons (Supplementary R1)." | 可以 | 加 "(numbered U1, U2, L1, M1, M2, H1–H3)" |
| 5.3 第 2 段 | 一段 9 组数字 | 拆成两段：一致面力一段；其他载荷类与 60 个几何一段 |
| 5.10 第 1 段（约 330 词，三条路线） | 过长 | 三条路线各成一小段或用 (a)(b)(c) 列表 |
| 6.4 第 2 句（约 80 词） | 枚举训练域 | 拆："It applies to the setting in which it was trained and tested: Schwarz-P-type cells of Eq. (1) with eight corner parameters in [0.175, 0.699], corner span and gradient norm at most 0.47 (Table ST02), and at most one planar cut whose normal is a cube-symmetry image of (cos ϑ, sin ϑ, 0). It is tied to one discretisation (n = 32, E_Y = 1, ν = 0.3, γ = 10⁻⁴), to which the 65/33/17/9 latent hierarchy is matched." |
| 全文 | "rigid modes / rigid motion / rigid kernel" | "rigid-body …" |
| 全文 | "internal / interior" | "interior" |
| 全文 | 英式拼写一致（optimisation、factorisation、modelling、neighbour、centre 均已是英式）；"cut band"（名词）/"cut-band"（定语）的用法是对的，不动 | — |

---

## 9. 建议的修改顺序与需要作者决定的事项

**第一批（机械，不需要决定，约半天）**：悬空引用 6.5；删 `<!-- restructured r2 -->`；§1.1 的重复保留意见按"保留一次"删改；§1.2 的九段改写；§1.3 的小措辞；§2.1 中标 [机械] 的残留删除；术语表中标 [机械] 的替换（read out、complete transpose、4.2/4.6 标题、relaxation→smoothing、macro-*、direction banks、predictor→variant、weights/updates/pool/checkpoint/weight selection、host、front end、development、rigid-body、interior）；图注（图 9、10、12、13、表 6）；表 1 行；§4 中的摘要拆句、引言第 4 段拆段、贡献 (iii)、§3 开头、4.2 相关工作句、6.1 第一段、§7 第三段、4.6 的成本划分两句。然后重建 PDF、FIGURES.md，查悬空引用。

**第二批（需要作者决定）**：
1. **NICE-post 取消命名**（建议取消）；图 9 是否去掉它的柱（去掉要重画 F05，脚本在 figures_src）。
2. **energy excess → energy error 统一**（建议统一；涉及图 5、图 8、表 3 标题与补充材料）。
3. **participation → energy share**（图 10 标题与 5.7 并入一起改）。
4. **retained coordinates → retained degrees of freedom**（全文约 40 处，含附录与补充材料；机械但量大）。
5. **表 5 删单线程列**（建议删，数据留 ST17b）。
6. **5.7 并入 5.6**；**X-y/X-z 段压成两句**。
7. **式 (18) 移到 3.2**（可选；要再跑一次编号脚本）。
8. **附录 J 解散**到 B/C/D/E/G/H 与补充材料；I 改题；R1 删 "Former label" 列、P0/S8 去留；R2 删。
9. 胞标签 H1/H2 与误差映射 H 的同名是否改（建议不改）。
10. 标题：现题可用；若想贴 L2 主线，可考虑 "Neural-initialised static condensation with equilibrium correction: learned substructures with variational structure for the analysis and thickness design of cut thin-walled TPMS lattices"（更长，不一定更好）。

**十条最要紧的改动**：(1) 5.1 训练后勤与选择机制压成五句；(2) 5.2 压成两段；(3) 取消 NICE-post；(4) 六条重复保留意见各留一次；(5) 5.9 残差段、6.2 差分段、表 6 图注三处"取证"压缩；(6) 6.1 第一段删引言重复；(7) §7 第三段减半并去掉 hedges；(8) 术语：energy error 统一、read out、complete transpose、predictor→variant、ML 训练词、host/front end；(9) 悬空引用 6.5 与 HTML 注释；(10) 附录 J 解散与补充材料 R1/R2 清理。
