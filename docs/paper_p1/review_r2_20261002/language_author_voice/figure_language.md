# P1 图注、图中文字与跨段重复语言审查

审查日期：2026-10-02。仓库：`/Users/fffffreeze/Desktop/curly-octo-goggles`。固定基线：`700af856f2b9f152ab2c263406c54132a11255bf`，现场 `git rev-parse HEAD` 与之相同，起始 `git status --short` 无输出。本报告只读检查仓库，唯一新增产物为本文件；没有构建、实验或论文修改。

## 1. 核验边界与结论

本项按 `MANUSCRIPT_EN.md` 和 `SUPPLEMENTARY_EN.md` 的实际 Markdown 图片链接建立清单，得到正文 13 张、补充 6 张，共 **19 张当前图**。19 段图注全部通读；19 张 PNG 全部通过图像工具目视检查；对应 19 个 SVG 的可提取文本均检查过。目视检查是当前 PNG 的屏幕显示尺度，部分大图被工具缩小，未声称逐像素印刷质量审校。未使用可能过时的 `FIGURES.md` 代替实际清单。

正文第 1–7 节的跨段重复采用关键主张检索及相关段落连读，重点为摘要、第 1 节、4 节导言、6–7 节。它不是正文公式逐项验证，也不是全篇数字审计。数字只保留源文值；局部读取 ST23b 的定义来解释 Figure S06 的措辞，没有重算任何数据。原始文献全文未检索或复核。先读当前源稿及图，再读 `docs/paper_p1/review_r1/HANDOVER_CODEX_20261002_CN.md`；交接仅用于核对作者措辞偏好，没有据此执行修改。

最优先的问题是 **Figure 2 把有谱条件的能量误差次序写成无条件短句**。其余主要为读者需要反复回看才能解码的图注，以及把具体误差类型缩略成泛称。没有发现应据语言理由删除图或改变作者定位的证据。

下列 `L-F01`–`L-F05` 为本辅助审查的稳定候选编号，供主报告映射至全局 `Lxx`。

## 2. 五项优先候选及中英对照

### L-F01：方法总览图遗漏“不会增大误差”的条件

- 严重性：**必须修**。
- 当前位置：`docs/paper_p1/MANUSCRIPT_EN.md:180–182`；实际图 `figures/F01_method_overview.png`；`figures/F01_method_overview.svg:885`；生成源 `figures_src/fig02_overview.py:117`。
- 原文：`S ≼ Ŝ_tg ≼ Ŝ_0 (correction cannot increase the error)`。
- 中文：`S ≼ Ŝ_tg ≼ Ŝ_0（校正不会增大误差）`。
- 问题：该句作为独立图内结论，没有写误差类型，也没有写条件。图注 182 行解释校正减少内部不平衡，但未补足定理前提。读者只看图容易把它理解为任何校正参数下对任何误差量都成立，尤其同图下方还出现厚度灵敏度。
- 现有证据：正文 221 行规定 `D^{-1/2} A D^{-1/2}` 的正谱须位于 `(0,b]`；244 行明确“exact coarse-grid correction”和 `||Φ_k||_A ≤ 1`，并说明该次序不延伸到灵敏度误差。附录 D.1（`APPENDICES_EN.md:350–360`）再次限定为能量次序。此处无需新结果，仅把现有条件带回图中。
- 建议英文图内短句：`Energy-error ordering under the spectral condition (Sections 4.2–4.3)`。
- 建议中文：`满足谱条件时的能量误差次序（第 4.2–4.3 节）`。
- 建议英文图注补句：`The ordering shown in (a) assumes an exact coarse-grid correction and a smoothing upper endpoint that bounds the positive spectrum; it concerns energy error, not sensitivity error.`
- 建议中文：`(a) 中的次序以精确粗网格校正以及光滑区间上端点覆盖正谱上界为前提；它约束能量误差，不约束灵敏度误差。`
- 修改理由：让图的语气与正文定理范围一致，避免缩略句抹掉最关键的条件。不要写成“the smoothing interval contains the spectrum”，因为低于下端点的模态仍可能存在。
- 是否影响定位／结论／删除：不改定位；澄清既有结论的适用范围；不删除内容。
- 确信度与范围：**高**。已目视 PNG、核对 SVG 与生成源、比对当前正文和附录相应条件；未独立重证完整数学推导。

### L-F02：Figure 12 下方条带同时比较不同设计与同一设计，长括注掩盖区别

- 严重性：**建议修**。
- 当前位置：`docs/paper_p1/MANUSCRIPT_EN.md:521`，Figure 12(a) 图注；实际 `figures/F13_optimisation.png`。
- 原文：`Lower strip: NICE compliance relative to the twin run at the same design iteration (line; the designs of the two runs coincide at iterations 0–4 and differ afterwards) and to the exact compliance of the same design (circles: −0.011%, −0.018%, −0.028%); at iterations 16 and 19 (labelled 'perturbed designs') the geometry-generation fallback had perturbed the NICE design (Supplementary Note S9.4, Table ST24a).`
- 中文：`下方条带：NICE 柔度相对于孪生运行在同一设计迭代的柔度（曲线；两次运行在第 0–4 次迭代的设计相同，之后不同），以及相对于同一设计的精确柔度（圆圈：−0.011%、−0.018%、−0.028%）；在第 16 和 19 次迭代（标为“扰动后的设计”），几何生成回退过程扰动了 NICE 设计（补充 Note S9.4、表 ST24a）。`
- 建议英文：`In the lower strip, the line compares NICE with the exact-condensation twin at the same iteration. The two runs use identical designs at iterations 0–4 and different designs thereafter. The circles instead compare NICE with exact condensation at the same design (−0.011%, −0.018%, −0.028%). The labels at iterations 16 and 19 mark NICE designs perturbed by the geometry-generation fallback (Supplementary Note S9.4, Table ST24a).`
- 建议中文：`下方条带的曲线比较 NICE 与精确凝聚孪生运行在同一次迭代的柔度。两次运行在第 0–4 次迭代使用相同设计，此后使用不同设计。圆圈则比较同一设计下 NICE 与精确凝聚的柔度（−0.011%、−0.018%、−0.028%）。第 16 和 19 次迭代的标注指出，几何生成回退过程扰动了这些 NICE 设计（补充 Note S9.4、表 ST24a）。`
- 修改理由：按“线是什么—设计是否相同—圆圈是什么—标注是什么”组织，保留全部数字与例外，减少读者误把整条曲线当成同设计代理误差的可能。
- 证据范围：图内纵轴为 `NICE vs exact (%)`；当前图注明确存在两种比较，PNG 中曲线、圆圈和两个扰动标注均可见。本项依据已写出的比较定义，未重新计算 ST24a。
- 是否影响定位／结论／删除：均否；只拆句与重排。
- 确信度与范围：**高**（句法和解码顺序）；数值真实性不在本项核验范围。

### L-F03：Figure S06 的“cubic to”没有说清楚检测了什么

- 严重性：**建议修**。
- 当前位置：`docs/paper_p1/SUPPLEMENTARY_EN.md:1205`，Figure S06；相同压缩表述也见 1039 行。
- 原文：`(a) Effective elasticity tensor C^H_11, C^H_12, C^H_44 (...), cubic to 3.1×10^-13`。
- 中文：`(a) 有效弹性张量 C^H_11、C^H_12、C^H_44（……），立方到 3.1×10^-13`。这里逐字译法本身暴露了省略的逻辑主语：三个量是分量，立方对称性是张量的性质。
- 建议英文：`(a) Components C^H_11, C^H_12 and C^H_44 of the effective elasticity tensor (Voigt notation, engineering shear strains, E_Y=1, ν=0.3). The computed tensor has cubic symmetry to the numerical precision reported in Table ST23b; the largest relative spread among symmetry-equivalent entries is 3.1×10^-13.`
- 建议中文：`(a) 有效弹性张量的 C^H_11、C^H_12 和 C^H_44 分量（Voigt 记号，工程剪应变，E_Y=1、ν=0.3）。计算所得张量在表 ST23b 报告的数值精度内满足立方对称性；对称等价分量之间的最大相对离散程度为 3.1×10^-13。`
- 修改理由：明确曲线画的是张量分量，数字衡量的是对称等价分量间的相对差别，避免把“cubic”同时读成拟合曲线的三次形式。后文另有 `cubic splines`，两者尤其需要区分。
- 证据：`SUPPLEMENTARY_EN.md:1086` 说明三个对称分量组的最大相对离散程度分别为 `3.1e-13`、`6.1e-14`、`1.4e-13`，并单列非立方耦合和周期平衡残差。建议句只重述该字段，不把数字当成新的范数或严格零。
- 是否影响定位／结论／删除：均否；定义性澄清。
- 确信度与范围：**高**。已读图注、PNG、SVG 标题以及 ST23b 的解释行；未复算张量或追查原始计算记录。

### L-F04：Figure 7 的“no error”应明确是所画的延拓误差

- 严重性：**建议修**；优先级低于前三项。
- 当前位置：`docs/paper_p1/MANUSCRIPT_EN.md:388`，Figure 7；实际 `figures/F12_field_error_M1.png`。
- 原文：`Cut-band elements carry no error because their DOFs are retained.`
- 中文：`切割带单元不含误差，因为它们的自由度被保留。`
- 建议英文：`The extension error is zero on the cut-band elements because all their DOFs are retained.`
- 建议中文：`切割带单元上的延拓误差为零，因为它们的全部自由度均被保留。`
- 修改理由：原句在该图语境下可理解且不应判为数学错误，但 `no error` 是比当前证据更宽泛的表达。显式点出误差对象，让独立读图者不会理解成这些单元没有离散误差、几何误差或任何组装后误差。
- 证据：该图注前文已限定同一保留位移下的精确场与近似场；正文 2.2 节以保留自由度确定延拓边界值，图中 (a) 区分 retained DOFs（保留自由度）与 interior DOFs（内部自由度）。不提出新误差结论。
- 是否影响定位／结论／删除：均否；保留原判断，缩窄其名词指代。
- 确信度与范围：**高**（措辞）；未核验所有单元的原始自由度映射。

### L-F05：口号保留，但第 4 节入口可改成具体阅读任务

- 严重性：**作者选择**。
- 当前位置：`docs/paper_p1/MANUSCRIPT_EN.md:178`；重复骨架见摘要 5 行、引言 17 行与贡献段 25 行、结论 549 行；Figure 2 另有三块短标签。
- 原文：`The construction has three parts. A learned extension supplies the trial interior field (Section 4.4); the energy form, applied with the transpose of the complete extension, supplies the mechanical structure (Section 4.1); and the correction W supplies the improvability (Sections 4.2 and 4.3). Figure 2 shows how the three parts define both the condensed action and the displacement recovered after assembly. We present the mechanical object first, then the correction that improves it, and then the network that initialises it.`
- 中文：`该构造有三部分。学习延拓提供内部试探场；能量形式与完整延拓的转置提供力学结构；校正 W 提供可改进性。图 2 显示这三部分如何同时定义凝聚作用和组装后恢复的位移。下文先给出力学对象，再给出改善它的校正，最后给出初始化它的网络。`
- 建议英文：`Figure 2 follows a retained displacement through the learned interior extension, the fixed correction W and the force calculation with the transpose of the complete extension. The same extension recovers the cell field after assembly. We first define the condensed energy (Section 4.1), then the two-grid correction (Sections 4.2 and 4.3), and finally the network that supplies its initial field (Section 4.4).`
- 建议中文：`图 2 按保留位移的传递过程，展示学习得到的内部延拓、固定校正 W，以及通过完整延拓的转置计算力的步骤。组装后，同一延拓用于恢复单胞场。下文依次定义凝聚能量（第 4.1 节）、双网格校正（第 4.2–4.3 节），以及提供初始场的网络（第 4.4 节）。`
- 修改理由：摘要、引言和结论中的口号可继续承担主线识别作用；方法节入口改为读图步骤和小节任务，可以减少再次宣告贡献的感觉，并让章节推进更具体。此建议不依据任何词汇推断写作来源。
- 证据：当前文本和图的步骤顺序；交接 2.1 节明确作者已决定在摘要、引言骨架和结论保留口号。此处只建议调整额外出现的第 4 节入口，是否采用由作者决定。
- 是否影响定位／结论／删除：不改定位与结论；涉及压缩一个重复表达，必须由作者选择；不建议删除既定口号。
- 确信度与范围：**高**（重复位置可核对）；**中**（阅读节奏是编辑判断）。主张跨节重复本身不是错误。

## 3. 19 张当前图覆盖表

下表所有条目均完成：源稿图注通读、PNG 目视、SVG 可提取文字检查。`无高优先级问题` 仅指本项语言范围，不意味着数字、绘图数据或数学已经通过独立审计。

| 当前图 | 实际图片文件（均位于 docs/paper_p1/figures/） | 源稿图注行 | 已核验内容与保留判断 |
|---|---|---:|---|
| Figure 1 | F08_geometry.png | MANUSCRIPT 52 | 颜色、同尺度和剩余盒体体积分数定义清楚；没有把盒体剩余率叫作材料体积分数。可保留。 |
| Figure 2 | F01_method_overview.png | MANUSCRIPT 182 | 三部件顺序、完整转置和设计回环可读；L-F01 要补谱条件。口号标签可保留。 |
| Figure 3 | F11_network_architecture.png | MANUSCRIPT 278 | 几何系数与线性位移路径区分清楚，图注最后一句明确 fixed geometry（固定几何）。`3 levels` 与四层网格的读法可由图 (c) 理解为三次下采样；不将其贸然报成架构错误。 |
| Figure 4 | F09_assembly_loads.png | MANUSCRIPT 356 | 载荷面、夹持面、目标／邻胞和共享 DOFs（自由度）对应清楚。可保留。 |
| Figure 5 | F02_validation.png | MANUSCRIPT 374 | 80／75 个几何体、几何均值、百分位范围与模型选择重叠均说明；无过度普遍化标题。可保留。 |
| Figure 6 | F03_spectrum.png | MANUSCRIPT 382 | 模态排序及每条曲线各自归一化写清楚，避免误读两种能量的绝对大小。可保留。 |
| Figure 7 | F12_field_error_M1.png | MANUSCRIPT 388 | bulk element energy（体单元能量）、归一化及共同颜色尺度已限定；L-F04 细化末句误差对象。 |
| Figure 8 | F04_correction.png | MANUSCRIPT 416 | 固定网络参数、固定保留位移、步数轴及比较预算写清楚。`Network only` 由标题和图注明确指 Base network（基础网络），无需制造新变体名。 |
| Figure 9 | F05_assembly.png | MANUSCRIPT 428 | “未对每个配置评估每个变体”明确，最大值口径和 3% line（3% 线）可读。可保留。 |
| Figure 10 | F10_energy_share.png | MANUSCRIPT 444 | energy share（能量份额）与局部误差乘积、精确组装位移均有定义；使用当前实际链接 F10_energy_share，未按旧图索引下结论。未核对每个点，也未独立核验 L1 的数据纳入。 |
| Figure 11 | F06_bernstein.png | MANUSCRIPT 452 | 明确两胞都用精确算子、只限制盒面表示；不把它称作对其他文献方法的复现。可保留。 |
| Figure 12 | F13_optimisation.png | MANUSCRIPT 521 | 精确同设计检查点、双运行比较、体积阶段和扰动标注可见；L-F02 重排下方条带图注。 |
| Figure 13 | F14_designs_scale.png | MANUSCRIPT 525 | `Scale demonstration`（规模展示）没有宣称超过 24 胞已验证精度；可保留 NICE 值与精确检查索引的区分。 |
| Figure S01 | S06_reference_verification.png | SUPPLEMENTARY 1185 | 明说 H1 非单调以及 n=48 差值不是 n=32 误差估计，是应保留的证据边界。 |
| Figure S02 | S01_distributions.png | SUPPLEMENTARY 1189 | 各载荷类样本数、几何均值、总体中位数与对数轴解释完整。五变体图例统一。 |
| Figure S03 | S03_sensitivity_diagnostics.png | SUPPLEMENTARY 1193 | 线性项范数占比与绝对贡献占比没有混同，六胞／五胞边界写清楚。可保留。 |
| Figure S04 | S02A_smoothing.png | SUPPLEMENTARY 1197 | learned／zero initialisation（学习／零场初始化）比较明确，零场灵敏度只记录 32 步未掩饰。可保留。 |
| Figure S05 | S02B_coarse_spaces.png | SUPPLEMENTARY 1201 | `Network`（网络）显式映射到 Base network（基础网络），保留 DOFs 不变写清楚；图注密度较高但未发现应优先改的推断跃迁。 |
| Figure S06 | S06_homogenised_law.png | SUPPLEMENTARY 1205 | 张量分量曲线与体积分数清楚；L-F03 把“cubic to”还原为可核验的对称性指标。 |

## 4. 跨段重复的保留与编辑顺序

| 主张 | 当前重复位置（MANUSCRIPT_EN.md） | 判断 |
|---|---|---|
| 学习给试探场、变分形式给结构、校正给可改进性 | 5、17、25、178、549；Figure 2 标签 | 这是作者已定主线。保留摘要、引言与结论；L-F05 仅提供方法节入口的可选替代。 |
| compliance accuracy（柔度准确性）不保证 sensitivity accuracy（灵敏度准确性） | 15、25、134、424、440、541、551 | 不应一概当重复删掉：15 是问题动机，134 是推导后果，424／440 是具体反例，541 是成本判据，551 是结论。优先保留具体反例；贡献段可由主审整体压缩。 |
| 无需重训的部署校正 | 17、25、346、416、533、549 | 方法定义、变体命名、实验解释各有作用；只有同一贡献串连续重复时才需要删并。 |
| 构造一般而训练网络专用 | 31、545、553 | 第 2 节前提与第 6 节限制有必要同时存在；末句能否称“transfers”应交定位和证据审查，不能仅凭润色扩大适用范围。 |

推荐顺序：先修 L-F01 的图文条件一致性，再处理 L-F02 的比较对象，随后澄清 L-F03／L-F04 的指标名词；最后由作者决定 L-F05 的节奏调整。不要以同义词替换来掩盖重复：`retained DOFs`、`interior`、`cut band`、`active element`、`weak support`、`energy share`、`two-grid correction`、`background element` 与 `3% line` 应保持一致。变体仍使用 NICE、Base network、Uncorrected、Smoothing-trained 和 'Base network, corrected'。

本项没有建议修改数字、改变文献口径、增加实验、删除图、变更结论或猜测 AI 使用声明。没有使用检测器，也没有凭语言特征判定写作来源。
