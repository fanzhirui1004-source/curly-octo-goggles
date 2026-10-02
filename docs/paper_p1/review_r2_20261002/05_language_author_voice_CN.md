# P1 英文语言与作者声音独立审查

日期：2026-10-02。固定基线：700af856f2b9f152ab2c263406c54132a11255bf。分支：claude/wizardly-euler-3m9cwx。

## 1. 判断与使用方式

稿件已有明确的作者判断：保留完整 retained DOFs（保留自由度），只近似 interior（内部）延拓；用离散能量形成可组装算子；用 two-grid correction（两网格校正）改善内部平衡；同时检查柔度与局部灵敏度。这条论证值得保留。最需要改的是信息组织和证据语气：引言反复铺陈同一组要求，长段落把比较、机制、保证和结果合在一起；结果段又把几何均值、方向极值、不同载荷和不同验证集连续堆叠，读者容易失去统计对象。

本报告给出24个优先段落或图文单元，编号 L01–L24。每项提供精确英文摘录、中文、建议英文和中文。建议通常只替换所引局部；未引部分及原有引用保持。英文建议是可讨论稿，不是已经执行的修改。涉及定位、结论范围、移段或删节的条目明确标记，须先与作者讨论。没有使用写作来源检测工具，也不根据词汇或句式推测作者身份。

必须先处理的内容是：L11 的“最终设计不变”、L12 的“达到局部最优”、L16 图2的无条件误差保证、L19 的3%线说明冲突、L20 的谱比值解释、L23 的相对差量纲，以及L24中数值谱核验与严格包含证明的区别。L15、L21、L22涉及证据能支持多强的推论，也应在定稿前核对。口号本身按作者偏好保留；减少重复属于作者选择。

## 2. 基线、文件和核验边界

实际工作目录是 /Users/fffffreeze/Desktop/curly-octo-goggles，未使用默认旧仓库的论文版本。开始时核对了提交、分支和干净工作树；先读三份英文源稿，之后才读本轮交接。交接称 Note S1–S10，当前补充源稿实际只有 S1–S9；ST26、ST27以粗体表题出现，不是 Markdown 标题，因此已另行纳入。图的范围由正文和补充材料中的实际19个图像链接确定，没有把派生 FIGURES.md 当作当前清单。

以下简写均指当前基线的完整绝对路径，行号为源 Markdown 行号：

- M：[MANUSCRIPT_EN.md](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/MANUSCRIPT_EN.md)，1–699行。
- A：[APPENDICES_EN.md](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/APPENDICES_EN.md)，1–826行。
- S：[SUPPLEMENTARY_EN.md](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/SUPPLEMENTARY_EN.md)，1–1205行。
- E：/Users/fffffreeze/Desktop/curly-octo-goggles/docs/paper_p1/evidence/。
- 交接：[HANDOVER_CODEX_20261002_CN.md](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/review_r1/HANDOVER_CODEX_20261002_CN.md)。只作为作者偏好和历史语境，不覆盖当前文本事实。

三份源稿的 SHA-256：

| 文件 | SHA-256 |
|---|---|
| M | 74e0db7e5f6773c8fed9e987a800c4e9657ea12a1b4a21af2b40fb7a7286b421 |
| A | 1af27edbb30f94a1472778717ec098e1ca0c49aca263074c79d7d80ebafb8e67 |
| S | 81d2a02a2ff7fe7f2c5acc7b478598c8e1eb33404ed3323b250d849a8e00c600 |

核验范围：

- 英文论述、表题、表注和图注：三份源稿完整阅读；正文1–7、附录A–I、ST01–ST27、实际全部Notes和19图全部覆盖。
- 数学：阅读公式上下文、检查语言是否保留假设；局部核对误差排序、统计口径、最优性措辞及谱比值反例。没有逐式独立重证全部推导。
- 数字：通读全部表格，交叉核对本报告涉及的数字与相应源表；原始记录只抽查最终设计柔度、交叉起点停止预算等。没有对全部实验记录重算，没有运行有限元、训练或服务器实验。
- 文献：通读当前引文及参考文献表，关注句子承载量和比较方式；没有重新查阅全部文献全文，因此 L01–L04 不构成对相关工作的独立事实认证。保留引用对象和原有文献口径。
- 图形：图文子审查逐张查看19张当前PNG，并检查图注和SVG文字；本主审亲自复核了图2。未重绘图、重算曲线、重建TeX/PDF或逐像素核验全部数据。
- 仅在指定缓存目录写报告。论文、代码、图及生成物未修改；未提交或推送。

## 3. 全篇覆盖表

### 3.1 正文和附录

| 范围 | 当前行号 | 阅读结论 / 对应建议 |
|---|---:|---|
| 标题、摘要、关键词 | M1–7 | 摘要叙事已完整；开头双重概括可具体化，L01 |
| 1 引言 | M9–27 | 完整阅读；要求、贡献、相关工作反复压入长段，L02–L05 |
| 2.1–2.3 离散模型、保留自由度、组装 | M29–88 | 定义具体，保留；四项要求与引言重复，L05 |
| 3.1 变分误差 | M90–111 | 保留残差可计算与精确误差需内部求解的区别 |
| 3.2 柔度、能量份额 | M113–143 | 公式后解释清楚；结果中的“累积”措辞见L09 |
| 3.3 灵敏度与完整导数 | M145–174 | 必须保留两种导数的区分；结尾路标可压缩，L05 |
| 4.1–4.3 能量、平滑、粗网格校正 | M176–244 | 条件写得清楚，图2须同步，L16 |
| 4.4–4.6 网络、训练、算子应用 | M246–304 | 具体动作优于贡献宣言；层次与算法可保留 |
| 5.1 设置 | M306–356 | 样本、变体和3%线定义应保留 |
| 5.2 验证 | M358–362 | 数值检查与谱条件应分段，L06 |
| 5.3 几何与载荷 | M364–374 | 分清分层均值与方向分布，L07 |
| 5.4 误差分量 | M376–392 | 机制与量纲解释有价值，建议保留 |
| 5.5 固定参数校正 | M394–416 | 控制变量明确，保留；不把预算样本趋势升格为逐步单调保证 |
| 5.6 两胞组装 | M418–444 | 对照解释清楚；“因此跟随”需具体，L08 |
| 5.7 保留表示消融 | M446–452 | 明确不是复现竞争方法，保留 |
| 5.8 全学习格架 | M454–460 | 加权和的语言需精确，L09 |
| 5.9 成本 | M462–485 | 路线和硬件边界已有说明；长结果段拆组，L10 |
| 5.10 优化 | M487–525 | 最终设计相似不等于未变；固定预算不等于局部最优，L11–L12、L23 |
| 6.1–6.3 讨论 | M527–541 | 作者判断可保留；口号重复由作者决定；6.2公式不宜删去条件 |
| 6.4 局限 | M543–545 | 范围完整但一段负担过重，L13 |
| 7 结论 | M547–553 | 保留主线及实测数字；外推的动词需分清，L14 |
| 可用性、参考文献、补充入口 | M555–699 | 完整读文本；未做全部外链/全文核验，暂不把未来归档写成已归档 |
| 附录A | A1–48 | 完整读；DOF顺序和统计定义保留；3%用语与L19统一 |
| 附录B | A50–165 | 完整读；刚体核假设、κ不是条件数的解释保留 |
| 附录C | A167–294 | 完整读；求解残差、作用与能量差的条件保留 |
| 附录D | A296–376 | 完整读；重复固定周期与改变平滑次数的区别保留；严格谱包络另需数学核验，L24 |
| 附录E | A378–424 | 完整读；完整转置及逆序操作解释具体，保留 |
| 附录F | A426–483 | 完整读；实现精度与实数算子的区别保留；实施、实测、条件可分段 |
| 附录G | A485–632 | 完整读；方向覆盖限制、统计量区分保留 |
| 附录H | A634–806 | 完整读；长段混合数值证据与逻辑判断，L15 |
| 附录I | A808–826 | 完整读；小下界不能确认小误差的表述简洁，保留 |

### 3.2 补充表

| 表 | 当前行号（含说明） | 语言与口径核验 |
|---|---:|---|
| R1及ST01 | S5–72 | 变体、几何、训练集合、选择集合；保留缺失选择分数的披露 |
| ST02 | S74–110 | 保留自由度和几何域；3%用语统一，L19 |
| ST03、ST03b–d | S112–167 | 均值/分位数/最大值对象，L07；不混合选择内外样本 |
| ST04 | S169–179 | 列定义过长，L17 |
| ST05a–b | S181–234 | 线性项范数份额和两个能量分母明确，可保留 |
| ST06a–b | S236–268 | 相同保留位移、不同内部初场，清楚 |
| ST07a–b | S270–326 | 保留“粗基列数不一定是空间维数”的说明 |
| ST08、ST08b | S328–386 | 相同预算及更大预算区分清楚，保留 |
| ST09、ST09b | S388–469 | 选择内/外、两胞评估顺序，L18–L19 |
| ST10 | S472–516 | 3%线与正文矛盾，L19 |
| ST11 | S518–530 | 三个替换误差不能相加，保留 |
| ST12 | S532–555 | 同载荷局部灵敏度和能量份额，保留 |
| ST13 | S557–563 | 单载荷加权界，保留 |
| ST14 | S575–584 | 有限细化对比，不能自动写成连续体误差证明 |
| ST15 | S586–599 | 谱比值的解释超出该比值，L20 |
| ST16a–b | S645–694 | 消融范围、载荷集合和最大值明确，保留 |
| ST17a–e | S710–765 | 阶段、精度和内存口径有定义，L10、L21 |
| ST18 | S767–776 | 准备/凝聚/应用和批大小分别解释，保留 |
| ST19 | S810–823 | 数值导数、离散开关与推论范围，L22 |
| ST20 | S829–846 | 参数/固定存储值及初始化分开，保留 |
| ST21 | S940–962 | 停止规则、目标与梯度不一致已写明，L12 |
| ST22a–c | S974–1031 | 同设计/异设计比較、相对差，L11、L23 |
| ST23a–c | S1043–1095 | 固定12次交叉起点不能称已达局部最优，L12 |
| ST24a–b | S1103–1130 | 计数未变不代表集合未变的限定可保留 |
| ST25 | S1136–1148 | 多实现差异不能孤立归因于精度，保留其表注 |
| ST26 | S1156–1166 | 2–4次分析计时与完整优化时间推算区分清楚 |
| ST27 | S1170–1179 | 仅六设计及最终两设计梯度已核验，保留范围 |

### 3.3 实际全部Supplementary Notes（补充说明）

| Note | 当前行号 | 覆盖 |
|---|---:|---|
| S1 | S565–567 | 可视化采样与力学离散区别，完整读 |
| S2 | S569–599 | 参考细化、导数谱，完整读；L20 |
| S3 | S602–615 | 富集列冗余与未验证求解，完整读 |
| S4 | S617–694 | 表示消融、条件、全结果，完整读 |
| S5 | S696–776 | 三条路线、计时、内存和精度，完整读；L21 |
| S6 | S778–823 | 预条件、两胞配置、复算和参数扫描，完整读；L22 |
| S7 | S825–846 | 重建网络所需设置，完整读 |
| S8 | S848–922 | 六个矩阵示例及第一例矩阵，完整读；未独立重算全部斜率 |
| S9 | S924–1179 | S9.1–S9.6全部读；L12、L23 |

### 3.4 当前19张图

全部图注均完整读；图文子审查已逐张查看PNG并核对SVG文字。以下是图文核验，不等于重算图数据。

| 当前图号 | 当前PNG文件（位于docs/paper_p1/figures/） | 图注行 | 结论 |
|---|---|---:|---|
| 1 | F08_geometry.png | M52 | 几何和保留体积定义具体，保留 |
| 2 | F01_method_overview.png | M182 | 主审复核PNG；排序缺条件，L16 |
| 3 | F11_network_architecture.png | M278 | 网络分支及线性位移路径说明可保留 |
| 4 | F09_assembly_loads.png | M356 | 支撑、载荷、私有DOF说明可保留 |
| 5 | F02_validation.png | M374 | 几何均值及选择样本说明可保留 |
| 6 | F03_spectrum.png | M382 | 误差与精确场分别归一化，保留 |
| 7 | F12_field_error_M1.png | M388 | 单方向、体积能量和相同保留位移清楚 |
| 8 | F04_correction.png | M416 | 固定参数与固定保留位移明确 |
| 9 | F05_assembly.png | M428 | 六载荷/两胞最大值明确，保留 |
| 10 | F10_energy_share.png | M444 | 用精确组装迹定义β，保留 |
| 11 | F06_bernstein.png | M452 | 对比范围明确，保留 |
| 12 | F13_optimisation.png | M521 | 同设计与同迭代索引区别重要；可拆分图注 |
| 13 | F14_designs_scale.png | M525 | 设计与规模计时分开；不写成110胞完整优化验证 |
| S01 | S06_reference_verification.png | S1185 | H1非单调细化警示应保留 |
| S02 | S01_distributions.png | S1189 | 方向类、样本数、几何均值明确 |
| S03 | S03_sensitivity_diagnostics.png | S1193 | 一阶项范数份额和绝对贡献不可混为带符号贡献 |
| S04 | S02A_smoothing.png | S1197 | 相同保留位移、零初场记录范围明确 |
| S05 | S02B_coarse_spaces.png | S1201 | 生成列与空间维数的区别及秩限制已读；保留ST07与Note S3的说明 |
| S06 | S06_homogenised_law.png | S1205 | 周期均匀化数据与样条插值分开，保留 |

## 4. 优先修改单元：英文原文、中文及建议

以下“必须修”表示目前句子需要在发表前处理；若涉及科学解释，仍由作者选择补证或收紧说法，不代表授权本审查直接改变结论。确信度针对识别出的语言/逻辑问题，不等于对全部实验的确信度。

### L01　摘要开头：让失效条件对应具体模型

- 严重性：建议修。位置：[M5](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/MANUSCRIPT_EN.md#L5)。
- 证据：同段；M11、495–497；S1041、1168。
- 影响：不改主线或数字；保留作者的比较框架，若扩大或缩小相关工作口径须作者讨论。确信度：高（可读性）；文献比较未独立全文核验。

**原文**

> Lattices are analysed by homogenisation or by assembling component models; when scale separation fails and component geometry varies, the first loses accuracy and the second reusability. For a single cut layer, homogenisation underestimates compliance by 27–37%.

**中文**：格架通过均匀化或组装组件模型分析；当尺度分离失效、组件几何变化时，前者失去精度，后者失去复用性。对单层切割结构，均匀化低估柔度27–37%。

**建议英文**

> Lattice analysis faces two difficulties: loss of scale separation limits homogenisation accuracy, while changing geometry limits reuse of component models. In the single cut layer studied here, homogenisation underestimates compliance by 27–37%.

**建议中文**：格架分析面临两种困难：尺度分离失效限制均匀化精度，几何变化限制组件模型复用。在本文研究的单层切割结构中，均匀化低估柔度27–37%。

**为何更清楚**：去掉“the first / the second”的回指负担，并让27–37%明确属于本文算例。保留比较的作者判断，不以同义词替换制造“自然感”。仅改摘要开头，整段词数需在作者选定后重新检查。

### L02　引言首段：把四个要求与回应分成可辨认的步骤

- 严重性：建议修。位置：[M11](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/MANUSCRIPT_EN.md#L11)后半。
- 证据：M11、88及Sections 3–4（第3–4节）。
- 影响：无数字或结论变化；仅句序及分段。确信度：高。范围：论证组织，不重新认证前半段文献。

**原文**

> A component model that is to replace static condensation in analysis and design must be symmetric and positive semidefinite with the rigid-body kernel, so that it can be assembled; it must apply to every new geometry, including one whose discrete space changes with the cut, without an interior factorisation; its error must be traceable to the compliance and to the sensitivities that analysis and design require; and it should be improvable at deployment without retraining.

**中文**：要在分析和设计中替代静力凝聚，组件模型须具有对称性、半正定性和刚体核以便组装；须无需内部因子分解即可用于每个新几何，包括离散空间随切割变化的几何；其误差须能追溯到柔度和灵敏度；并应能在部署时不重训地改善。

**建议英文**

> We require a component model to retain symmetry, positive semidefiniteness and the rigid-body kernel, so that it can be assembled. It must accommodate a new geometry without an interior factorisation, even when the cut changes the discrete space. We also require its error to be traceable to compliance and local sensitivity, and its accuracy to be improvable at deployment without retraining.

**建议中文**：我们要求组件模型保持对称性、半正定性和刚体核，以便组装。即使切割改变了离散空间，它也必须无需内部因子分解即可处理新几何。我们还要求误差能追溯到柔度和局部灵敏度，并能在部署时不重训地改善精度。

**为何更清楚**：“We require”保留作者的取舍和研究标准；三个句子按结构、几何适用性、误差与校正组织。首段前面的背景/文献应另成段，不把整段改成一套均匀长短的模板。

### L03　PIML比较：先说表中证据，再说本文选择

- 严重性：建议修。位置：[M19](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/MANUSCRIPT_EN.md#L19)。
- 证据：当前段引用的Guo et al. (2026a)及其中所述Table 1；本轮未重新查全文。原始链接：[论文DOI](https://doi.org/10.1016/j.cma.2026.118955)。
- 影响：不改竞争方法口径或数值；长段拆分。确信度：高（行文），文献事实仍需文献审查确认。

**原文**

> Their Table 1 shows this: a learned interior with every boundary node retained, 152 nodes for \(5\times5\times5\)-element substructures, was the most accurate option (displacement error 0.84% against 1.95% with the Bézier boundary), but with 602 nodes for \(10\times10\times10\)-element ones it was not (3.46% against 2.22%); in both cases it was seven to fifteen times slower, so they adopted the Bézier boundary as the better balance of accuracy and cost.

**中文**：其表1展示了这一点：保留全部边界节点的学习内部，在5×5×5单元子结构、152节点时最准确（0.84%，Bézier边界为1.95%）；在10×10×10单元、602节点时则不是（3.46%对2.22%）；两种情形下均慢7–15倍，因此作者采用Bézier边界折中精度与成本。

**建议英文**

> Their Table 1 compares both choices. For \(5\times5\times5\)-element substructures with 152 boundary nodes, retaining every node gives a displacement error of 0.84%, compared with 1.95% for the Bézier boundary. For \(10\times10\times10\)-element substructures with 602 boundary nodes, the errors are 3.46% and 2.22%, respectively. The full-node option is seven to fifteen times slower in these comparisons, which motivated their choice of the Bézier boundary.

**建议中文**：其表1比较了两种选择。5×5×5单元、152边界节点时，全节点方案的位移误差为0.84%，Bézier边界为1.95%。10×10×10单元、602边界节点时，两者分别为3.46%和2.22%。在这些比较中，全节点方案慢7–15倍，因此他们选择Bézier边界。

**为何更清楚**：两种规模形成一组可比较的句子；最后再给选择理由。原段前部的PIML发展清单、此处的具体证据、后部的本文取舍宜分成三段。此举不删除先行研究，也不把不同规模直接解释为单一因果证明。

### L04　贡献段：让每项贡献有一个主语和一个核心动作

- 严重性：作者选择。位置：[M25](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/MANUSCRIPT_EN.md#L25)，第(i)项。
- 证据：M96–111、186–244；完整假设在A56–62。
- 影响：涉及贡献呈现与文字压缩，须作者讨论；不主张改定位、删科学内容或改变保证。确信度：高。范围：叙事去重。

**原文**

> The condensed stiffness thereby keeps the properties of static condensation, including the lower bound by the exact Schur complement, remains assemblable and differentiable, and is improved by the correction at deployment without retraining and without loss of these properties; the correction cannot increase the error when the upper end of its smoothing interval bounds the spectrum.

**中文**：凝聚刚度因此保持静力凝聚的性质，包括以精确Schur补为下界，仍可组装和求导，并可在部署时不重训、也不丢失这些性质地改善；当平滑区间上端点界定谱时，校正不能增加误差。

**建议英文**

> This construction preserves the variational structure of static condensation on the complete retained space. The condensed operator can be assembled and differentiated on the stated design intervals. Its error can be reduced at deployment without retraining; the correction cannot increase the error when the upper end of its smoothing interval bounds the spectrum.

**建议中文**：这一构造在完整保留空间上保持静力凝聚的变分结构。凝聚算子可组装，并可在所述设计区间内求导。部署时无需重训即可降低误差；当平滑区间上端点界定谱时，校正不能增加误差。

**为何更清楚**：将结构、设计求导条件、部署校正各放在一个句子。完整的半正定性、刚体核和下界保留在该项前文或第4.1节，不在引言多个位置逐字重列。若作者希望贡献段自足，可保留性质清单，优先分句而非删减。

### L05　口号与章节路标：保留标志句，减少邻近重复解释

- 严重性：作者选择。位置：[M178](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/MANUSCRIPT_EN.md#L178)；重复链M5、17、25、88、174、549。
- 证据：上述当前段落；交接2.1明确口号为作者偏好。
- 影响：只涉及重复与篇幅，任何删节先由作者选择；不改口号或定位。确信度：高。范围：跨节重复阅读。

**原文**

> The construction has three parts. A learned extension supplies the trial interior field (Section 4.4); the energy form, applied with the transpose of the complete extension, supplies the mechanical structure (Section 4.1); and the correction \(\mathcal W\) supplies the improvability (Sections 4.2 and 4.3).

**中文**：构造分为三部分：学习延拓提供内部试探场；能量形式连同完整延拓的转置提供力学结构；校正提供可改进性。

**建议英文**

> Figure 2 connects the learned interior field to the condensed operator. We first define the energy and its work-conjugate force, then the fixed correction, and finally the network that supplies the initial field.

**建议中文**：图2把学习内部场与凝聚算子连接起来。下文先定义能量及其功共轭力，再介绍固定校正，最后介绍提供初场的网络。

**为何更清楚**：第4节开头承担阅读导航即可。摘要、引言和结论中的作者口号可以照留；避免在口号之外又叠加“四要求→三条件→三部分→三贡献”的连续宣言。这里没有理由把每段都改成“问题—方法—意义”的统一模板。

### L06　算子验证：先分两类核验，再给数值

- 严重性：建议修。位置：[M362](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/MANUSCRIPT_EN.md#L362)。
- 证据：S169–179的ST04；A340的80几何谱检查。
- 影响：不改数值、假设或结论；分段。确信度：高。范围：源表对照；未重算80个谱。

**原文**

> To verify the operator (arithmetic as in Table 1): on U2, M1, M2, H1 and H2, over the consistent-traction validation directions, the bilinear form \(q_i^T\widehat Sq_j\) is symmetric to a relative \(9\times10^{-9}\), the returned work \(q^T\widehat Sq\) agrees with the recovered-field energy \(q^TF^TKFq\) to \(5\times10^{-9}\), the deployed operator reproduces the training-time field to \(1.1\times10^{-7}\), and the energy of normalised rigid-body modes is at most \(2\times10^{-11}\) of a typical deformation energy;

**中文**：在五个胞的一致牵引验证方向上，双线性形式的相对对称误差为9×10⁻⁹，返回功与重建场能量差为5×10⁻⁹，部署与训练场差为1.1×10⁻⁷，归一化刚体模态能量至多为典型变形能量的2×10⁻¹¹。

**建议英文**

> We checked the deployed operator on U2, M1, M2, H1 and H2 using the arithmetic of Table 1. Under the consistent-traction validation directions, relative asymmetry is at most \(9\times10^{-9}\), and returned work agrees with recovered-field energy to \(5\times10^{-9}\). The deployed and training-time fields agree to \(1.1\times10^{-7}\). Normalised rigid-body modes carry at most \(2\times10^{-11}\) of a typical deformation energy (Table ST04).

**建议中文**：我们按表1的精度设置，在五个胞上核验部署算子。一致牵引验证方向上的相对不对称性至多为9×10⁻⁹，返回功与重建场能量在5×10⁻⁹内一致。部署场与训练场在1.1×10⁻⁷内一致；归一化刚体模态至多携带典型变形能量的2×10⁻¹¹。

**为何更清楚**：四类性质可逐一辨认。紧接的节点力结果保留；从“The contraction of Section 4.2 requires…”起另成一段，明确这是另一项、覆盖80几何的谱条件核验。保持最后关于更大格架未检查谱的说明。

### L07　几何误差段：不要让分层统计和方向极值共用一条叙述线

- 严重性：建议修。位置：[M366](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/MANUSCRIPT_EN.md#L366)。
- 证据：S114、ST03b S132–135、ST03d S155。
- 影响：不改任何数字；按统计对象分句和分段。确信度：高。范围：源表/文字交叉核对，非原始方向记录重算。

**原文**

> Under consistent tractions, its mean directional energy error is 0.016%, 0.079%, 0.091% and 0.109% across the uncut, lightly, moderately and heavily cut strata (Supplementary Table ST03b; 0.074% overall, largest geometry mean 0.65%), against 1.03%, 5.33%, 7.82% and 11.2% for the Uncorrected continuation (6.33% overall, largest 42%) and 6.89% for the base network (largest 48.6%).

**中文**：一致牵引下，四个几何分层的平均方向能量误差为0.016%、0.079%、0.091%、0.109%，整体0.074%、最大几何均值0.65%；Uncorrected分别为1.03%、5.33%、7.82%、11.2%，整体6.33%、最大42%；基础网络整体6.89%、最大48.6%。

**建议英文**

> Under consistent tractions, NICE has mean directional energy errors of 0.016%, 0.079%, 0.091% and 0.109% in the uncut, lightly, moderately and heavily cut strata, respectively (Table ST03b). The corresponding Uncorrected means are 1.03%, 5.33%, 7.82% and 11.2%. Across all geometries, the mean is 0.074% for NICE, 6.33% for Uncorrected and 6.89% for the base network; their largest geometry means are 0.65%, 42% and 48.6%, respectively.

**建议中文**：先分别列出NICE和Uncorrected的四个分层均值，再单独报告三变体的整体均值和最大几何均值，所有数值不变。

**为何更清楚**：先回答切割程度的影响，再回答总体性能。原段随后5,120个方向的95/99分位数和最大值另成一句/段，明确其单位是方向，不能用“largest”无标记接在几何均值之后。

### L08　两胞结果：用观察关系替代宽泛的“therefore follows”

- 严重性：建议修。位置：[M420](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/MANUSCRIPT_EN.md#L420)。
- 证据：S457–469，ST09b的W1–W5和四个随机胞；M420前句。
- 影响：仅明确已观察关系，不改通过3%线的结果；若将其视作普遍预测律，则须作者讨论结论范围。确信度：高。范围：源表对照，非所有误差相关性分析。

**原文**

> Assembled accuracy therefore follows the single-cell error, and stays below both 3% lines in every held-out configuration.

**中文**：因此，组装精度跟随单胞误差，并在每个留出配置中均低于两条3%线。

**建议英文**

> In these held-out configurations, the five cells with the largest single-cell errors account for all compliance errors above 0.02%. All configurations remain below both 3% lines.

**建议中文**：在这些留出配置中，单胞误差最大的五个胞涵盖了所有超过0.02%的柔度误差。全部配置仍低于两条3%线。

**为何更清楚**：直接说明“跟随”具体指什么，不暗示已经建立组装误差对单胞误差的通用单调关系。0.02%来自同段，不是新增阈值。采用时将前一句中的同一事实合并，避免产生新的重复。

### L09　格架误差：加权相加仍然是求和

- 严重性：建议修。位置：[M458](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/MANUSCRIPT_EN.md#L458)。
- 证据：M127–134，Eq. (7)；A229–254。
- 影响：不改变误差界或实测结果；明确数学关系。确信度：高。范围：公式与解释对照，未重算格架。

**原文**

> These errors are below the largest two-cell errors: the cells' energy errors combine through their shares of the assembled energy rather than accumulate.

**中文**：这些误差低于最大的两胞误差：各胞能量误差通过其在组装能量中的份额组合，而不是累积。

**建议英文**

> These errors are below the largest two-cell errors. For compliance, Eq. (7) combines the local energy errors as a weighted sum, with weights given by the cells' shares of the exact assembled energy.

**建议中文**：这些误差低于最大的两胞误差。对柔度而言，式(7)将局部能量误差按加权和组合，权重由各胞在精确组装能量中的份额给出。

**为何更清楚**：“rather than accumulate”容易被读成胞数增加时误差不会累积，且前面的“these errors”还包括灵敏度和位移。显式限定柔度、给出加权和，保留作者解释又不把同一机制套到所有响应量上。

### L10　成本段：先说明谁慢，再分别谈准备和内存

- 严重性：建议修。位置：[M483](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/MANUSCRIPT_EN.md#L483)。
- 证据：M480–481；S731、733、744–745、755–756。
- 影响：不改数字或比较范围；拆分长段。确信度：高。范围：源表一致性；没有重新计时。

**原文**

> For the eight-cell lattices, NICE takes 81 and 110 s against 868 and 1,042 s for the direct solution, about ten times longer, and 950 and 1,218 s for conventional condensation.

**中文**：对八胞格架，NICE耗时81和110秒，直接求解为868和1,042秒，大约长十倍，常规凝聚为950和1,218秒。

**建议英文**

> For the two eight-cell lattices, NICE takes 81 and 110 s per design iteration. The whole-lattice direct solution takes 868 and 1,042 s, about ten times as long; conventional exact condensation takes 950 and 1,218 s.

**建议中文**：对两个八胞格架，NICE每次设计迭代耗时81和110秒。整格架直接求解耗时868和1,042秒，约为前者十倍；常规精确凝聚耗时950和1,218秒。

**为何更清楚**：避免“about ten times longer”靠近NICE而造成回指歧义。整段宜按总耗时、准备成本、计时路径的误差、内存分别成段。M472的不同处理器比较说明保留。正文硬件只用AMD EPYC 9654和RTX 5090；不补机器负载、共享主机等运行叙事。

### L11　最终设计：“非常接近”不等于“没有改变”

- 严重性：必须修。位置：[M491](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/MANUSCRIPT_EN.md#L491)末句。
- 证据：同段最大参数差0.0051；S968、1031；E/opt/optA/optA/check_023.json的C_exact与E/opt/optA/optAx/history.jsonl最终C。
- 影响：涉及结论语气，须作者确认；保留复现最终设计的主判断，不改数字。确信度：高。范围：当前描述和最终柔度记录抽查，未重跑优化。

**原文**

> Both errors grow along the path as the design approaches the bounds, without changing the final design.

**中文**：随着设计接近边界，两种误差沿路径增大，但没有改变最终设计。

**建议英文**

> Both errors grow along the path as the design approaches the bounds. The final designs nevertheless remain close: their corner parameters differ by at most 0.0051.

**建议中文**：随着设计接近边界，两种误差沿路径增大。最终设计仍非常接近，其角点参数最大相差0.0051。

**为何更清楚**：陈述实际观察到的接近程度；不声称误差对优化路径没有因果影响，也不暗示两个最终参数向量完全相同。为避免重复，可将同段已有0.0051移至此处，不能再据此宣称任一设计更优。

### L12　交叉起点：“不同终点”尚未证明“局部最优”

- 严重性：必须修。位置：[M497](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/MANUSCRIPT_EN.md#L497)。
- 证据：S1041明确两次延续均未达到停止规则；ST23a S1050、1057；E/opt/plates/xstartH_y/meta.json与xstartH_z/meta.json的args.stop_after=12；两份history.jsonl最后均为k=11，dx分别约0.025389、0.003340。
- 影响：限定优化结论，须作者讨论；无需删除交叉起点结果。确信度：高。范围：停止预算和最终记录已抽查，未判定任何设计满足或违反最优性条件。

**原文**

> X-\(z\) ends 0.87% below B2, so the optimisations reach local optima that depend on the start.

**中文**：X-z最终比B2低0.87%，所以优化到达了依赖起点的局部最优解。

**建议英文**

> After 12 continuation iterations, X-\(z\) has 0.87% lower compliance than B2. The terminal designs therefore depend on the starting design within the reported iteration budgets; neither continuation met the stopping rule (Table ST23).

**建议中文**：延续12次迭代后，X-z的柔度比B2低0.87%。因此，在已报告的迭代预算内，终止设计依赖起点；两次延续都未满足停止规则。

**为何更清楚**：保留起点影响这一有价值的作者判断；把观察到的终点差异与未验证的局部最优性分开。不能把“local optima”仅换成“converged designs”，后者仍与当前记录冲突。

### L13　局限段：按四种边界分段，避免读者把保证与精度混在一起

- 严重性：建议修。位置：[M545](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/MANUSCRIPT_EN.md#L545)。
- 证据：M31、350、362、501；S139、155、ST26及ST27。
- 影响：分段无内容删除；保留全部局限及文献，不改变结论。确信度：高。范围：全段阅读与对应源表，非域外泛化实验。

**原文**

> The construction and the error relations are not specific to the cells studied: they apply to any discrete model with a symmetric positive semidefinite stiffness and a positive definite interior block. The trained network is specific to them. It applies to the setting in which it was trained and tested:

**中文**：构造及误差关系不局限于所研究胞，适用于具有对称半正定刚度和正定内部块的任意离散模型。训练网络则专用于这些胞。它适用于训练和测试时的设置：

**建议英文**

> The algebraic construction applies to discrete models with a symmetric positive semidefinite stiffness and a positive definite interior block. The trained network has a narrower scope: the Schwarz-P cell family and discretisation specified below.

**建议中文**：代数构造适用于具有对称半正定刚度和正定内部块的离散模型。训练网络的范围更窄，仅包括下述Schwarz-P胞族及离散设置。

**为何更清楚**：去掉“them / It”的回指。余下内容保持原数值和限定，分为四段：（1）几何、材料、离散与方向；（2）没有上界证书及离散参考精度；（3）组装/优化证据、单随机种子和未孤立的损失项；（4）未比较求解器与超过24胞的精度范围。不是把每个限制都套成同一种句式，也不是削弱作者结果。

### L14　结论外推：区分可用于新胞族的构造与已验证的网络迁移

- 严重性：作者选择；若作者意指已实测迁移，则必须补证或修改。位置：[M553](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/MANUSCRIPT_EN.md#L553)末句。
- 证据：M31、545；A56–62。当前稿明确未测试其他TPMS族。
- 影响：触及定位/结论适用范围，须作者讨论；不建议删一般性构造主张。确信度：高（可发生歧义），中（作者预期含义）。范围：源内范围对照，没有域外实验。

**原文**

> The construction transfers to other component families with new training data; the structure and the error relations do not depend on the family.

**中文**：有新训练数据，构造即可迁移到其他组件族；其结构及误差关系不依赖组件族。

**建议英文**

> The construction can be applied to other component families under the same algebraic assumptions. The structural properties and error relations remain applicable, while a new family requires its own training data and accuracy validation.

**建议中文**：在相同代数假设下，该构造可用于其他组件族。结构性质和误差关系仍适用，而新的组件族需要相应训练数据和精度验证。

**为何更清楚**：保留一般方法主张，并指出已证明的结构迁移与待验证的数值精度是不同层面。不要把“can be applied”改成更强的“generalises”作为纯润色。

### L15　附录H.2：差分趋势与积分分支未变是两个判断

- 严重性：必须修或补直接证据；先交数学审查及作者。位置：[A795](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/APPENDICES_EN.md#L795)。
- 证据：A692已说明重新积分可能改变分支；S1185及S588–599；子审查抽读E/ref_valid.json、ref_valid_h1.json的差分记录，数字与所述趋势吻合，但没有据此核实所有分支标记。
- 影响：涉及“无分支变化”的证据强度，不改灵敏度数值或嵌套域理论；须作者决定补证还是限缩。确信度：高（逻辑间隙）；实际是否有开关未穷尽核查。

**原文**

> the hundredfold reduction per decade is the second-order truncation of the central difference, so no integration branch changes within \(\pm10^{-3}\tau_c\) at the fixed active set on these cells.

**中文**：每缩小一数量级步长，误差降低百倍，这是中心差分的二阶截断，因此这些胞在固定活跃集合、±10⁻³τc内没有积分分支改变。

**建议英文**

> The hundredfold reduction per decade is consistent with second-order central-difference truncation on these cells. This step study supports the numerical derivative at the tested designs; establishing that no integration branch changes requires a separate check of the integration choices.

**建议中文**：步长每缩小一数量级、误差约降低百倍，与这些胞上的中心差分二阶截断相符。该步长研究支持测试设计上的数值导数；要确认没有积分分支改变，还需直接核对积分选择。

**为何更清楚**：平滑的数值趋势可以支持导数一致性，但不能单凭这一趋势排除相互抵消或未影响所测量值的离散开关。若已有逐分支记录，英文应直接写明检查了什么，而不是让“so”承担这一步证明。整个A795宜按精确积分结论、步长核验、元素谱观察三个段落组织，保留负特征值披露。

### L16　图2：将有条件能量排序完整带入总览图

- 严重性：必须修。位置：[M180–182对应图2](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/MANUSCRIPT_EN.md#L182)；[生成源117行](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/figures_src/fig02_overview.py#L117)；[SVG885行](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/figures/F01_method_overview.svg#L885)。
- 证据：主审已查看F01_method_overview.png；M221、244；A350–360。
- 影响：不改定理，只修图文适用范围；不删图。确信度：高。范围：PNG、SVG、源脚本和正文条件，未独立重证全部数学。

**原文（图中文字转录）**

> \(S\preceq\widehat S_{\rm tg}\preceq\widehat S_0\) (correction cannot increase the error)

**中文**：S≼Ŝtg≼Ŝ0（校正不会增加误差）。

**建议英文（图内）**

> Energy-error ordering under the conditions of Sections 4.2–4.3

**建议中文**：满足第4.2–4.3节条件时的能量误差排序。

**建议英文（图注补句）**

> The ordering in (a) assumes an exact coarse-grid correction and a smoothing upper endpoint that bounds the positive spectrum. It concerns energy error; sensitivity error is not ordered by this result.

**建议中文**：(a)中的排序以精确粗网格校正及平滑区间上端点界定正谱为前提。该结果约束能量误差，不给灵敏度误差排序。

**为何更清楚**：读者可单独读取图2，目前图内结论没有条件，也没说误差类型。同图还展示灵敏度，泛称“the error”尤易误读。不能写成“the smoothing interval contains the spectrum”，低于区间下端点的模态并未被排除。

### L17　ST04：把列说明按诊断目的分组

- 严重性：建议修。位置：[S171](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/SUPPLEMENTARY_EN.md#L171)。
- 证据：ST04表头S173及M362。非对称性只用前八方向，不能改成全部方向证明。
- 影响：不改指标公式和数值，不删诊断。确信度：高（组织）；聚合字段沿用原文，未重读全部生成实现。

**原文**

> Columns: \(\lambda_{\max}\) of \(D^{-1}K_{II}\) by Lanczos; a power-iteration estimate \(b\) of the upper smoothing endpoint; the Gershgorin bound; the maximum relative asymmetry \(\max_{i,j}|G_{ij}-G_{ji}|/\sqrt{|G_{ii}G_{jj}|}\) of \(G=Q^T\widehat SQ\), where the columns of \(Q\) are the first eight retained-displacement directions of the cell's consistent-traction (force_c) validation set;

**中文**：各列为兰索斯最大特征值、幂迭代平滑上端点、盖尔圆界，以及由该胞一致牵引验证集前八个保留位移方向构成的G矩阵的最大相对不对称性。

**建议英文**

> The table separates spectral checks, operator checks and energy diagnostics. Spectral checks compare the computed \(\lambda_{\max}(D^{-1}K_{II})\), the smoothing endpoint \(b\), and the Gershgorin bound. For the asymmetry check, \(G=Q^T\widehat SQ\), where \(Q\) contains the first eight retained-displacement directions of the cell's consistent-traction validation set. The reported metric is \(\max_{i,j}|G_{ij}-G_{ji}|/\sqrt{|G_{ii}G_{jj}|}\).

**建议中文**：本表分为谱检查、算子检查和能量诊断。谱检查比较计算得到的最大特征值、平滑端点b及盖尔圆界。不对称性检查使用一致牵引验证集前八个保留位移方向构成Q，并按所给公式报告G的最大相对不对称性。

**为何更清楚**：该表同时回答不同问题，按目的分组比一长串分号容易检索。其余返回功/场能量、部署/训练场、刚体能量、鬼罚能量份额、δ和κ说明按原定义接在相应组中，保留各自平均或最大值口径，不制造新统计量。

### L18　ST09b：明确“之前”是哪一级评价之前

- 严重性：建议修。位置：[S457](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/SUPPLEMENTARY_EN.md#L457)，同步M420。
- 证据：S155的五个最大单胞误差几何；ST09b的Selection（选择方式）列。
- 影响：不改变留出集合或结果；澄清选择时序，不能未经记录宣称事先登记。确信度：高（歧义），实际选择时戳未核验。

**原文**

> Cells outside model selection, fixed before evaluation: the five validation geometries with NICE's largest single-cell errors and one random cell per stratum.

**中文**：模型选择外、在评价前固定的胞：NICE单胞误差最大的五个验证几何，加上每个分层随机一个胞。

**建议英文**

> The two-cell tests use cells outside model selection: the five validation geometries with NICE's largest single-cell errors and one random cell from each stratum.

**建议中文**：两胞测试使用未参与模型选择的胞：NICE单胞误差最大的五个验证几何，以及每个分层随机一个胞。

**为何更清楚**：这组胞显然已经利用单胞误差选择，不能让“before evaluation”暗示从未看过任何评价。若已有时间记录，另加“selected before the two-cell evaluations”；否则采用上面的无时序版本即可。极端胞加随机胞是一项有意义的压力测试，明确设计比强调“independent”更有说服力。

### L19　ST10及统一术语：3%线已用于正文的切面载荷比较

- 严重性：必须修。位置：[S474](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/SUPPLEMENTARY_EN.md#L474)；关联S94、390、392–451、522及A48。
- 证据：M422明确写切面载荷下超过3%线的四个Uncorrected配置；ST10本身用粗体标值。
- 影响：不改阈值、数字或总体结论，不删结果。确信度：高。范围：已逐字核正文和全部ST09、ST10行。

**原文**

> The 3% reference is not applied to these loads in the main text; values above it are marked in bold.

**中文**：正文不对这些载荷应用3%参考；超过它的数值加粗。

**建议英文**

> Values above the common 3% line are marked in bold, consistent with the cut-surface-traction comparison in Section 5.6.

**建议中文**：超过共同3%线的数值加粗，与第5.6节切面牵引的比较一致。

**为何更清楚**：修当前源内矛盾，同时将3%比较线与reference（精确离散参考解）区分。A48与S94、390、522统一用“3% line”。ST09的Outcome（比较结果）列可改成“Relative to 3% line”，值为“Below line / Above line”；不把参考线变成验收证书或改变任何比较判定。

### L20　ST15：谱比值为−1不足以说明整个导数矩阵负半定

- 严重性：必须修。位置：[S588](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/SUPPLEMENTARY_EN.md#L588)；A795的“some of them negative semidefinite”需联动核验。
- 证据：ST15定义为\(\lambda_{\min}/\max|\lambda|\)；对角矩阵diag(−2,1)的该比值也是−1，却含正特征值。该反例只检查逻辑，不是论文实际矩阵。
- 影响：局部数学解释必须修；未判定实际矩阵是否负半定，不据此推翻整体结论或删除数据。确信度：高（逻辑）；实际全谱待核验。

**原文**

> A ratio of −1 means that the element derivative is negative semidefinite.

**中文**：比值为−1意味着元素导数矩阵负半定。

**建议英文**

> A ratio of −1 means that the most negative eigenvalue has the largest absolute magnitude. The ratio alone does not establish negative semidefiniteness.

**建议中文**：比值为−1说明最负特征值的绝对值达到全谱最大值。仅凭这一比值不能判定负半定。

**为何更清楚**：让每个名词判断对应实际统计信息。若作者要保留“negative semidefinite”，需核实这些实际矩阵的最大特征值和容差。还应检查A795及S588末句“therefore carry derivatives of at most this relative size”的归一化对象；本报告不借此自行改数值或修矩阵。

### L21　S5精度比较：用观察到的微小差异代替“不改变精度”

- 严重性：建议修。位置：[S706](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/SUPPLEMENTARY_EN.md#L706)。
- 证据：S704–706；S806还给出两个八胞格架的对应精度检查，不能误说只有一个复算。
- 影响：限定保证措辞到已评估工况；不改数字或精度设置，作者确认后采用。确信度：高（语气范围）；成对差值未独立全量复算。

**原文**

> The single-precision correction itself does not change the operator accuracy: repeating the \(2\times2\times2\) solve of Section 5.8 with the correction in single precision (to \(10^{-10}\)) changes its compliance errors by less than \(3\times10^{-9}\).

**中文**：单精度校正本身不改变算子精度：把2×2×2算例的校正改为单精度、求解到10⁻¹⁰，柔度误差变化小于3×10⁻⁹。

**建议英文**

> The evaluated lattice re-solves show a much smaller effect of correction precision. Repeating the \(2\times2\times2\) solve of Section 5.8 with the correction in single precision, to a recursive residual of \(10^{-10}\), changes its compliance errors by less than \(3\times10^{-9}\). Supplementary Note S6.3 reports the corresponding checks for both eight-cell lattices.

**建议中文**：已评估格架复算中，校正精度的影响要小得多。在第5.8节2×2×2算例中，把校正改为单精度并求解到10⁻¹⁰递归残差，柔度误差变化小于3×10⁻⁹。S6.3报告了两个八胞格架的对应检查。

**为何更清楚**：说明这里的实际论证：计时路径的误差增量与求解停止误差有关，单/双精度校正差异在已测工况很小。不要用“does not change”把微小但非零的已测差别变成任意工况的保证。

### L22　S6.3非光滑性：保留机制判断，标明来自哪些扫描

- 严重性：作者选择；若意指排除了所有网络开关贡献，则须补证。位置：[S823](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/SUPPLEMENTARY_EN.md#L823)末句。
- 证据：同段M1/x及U1/y三个扫描；S808、823承认网络二值指示器和粗因子移位还有自身开关。子审查抽读当前基线review_r1/results/X3/E13/E13_SUMMARY.json中的m1x/u1y/u1ym.n、n_switch及ref.jumps；这些是记录字段，不沿用历史评论的判断。
- 影响：收窄解释的普遍性，须作者讨论；不改实测跳变量或删扫描。确信度：高（范围），中高（“主要来源”的归因）；未重新扫描全部参数域。

**原文**

> The non-smoothness of the surrogate objective is therefore that of the discrete model, which an optimiser driven by exact condensation meets in the same way.

**中文**：因此，代理目标的非光滑性就是离散模型的非光滑性，精确凝聚驱动的优化器也会以同样方式遇到。

**建议英文**

> In these sweeps, the surrogate and exact discrete compliances depart from the sensitivity-predicted changes by similar amounts. This points to switches in the discrete model as the main source of the observed non-smoothness; the corresponding exact-condensation evaluations show the same behaviour.

**建议中文**：在这些扫描中，代理柔度和精确离散柔度相对于灵敏度预测变化的偏离幅度相近。这表明离散模型开关是所观察非光滑性的主要来源；对应的精确凝聚评价也表现出同样行为。

**为何更清楚**：先报告直接对照，再给解释。有限扫描支持“这些观察主要源于离散模型”，不能抹去正文自己承认的其他开关。原段很长，可依次分为扫描设计、实际开关和差异、机制判断三段。

### L23　S9.2最终柔度差：补“相对”并避免暗示优化优越性

- 严重性：必须修（量纲）；作者选择（是否强调差值方向）。位置：[S968](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/SUPPLEMENTARY_EN.md#L968)。
- 证据：ST22c S1023、1031；M491。原始抽查：E/opt/optA/optA/check_023.json的C_exact=24.110504299133062，E/opt/optA/optAx/history.jsonl最终C=24.110806009123603；相对差约−1.25135×10⁻⁵，绝对差约−0.00030171。
- 影响：只补量纲，不改结果或定位，不删除数值。确信度：高。范围：已做存档数值的直接四则核算，未做新精确求解。

**原文**

> Their corner parameters differ by at most 0.0051 (root mean square 0.0011), and the exact compliance of the NICE design, 24.110504, is \(1.25\times10^{-5}\) below that of the twin's design, 24.110806.

**中文**：其角点参数最大差0.0051、均方根差0.0011；NICE设计的精确柔度24.110504比孪生设计的24.110806低1.25×10⁻⁵。

**建议英文**

> Their corner parameters differ by at most 0.0051 (root mean square 0.0011). The exact compliances of the final designs are 24.110504 for NICE and 24.110806 for the twin, a relative difference of \(1.25\times10^{-5}\).

**建议中文**：两者角点参数最大差0.0051，均方根差0.0011。NICE和孪生运行最终设计的精确柔度分别为24.110504和24.110806，相对差为1.25×10⁻⁵。

**为何更清楚**：“below”加无单位数字容易被读成绝对差。该处只应支持两设计接近，不借这一微小差别主张NICE获得更优最优解。紧接的2.0%改善及25%初始材料差保留。

### L24　附录D：数值谱核验不能靠连接词升级成严格包含证明

- 严重性：必须修证据级措辞或补认证依据；交数学审查与作者确认。位置：[A340](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/APPENDICES_EN.md#L340)，同步M362。
- 证据：E/lam_check_val80.json，80条记录的op_lmax、lanczos_lmax、ritz_residual、lanczos_steps、margin和gershgorin_over_op。主审复核margin最小0.02070217、中位0.03924660、最大0.04997696，与2.1–5.0%及3.9%舍入一致；子审查亦核对了相对Ritz残差。
- 影响：不否定实际谱条件成立；把数值支持与认证保证分开，可能影响保证的证据口径，必须由作者讨论。确信度：高（文字跳步）；未穷尽仓库有无另一个严格认证证据。

**原文**

> The operational endpoint exceeds the converged value by 2.1–5.0% (median 3.9%) on every geometry, so the containment required by Eq. (D.2) holds for all evaluated cells.

**中文**：实际端点在每个几何上超过收敛值2.1–5.0%（中位数3.9%），所以所有已评价胞都满足式(D.2)要求的谱包含。

**建议英文**

> On all 80 validation geometries, the operational endpoint exceeds the largest Ritz value computed by the stated Lanczos procedure by 2.1–5.0% (median 3.9%). These checks provide numerical evidence for the spectral condition on the evaluated validation cells.

**建议中文**：在全部80个验证几何上，实际端点比所述兰索斯程序计算的最大Ritz值高2.1–5.0%，中位数3.9%。这些核验为已评估验证胞上的谱条件提供数值证据。

**为何更清楚**：最大Ritz值及其残差不是自动得到的最大特征值认证上界；小残差可以表示已找到某个特征对，不独自证明没有漏掉更大的特征值。保留前后有关40步估计、实际残差和保守盖尔圆界的内容。若另有严格最大特征值包络，直接引用该证据即可保留较强句子。此条不建议更改谱条件、加移位或调整计算参数。

## 5. 应明确保留的内容

| 当前位置 | 应保留的短英文判断 | 为何重要 |
|---|---|---|
| M46 | “band parameters, not pointwise wall thicknesses” | 直接阻止把隐式带参数误当实际局部壁厚，属于必要定义 |
| M103–111 | “the absence of a first-order term follows from equilibrium of the reference field” | 有因果机制，作者判断具体 |
| M125、134 | “Compliance underestimation is therefore the total reconstructed error energy”及小energy share的后果 | 理论解释与后续反例衔接；不应为去重删掉 |
| M170 | “they estimate the exact sensitivity and are not the derivative of the surrogate compliance” | 全文最关键的评价对象区分，必须保留 |
| M244 | “The ordering is guaranteed; the size of the reduction is not” | 在已写明的条件下，保证与实测改善幅度分开，语气恰当 |
| M370 | “most of the improvement comes from the correction itself” | 有固定参数对照支持，体现作者解释，优于再宣告总体优越 |
| M400 | 同一校正预算下比较零场、调和场和学习场 | 比抽象的“complementary”更能说明学习贡献 |
| M448、S619 | “not reproduced here”及消融范围说明 | 公允限定与竞争方法的关系；不是可删的防御性句子 |
| M497、S1041 | 均匀化响应误差27–37%，但其细尺度设计效果仍在约1%范围 | 避免把响应预测失准夸成设计完全失败，作者判断应保留 |
| A97 | “Positive semidefiniteness alone does not imply rigid reproduction” | 半正定、刚体复现、没有额外零模态是不同条件 |
| A154–165 | κ不是矩阵条件数；缩放低模态不是自由胞的弯曲响应 | 具体区分物理对象，禁止靠换词模糊 |
| A350 | 界几乎只保证不扩张，实测才给出更大降低 | 证据尺度清楚，不应润色成统一快速收敛保证 |
| A434、S615 | 富集列独立性及粗求解精度未认证 | 把数值观察与已验证投影分开，须保留 |
| A628–630 | 方向未覆盖可能隐藏大误差的反例 | 完整DOF目标不等于全方向精度证明 |
| A751、S520 | 理论界不是百分比拟合归因；替换误差范数不可相加 | 已经避免了常见的机制过度解释 |
| A826 | “A large value detects an inaccurate direction; a small value alone does not bound the error from above.” | 简洁、具体，应原样保留 |
| S1005、M521 | 相同迭代号不一定是相同设计 | 避免把优化路径差误当代理同设计误差 |
| S1130 | 只按计数识别的开关数是下界 | 直接说明观测方法边界，值得保留 |
| S1154、ST26 | 规模展示只完成早期分析；135胞在首解前失败 | 保留实测和外推区别，不写成110胞完整优化已验证 |

这些段落并不具有统一的“最好句长”，也无需全部改成短句。公式后的必要条件、跨章节重复定义和独立图注中的解释有功能，应按读者需求保留。

## 6. 统一语言原则

1. **让判断有明确对象。** 写出“哪一类载荷、哪一组几何、哪一种误差、哪个对照”，再给解释。特别避免把柔度排序推广到灵敏度，或把单胞测试推广成所有格架。
2. **让动词说明作者做了什么。** “We retain / compare / check / choose”可直接体现作者决策。没有必要把每段都写成“the construction provides / supplies / enables”的抽象名词链，也无需到处补“we”。
3. **拆开不同证据层级。** 数学恒等式、在假设下的保证、有限样本数值核验、作者机制解释和规模外推分别说清楚。优先处理“so / therefore / does not / every / transfers / local optima”所连接的推论，不做禁词式替换。
4. **避免机械复制贡献清单。** 摘要负责结果概貌，引言说明为何作出选择，方法给定义与步骤，结果用对照解释，讨论判断代价与局限，结论保留最重要判断。口号按作者偏好保留，尤其不要以语言审查之名改变主线。
5. **保留统计单位。** geometry mean（几何内方向均值）、population mean（几何等权总体均值）、directional maximum（采样方向最大值）及all-direction supremum（全方向上确界）不可互换。bootstrap（自助重采样）的单位若是几何，就直接说几何。
6. **保留固定术语，不追求词汇多样性。** retained DOFs（保留自由度）、interior（内部）、cut band（切割带）、active element（活跃背景单元）、weak support（弱支撑）、energy share（能量份额）、two-grid correction（两网格校正）、background element（背景单元）、3% line（3%参考线）。cell专指TPMS胞，element指背景有限元单元。
7. **变体名只用既定五种。** NICE；Base network（基础网络）；Uncorrected（未校正续训）；Smoothing-trained（经平滑训练）；'Base network, corrected'（部署时校正的基础网络）。不要为行文变化创造新变体名；句内“base network”大小写可服从普通句法，表/图标签保持统一。
8. **比较对象及硬件只写必要信息。** 主文使用AMD EPYC 9654和RTX 5090；成本比较保留求解载荷数、算术路径和是否包含准备等直接影响可比性的定义，不加入负载、限流、共享主机等运行叙事。
9. **不要为润色修数学。** 不能通过加jitter（对角微扰）、伪逆、特征值裁剪或放宽容差来消除措辞问题。发现保证缺条件，应补条件或补证据。
10. **AI使用声明由作者提供事实。** 本报告不猜工具、用途或使用程度，不代填真实性声明。声明文本需依据作者实际提供的信息另行完成；本轮没有把这一未知项伪装成已完成。

## 7. 修改先后顺序与待讨论事项

| 顺序 | 建议动作 | 条目 | 作者须决定什么 |
|---|---|---|---|
| 1 | 修源内矛盾、量纲和图文条件 | L11、L16、L19、L20、L23 | 用既有结果准确表述；L20先确认实际全谱支持什么 |
| 2 | 对齐推论与证据范围 | L12、L15、L21、L22、L24 | 保留强判断并补直接证据，还是采用当前证据支持的较窄表述 |
| 3 | 让统计及比较对象显式 | L06–L10、L17–L18 | 不改数字；确认选择时序是否已有记录 |
| 4 | 调整段落功能与读者负担 | L01–L04、L13 | 是否移段；引言文献均保留，先分段再考虑压缩 |
| 5 | 最后讨论口号出现次数与结论外推 | L05、L14 | 口号保留；只由作者决定是否压缩额外重复及如何表达一般性 |

执行改稿前，建议先选定第1–2组。L12、L15、L20、L22、L24应与数学/证据审查交叉合并，避免不同审查各自给出不同强度的句子。没有必要先做全篇同义词替换；那样会使固定术语变动，却不能解决目前最影响可读性和可信度的连接问题。

## 8. 审查产物与交叉核对

本主报告为唯一整合产物。辅助报告用于保留审查范围和候选，不代表全部候选都需要采用：

- [补充材料语言辅助审查](language_author_voice/supplement_language.md)：主报告主要采用其ST04、ST09b、ST10、ST15、S5、S6.3及S9.2候选。
- [附录语言辅助审查](language_author_voice/appendix_language.md)：主报告采用H.2、D的证据级语言问题；已交叉纠正其初稿中沿用的未确认负半定措辞。
- [19图文字与重复辅助审查](language_author_voice/figure_language.md)：全图覆盖、图2条件问题及口号重复位置。图12比较对象、图S06立方对称性、图7误差类型的局部澄清可留作下一轮，未挤入本轮24项优先清单。

主报告复核了图2的实际图中文字、最终柔度差的原始字段、交叉起点的停止预算、80条谱记录的端点裕量，并保留了未核验范围。除报告文件外没有产生论文修改。
