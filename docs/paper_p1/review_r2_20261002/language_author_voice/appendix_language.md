# 附录 A–I 英文与作者论证声音独立审查

本报告只读审查 `/Users/fffffreeze/Desktop/curly-octo-goggles`，固定基线 `700af856f2b9f152ab2c263406c54132a11255bf`，分支 `claude/wizardly-euler-3m9cwx`。开始时 `git status --short` 无输出；未构建、未运行有限元或训练、未修改仓库。审查日期：2026-10-02。

已逐行读完 `docs/paper_p1/APPENDICES_EN.md` 的 1–826 行；先独立阅读附录并对照正文定义，再读 `review_r1/HANDOVER_CODEX_20261002_CN.md`。交接文件只用于识别作者偏好；以下判断以当前源文与明确列出的证据为准。候选编号 `APP-L01`–`APP-L06` 为辅助报告稳定编号，主审可映射至全篇的 L 系列编号。

## 判断与覆盖边界

附录最值得保留的是明确区分数学对象和评价对象的段落，例如 B.3 区分位移范数和能量误差、C.1 区分总重构误差与 retained solution（保留自由度解）的误差、H 区分 field-based sensitivity（基于重构场的灵敏度估计）与 complete design derivative（完整设计导数）。这里不宜为“去 AI”而消除必要的限定或重复公式定义，也不凭词汇判断写作来源。

改写优先级最高的是：将 D 的数值谱核验与数学保证分开；将 H.2 的差分步长证据与“没有积分分支改变”的判断分开；将 C.2 的误差归因限定在实际核验的柔度工况。其余主要是把定义、实施细节、实测记录和理论解释分段。未发现需要因英语问题推翻论文定位或删除完整附录的理由。

| 部分 | 当前行号 | 覆盖 | 可保留 / 修改重点 |
|---|---:|---|---|
| A | 1–48 | 全读 | A.1 的 retained DOFs（保留自由度）构造可保留；L5 拆分认证、离散与保留集定义；L48 统一 3% line（3% 线）。 |
| B | 50–165 | 全读 | B.1 假设清楚；L97 不把刚体复现等同于无额外零模态；L154–165 对两类“软”的区分应保留。 |
| C | 167–294 | 全读 | 误差正交分解与 load-specific（针对给定载荷）的 energy share（能量份额）解释清楚；L294 限定归因范围。 |
| D | 296–376 | 全读 | 多项式的谱条件、固定循环与改变次数的区别清楚；L340 数值核验与严格保证的衔接需改。 |
| E | 378–424 | 全读 | 完整转置与 retained force（保留自由度上的力）贡献说明清楚；无需为压缩而删掉 L406。 |
| F | 426–483 | 全读 | 三种算术路径表格值得保留；L442 三层信息应分段；L483 一处主谓一致。 |
| G | 485–632 | 全读 | 层级操作表、参数维度、方向覆盖反例均可保留；L610 对照条件、求导路径、模型选择应分开。 |
| H | 634–806 | 全读 | 灵敏度对象与导数差异需要完整保留；L795 过长，并有“收敛趋势推出分支不变”的跳步。 |
| I | 808–826 | 全读 | 下界大可以检出问题、下界小不能认证精度的收束句准确清楚。 |

对照正文的范围是 §§2–3 全部、§4 中与延拓和校正相关的定义、§5.2；补充材料只核查 ST15、S2、S6.3/ST19 及相关表的定位。未声称本辅助报告覆盖正文 1–7、全部补充材料或全部图。没有看图像像素、没有读外部文献全文、没有完整重做数学推导。数值追溯仅包括：`lam_check_val80.json` 的 80 条记录的列汇总；`ref_valid.json` 与 `ref_valid_h1.json` 中四胞差分记录；`time_setup.json` 的七项 `shift`。其余数字仅按当前稿保留，不能据此称为独立数值验证。

## APP-L01：A.1 先定义 active element，再说明离散与 retained set

- **位置**：`docs/paper_p1/APPENDICES_EN.md:5`。
- **严重性**：建议修。
- **短英文原文**： “This is decided with closed-form interval enclosures of \(\phi\) and \(\tau\) over the element clipped by the cut plane: the element is excluded if an enclosure shows that a constraint is violated everywhere, accepted if both constraints hold strictly at its centre or a vertex, and subdivided otherwise.”
- **中文**：通过切平面裁剪后单元上 \(\phi\) 与 \(\tau\) 的闭式区间包络作出判断：若包络显示某约束处处不满足则排除，若中心或顶点严格满足两个约束则接受，否则细分。

**问题与方向**：段落同时完成 active element（活跃背景单元）认证、Q2 离散、节点存储、两个保留集定义及虚功解释。尤其 “accepted” 的宾语隐含在长句中，读者需自己判断是接受为活跃单元，还是接受为完全充满材料的单元。建议明确 “certified active”，分三段，避免改动任何集合定义。

**建议英文（替换 L5；原数字和构造保持）**：

> A background element is active if its intersection with \(\Omega(\eta)\) has positive measure. We certify this property using closed-form interval enclosures of \(\phi\) and \(\tau\) over the element clipped by the cut plane. An element is excluded if an enclosure shows that a constraint is violated everywhere. It is certified active if both constraints hold strictly at its centre or a vertex; otherwise, it is subdivided.
>
> Active background elements use tensor-product \(Q_2\) displacements, with 27 nodes and 81 displacement DOFs per element. Active DOFs are stored in a fixed node-major \(x,y,z\) order.
>
> The retained box set contains the nine face nodes of every certified positive-area material patch. The cut set contains all 27 nodes of each active element carrying a positive-area cut-surface patch. We take their union and deduplicate it in active-node order. The off-plane cut-band DOFs remain retained because their basis functions contribute to displacement and virtual work on the cut plane.

**建议中文**：背景单元与材料域的交集具有正测度时，该单元为活跃单元。用裁剪后单元上的闭式区间包络认证这一性质：处处违反约束时排除；中心或一个顶点严格满足两个约束时认证为活跃；否则细分。随后单列 Q2 自由度和节点优先的存储顺序。最后分别定义保留盒面集和切割集，按活跃节点顺序对并集去重，并以虚功解释保留离开切平面的 cut-band DOFs（切割带自由度）。

**为何更清楚**：每段只回答一个问题；作者使用 “We certify / We take” 说明确定性构造；不把决定性的虚功理由淹没在实现列表中。

- **证据**：正文 `MANUSCRIPT_EN.md:48,56–58`；附录 L5、L19 中 “certified as completely filled” 是另一种性质，故这里明确 active 尤其有用。
- **影响**：定位否；结论否；删除否。只是澄清 “accepted” 的对象。
- **确信度**：高。
- **核验范围**：全文定义相互对照，未复核区间包络实现。

## APP-L02：C.2 将误差归因限定为核验过的柔度工况

- **位置**：`docs/paper_p1/APPENDICES_EN.md:292–294`，核心为 L294 末句。
- **严重性**：建议修；若全篇把它用作所有误差的普遍归因，则必须修该范围表述。
- **短英文原文**：“at the final iterates, \(|\bar U^T\rho|\) and \(|\omega|\) stay below \(6\times10^{-8}\) of the compliance and the bound below \(2\times10^{-5}\), so the reported errors are those of the operator.”
- **中文**：最终迭代时，两项都低于柔度的 \(6\times10^{-8}\)，该界低于 \(2\times10^{-5}\)，所以所报告的误差就是算子的误差。

**问题与方向**：此前的恒等式只解释柔度误差，并且这些记录来自指定的重新求解工况。“the reported errors” 容易把结论扩大到全文的位移、灵敏度、所有组装和所有算术路径。作者真正有力的判断是：在这些核验中，求解残差及作用量与能量不一致项都远小于被比较的柔度误差。应直接说出这个判断。

**建议英文（替换 L294 的测量与结论部分）**：

> Supplementary Note S6.3 records the signed residual work, the action–energy inconsistency and the dual-norm bound for the checked pair configurations and lattices. At the final iterates, \(|\bar U^T\rho|/C\) and \(|\omega|/C\) are below \(6\times10^{-8}\), and the normalised bound is below \(2\times10^{-5}\). These terms are small relative to the compliance errors in those checks, supporting their attribution primarily to the approximate condensed operator.

**建议中文**：S6.3 对所检查的两胞和格架工况记录了有符号残差功、作用量与能量不一致项以及对偶范数界。最终迭代时，两个直接项相对柔度均低于 \(6\times10^{-8}\)，归一化后的界低于 \(2\times10^{-5}\)。这些项相较于这些核验中的柔度误差很小，因此支持把该柔度误差主要归因于近似凝聚算子。

**为何更清楚**：结论保留力度，同时明确对象、分母与核验范围。公式和具体上界保持；将一句笼统的 “errors are” 改成有证据对象的作者判断。

- **证据**：正文 §3.2 `MANUSCRIPT_EN.md:136–143`；`SUPPLEMENTARY_EN.md:804–823`，尤其 ST19 的 `max residual work / C`、`max omega / C` 及 L821 的两胞汇总。
- **影响**：定位否；结论轻微限定范围，不改变所测数值；删除否。需作者同意将笼统误差对象明确为柔度。
- **确信度**：高（范围问题），中高（“主要归因”强度；本报告只表内抽查）。
- **核验范围**：核对数学对象与补充表列义，未重做残差计算或穷尽全部组装工况。

## APP-L03：D 中应把经验谱核验和严格包络的语气分开

- **位置**：`docs/paper_p1/APPENDICES_EN.md:340`；正文同步点 `MANUSCRIPT_EN.md:362`。
- **严重性**：必须修（证据级措辞）；先交由数学主审确定是否存在额外的认证证据，再选择补证或限定措辞。
- **短英文原文**：“The operational endpoint exceeds the converged value by 2.1–5.0% (median 3.9%) on every geometry, so the containment required by Eq. (D.2) holds for all evaluated cells.”
- **中文**：实际使用的上端点在每个几何上都超过收敛值 2.1–5.0%（中位数 3.9%），所以所有已评价胞都满足式 (D.2) 所要求的谱包含条件。

**问题与方向**：这一段前半明确说有限次幂迭代加安全因子不是严格包含保证，后半却将带残差的最大 Ritz 值称为 “converged value”，再用 “so … holds” 推到严格条件成立。这里需区分数值上有力支持和数学认证；这不是说记录的谱值错误。Lanczos（兰索斯迭代）结果的残差收敛不能由一句话自动升级为最大特征值的认证上界。如果已有独立认证，应补确切证据；没有时可用下述改法。

**建议英文（替换 L340 的 Lanczos 核验段，前后实施细节保留）**：

> We checked the operational endpoint on all 80 validation geometries against the largest Ritz value from fully reorthogonalised Lanczos iteration on \(D^{-1/2}AD^{-1/2}\). The runs used 78–150 steps; the relative Ritz residual was at most \(9.6\times10^{-4}\), with median \(2.2\times10^{-6}\). The endpoint exceeded the computed value by 2.1–5.0% (median 3.9%). These checks provide numerical evidence for the spectral condition on the evaluated validation cells. The strict Gershgorin bound \(\max_i\sum_j|A_{ij}|/A_{ii}\) is available but is 4.7–19.4 times the operational endpoint (median 18.4), which would stretch the target interval by that factor and slow contraction of the upper spectrum.

**建议中文**：在全部 80 个验证几何上，用完全重正交兰索斯迭代得到的最大 Ritz 值检查实际端点；给出迭代次数、相对残差和端点裕量。将结论写成“这些核验为已评价验证胞上的谱条件提供数值证据”。随后保留严格 Gershgorin（盖尔圆）上界的数值和其代价解释。

**为何更清楚**：论证顺序变为“怎么检查—得到什么—这支持到哪一级结论—为何没有直接使用严格但保守的界”。作者的经验判断仍明确，同时不会混淆 proof（证明）与 verification（数值核验）。不改变条件 \(b\ge\lambda_{\max}\)，不改成错误的“谱都在 \([b/30,b]\)”。

- **证据**：`APPENDICES_EN.md:338–340`；`evidence/lam_check_val80.json` 的 `op_lmax`、`lanczos_lmax`、`ritz_residual`、`lanczos_steps`、`margin`、`gershgorin_over_op`。本次汇总全部 80 条：`margin` 最小 0.0207021701、中位 0.0392466033、最大 0.0499769572；`ritz_residual/lanczos_lmax` 最大 0.0009640684、中位 0.0000022263，与原文舍入吻合。
- **影响**：定位否；结论证据强度可能需调整，必须交作者和数学主审决定；删除否。
- **确信度**：高（文字把两层证据连成了严格推论）；中高（尚未穷尽仓库中是否另有认证证明）。
- **核验范围**：核查全部 80 条 JSON 的上述字段与统计；未执行特征值求解，未查完整实现，未否定实验中实际谱条件成立。

## APP-L04：F.2 分开“算法做什么”“存档观察到什么”“公式在什么条件下成立”

- **位置**：`docs/paper_p1/APPENDICES_EN.md:442`，与 L450 连读。
- **严重性**：建议修。
- **短英文原文**：“The selected value is stored with the geometry factorisation (the coarse-factor shift); in the recorded setups it was 0 on six cells and \(10^{-12}\) on H1 (data archive record of the setup timings).”
- **中文**：选定的值随几何分解一起存储，称为粗层分解偏移；记录中六胞为 0，H1 为 \(10^{-12}\)，见设置耗时的数据档案。

**问题与方向**：L442 将缩放、分解的取三角约定、逐级试探、七胞记录、三胞矩阵一致性和六个新符号定义挤在一段内。括号式 “data archive record of …” 既打断阅读，又不如具体证据名称可追踪。三段即可保留全部事实，让数值观察不会被误读成代数恒等式的证明。

**建议英文（替换 L442；接原 Eq. F.1）**：

> The recovered matrix is Jacobi scaled, and the columns of \(V\) are scaled consistently. Cholesky factorisation reads the lower triangle and tries the diagonal shifts \(0,10^{-12},10^{-10},10^{-8},10^{-6},10^{-4}\) in that order. The selected coarse-factor shift is stored with the geometry factorisation.
>
> In the seven recorded setups, the shift was zero on six cells and \(10^{-12}\) on H1. On three lattice cells, the probed factor agreed with the element-assembled Galerkin factor to a relative \(1.3\times10^{-11}\) or better. These records document the numerical implementation; the following identity states the condition under which such a solve decreases the energy.
>
> Let \(V_s\) be the scaled basis and \(A_s=V_s^TAV_s\succeq0\). For \(\xi\ge0\) such that \(A_s+\xi I\succ0\), set \(G_s=(A_s+\xi I)^{-1}\) and \(b_s=V_s^Tr_I\). Then …

**建议中文**：先说明 Jacobi（雅可比）缩放、基的相同缩放、读取下三角及原有偏移试探顺序。另起一段报告七胞和三胞中的观测，并明确下式是在指定矩阵相等条件下成立的代数性质。第三段再逐一定义缩放基、Galerkin（伽辽金）矩阵及逆，并接原式 (F.1)。

**为何更清楚**：读者可以分别评估实现约定、样本核验和理论条件。保留已有偏移事实的透明披露，不提出添加偏移、伪逆或特征值裁剪来修补数学。

- **证据**：`APPENDICES_EN.md:440–450`；`evidence/time_setup.json` 七个记录的 `shift` 已全部核对，六个为 0，H1 对应记录为 `1e-12`。三胞 \(1.3\times10^{-11}\) 数据本次未找到并读取原始记录，标为**待主审追溯**；建议英文仅忠实保留当前稿数值，不代表本报告认证。
- **影响**：定位否；结论否；删除否。若更换存档括号的具体引用，须使用实际可追溯的记录，不可杜撰引用名。
- **确信度**：高（组织与七胞数值）；三胞探针记录未核验。
- **核验范围**：源文及七项 `shift`；未复核粗层矩阵及 Eq. (F.1) 的数值实现。

## APP-L05：G.3 先讲公平对照，再讲求导路径，最后交代模型选择

- **位置**：`docs/paper_p1/APPENDICES_EN.md:610`。
- **严重性**：建议修。
- **短英文原文**：“NICE and the Smoothing-trained variant continue the base network from its selected parameters for 15,000 training steps with the same schedule, class weights, training set and seed as the Uncorrected continuation, so that the three continuations differ only in the correction used in the forward map of Eq. (18) …”
- **中文**：NICE 与 Smoothing-trained 从选定的基础网络参数继续训练 15,000 步，与 Uncorrected 延续训练使用相同日程、类别权重、训练集与种子，所以三种延续训练仅在式 (18) 前向映射所用校正上不同。

**问题与方向**：这段承担了最重要的对照实验解释，却马上转入每个几何的校正构建、反向传播精度、显式伴随，末尾又跳到检查点选择。建议保留“只有校正不同”这个作者判断，把对照关系先完整说完。

**建议英文（替换 L610）**：

> The three continuations start from the selected Base network parameters and train for 15,000 steps with the same schedule, class weights, training set and seed. They differ in the correction included in the forward map of Eq. (18): Uncorrected uses none; Smoothing-trained uses eight Chebyshev steps; and NICE uses eight smoothing steps, a \(Q_1(17)\) coarse-grid correction and eight further smoothing steps.
>
> For each geometry held on the GPU, we recompute the smoothing interval by power iteration and the coarse matrix by coloured probing (Appendix F.2). The resulting correction is held fixed as a linear map of the network output. Automatic differentiation propagates the loss gradient through the smoothing and coarse-grid correction in double precision. At evaluation, the explicit transposes in Appendix E follow the same operations.
>
> Model selection compared the parameters after 7,500 and 15,000 steps for each continuation and after 10,000, 20,000, 30,000 and 40,000 steps for the Base network (Section 5.1; Table ST01). The selected parameters are those at 15,000 steps for the continuations and at 30,000 steps for the Base network.

**建议中文**：第一段并列说明三种延续训练的共同起点、相同设置和三种校正；第二段说明如何在每个几何上重建校正、哪些内容作为网络输出的固定线性映射、梯度怎样穿过校正；第三段统一说明检查过哪些训练步以及最终选中哪组参数。

**为何更清楚**：每个读者问题均有对应段落；数据和比较关系保持；“we recompute” 给作者一个明确操作，而不是连续的被动结构。变体名称固定，不引入新同义名。

- **证据**：当前 L606–610；`SUPPLEMENTARY_EN.md:44–72` 的 ST01 及模型选择说明可供主审交叉核对。本辅助报告未独立读取所有训练日志。
- **影响**：定位否；结论否；删除否。
- **确信度**：高（语言组织）；日志层面未核验对照条件完全相同。
- **核验范围**：当前正文与附录的对象、命名和训练说明；不认证训练记录。

## APP-L06：H.2 的长段应分为精确结论、差分核验、离散单调性核验

- **位置**：`docs/paper_p1/APPENDICES_EN.md:795`。
- **严重性**：必须修（推论范围）；分段为建议修。若存在积分分支的直接记录，可补记录后保留更强判断。
- **短英文原文**：“the hundredfold reduction per decade is the second-order truncation of the central difference, so no integration branch changes within \(\pm10^{-3}\tau_c\) at the fixed active set on these cells.”
- **中文**：每缩小一个数量级，误差降低百倍，这是中心差分的二阶截断，因此这些胞在固定活跃集下、\(\pm10^{-3}\tau_c\) 范围内没有积分分支改变。

**问题与方向**：该段从精确积分下的 PSD（半正定）性质，进入四胞的差分步长变化，再进入八胞的单元导数谱，最后限定本研究参考灵敏度的含义。这一安排隐藏了它本来很有价值的结论：“数值导数可很稳定地逼近离散模型的导数，但这不自动使离散模型继承精确积分的半正定性质”。此外，几个步长上二阶趋势不能单独证明整个区间内没有分支改变；这是证据级跳步，而非修辞偏好。

**建议英文（建议将 L795 分为三段；全部当前数值保留）**：

> In Eq. (H.10), \(\nabla^{\rm s}\) is the symmetric gradient; the elasticity tensor and basis are fixed, and the ghost contribution cancels. The differentiable limit gives a positive-semidefinite thickness derivative. A centred difference assembled from exactly nested domains has the same property. Adaptive subdivision, approximate moments or independently selected stabilisation can disrupt this nesting, so fixing the active topology alone does not establish the sign of the numerical derivative.
>
> We first check the finite-difference step on four cells (Figure S01(c)). Relative to the production step \(10^{-5}\tau_c\), the sensitivity changes by at most \(2.6\times10^{-7}\), \(2.5\times10^{-9}\) and \(1.6\times10^{-10}\) at steps \(10^{-3}\tau_c\), \(10^{-4}\tau_c\) and \(10^{-6}\tau_c\), respectively. The decrease at the larger steps is consistent with second-order truncation. The stiffness-derivative sensitivities also agree with differences of the re-solved compliance to \(4\times10^{-8}\). These checks support the numerical derivative at the tested designs and steps.
>
> We then test whether the discrete moments preserve the sign predicted by exact integration (Table ST15). The element stiffnesses are positive semidefinite to rounding, but differentiating the discrete moments at fixed clipping topology produces some element derivative matrices with negative eigenvalues. In the uncut cells, the smallest ratio \(\lambda_{\min}/\max|\lambda|\) is \(-2.3\times10^{-3}\). In four of the six cut cells, between 2 and 11 partially filled elements, out of 762 to 6,033, have a uniform-thickening derivative with a negative eigenvalue exceeding \(10^{-6}\) of the largest absolute eigenvalue; the sign of the full spectrum requires the additional check discussed in the supplementary-material review. The production central difference agrees with the exact discrete derivative to \(2\times10^{-8}\). Equations (H.10) and (H.11) therefore describe exact integration, while the reference sensitivities reported here are derivatives of the discrete model, which preserves this structure only approximately.

**建议中文**：第一段只说明精确积分下何以半正定，以及离散积分为何需要另查。第二段只报告四胞差分步长试验，判断为“较大步长下的减少与二阶截断一致”，不由此推出整个区间内积分分支绝不变化。第三段再报告单元导数谱：刚度本身半正定与其数值厚度导数半正定是不同性质；中心差分准确逼近离散导数，并不消除离散导数局部负特征值。末句保留精确积分公式与本文离散参考灵敏度的边界。

**为何更清楚**：从“公式保证什么”到“差分算准没有”再到“离散模型是否继承单调性”，每一步对象不同但相互衔接。作者的判断比原来更突出；不回避已记录的负导数，也不暗示负值需要人为修复。

- **证据**：`APPENDICES_EN.md:692,786–795`；`SUPPLEMENTARY_EN.md:586–599` 的 ST15；`evidence/ref_valid.json` 与 `evidence/ref_valid_h1.json` 的 `per_case.*.fd.rel_to_h1e_5`、`direct_vs_sens`。四胞汇总的三个步长最大值为 `2.5580023126e-7`、`2.5289345704e-9`、`1.5569645179e-10`；重新求解导数对比最大 `3.8424593530e-8`，与稿中舍入吻合。所读 `fd` 记录含 `sens_by_h`、`rel_to_h1e_5`、`direct`、`direct_vs_sens`，未提供分支计数或分支签名。这里不声称整个仓库不存在另一个分支核验文件。
- **影响**：定位否；结论对“无分支改变”的证据强度有影响，应由作者决定改为数值支持或补直接记录；删除否。
- **确信度**：高（推论范围、四胞数字与结构组织）；未独立认证所有单元谱数字。
- **核验范围**：四胞存档数字全汇总、ST15 文字及表列检查；未重做积分、特征值计算、全区间分支追踪。

## 小型统一项：可直接并入主报告的语言清单

| 项 | 当前位置与原文 | 建议 | 分类与边界 |
|---|---|---|---|
| APP-U01 | A.2 L48：“Its six face-load cases define the common 3% reference …” / 六个面载荷工况定义共同 3% 参考。 | “Its six face-load cases use the common 3% line for compliance and sensitivity …” / 六个工况使用柔度与灵敏度共同的 3% 线。 | 必须修作者已定命名；`MANUSCRIPT_EN.md` 的精确离散参考另有含义。交接 L34 解释了偏好。无定位、结论、删除影响。 |
| APP-U02 | F.4 L483：“The reference integration initialises … and refine …” | `refine` → `refines`。 | 明确主谓一致；建议修。无科学影响。 |
| APP-U03 | B.3 L154：“the stiffness content of the error” / 误差的刚度内容。 | 可保留；若作者要具体化，写成 “the Rayleigh quotient of the error relative to that of the equilibrium response”。 | 作者选择。原文下一句已有物理解读，不宜只为去抽象名词而拉长句子。 |

## 应保留的作者判断与解释

1. **B.2 L97**：明确 “Positive semidefiniteness alone does not imply rigid reproduction …” 并指出参考矩阵的额外核问题。这是必要的机械条件，不是可删的防御式文字。
2. **B.3 L165**：用 \(A=\operatorname{diag}(\epsilon,1)\) 的短例子区分未经缩放的软方向与 Jacobi（雅可比）缩放后的低模态。具体例子优于抽象宣言。
3. **C.1 L220、252、270**：分别解释柔度误差的正交贡献、energy share（能量份额）在同一精确组装迹上评价、渐近阶次不能自动覆盖几何族。三个对象不同，不能简单按“重复限定”删掉。
4. **D L350**：坦率说明现有收缩界几乎只保证不扩张，而一到两个数量级的降低来自实测。作者判断清楚，证据力度合适。
5. **E L406**：前向保持 retained displacement（保留自由度位移）不代表转置没有 retained force（保留自由度力）贡献。短而有解释力。
6. **F.1 L434**：将未认证独立性的富集列结果作为数值观察，明确主结果采用 \(Q_1(17)\)。这一区别应完整保留。
7. **G.4 L628–630**：未覆盖方向可藏任意大误差的反例，清楚说明完整 retained DOFs（保留自由度）与足够方向覆盖是两件事；不要压成一句“覆盖有限”的套话。
8. **H.1 L751、753–755**：理论 bound（界）不是对记录百分比的拟合归因；替换试验的三个输出范数不是可加的误差分解。它们阻止读者误用结果。
9. **H.2 L784**：解释一阶场估计误差与二阶完整导数为何可以同时出现，建立真正的逻辑联系。
10. **I L826**：“A large value detects an inaccurate direction; a small value alone does not bound the error from above.” 判断具体，保留。

## 统一语言原则与建议修改顺序

先处理 APP-L03、APP-L06 的证据级措辞，与数学/证据主审对齐；再处理 APP-L02 的误差对象与工况范围；随后分段 APP-L01、APP-L04、APP-L05；最后统一 APP-U01、APP-U02。未获得作者讨论前，不实施会影响结论口径的建议。

附录优先使用“确定性构造/实施步骤 → 数值观测 → 支持的判断”顺序，但不把每一段都机械改成相同三句模板。保留直接的作者动作（we certify、we compare、we check）及其动机。公式附近允许重申必要条件，尤其是固定 retained DOFs（保留自由度）、interior（内部自由度）、固定载荷、完整转置和谱上端点条件。术语 cut band（切割带）、active element（活跃背景单元）、background element（背景单元）、weak support（弱支撑）、energy share（能量份额）、two-grid correction（两网格校正）不作同义替换。

作者选定口号在附录中并未构成突出的机械重复问题；本报告不建议删除口号。所有建议均未写入论文，未改任何数字、公式、图像或结论。
