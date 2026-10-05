# P1 提升方案：具体改法与中英改稿（2026-10-05）

依据 `CMAME_BENCHMARK_CN.md` 的诊断。2026-10-05 已按作者决定落稿：A–J、K1 已写入中英文源稿；K2（成本账）取消；L–P 在服务器上推进；标题保留 NICE，候选待定。

## 总原则

要做的是三件事，不加形容词，不说首创，可读性优先。

1. **抬高度。** 先把原理说出来，再讲实现；理论结果写成 Proposition。
2. **定姿态。** 适用范围只说一次，之后的结论不再挂限定语；局限写成"事实 + 下一步"。
3. **亮能力。** 头条放能力和最大的结果；补一张展示图；设计结果要有对照。

## 分批计划

| 批次 | 内容 | 性质 | 工作量 |
| --- | --- | --- | --- |
| 第一批 | A 核心思想句；B 摘要；C 贡献；D 第 4 节开头的统一视角；J Highlights | 纯写作，改动集中 | 小 |
| 第二批 | E 假设编号 + Proposition 1–4；F 条件只说一次 | 呈现方式，涉及 §3、§4.4、附录引用 | 中 |
| 第三批 | G 引言拆出 1.1 Related work；H 第 5 节开头与 5.2 压缩；I 讨论与结论收尾；K 设计问题公式与成本账 | 结构调整，只移动不删除 | 中 |
| 第四批 | L 图 1 改为问题图；M 5.10 三维展示；N 均匀厚度对照；O 110 胞元精度抽查；P 补引 McBane & Choi | 实质补充，需要算或渲染 | 中到大 |
| 待你拍板 | 标题（见 Q） | 取舍 | 小 |

---

## A. 引言第 5 段：核心思想（D1 + D5）

**问题。** 核心思想刚出场，下一句就是 "Each part rests on established ground"，读者第一印象是"都是已有的"。最好的一句三分工只出现在结论里。

**改法。**

- 首句后直接接三分工。
- 把"都有出处"改成"这种分工换来了什么"：网络只需要准，结构由能量形式保证。
- 点出精确凝聚是特例。
- 出处放到最后，原有引用全部保留。

**EN（替换第 5 段）**

> The central idea is a division of tasks within static condensation: learning supplies the trial field, the variational form supplies the structure, and a fixed correction supplies the improvability. The approximation is placed in the interior extension; the extension is a network that is exactly linear in the retained displacement and admissible by construction, so that its energy defines a Ritz approximation of the condensed stiffness; a fixed equilibrium correction inside that energy removes much of the error the network leaves; and the error that remains is followed through the energy share and the stiffness derivative to the compliance and the design sensitivity. With this division the network has to be accurate, but it does not have to preserve structure. Whatever admissible extension it produces, as long as it reproduces rigid-body motion, the energy form yields a symmetric positive semidefinite condensed stiffness with the rigid-body kernel, bounded below by the exact Schur complement with an error quadratic in the interior error; exact static condensation is the case in which the extension is exact. The same Ritz property underlies static condensation itself [Toselli & Widlund (2005)], multiscale finite elements [Hou & Wu (1997)], component reduced-basis methods [Huynh et al. (2013)] and learned shape functions evaluated as \(N^TKN\) [Huang et al. (2023)]; here it carries a learned extension on a retained space that changes with the cut.

**CN**

> 核心思想是在静力凝聚内部进行分工：学习提供试探场，变分形式提供结构，固定校正提供可改进性。近似被置于内部延拓之中；延拓由一个对保留位移精确线性、由构造满足容许性的网络给出，因此其能量定义了凝聚刚度的 Ritz 近似；同一能量内部的固定平衡校正去除网络留下的大部分误差；剩余误差则经由能量份额和刚度导数追踪到柔度与设计灵敏度。在这种分工下，网络只需要准确，而无需自己保持结构：只要它给出的延拓是容许的并再现刚体运动，能量形式就保证凝聚刚度对称半正定、以刚体模态为零空间、以精确 Schur 补为下界，且误差是内部误差的二次型；精确静力凝聚正是延拓精确时的特例。同一 Ritz 性质也是静力凝聚本身 [Toselli & Widlund (2005)]、多尺度有限元 [Hou & Wu (1997)]、构件约化基方法 [Huynh et al. (2013)] 以及以 \(N^TKN\) 评估的学习形函数 [Huang et al. (2023)] 的基础；在本文中，它承载的是一个定义在随切割变化的保留空间上的学习延拓。

---

## B. 摘要（D3）

**问题。**

- 到第 4 句才说出自己的贡献，全文没有出现 NICE 这个名字。
- 约 9 个数字。
- 结尾挂着 "under stated conditions" 和 "at the checked designs"。

**改法。**

- 前三句不动。
- 第 4 句点出 NICE，并用三分工表述。
- 条件写进名词里："a correction nonexpansive in this energy"。
- 柔度与灵敏度的区分单独成句。
- 数字只留 27%、0.074%、约十倍、0.03%/0.3%。
- 删掉 "assembled compliance and sensitivity errors stay below 0.28% and 1.5%"，正文里仍有。
- 110 胞元那条，等 O 的精度抽查做完再决定加不加。

**EN（250 词）**

> When a lattice is trimmed to the shape of a part, the cells cut by the boundary can carry its supports and loads, and each has a geometry of its own. Resolving every wall makes each design iteration a fine-scale analysis of the whole structure, whereas homogenisation loses accuracy where scales do not separate, underestimating the compliance of a 24-cell layer clamped at its cut by 27%. Static condensation keeps each cell's fine scale, but needs an interior factorisation for every geometry, whose discrete space changes with the cut. Neural-initialised static condensation with equilibrium correction (NICE) avoids it by dividing the tasks of condensation: a geometry-conditioned network supplies the interior field, the energy form the structure, and a fixed two-grid correction the improvability. The network, exactly linear in the retained displacement and admissible by construction, keeps every box-face and cut-band degree of freedom, tens of thousands per cell, and forms no matrix. The condensed stiffness is a Ritz approximation, bounded below by the exact Schur complement with error quadratic in the interior error, and a correction nonexpansive in this energy cannot increase that error. Accurate compliance does not imply accurate sensitivity, so NICE is trained and checked on both. For cut Schwarz-P cells, the mean energy error on 80 validation geometries is 0.074%, and an eight-cell analysis with sensitivities is about ten times faster than direct solution. Thickness optimisation reproduces the exact-condensation design of an eight-cell lattice; on the clamped layer, compliance and gradient agree with exact condensation to 0.03% and 0.3%.

**CN**

> 当点阵被修剪为零件形状时，被边界切割的胞元可能承担零件的支承与载荷，且各自具有不同的几何。逐一解析每道壁面，会使每次设计迭代都成为整个结构的细尺度分析；而均匀化在尺度不分离处失去精度，对一个在切割处固支的 24 胞元单层，其柔度被低估 27%。静力凝聚保留每个胞元的细尺度，但需要对每个几何进行内部分解，而其离散空间随切割变化。带平衡校正的神经初始化静力凝聚（NICE）通过对凝聚进行分工来避免这一分解：几何条件化网络提供内部场，能量形式提供结构，固定的两重网格校正提供可改进性。该网络对保留位移精确线性、由构造满足容许性，保留全部胞元边界面与切割带自由度（每个胞元数万个），且不形成任何矩阵。凝聚刚度是 Ritz 近似，以精确 Schur 补为下界，误差是内部误差的二次型；在该能量下非扩张的校正不会增大这一误差。柔度精确并不意味着灵敏度精确，因此 NICE 对二者分别训练与检验。对切割 Schwarz-P 胞元，80 个验证几何上的平均能量误差为 0.074%，八胞元带灵敏度的分析比直接求解快约十倍。厚度优化再现了八胞元点阵的精确凝聚设计；在该固支单层上，柔度与梯度与精确凝聚的偏差分别为 0.03% 与 0.3%。

**注意。** 去掉 "at the checked designs" 之后，"0.03%/0.3%" 读起来像是对所有设计都成立。正文和表 6 里已经写明是三个经过精确核查的设计。如果你觉得摘要里也必须写明，就恢复成 "at the three checked designs"（多 3 个词，需要在别处再删 3 个词）。

---

## C. 贡献：改成列表，每条一句断言（D8）

**问题。** 现在是一个约 450 词的整段，每条贡献三到五句，混着代价、条件和七个章节指引。

**改法。** 每条先用一句加粗的断言，再跟一两句证据，最后给一个章节指引。(i) 里的代价句删掉，PIML 那段已经讲过"装配系统更大"。

**EN**

> The contributions of this work follow this division.
>
> - **(i) Condensation on the complete, geometry-dependent retained space.** Every degree of freedom through which a cell exchanges displacement with its neighbours, supports and loads is kept, and the approximation is confined to the interior response; cut-surface supports and loads therefore act on the original degrees of freedom without retraining, although these change with the geometry (Sections 2, 5.7 and 5.10).
> - **(ii) A learned extension whose size does not depend on the retained set.** The network is nonlinear only in the geometry and exactly linear in the retained displacement, with the retained values and rigid-body motion imposed by construction, so that the relations of Section 3 apply to it and its transpose returns the work-conjugate force. One network of about \(6\times10^5\) parameters serves retained sets of 2,679 to 45,900 degrees of freedom without forming a matrix (Sections 4.2 and 4.6).
> - **(iii) A correction that improves a deployed network without retraining.** One fixed two-grid cycle inside the condensed energy keeps the properties of static condensation for any discrete model with a symmetric positive semidefinite stiffness and a positive definite interior block, and under the condition of Proposition 4 cannot increase the error (Sections 4.3–4.4 and 5.5).
> - **(iv) Compliance and sensitivity as separate acceptance criteria.** The interior error reaches the compliance weighted by each cell's energy share, but the thickness sensitivity through a term linear in that error, which the energy error bounds only at the order of its square root. NICE is therefore trained and checked on both, and its designs are verified against exact condensation (Propositions 2 and 3; Sections 5.6, 5.8 and 5.10).

**CN**

> 本文的贡献与上述分工一一对应。
>
> - **(i) 在完整且随几何变化的保留空间上进行凝聚。** 保留胞元与邻居、支承和载荷交换位移所经由的全部自由度，近似仅限于内部响应；因此，尽管这些自由度随几何变化，切割面上的支承与载荷仍作用于原始自由度，无需重新训练（第 2、5.7 和 5.10 节）。
> - **(ii) 规模与保留集无关的学习延拓。** 网络仅对几何非线性、对保留位移精确线性，并由构造施加保留值与刚体运动，因而第 3 节的关系直接适用于它，其转置返回功共轭力。一个约 \(6\times10^5\) 参数的网络服务于 2,679 至 45,900 个自由度的保留集，且不形成任何矩阵（第 4.2 和 4.6 节）。
> - **(iii) 无需重新训练即可改进已部署网络的校正。** 凝聚能量内部的一次固定两重网格循环，对任意刚度对称半正定、内部块正定的离散模型都保持静力凝聚的性质，并在命题 4 的条件下不会增大误差（第 4.3–4.4 和 5.5 节）。
> - **(iv) 将柔度与灵敏度作为两项独立的验收标准。** 内部误差以各胞元的能量份额为权重进入柔度，却以一个关于该误差的线性项进入厚度灵敏度，而能量误差只能在其平方根量级上界定这一项。因此 NICE 对二者分别训练与检验，其设计均与精确凝聚对照验证（命题 2 和 3；第 5.6、5.8 和 5.10 节）。

（4.2 GiB 那句移到 4.6，原文那里已经有了。）

---

## D. 第 4 节开头：以 F 为中心的统一视角（D5）

**EN（替换第 4 节首段的前两句）**

> NICE is defined by one object, the corrected extension \(F\), which maps a retained displacement to a displacement of the whole cell; its energy \(\widehat S=F^TKF\) is the condensed stiffness, and exact static condensation is the case \(F=E\). The construction divides the work on \(F\). The energy form, applied with the transpose of the complete extension, supplies the mechanical structure for any admissible extension (Section 4.1). …（后文不变）

**CN**

> NICE 由一个对象定义，即校正后的延拓 \(F\)：它把保留位移映射为整个胞元的位移；其能量 \(\widehat S=F^TKF\) 就是凝聚刚度，精确静力凝聚是 \(F=E\) 的情形。构造在 \(F\) 上进行分工。能量形式配合完整延拓的转置，为任意容许延拓提供力学结构（第 4.1 节）。……

可选：在式 (4) 之后补一句 "Exact static condensation is the case \(F=E\), \(H=0\)." / "精确静力凝聚即 \(F=E\)、\(H=0\) 的情形。"

---

## E. 理论骨架：假设编号 + 命题（D4）

**问题。** 正文和附录里 Proposition、Theorem、Lemma 都是 0 个；条件散在七处，每处都在复述。

**改法。**

- §3 的消息式小节标题、方框公式和行文全部保留。
- 只在 §3 引言末尾加一个编号的假设表（实质是把附录 B.1 前移成摘要）。
- 把四个已有结果写成加粗的 Proposition 段落，每个命题后注明证明所在的附录，并跟一句工程含义。
- Markdown 写成 `**Proposition 1 (Ritz identity).** *…*`，LaTeX 和 Word 的构建脚本都不需要改。

**§3 引言末尾新增（EN）**

> The results below hold under the following assumptions, collected in Appendix B.1.
> (A1) The cell stiffness \(K\) is symmetric positive semidefinite, its null space consists of the rigid-body modes, \(A=K_{II}\succ0\), and interior body loads vanish (Section 2.2).
> (A2) The extension \(F\) is linear and admissible, \(J_PF=I_p\), and reproduces rigid-body motion, \(FR_P=R\).
> (A3) Cells share only retained degrees of freedom, and the supported assembled stiffness is positive definite (Section 2.3).
> (D) On a design interval, the discrete choices listed in Appendix B.1 are fixed.

**CN**

> 以下结果在如下假设下成立（汇总于附录 B.1）。
> (A1) 胞元刚度 \(K\) 对称半正定，其零空间由刚体模态组成，\(A=K_{II}\succ0\)，内部体力为零（第 2.2 节）。
> (A2) 延拓 \(F\) 线性且容许，\(J_PF=I_p\)，并再现刚体运动，\(FR_P=R\)。
> (A3) 胞元之间只共享保留自由度，带支承的装配刚度正定（第 2.3 节）。
> (D) 在设计区间内，附录 B.1 所列离散选择保持不变。

**四个命题（陈述沿用现有公式，不增加新内容）**

| 编号 | 位置 | EN 陈述 | 证明 |
| --- | --- | --- | --- |
| Proposition 1 (Ritz identity) | §3.1，包住式 (4)(5) | Under (A1)–(A2), \(\widehat S-S=H^TAH\succeq0\) with \(H=J_I(F-E)\), and the relative directional energy error is \(\varepsilon(q)=r_I^TA^{-1}r_I/q^TSq\). | App. B |
| Proposition 2 (compliance) | §3.2，包住式 (6)(7) | Under (A1)–(A3), \(C-\widehat C=\|\widehat U-U\|_{\mathbb K}^2+\sum_m\|H_mB_m\widehat U\|_{A_m}^2\) and \(0\le (C-\widehat C)/C\le\beta/(1+\beta)\le\beta\). | App. C |
| Proposition 3 (sensitivity) | §3.3，包住式 (9) | Under (A1), (A2) and (D), at a common retained displacement, \(\widetilde s_c-s_c=-2d^TK_{,c}u-d^TK_{,c}d\); the linear term is bounded by the energy error only at order \(\sqrt\varepsilon\) (Eq. (H.4)). | App. H |
| Proposition 4 (correction ordering) | §4.4，替换现在的条件段 | Let \(F=\mathcal W\widehat E\) with \(\mathcal W\) linear, fixed per geometry and preserving retained values. If the interior error map \(\Theta\) of \(\mathcal W\) satisfies \(\Theta^TA\Theta\preceq A\), which holds for an exact or nonnegatively shifted coarse solve with \(\|\Phi_k\|_A\le1\), then \(S\preceq\widehat S\preceq\widehat S_{\rm net}\). | App. D.1 |

每个命题后面跟一句工程含义，基本就是现有正文的那句。例如命题 1 后面是 "An approximate extension therefore adds stiffness in proportion to the energy of its interior error."

**CN 措辞**：命题 1（Ritz 恒等式）、命题 2（柔度）、命题 3（灵敏度）、命题 4（校正排序），陈述照上表逐句翻译。

**连带改动。**

- §3 末尾的"三项条件"列表保留，用命题编号引用。
- 引言、图 2 图注、6.2、结论里复述条件的地方，统一改成引用命题 4（见 F）。
- 附录各节标题加上 "Proof of Proposition n"。

---

## F. 条件只说一次（D9）

| 位置 | 现在 | 改为 |
| --- | --- | --- |
| 图 2 图注 | The ordering in (a) assumes an exact or nonnegatively shifted coarse solve and a smoothing interval whose upper end bounds the spectrum; it orders the energy error, not the sensitivity error. | The ordering in (a) is that of Proposition 4; it concerns the energy error, not the sensitivity error. ／ (a) 中的排序即命题 4；它针对能量误差，而非灵敏度误差。 |
| 引言第 7 段 | Its error cannot exceed that of the uncorrected one when the coarse solve is exact or nonnegatively shifted and the upper end of the smoothing interval bounds the spectrum [refs] | Under a condition on the coarse solve and the smoothing interval (Proposition 4), its error cannot exceed that of the uncorrected one [refs] ／ 在关于粗求解与光滑区间的条件下（命题 4），其误差不会超过未校正者 [refs] |
| 结论第 1 段 | a correction with an exact or nonnegatively shifted coarse solve and a smoothing interval that bounds the spectrum from above cannot increase the error | a correction satisfying Proposition 4 cannot increase the error ／ 满足命题 4 的校正不会增大误差 |
| 全文 "examined" (10 处) | in the cut cells examined / on the five cells examined … | 只保留确有必要的 3–4 处；其余改为直接陈述，范围已由第 5.1 节和 6.4 节交代 |

---

## G. 引言结构：拆出 1.1 Related work（D6）

- 第 11–23 行不动：矛盾 → 论点 → 保留空间 → 柔度/灵敏度 → 核心思想 → 三个障碍 → 校正与命名。
- 命名之后插入一段定位（见下）。
- 接着是贡献列表（C）和路线图。
- 原第 25–33 行原样移到新小节 "1.1 Related work"，引用和 PIML 定性句全部保留，只把"边比边认错"的句子改成平陈。

**定位段 EN**

> Learned and reduced component models differ in three choices: what a component exchanges with its neighbours, how its interior response is represented, and where its approximation is improved. Methods with few boundary degrees of freedom exchange restricted displacement patterns, in return for small assembled systems; component reduced-basis methods keep the interface but represent the interior on a discretisation shared by the parameter family; and inexact substructuring improves approximate local solves inside a preconditioner. NICE keeps the complete retained space although it changes with the cut, represents the interior by a geometry-conditioned network, and improves the approximation inside the condensed operator that is assembled. Section 1.1 relates it to each family.

**CN**

> 学习型与约化型构件模型的区别在于三项选择：构件与邻居交换什么，内部响应如何表示，近似在何处改进。边界自由度很少的方法以受限的位移模式进行交换，换来较小的装配系统；构件约化基方法保留完整界面，但在参数族共享的离散上表示内部；非精确子结构方法则在预条件子内部改进近似的局部求解。NICE 保留随切割变化的完整保留空间，以几何条件化网络表示内部，并在被装配的凝聚算子内部改进近似。第 1.1 节逐一说明它与各类方法的关系。

---

## H. 第 5 节：开头先给结论，验证细节下沉（D10）

**第 5 节开头（替换现在的路线图段）EN**

> The examples show three things. On single cells, the corrected learned extension reaches a mean energy error of 0.074% on 80 validation geometries, and the network and the correction each remove error that the other leaves (Sections 5.3–5.5). After assembly, compliance follows the share-weighted relation of Proposition 2, whereas the local sensitivity has to be checked on its own (Sections 5.6–5.8). In design, an eight-cell analysis with sensitivities is about ten times faster than direct solution, and the optimised designs agree with exact condensation in compliance and gradient (Sections 5.9 and 5.10). Sections 5.1 and 5.2 first state the settings and verify the discrete reference.

**CN**

> 算例说明三点。在单胞上，校正后的学习延拓在 80 个验证几何上的平均能量误差为 0.074%，网络与校正各自去除对方留下的误差（第 5.3–5.5 节）。装配之后，柔度遵循命题 2 的份额加权关系，而局部灵敏度必须单独检验（第 5.6–5.8 节）。在设计中，八胞元带灵敏度的分析比直接求解快约十倍，优化设计在柔度与梯度上均与精确凝聚一致（第 5.9 和 5.10 节）。第 5.1 和 5.2 节先给出设置并验证离散参考解。

**5.2 第 2 段整段压成一句**

> EN: The deployed operator is symmetric, satisfies the work–energy identity and annihilates rigid-body modes to round-off, and reproduces the training-time field to \(1.1\times10^{-7}\) (relative deviations at most \(3\times10^{-8}\); Table ST04).
> CN: 部署算子的对称性、功能一致性与刚体模态零能量均达到舍入精度，并以 \(1.1\times10^{-7}\) 的精度再现训练时的场（相对偏差至多 \(3\times10^{-8}\)；表 ST04）。

**其他调整。**

- 5.1 中的样本池重叠、训练步数、GPU 时等记账移到 ST01；"20 个几何用于检查点选择、16 个点阵胞元未参与"这句保留。
- 5.3–5.10 每个小节首句改成一句结论，例如 5.3："NICE is 64 to 102 times more accurate than the uncorrected continuation in every stratum, and its error grows about sevenfold from uncut to heavily cut cells."（这已基本是原文首句，其他小节照此调整。）
- 正文用力学描述代替编号：uncut / moderately cut / heavily cut cell。U1、M1、H2 这类编号只在图表里出现。

---

## I. 讨论与结论：落在"能走多远"（D13）

**讨论部分。**

- 6.2 的推导（公共 q 处的两个误差表达式，H_{,c} 与 E_{I,c} 的比较）移到 3.3 式 (10) 之后，或移到附录 H.2。
- 6.2 的证据（ST14 共享变量梯度、中心差分失效、MMA 不做线搜索）并入 6.3 或 6.4。
- 讨论由此变为三个小节：6.1 保留什么、表示什么、校正什么；6.2 计算价值（原 6.3，加上 K 里的成本账）；6.3 局限（原 6.4）。
- "构造与误差关系适用于任意 SPSD 刚度、正定内部块"这句，从局限首句挪到 6.1 末尾，作为结构性事实。

**局限部分**：内容全部保留，句式改成"事实 + 下一步"。例如：

> EN: The size of lattice that one GPU can analyse is set at present by the direct factorisation of \(\mathbb K_{PP}\) in the preconditioner; a further level of the multilevel correction or an iterative coarse solve would remove this limit.
> CN: 目前单张 GPU 可分析的点阵规模受限于预条件子中 \(\mathbb K_{PP}\) 的直接分解；在多层校正中再加一层，或改用迭代粗求解，即可去除这一限制。

**结论**：从四段压到三段——方法与性质、误差关系、实例结果。数字只留关键的几个，最后一段换成下面这段。

**结论末段 EN**

> Between homogenisation, which loses accuracy where scales do not separate, and the resolution of every wall, whose cost each design iteration pays again, NICE provides a component model that keeps the structure of static condensation without its interior factorisation. Folding the coarse-grid correction into the network, as a coarse level that solves the Galerkin problem, would leave the smoothing as the only fixed stage. Because the structure comes from the energy form and not from the network, the error relations and the correction carry over to any component family with a symmetric positive semidefinite stiffness and a positive definite interior block: a new family needs new training, not a new theory.

**CN**

> 在尺度不分离处失去精度的均匀化，与每次设计迭代都要重新付出代价的逐壁解析之间，NICE 提供了一种构件模型：它保持静力凝聚的结构，而不需要其内部分解。将粗网格校正并入网络，作为求解 Galerkin 问题的粗层，将使光滑成为唯一的固定环节。由于结构来自能量形式而非网络，误差关系与校正可以推广到任何刚度对称半正定、内部块正定的构件族：换一族胞元，需要的是重新训练，而不是新的理论。

（"a new family needs new training, not a new theory"是事实陈述，不涉及首创，也不夸大精度，可以作为全文的最后一句。）

---

## J. Highlights（每条 ≤ 85 字符）

| # | Highlight | 字符数 |
| --- | --- | ---: |
| 1 | Learned interior extension keeps the structure of static condensation in cut cells | 82 |
| 2 | Condensed stiffness error is one-sided and quadratic in the interior error | 74 |
| 3 | Accurate compliance does not imply accurate sensitivity; both are checked | 73 |
| 4 | A fixed two-grid correction improves a deployed network without retraining | 74 |
| 5 | Optimised cut TPMS lattices agree with exact condensation in both measures | 74 |

**图形摘要**：直接用新图 2 的 (a) 行（三个职责标签加公式），或用 L 的问题图。

---

## K. 补两处正文

**1. 在 2.3 节写出设计问题**（目前只在 5.10 用文字描述）

\[
\min_{\boldsymbol\tau}\; C(\boldsymbol\tau)\quad\text{s.t.}\quad V(\boldsymbol\tau)\le V^*,\quad \tau_{\min}\le\tau_c\le\tau_{\max},\quad\text{corner-span and gradient-norm limits},
\]

后面接一句共享变量梯度的装配公式。3.3 节和 5.10 节引用这个式号。

**2. 成本账：已取消（作者决定，正文不写离线/数据集成本）。**

---

## L–P. 实质补充（需要计算或渲染，开跑前逐项确认）

| 项 | 内容 | 资源 | 产出 |
| --- | --- | --- | --- |
| L | 新图 1"问题图"：24 胞元切割板三维渲染（切割带固支、端面加载），加一个切割胞元特写（保留面与切割带着色）。现在的图 1 挪到 5.1 | CPU，用现有几何生成代码渲染 | 第一眼就是问题，而不是测试样本 |
| M | 5.10 开头加三维展示：均匀初始设计 vs NICE 最终设计（壁面按厚度着色），再加固支旁一个切割胞元上 NICE 与精确凝聚的壁面场对比，图注写相对差 | CPU；用已有精确核查的设计（ST21） | 全文第一张"结构级"的场图 |
| N | 同体积的均匀厚度设计：用 NICE 算一次、精确核查一次，与优化设计和均匀化设计并列 | 几次分析，单卡或 CPU | 让 2.44% 有参照，并给出优化本身的收益 |
| O | 110 胞元（或 51 胞元）精度抽查：抽若干胞元做精确凝聚，比较局部能量误差；或对 51 胞元板做整体精确核查 | GPU 或大内存 CPU，开跑前告知 | 决定 32.7M 那条能否进摘要和贡献 |
| P | 补引 McBane & Choi (2021, CMAME, doi 10.1016/j.cma.2021.113813)，在构件约化基一段加一句对比 | 核实 DOI | 补上审稿人所在圈子的工作 |

---

## Q. 标题（待你拍板）

| 选项 | 标题 | 词数 |
| --- | --- | ---: |
| 现状 | Neural-initialised static condensation with equilibrium correction for the analysis and thickness design of cut thin-walled TPMS lattices | 17 |
| A | Learning the interior extension of static condensation: analysis and thickness design of cut TPMS lattices | 15 |
| B | Static condensation with a learned, corrected interior extension for the thickness design of cut TPMS lattices | 16 |
| C | 保留现标题，只把 NICE 的命名和三分工放进摘要（B） | — |

- A 把"学的是什么"放到了最前面；方法名 NICE 仍在摘要第 4 句出现。
- 之前否决过 "Learned static condensation"，A 和 B 都不回到那个说法。
