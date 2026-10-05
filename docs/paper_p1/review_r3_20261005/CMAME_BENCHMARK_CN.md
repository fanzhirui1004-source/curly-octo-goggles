# P1 与 CMAME 代表论文的对照诊断（2026-10-05）

## 0. 方法

本次诊断分四步：

1. 从四个角度通读 P1 正文和附录：叙事与定位、方法与理论的呈现、结果与图、行文语气。
2. 上网阅读 17 篇 CMAME 论文的全文（arXiv 版本），DOI 均核实为 10.1016/j.cma，见第 4 节。
3. 两边逐条对照。
4. 每一条差异都单独做了反驳式核查：P1 引文是否原文、是否有代表性，对方证据是否可信，改法是否违反我们已定的约束。共 14 条差异，保留 11 条，删去 3 条；另有一轮补漏，补出 6 条。

下面引用的 P1 原文都已在 `MANUSCRIPT_EN.md` 里逐字核对过。

## 1. 总诊断

**实质不弱。** 我们有：

- 精确的 Ritz 恒等式；
- 按能量份额加权的柔度关系；
- 灵敏度误差的 √ε 界；
- 校正的排序结果；
- 80 个未见几何；
- 与精确凝聚的孪生优化；
- 3270 万自由度的设计迭代。

拿最接近的 CMAME 文章比，McBane & Choi (2021) 是格子结构的分量式降阶模型，它的三个"定理"只用了 Hölder 不等式和范数等价；Samaniego (2020) 和 Linka & Kuhl (2023) 一个定理都没有，算例规模也不大。

**差距在高度和姿态。** 代表论文的共同做法：

- 用一句不带符号的话说出核心思想，并在摘要、引言、结论里反复出现；
- 给方法一个名字和一张概念图；
- 适用条件在假设里只说一次，之后的结论不再带限定语；
- 用一张决定性的图和一个好记的数字展示结果；
- 结尾把视野打开。

P1 在读者形成印象的几乎每个位置都反着来：

- **标题**讲的是操作步骤，落点是最窄的测试对象。
- **摘要**从场景开始，罗列性质和约十个数字，最后以 "under stated conditions" 和 "at the checked designs" 收尾。
- **核心思想**到引言第 5 段才出现，紧接着就是 "Each part rests on established ground"，把功劳先让了出去。
- **定位**用了四段长文，逐家说明 NICE 和十来个相邻方法有什么不同。
- **贡献**每一条都自带代价或限定。
- **第 5 节**以测试计划开头，第一个格子结构出现之前已经写了约 4,800 词的验证。
- **讨论**写"能做什么"的篇幅少于"保证不覆盖什么"。
- **结论**以 "remain to be established" 结尾。

**最好的东西都在文中，但放在了失去力量的位置：**

| 内容 | 现在的位置 |
| --- | --- |
| 三分工"学习给试探场、变分形式给结构、校正给可改进性" | 只在结论里（图 2 上有标签） |
| F 是几何条件化的延拓（prolongation），Ŝ = FᵀKF 是其 Galerkin 算子，精确凝聚是 F = E 的特例 | 引言一段相关工作的最后一个分句 |
| 任意 A-非扩张校正的一般结论 | 附录 D.1 |
| "误差比解更硬"的放大机制 | 第 5.4 节的一段 |
| 110 胞元、3270 万自由度、一次设计迭代 19 分钟 | 结论最后一句和表 6 最后一行 |
| 所有理论结果 | 没有一条写成 Proposition，读起来像推导笔记 |

**大部分是写法问题，但有两处是实质缺口：**

1. **没有全尺度的展示。** 没有一张图把装配后或优化后的切割点阵渲染出来，也没有一张图给出点阵壁面上的场。110 胞元那次运行既没有精度核查（表 6 写着 "Not checked"），在那个尺度上也没有成本对比。
2. **设计层面的收益显得不大。** 对均匀化只赢了 2.44%，两张设计图看上去也差不多。真正有说服力的是预测层面：均匀化把固支层的柔度低估了 27%。另外，我们没有和"同体积均匀厚度设计"做对比，所以无法判断 2.44% 到底算大算小。

这两处都不需要新方法，只需要用现有流程补一次针对性计算或渲染。

## 2. 逐条差异（按优先级）

### D1 核心思想：要说成自己的原理，不要一出场就把功劳让出去（小改）

- **他们**：
  - Boon et al. 2025："We herein adopt the viewpoint that, while some error in constitutive laws are tolerable, physical conservation laws need to be satisfied precisely."
  - Samaniego 2020："The energy of a mechanical system seems to be the natural loss function for a machine learning method to approach a mechanical problem."
- **我们**：引言第 5 段 "The central idea is a division of tasks within static condensation. … Each part rests on established ground." 最干净的一句表述只在结论里出现。
- **改法**：在第 5 段首句后用冒号接上三分工。把 "Each part rests on established ground" 换成"这样分工换来什么"：网络只需要准，结构由能量形式保证。然后再说 Ritz 性质是凝聚、MsFEM、构件约化基、NᵀKN 共有的，原有引用全部保留。

### D2 展示结果与规模（实质缺口 + 呈现，中等工作量）

- **他们**：
  - Groen et al. 2020："a reduction in computational cost of 3 orders of magnitude, paving the way for giga-scale designs on a standard PC"；
  - 大尺度三维渲染，图注里写网格规模。
- **我们**：最大的结果只在结论末句一笔带过："A design iteration of a 110-cell plate with 32.7 million degrees of freedom takes 19 min." 表 6 最后一行写着 "Not checked"。
- **改法**：
  - (a) 在 5.10 开头加一张三维图：24 胞元切割板，经切割带固支；左边是均匀初始设计，右边是 NICE 最终设计，壁面按厚度着色；再加一个固支旁的切割胞元，NICE 和精确凝聚的壁面场对比。这些设计都已有精确核查（ST21）。
  - (b) 110 胞元结果先补一次精度抽查，再决定能否进摘要。
  - (c) 设计时间和分析时间不能画在同一条坐标轴上（表 6 的图注已经说明两者不是同一个量）。

### D3 摘要：从"性质清单 + 数字清单"改成"能力陈述"（小改）

- **他们**：先给问题类，再给思想，再给它带来了什么；只留一两个取整后的头条数字；结论用定性判断；条件写进对象的定义里，不挂在句尾。
- **我们**：到第 4 句才说出自己的贡献；带了约 9 个数字；全篇没有出现 NICE 这个名字；以限定语收尾。
- **改法**：
  - 前三句保留：切割点阵、27% 的代价、凝聚的瓶颈。
  - 在引出方法的那一句直接写出 NICE 全名。
  - 把条件放进名词："a fixed two-grid correction that is nonexpansive in this energy … cannot increase the error"，然后删掉 "under stated conditions"。
  - 数字只留 3–4 个。
  - 板的 0.03%/0.3% 精确核查必须留下，这是我们独有用途的唯一精度证据。
  - 110 胞元那条，有了精度抽查才写。

### D4 理论要有形式骨架（呈现，中等工作量）

- **他们**：
  - McBane & Choi 2021："Error bounds for the displacement, compliance, and its sensitivity for the CWROM is derived."
  - 即使是初等结果，也写成 Assumption / Proposition / Lemma，附一句工程含义。
- **我们**：正文和附录里 Proposition / Theorem / Lemma 都是 0 个。条件分散在 2.2、3.1、4.4、附录 B.1、图 2 图注、6.2 和结论里。
- **改法**：§3 的消息式小节标题和方框公式都保留。
  - 在 §3 引言末尾加一个带编号的假设列表：(A1) K 半正定，零空间为刚体模态，A 正定；(A2) F 容许且再现刚体；(A3) 装配条件；(D) 设计区间内离散选择不变。
  - 把式 (4)(7)(9) 以及校正排序写成 Proposition 1–4，每条后面注明证明在哪个附录，并跟一句工程含义。
  - 后文不再复述条件，只引用编号。

### D5 统一视角：F 是延拓，经典凝聚是特例（小改）

- **他们**：
  - LOD："it recovers the mathematical theory of homogenization … and even bridges to the theory of iterative solvers"；
  - 多层 FBPINN 直接说明自己受多层 Schwarz 启发。
- **我们**：最有统一性的一句（"F is in effect a geometry-conditioned prolongation … Ŝ = FᵀKF as its Galerkin operator"）是引言里一段相关工作的最后一个分句。
- **改法**：
  - 把 §4 开头两句换成以 F 为中心的表述，点出两个端点：F = E 就是精确凝聚，Ŝ_net 是未校正的情形。
  - 可选：在式 (4) 后补一句 "exact static condensation is the case F = E, H = 0"。
- 核查人提醒：文中其实已经反复用"能量形式"这条主线，所以这条只需小改，不必重构。

### D6 引言结构：一张定位图，代替逐家说不同（中等工作量）

- **他们**：引言 600–2,200 词，对一两种范式各用一句话对比；细致的比较放进 Related work 小节或讨论。
  - McBane & Choi："All of the above methods share a common structure … Our contribution is …"
- **我们**：引言 2,828 词，63 处引用。命名 NICE 之后，还有约 960 词（第 25–33 行）逐家对比，而且常常一边定位一边承认自己的弱点。
- **改法**：
  - 第 11–23 行保持不动。
  - 之后插入一段定位，用全文已经在用的三个维度：胞元与邻居交换什么、内部响应怎么表示、近似在哪里改进。
  - 接着写贡献、路线图。
  - 原第 25–33 行原样移到新的 "1.1 Related work" 小节，引用和 PIML 定性句全部保留。

### D7 标题：讲概念或保证，不讲操作步骤（小改）

- **他们**：
  - "Neural network solvers for parametrized elasticity problems that conserve linear and angular momentum"
  - "A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics"
- **我们**："Neural-initialised static condensation with equilibrium correction for the analysis and thickness design of cut thin-walled TPMS lattices"
- **候选**：
  - "Learning the interior extension of static condensation: a variational component model for the analysis and thickness design of cut thin-walled TPMS lattices"
  - 较短版本："Static condensation with a learned, two-grid-corrected interior extension for …"
- 注意：之前已经否决过 "Learned static condensation" 的写法，这里不回到那个方向。候选一共 22 词，比现在长；如有需要，可以去掉 "thin-walled" 或 "analysis and"。

### D8 贡献：写成主张，不要写成谈判过的规格书（小改）

- **他们**：一行一条，句式平行，不带内部限定。
  - McBane & Choi："The reusability of the trained components for the lattice structure is demonstrated in the design optimization problems."
- **我们**：一个约 450 词的整段，四条贡献，每条三到五句，混着主张、代价、条件和七个括号里的章节指引。
- **改法**：改成四个列表项。每项先用一句断言点明思想和它带来什么，再跟一两句带关键数字的证据，最后统一给一个章节指引。(i) 里关于代价的那句移走，PIML 那段已经讲过。

### D9 语气：适用范围只说一次，不要给每个结论都挂限定（中等工作量）

- **他们**：
  - Boon et al.："We emphasize that we only consider models that feature linear conservation laws …"，范围说一次。
  - DD-LSPG 承认自己效率较低时，紧接着说明在哪里仍然占优。
- **我们**：
  - 排序的适用条件在图 2 图注、4.4、6.2、引言和结论里各写了一遍；
  - "examined" 出现 10 次；
  - 结果段落里有约 8 处 "was/were not"。
- **改法**：
  - 在 4.4 给排序条件起名并编号，其他地方只引用编号。
  - 图 2 图注里的条件句删掉，或换成 "under the condition of Section 4.4"。
  - 6.2 是专讲保证范围的小节，保留一句。

### D10 结果叙事：以能力为主线，而不是一份验证协议（中等工作量）

- **他们**：数值部分写成"难度递增的算例"，每小节先给一句结论，数字放进表和图；代码正确性检查只用一句话交代。
  - Linka & Kuhl："Classical Neural Networks can describe data well but cannot predict beyond the training regime."
- **我们**：
  - 第 5 节开头是一段路线图。
  - 5.2 第 2 段整段讲舍入误差级别的检查（对称性 9×10⁻⁹、功能一致 5×10⁻⁹）。
  - 第 5 节共 8,697 词，其中 4,832 词出现在第一个全学习点阵之前。
- **改法**：从单胞到设计的顺序不变。
  - 第 5 节开头换成一段"结果是什么"。
  - 5.2 第 2 段压成一句，指向表 ST04。
  - 5.1 里的样本池重叠、训练步数、GPU 时等记账移到 ST01。
  - 每个小节以一句结论开头。

### D13 收尾：落在"能走多远"，而不是"边界在哪"（中等工作量）

- **他们**：
  - 局限写得短，放在最后，并写成"下一个研究问题 + 对策"。DD-LSPG："our method introduces a way … that can be useful for truly large-scale problems"。
  - 结论一到三段，落在这个思想对领域意味着什么。
- **我们**：讨论四个小节里有两个讲边界（6.2、6.4），6.3"计算价值"只有一段。结论四段，重复了约 15 个数字，最后一句以限定语结尾。
- **改法**：
  - 6.2 的推导移到 3.3 或附录 H.2，证据移到 6.3 或 6.4。
  - "构造与误差关系适用于任意 SPSD 刚度、正定内部块"这句，从 6.4 首句挪到 6.1 末尾，作为结构性事实陈述。
  - 结论压到两三段，最后一句落在这个方法打开了什么。

### 补漏一轮（另一个智能体检查"还漏了什么"）

- **图 1 的作用**：现在的图 1 是四个编号胞元，读者第一眼会以为这是"单胞数值研究"。建议图 1 改成问题图：24 胞元切割板经切割带固支，旁边放一个切割胞元特写。现在的图 1 挪到 5.1。这张图可以和 D2 的展示图共用渲染。
- **Highlights 和图形摘要**：现在的 Highlights 也是规格书口吻，建议改成能力陈述，每条不超过 85 字符。图 2 本身就可以作为图形摘要。
- **漏引 CMAME 自家的点阵降阶工作**：McBane & Choi (2021, CMAME 113813) 是 CMAME 上分量式降阶做点阵设计的代表作，我们没有引。审稿人往往从参考文献和本刊文献里挑，这条应该补，用一句对比，写法参照对 Huynh et al. 的那句。
- **设计问题没有正式写出**：标题里有 "design"，但文中没有一个显式的优化问题（min C(τ)，体积约束，边界，跨度与梯度限制）。建议在 2.3 加一行，3.3 和 5.10 引用它。同时补一个对照：同体积的均匀厚度设计（见上面实质缺口 2）。
- **离线/在线成本账**：读者最先会问训练多久能回本。建议在 6.3 用两三句交代：一次性的离线成本（60 GPU 小时）、每次胞元分析节省多少、多少次后回本。
- **私有词汇太多**：正文里用 U1/M1/H2 这类编号、多个变体名，还频繁出现 ST 指引，读起来像实验记录。建议正文用力学描述（未切割 / 中度切割 / 重度切割），编号只在图表里出现；正文只保留真正支撑论点的变体。

### 核查后删去的三条

- **D11**："把 5.4–265 倍写进引言"——引言第 9 段已经写了。
- **D12**："两个可复用结论被当成次要检查"——柔度与灵敏度的区分本来就是贡献 (iv)。
- **D14**："通用性只在局限里出现一次"——不实，通用代数类在前文多处已经说明。

## 3. 不要学的

- **优先权和新颖性的用词**：novel、first、paradigm shift、radically different 等。违反我们"不说首创"的约定，也没有必要。
- **评价性形容词和感叹号**：remarkable、drastically 之类。用一个选得好的数字代替。
- **完全不带数字的摘要**：我们的数字是优势，应该精简到两三个，而不是去掉。
- **删掉局限**：6.4 要完整保留，只改位置、去掉重复，写法改成"事实 + 下一步"。
- **没有支撑的广度宣称**，例如 "directly applicable to acoustic and photonic band gaps"。我们只说已经证明的结构性结论，并直接说明换一族胞元需要重新训练网络。
- **挑衅式或历史长镜头式开头**：例如 Linka & Kuhl 的 "Nothing."、"For more than 100 years"。
- **靠删验证来缩短篇幅**：内容挪到附录或补充材料，不删。
- **与 PINN/PIML 做数值对比**：维持现在的一句定性定位。

## 4. 读过的 CMAME 论文

| 簇 | 论文 |
| --- | --- |
| 力学中的机器学习 | Kirchdoerfer & Ortiz 2016（数据驱动计算力学）；Samaniego et al. 2020（能量法）；Haghighat et al. 2021（固体力学 PINN）；Linka & Kuhl 2023（CANN） |
| 凝聚 / 构件降阶 / 多尺度 | Chung–Efendiev–Leung 2018（CEM-GMsFEM）；McBane & Choi 2021（点阵分量式降阶）；Hoang–Choi–Carlberg 2021（DD-LSPG）；Maia et al. 2023（物理递归网络）；Engwer et al. 2019（LOD 实现） |
| 点阵 / 拓扑优化 | Groen et al. 2020（三维去均匀化）；Kumar et al. 2020（spinodoid 数据驱动拓扑优化）；Ferrari & Sigmund 2020（大规模屈曲约束拓扑优化）；White et al. 2019（神经网络代理的多尺度拓扑优化） |
| 学习 + 经典数值（最接近的体裁） | Badia–Li–Martín 2024（FEINN）；Boon–Franco–Fumagalli 2025（守恒动量的 NN 求解器）；Fresca–Manzoni 2022（POD-DL-ROM）；Dolean–Heinlein–Mishra–Moseley 2024（多层 FBPINN） |

完整的引文、核查记录和每篇论文的拆解，保存在会话临时目录的 `bench_result.json` 中，没有入库。
