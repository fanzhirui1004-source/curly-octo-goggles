# R6 文风规范：以 PIML 课题组论文为范本

本文件用于第六轮改写。R5 稿在可读性改写之后偏于口语化和讲解式：短句多，"即""称为"一类解释性插入多，并夹有"做法""有的""却""多得多""反过来""那部分"等口语词。本轮的目标是把正文改回学术书面语，行文向郭旭、刘畅、杜宗亮等（以下简称 PIML 课题组）的论文靠拢，同时不回到自造术语。本轮只调整表述与段落组织，不改动内容、数值和结论。

R5_BRIEF 第 2 节（硬性约束）和第 3 节（术语）继续有效。本文件第 4.1 节说明它与 R5 第 1 节（文风规则）的关系。

## 0. 依据、方法与现稿诊断

### 0.1 依据

| 记号 | 论文 | 本稿中的引用名 |
|---|---|---|
| [H22] | Huang et al. 2022，Extreme Mech. Lett. 101887，PIML 拓扑优化 | Huang et al. (2022) |
| [H23] | Huang et al. 2023，Extreme Mech. Lett. 102041，PIML 增强的子结构法 | Huang et al. (2023) |
| [H24] | Huang et al. 2024，J. Mech. Phys. Solids 105893，基于力学的无数据 PIML | Huang et al. (2024) |
| [J26] | Jiang, C. et al. 2026，Compos. Struct. 120865，几何对称增强的轻量网络 | Jiang, C. et al. (2026) |
| [G26a] | Guo et al. 2026a，Comput. Methods Appl. Mech. Eng. 118955，三次 Bézier 边界插值 | Guo et al. (2026a) |
| [Z26] | Zhang et al. 2026，Int. J. Mech. Sci. 112007，问题无关迁移学习（PITL） | Zhang et al. (2026) |
| [G26b] | Guo et al. 2026b，arXiv 2607.22019，PIML-OFEM，文末附中文摘要 | Guo et al. (2026b) |

### 0.2 方法

四组分析员各自通读上述论文全文，统计了句长、人称、连接词频次、定义方式、结果表述、图表引用和术语。本文件抽查了其中引用的原文短例，并以 [G26b] 文末的中文摘要（11 句）作为中文语体的参照。这是课题组成员撰写的唯一一段中文原文。文中英文短例均为原文，不超过 25 词，"…"表示省略。[H23] 的文本提取丢失了空格，引用时已复原。

### 0.3 现稿诊断

统计范围为正文（不含公式、表格、题注和列表）。中文句子以"。"和"；"切分，英文以句号和分号切分。

| 指标 | PIML 论文 | 现稿 EN | 现稿 CN |
|---|---|---|---|
| 平均句长 | 引言约 29–34 词（[Z26] 21.6，[J26] 24.2）；算例约 19–35 词；结论约 23–37 词。[G26b] 中文摘要平均 62 字/句（43–95 字） | 17.9 词（引言 21.6，方法 16.4，算例 17.4，讨论 18.1） | 34.1 字（方法 30.2，算例 33.4，讨论 36.4） |
| 短句与长句 | ≥35 词的句子约占四分之一（[H24]、[G26a]）；≥40 词约占三分之一（[H22]、[G26b]） | ≤15 词的句子占 40%；≥35 词的仅占 1% | ≤30 字的句子占 48%；≥80 字的仅占 2% |
| 成对数值 | "respectively"每篇 9–18 次 | "respectively" 1 次 | 多拆成两句 |
| 解释性插入 | "(i.e., …)"较多（[H23] 13 次），多用于符号或范围限定 | "that is," 5 次，"called" 11 次 | "，即"31 次，"即为"4 次，"称为"9 次 |
| 句间回指 | 少，衔接靠从句与连接词 | 以 This、These、It 开头的句子 93 个（其中 This 37 个） | 以"这些""这一""该"开头的句子 91 个；全文"这些"61 次，"这一"41 次 |

结论：现稿句子明显短于 PIML 论文。R5 规则 1（一句一论断，EN 15–30 词，CN 30–60 字）把条件、原因和结果拆成了独立短句，再用"这些""这一""因此"回指衔接，因而读起来像讲解。PIML 论文把条件、原因和结果放进同一个复句，靠从句和连接词衔接。

## 1. 一页概要：PIML 论文的文风特征

| 方面 | PIML 的写法 | 原文短例 | 对本稿的要求 |
|---|---|---|---|
| 句长与句式 | 以多分句的长句为主：前置状语（Since / In order to / Under …）+ 主句 + 后置分词或 which / where 从句。结果段的句子较短 | "However, existing methods struggle to enforce strict physical consistency, limiting the accuracy and generalization of the machine learning models." [J26]；"The classical FEM analysis takes 12.39 s; the PIML-OFEM analysis takes 2.64 s, of which solving the condensed linear system takes only 1.04 s." [G26b] | 合并短句。条件与原因写入从句，并列数值用分号或"分别" |
| 人称与主语 | 被动语态为主。"we"只用于提出、选择、假设和推导。自称用 "in the present work""this paper""the proposed method"。方法缩写可作主语 | "In the present work, we recast the PIML approach in a more general substructure-based FEA framework" [H23]；"a projection-correction strategy is introduced to enforce the rigid-body constraint" [J26] | 中文用"本文""该方法"，不用"我们"。英文用被动语态和 "this paper"，NICE 可作主语 |
| 连接与衔接 | Therefore、Furthermore、In addition、Consequently；However 常置于句中；列举用 First / Second / Finally；解释用 "This is due to …"、"… because …" | "This deviation arises because the linear interpolation of boundary displacements in the substructure method inherently smooths the displacement field." [Z26] | 连接词放在复句内部，不靠句首"这些""这一"回指。"值得注意的是"仍按 R5 禁用（见 4.1） |
| 定义与记号 | 方程之后用 "where … denote …, respectively"；下标用 "denoted by"；命名用 "referred to as""termed""is called"，各用一次。缩写可在各节首次使用时重新展开 | "the nodes of each substructure are divided into internal nodes and boundary nodes, denoted by subscripts i and b, respectively." [H24]；"we partition the domain into a set of non-overlapping subdomains, referred to as substructures (Fig. 1b)." [Z26] | 术语只在定义处命名一次，用"记为""以下称为"。不对普通词加"即……"释义 |
| 方程 | 引入句以 "can be obtained as:""is given by:""yields:" 结尾；式后接 "where …"，再用一句说明其物理含义 | "Further substituting Eq. (3) into Eq. (2), the condensed stiffness matrix K_s^j that only includes the DOFs of the boundary nodes can be obtained as:" [H24]；"where ue and fe are the nodal displacement and force vectors, respectively." [Z26] | 现稿"式中，……"的写法已符合。引入句写"……可表示为""……为" |
| 结果表述 | 先给本方法的数值和参考值，再给相对误差或倍数。百分数多保留两位小数。定性比较用 "almost the same as""consistent with"，七篇论文中未见 "agree well"。数值之后一般接一句原因 | "The relative difference of the corresponding structural compliance values is 6.20% (C_EMs = 89.46 and C_PIML = 83.89)." [H24]；"This reference solution yielded a compliance of CRef = 430.59, corresponding to a relative error of 1.2%." [Z26] | 成对数值写"分别为……和……"，保持原有小数位。原文给出原因时，在数值之后紧接一句原因 |
| 图表引用 | 以 "as shown in Fig. N" 为主，也以图作主语（"Fig. 3 presents …"）。题注标题为名词短语，不写结论；表题位于表格上方 | "Fig. 3 presents the flowchart of the PIML model for two-dimensional substructures." [H24]；"The corresponding optimized designs are shown in Fig. 21(a)–(d)." [G26a] | 中文写"如图 N 所示""图 N 给出了……"。图 1、图 11 的标题改为名词短语 |
| 局限 | 平实陈述适用范围（"only … is considered"），紧接原因或补救，再提今后工作 | "In the present work, only linear displacement interpolation along the boundary of substructure is explored. This, however, will overestimate the stiffness" [H23]；"This issue can be alleviated by increasing the number of substructures, as shown in Table 2." [H24] | 第 6.3 节大体符合。每项局限统一按"范围、后果、补救或今后工作"组织 |
| 结论 | 全用散文：先以"本文提出了……"概括方法与核心思想，再写主要结果（多数带数值），最后写局限与展望 | "In this work, a mechanics-based data-free Problem Independent Machine Learning (PIML) model for large-scale structural analysis and design optimization is proposed." [H24]；"There remain several issues in this method worth further investigation." [G26b] | 第 7 节首句改为"本文针对……提出了……"，末段以展望收束 |
| 段落走向 | 方法节首段依次写上一步的结果、"然而"引出的困难、"为此"给出的处理和下文安排。算例依次写目的、设置与图、结果、与基准比较、原因、限定 | "To address these issues, this section proposes a symmetry-enhanced lightweight learning scheme …" [J26]；"To verify the accuracy of the proposed method, a cantilever beam … is investigated." [J26] | 第 3.2 节和第 5.3 节的首段按此重排（见第 6 节示例） |

中文语体可参照 [G26b] 的中文摘要。其句子平均 62 字，多为"为……，……""由于……，……可……""该方法……，并通过……，构造……"一类复句，例如"该方法仅保留每个子结构的角节点自由度，并通过在扩展域上求解局部边值问题，构造……数值基函数"。该摘要也使用"显著提升""高度吻合""新范式"等评价词，本稿不采用这些词（见第 5 节）；摘要中两处用"我们"，本稿仍按 R5 用"本文"。

## 2. 结构范式与逐节对照

### 2.1 PIML 论文各部分的段落走向

**引言（6–9 段）**

1. 背景：说明问题的工程意义与计算代价，1–2 段。
2. 已有方法：按方法族概括，成组引用，例如 "multi-resolution topology optimization methods [13–17], design variable reduction approaches [18,19] …" [H23]。1–2 段。
3. 机器学习方法：先述端到端方法，再述面向有限元子过程的方法。课题组的前期工作放在综述末尾，以第三人称叙述（"Huang et al. proposed …"）。1–3 段。
4. 缺口：以让步句开头（"Despite the fact that …, there still exist some challenging issues …" [H23]），问题用 First / Second / Finally 或 (1)(2)(3) 在行内列举。1 段。
5. 本文工作：以 "In the present work …""The central idea is …" 开头，可附一个代表性结果。1 段。
6. 贡献：多数论文不单列。[H24] 和 [G26b] 单列时写成散文（"… can be summarized as: (1) …"，"embodied in the following four aspects. First, we …"）。七篇论文均不用项目符号列表。
7. 结构安排：1 段，每节一句，或用分号串联（[J26]）。

**方法**

- 先重述经典子结构法：自由度按边界与内部分块，写出凝聚方程、凝聚刚度矩阵、装配，以及形函数形式 \(K_s=N^TKN\)。每个式子之后紧接 "where …"。
- 每个主要的节以一段动因开头，依次写上一步的结果、"However"引出的困难、"To address / In order to"给出的处理和下文安排。
- 网络写得朴素（"just an ordinary feedforward neural network" [H23]），网络的设计选择以力学要求论证（刚体运动、单位分解）。
- 训练细节平铺直叙（学习率、样本划分）。流程用流程图或文字表达，七篇论文都没有伪代码框。
- 用 "Remark" 段说明推广或注意事项。

**算例**

- 节首段先逐一说明各算例的目的（First … Subsequently … Finally …），再给共同设置：材料、无量纲约定、优化器、停止准则和硬件。
- 每个算例依次写目的、设置（几何、边界条件、载荷、离散）与图、定性结果（构型图）、与基准的定量比较、"This is due to …"一类解释、残差或局限及其补救，最后一句小结。
- 基准分两层报告：一是同一边界假设下的精确解（[J26] 的 "Sub"，[H24] 的 EMsFEM），二是全尺度解，以此区分学习误差与边界假设误差。

**讨论与结论**

- 七篇论文都没有独立的"讨论"或"局限"节。讨论分散在各算例中，局限放在结论末段或 Remark 中。
- 结论为 3–6 段散文，依次写方法与核心思想、主要结果、局限与展望。

### 2.2 逐节对照

下表只涉及表述与段落组织，不涉及内容、数值和结论。凡调整一处，EN 与 CN 须同步，段落数保持一致。

| 现稿位置 | PIML 范式 | 现稿状况 | 调整建议 | 优先级 |
|---|---|---|---|---|
| 摘要 | 背景、缺口、"本文提出"及核心思想、机理、结果。[H24]、[G26a] 不含数值，[H22]、[H23]、[J26] 含一个代表性数值 | 结构已相符。EN 恰为 250 词 | 不改结构。润色不得增加词数 | 低 |
| 1 引言 ¶1 | 背景 | 内容相符；"却"一处 | 删"却"，合并短句 | 中 |
| 1 ¶2 子结构法 | 已有方法及其代价 | 内容相符，句子过碎 | 见示例 6.1 | 高 |
| 1 ¶3 学习组件模型 | 已有机器学习方法和缺口 | 缺口分散在各句中；有"有的""做法" | 以让步句承接上文，把切割胞元面临的两点困难写成"一是……；二是……" | 高 |
| 1 ¶4 本文方法 | "In the present work … The central idea is …" | 相符 | 首句"本文提出……"之后接"其基本思路是……"；把网络性质的三个短句合为一句 | 中 |
| 1 ¶5、¶6 性质、柔度与灵敏度 | 方法的优点写在方法节或"本文工作"段 | 相符；"因此"连用 | 合并短句，"因此"不连用 | 中 |
| 1 主要工作 | 不用列表；单列时写成散文"……以下四个方面。首先……其次……" | 项目符号列表 | 改为一段散文，四项依次以"首先""其次""再次""最后"引出，节号全部保留 | 高 |
| 1 结构安排 | 每节一句或分号串联 | 偏长，含算例细节 | 用分号串联各节；第 5 节的算例清单保留，并入一句 | 中 |
| 1.1 相关工作 | 不单设一节；PIML 工作以第三人称、按其原用语叙述 | 单设一节；描述 PIML 时用"学习子结构""形函数标签""得不偿失""多得多" | 保留该节。每段首句点明方法族和本段要点；按第 3 节的对照表用 PIML 的原词；删除口语词 | 高 |
| 2 模型与设计问题 | "Revisiting the substructure method"：分块、凝聚、装配，"where … respectively" | 已相符 | 节首可加一句本节安排；合并短句；第 2.4 节的假设列表保留（各条有编号，供后文引用） | 低 |
| 3 节首 | 节首给出本节安排 | 内容相符，但由 10 个短句组成 | 合为 4–5 句，依次写算子定义、性质、网络、修正、训练与施加 | 中 |
| 3.1 | 推导、"where"、物理含义 | 相符 | 合并短句。"称为测试位移"只出现这一次 | 低 |
| 3.2 | 动因段：目标、"然而"困难、"为此"处理 | 首两段句子过碎 | 见示例 6.2。其余各段写每个设计选择时，先写其目的，再写实现 | 高 |
| 3.3、3.4 | 动因段、推导、Remark | 相符 | 段首以"由式 (7) 和式 (14) 可知……"直接引出动因；注 1 保留 | 低 |
| 3.5 训练 | 平实陈述超参数 | 相符 | 合并短句 | 低 |
| 3.6 施加 | 流程图或文字 | 算法 1 为编号列表 | 保留。算法步骤不属于叙述性列表 | 低 |
| 4 误差分析 | 不单设一节，最接近 [H24] 第 2 节和 [G26a] 第 2.4 节；命题之后用一句说明物理含义 | 相符 | 节首段合并短句。每个命题之后保留"因此，……"一句 | 低 |
| 5 节首 | 先说明各算例的目的与顺序，再给共同设置 | 以结果数值开头 | 先写各节的目的与顺序（5.1–5.2 设置与验证，5.3–5.5 单胞元，5.6–5.8 装配，5.9 成本，5.10 设计），现有结果句保留在其后 | 高 |
| 5.1、5.2 | 共同设置，参考解的验证 | 相符 | 合并短句 | 低 |
| 5.3 | 目的或图、结果（基准在前）、分组、总体、极值 | 以结论句开头，数值句过碎 | 见示例 6.3 | 高 |
| 5.4 | 目的、结果、解释 | 首段用"前两个方面……第三个方面……"讲解式列举 | 改为"本节从……、……、……和……四个方面考察修正前的误差"一句 | 中 |
| 5.5、5.6 | 相符 | 相符；5.6 首段已是目的句 | 合并短句 | 低 |
| 5.7 | 目的、设置、结果、解释 | 以结论开头 | 首句改为"为单独考察……，本节……"，结论移至段末 | 中 |
| 5.8 | 背景，"为检验……" | 相符 | 合并短句 | 低 |
| 5.9 | 目的、设置（硬件）、结果、耗时分解、限定 | 以结论开头 | 首句改为"本节比较……的计算成本"，结论放在表 5 之后 | 中 |
| 5.10 | 目的、设置、结果、力学解释、与基准比较 | 相符 | 合并短句。"对照运行"等名称各定义一次 | 低 |
| 6 讨论 | 不单设一节 | 分为 6.1–6.3 三小节 | 保留三小节。6.1、6.2 改为论点先行的复句（示例 6.4）；6.3 每项局限按"范围、后果、补救或今后工作"组织 | 中 |
| 7 结论 | 依次写"本文提出了……"、结果、展望 | 首句为"NICE 是一种……" | 首句改为"本文针对……提出了……"；末段以展望收束 | 中 |
| 图表题注 | 标题为名词短语；题注长短不一（[H23] 约 11 词，[G26a] 中位数约 30 词，最长 147 词） | 图 1、图 11 的标题是判断句 | 两图标题改为名词短语，例如"图 11. 柔度误差、灵敏度误差与被测胞元应变能占比的关系"。题注其余文字不删 | 低 |
| EN 正文中的图号 | "Fig. N"，从不写 "Figure N" | "Figure N" | 可选：正文改为 "Fig. N"。构建脚本只解析题注行 `**Figure N. …**`，题注行不得改动 | 低 |

### 2.3 不照搬的结构特征

- 不删除第 1.1 节"相关工作"和第 6 节"讨论"。这两节在 CMAME、SMO 中常见，R5 结构也已获作者认可，本轮只改写法。
- 不加离线成本表（[Z26] 表 2 的做法）。这违反硬性约束。
- 不在方法节另列优点清单（[H22] 第 4.2 节、[H23] 第 2.3 节的写法），以免与"主要工作"重复。
- 不模仿其在多篇论文之间逐字复用的套句，例如停止准则和硬件说明。
- 不写规模性的标题式结论，例如在笔记本电脑上求解十亿个设计变量一类的表述。
- 不把题注删减为只有标题。本稿部分数值只出现在题注中，例如图 11 的 0.963 至 1.004 倍、图 13(c) 的 1.04%、0.76% 和 1.72%，删去会丢失内容。

## 3. 术语对照

### 3.1 对照表

建议分三类：**沿用**（双方一致，或本稿用语已与之相同）、**保留本稿**（概念不同，或本稿用语更准确）、**并列说明**（首次出现时写出 PIML 的对应说法，此后用一种）。

| # | 概念 | PIML 用语（出处） | 本稿现用（EN / CN） | 概念是否相同 | 建议 |
|---|---|---|---|---|---|
| 1 | 被凝聚的对象 | substructure（[H23] 67 次，[H24] 166 次，[G26b] 228 次）；superelement 只用于他人方法 | cell / 胞元（几何对象）；superelement / 超单元（引言、关键词）；learned substructure / 学习子结构 | 凝聚对象相同；"胞元"指几何对象 | 并列说明：引言首次写"子结构（超单元）"，此后凝聚对象称"子结构"，几何对象称"胞元"。关键词保留 superelement |
| 2 | 静力凝聚 | static condensation [H24][J26][G26b]；Schur complement [G26a][Z26] | static condensation / 静力凝聚；the Schur complement / 即 Schur 补（R5 已定，首次） | 相同 | 沿用 |
| 3 | 单个子结构的凝聚刚度矩阵 | condensed stiffness matrix（[H23] 25 次，另见 [H24][G26a][G26b]）；[Z26] 用 export stiffness matrix | condensed stiffness (matrix) \(S\)、\(\widehat S\) / 凝聚刚度矩阵 | 相同 | 沿用；不用 export stiffness |
| 4 | 装配后的整体矩阵 | global condensed stiffness matrix [H23][H24][G26b] | assembled matrix \(\mathbb K\) / 装配刚度矩阵、装配系统 | 相同 | 并列说明：第 2.3 节式 (3) 处写"整体凝聚刚度矩阵（装配刚度矩阵）\(\mathbb K\)"，此后任选一种并全文统一 |
| 5 | 保留的自由度 | boundary nodes / DOFs，下标 b [H24][G26a]；经插值缩减后为 corner nodes [H23][G26b]、export nodes [Z26]、control points [G26a]；动词用 retain [G26a][G26b] | retained (master) DOFs \(P\) / 主自由度 | **不同**。本稿 \(P\) 由胞元表面上各正面积材料片所含节点的自由度和切割平面单元的全部自由度组成；后者多数不在子结构的界面上 | 保留"主自由度"，不得改称"边界自由度"。第 2.2 节定义处并列说明一次："其中胞元表面部分为相邻胞元之间的界面自由度，对应经典子结构法中的边界节点自由度"。叙述 PIML 时用其原词"边界节点自由度""角节点自由度" |
| 6 | 被消去的自由度 | internal nodes / DOFs [H24][Z26]；interior nodes / DOFs [G26a][G26b] | interior DOFs \(I\) / 内部自由度 | 相同 | 沿用 |
| 7 | 由保留位移求内部位移 | "the interior DOFs are subsequently recovered via numerical shape functions" [G26a]；recover / recovery 33 次 [G26b] | displacement recovery / 位移恢复；exact recovery \(E\) / 精确位移恢复 | 相同 | 沿用（与 PIML 的用词一致） |
| 8 | 由保留自由度到全部自由度的映射矩阵 | (multiscale) numerical shape function matrix \(N\)，\(K_s=N^TKN\) [H24][G26a][J26]；numerical basis functions [G26b] | recovery operator \(F\)，\(\widehat S=F^TKF\) / 位移恢复算子 | 数学上对应：精确位移恢复 \(E\) 的内部块 \(E_I=-A^{-1}K_{IP}\) 即 [G26a] 中"由边界节点位移映射到内部节点位移"的 \(N_s\)。但本稿不形成 \(F\)，网络输出位移 \(Fq\)，而非 \(N\) 的各列 | 保留"位移恢复算子"。第 3.1 节首次出现处并列说明一次："\(\widehat S=F^TKF\) 与子结构法中由数值形函数矩阵计算凝聚刚度矩阵的形式 \(K_s=N^TKN\) 相同，但本文不形成 \(F\)"。不得把网络输出称为"形函数"，第 1.1 节正是以此区分两类方法 |
| 9 | 刚体运动 | rigid body displacements / modes [H23][Z26]；rigid-body invariance [J26][G26a]；physical constraints [H24]；hard constraints, satisfied by construction [Z26] | reproduces rigid-body motions / 精确再现刚体运动；刚体模态；由构造保证 | 相同 | 沿用。第 3.2 节可写"由网络结构严格满足（硬约束）"。不用"单位分解"：PIML 的单位分解只对应平移，且本稿另有 \(J_PF=I_p\) 的条件 |
| 10 | 在保留自由度上取给定值 | Kronecker-δ property（numerical basis functions）[G26b] | reproduces the retained displacements, \(J_PF=I_p\) / 在主自由度上等于给定位移 | 相同 | 保留本稿表述。可选：首次出现时括注"（相当于形函数的插值性质）"。Kronecker-δ 一词只见于一篇，不引入 |
| 11 | 边界位移的近似 | linear displacement interpolation along the boundary、linear boundary displacement assumption [H23][H24][Z26]；cubic Bézier interpolation [G26a]；oversampling [G26b] | 以少量自由度描述边界位移；第 5.7 节"以 \(r\) 次 Bernstein 多项式限制胞元表面位移" | 描述的对象相同；本稿的消融不是对其方法的复现 | 叙述 PIML 时用其原词："边界位移线性插值假设""三次 Bézier 插值""过采样"。本稿消融保留现有说法，并保留"本文不复现这些方法"一句（"做法"改为"方法"） |
| 12 | 对整个结构建立细尺度模型 | full-scale analysis（[H24] 25 次）；full-scale FEA [Z26]；direct FE analysis [G26a]；classical (fine-scale) FEM [G26b]；中文摘要用"直接有限元分析""细尺度有限元" | fine-scale analysis of every wall / 对全部胞元壁的细尺度分析；whole-lattice direct solution / 整体点阵直接求解（第 5.9 节路线 (a)） | 一般概念相同。"整体点阵直接求解"专指稀疏 Cholesky 路线 | 并列说明：引言中的一般概念写"对全部胞元壁建立细尺度模型的全尺度分析"，此后可称"全尺度分析"。路线 (a) 的名称保留 |
| 13 | 参考解 | 以 standard substructure method 为参考解，记为 Sub；该文第 2.1 节将子结构法与边界位移假设一并定义，算例在线性边界位移假设下划分子结构 [J26]；exact substructure method [G26a]；EMsFEM [H24] | exact static condensation / 精确静力凝聚，不缩减边界，其装配矩阵与先装配细尺度系统再凝聚所得的矩阵相同（第 2.3 节） | **不同**。PIML 的"标准子结构法"含边界位移假设，本稿的精确静力凝聚不含 | 保留"精确静力凝聚"，不得改称"标准子结构法"。如需对应，写"不缩减边界的精确静力凝聚" |
| 14 | "粗" | coarse-resolution mesh / element [H22][H23]；coarse mesh (the substructures) [H24]；coarse-scale、coarse space [J26]，均指子结构层级 | coarse grid / 粗网格；coarse-grid correction / 粗网格校正；coarse space / 粗空间，均指胞元内部两重网格的粗层 | **同词异义** | 保留本稿含义。子结构层级称"装配系统""主自由度装配系统"，不称"粗网格"。转述 PIML 时写"以子结构为粗单元"，不写"粗网格" |
| 15 | 细尺度 | fine-scale elements / mesh（全部论文）；中文"细尺度" [G26b] | fine-scale / 细尺度 | 相同 | 沿用 |
| 16 | 柔度 | structural compliance [H22][H24]；compliance | compliance / 柔度 | 相同 | 沿用；首次可写"结构柔度" |
| 17 | 灵敏度与设计变量 | sensitivity (analysis)，对设计变量（单元密度，SIMP）求导 [H22][H24][Z26] | thickness sensitivity / 厚度灵敏度，对角点厚度参数求导 | 用词相同，设计变量不同 | 保留本稿。不写"密度"。灵敏度定义处说明一次它不计网络对设计的依赖（R5 已定） |
| 18 | 训练数据 | samples、training samples、sample generation、dataset、test set（全部论文）；label 基本不用（仅 [Z26] 的 labeled data 和 [G26b] 的 label 少量出现） | training geometries / 训练几何；test displacements / 测试位移；checkpoint selection / 检查点选择 | 不同。PIML 的一个样本是一个材料分布及其精确输出；本稿每个几何含一组测试位移及其精确能量与灵敏度 | 保留本稿。第 1.1 节"无需形函数标签"改为"无需预先计算的形函数数据"，可括注 [H24] 的自称 data-free。全文不用"标签" |
| 19 | 损失函数 | loss function；mean square error [H22][H23]；mechanics-based loss function（最小势能）[H24] | loss / 损失函数（能量比的对数加灵敏度项） | 本稿损失以精确凝聚刚度矩阵归一化，与 [H24] 的无数据势能损失不同 | 沿用"损失函数"，不称"基于力学的损失函数" |
| 20 | 误差度量 | relative error（位移的 Frobenius 范数、柔度）；total strain energy 的相对误差 [J26]；relative elemental strain energy error [G26b] | relative energy error \(\varepsilon(q)\) / 相对能量误差（能量误差）；\(e_C\) 柔度误差；\(e_s\) 灵敏度误差 | \(\varepsilon(q)\) 是给定主自由度位移下凝聚应变能的相对误差，与 [J26] 的整体应变能误差相近，但不相同 | 沿用。首次写全称"相对能量误差"，此后称"能量误差" |
| 21 | 计算效率 | time cost、solution efficiency、average time per iteration、speedup (ratio)、N times、orders of magnitude、time breakdown | 耗时、快约十倍、每次设计迭代耗时 | 相同 | 沿用。中文可用"计算耗时""单次设计迭代平均耗时""求解效率"，数值写法不变 |
| 22 | 在线与离线 | online / offline（普遍使用）；中文摘要用"在线计算""在线构造" | at deployment / 部署时；数据生成和训练的一次性成本 | "部署"即在线阶段 | 并列说明：首次写"部署（在线计算）阶段"。不得据此增写离线成本或 GPU 机时 |
| 23 | 问题无关 | problem-independent machine learning (PIML)：所学映射不依赖于边界条件、载荷和设计域 | 无对应名称；本稿写"同一网络未作改动，即用于……" | 这是其方法的名称 | 不用"问题无关"描述 NICE，保留现有的事实陈述 |
| 24 | 对他人学习型子结构的泛称 | PIML、PIML-enhanced substructure、ML-enhanced methods | learned substructures / 学习子结构；learned cell / 学习胞元（本稿胞元） | — | 指课题组工作时写"PIML 方法"或"基于 PIML 的子结构方法"。"学习子结构"只作泛称，本稿胞元称"学习胞元" |
| 25 | 修正 | [G26b] 在全局平衡方程上进行少量 Jacobi 预条件共轭梯度迭代，修正重构的细尺度场 | two-grid correction / 两重网格修正（方法名保留"平衡校正"） | 不同：本稿的修正在胞元内部、主自由度位移固定的条件下进行，并位于凝聚之中 | 保留。不把二者等同，第 1.1 节现有的区分保留 |
| 26 | 装配映射 | location matrix \(G^j\) [H24]；Boolean matrix [J26]；assembly matrix [G26a] | Boolean assembly map \(B_m\) / 布尔装配映射 | 相同 | 沿用 |
| 27 | 自由度缩写 | degrees of freedom (DOFs) | DOFs / 自由度 | 相同 | 沿用。EN 在引言首次出现时展开 |
| 28 | 载荷 | 中文摘要用"荷载工况" [G26b] | 载荷 | 相同 | 可选。两者均为规范用语；若改用"荷载"，须在正文、附录、补充材料和图题中全部统一 |
| 29 | 附注 | Remark [H24][G26a] | 注 1 | 相同 | 沿用 |

### 3.2 必须避免混淆的五处概念差异

1. **主自由度不是边界自由度。** PIML 的边界自由度位于子结构界面上，且通常再经插值缩减为角节点或控制点。本稿的主自由度包括切割平面单元的全部自由度，其中多数节点不在任何界面上（第 5.8、5.9 节点阵中，这部分占未约束主自由度的 37% 至 50%）。"主自由度"一词保留，只在定义处说明其中的胞元表面部分与经典子结构法的边界节点自由度相对应。
2. **胞元表面不是子结构边界。** 胞元表面指参考立方体的六个面，即相邻胞元之间的界面。切割胞元的切割面以切割平面单元的形式保留，TPMS 壁面属于自由表面，不保留。因此不把"胞元表面"改写成"子结构边界"。
3. **"粗网格"同词异义。** PIML 用"粗网格"指子结构层级；本稿用于胞元内部的两重网格修正。本稿中"粗网格"只用于后者。
4. **位移恢复算子与数值形函数矩阵只是形式对应。** 二者给出同一形式的凝聚刚度矩阵，但本稿不形成矩阵，网络也不输出形函数。说明一次对应关系即可，不改名。
5. **精确静力凝聚不是 PIML 意义上的"标准子结构法"。** 后者含边界位移假设，前者不缩减边界。

### 3.3 叙述 PIML 工作时的用语

| 对象 | 建议写法（CN / EN） | 避免 |
|---|---|---|
| 课题组 | 第三人称："Huang 等 [ ] 提出了……" / "Huang et al. proposed …"，与其自引方式一致 | "PIML 发展得最为充分"中的"得"字结构，以及对其工作的评价性形容词 |
| 预测对象 | 子结构的数值形函数 / numerical shape functions of the substructure | "形函数标签" |
| 边界处理 | 边界位移线性插值假设；三次 Bézier 插值；过采样数值基函数与重叠有限元 | "以少量自由度描述"之外的自造说法 |
| 无数据训练 | 基于最小势能原理的无数据（data-free）训练 [H24] | "无需标签" |
| 对称性 | 利用子结构几何对称性的形函数特征解耦 [J26] | — |
| 迁移 | 问题无关迁移学习（PITL），用于复杂三维设计域 [Z26] | — |
| 与本稿的关系 | 只用文字描述两类方法在保留什么、学习什么上的差别 | 任何数值对比 |

## 4. 中文写作规则

### 4.1 与 R5 第 1 节的关系

- **取代**：R5 规则 1（一句一论断；EN 15–30 词；CN 30–60 字）由本文件第 4.2 节取代。
- **修订**：R5 规则 3 要求技术条件"放在从句或后续句中"。本轮改为优先放进同一句的从句。
- **保留**：R5 规则 2、4、5、6、7 继续有效。R5 列出的禁用模式全部继续有效，包括排比口号、格言式句子、破折号、反问、为修辞效果而写的"不是……而是……"、任何"首次"类声明，以及"值得注意的是""这正是""正是""换言之"和装饰性的"由此可见"。
- **请作者决定**：PIML 论文常用 "It is worth noting that"（[H23] 6 次，[G26a] 5 次），R5 已禁用"值得注意的是"。建议维持禁用。若作者希望语气更接近 PIML，可允许"需要指出的是"在全文出现至多 3 次，且只用于引出限定条件。

### 4.2 句长与句群

- **中文**：以"。"或"；"计，句均 45–60 字，多数句子在 30–90 字之间。30 字以下的短句不超过两成，只用于参数设定（如"参数界限为 \(0.18\le\tau\le0.69\)"）或段末结论。同一段中不连续出现三个 30 字以下的句子。超过 100 字的句子只用于行内列举（"（1）……；（2）……"），其余应拆分。
- **英文**：句均 22–30 词，引言和讨论取上限，算例取下限。15 词以下的句子不超过两成；35–50 词的句子可占一至两成；50 词以上的句子只用于列举。
- **组句原则**：一句仍只有一个主论断，但论断的条件、原因、方式和对比都以从句写入同一句，不再拆成以"这些""这一"开头的后续短句。
- **判断标准**：若后一句以"这些""这一""该""因此"开头，且只是补充前一句的原因、条件或数值，就与前一句合并。

### 4.3 复句组织

下表第二列为 PIML 的英文句式，第三列为对应的中文句式。用例取自本稿现有内容，不含新信息；用例只演示句式，个别用例略去了原句中的部分数值，改写正文时须保留原句的全部数值。

| 功能 | PIML 句式（出处） | 中文句式 | 本稿用例 |
|---|---|---|---|
| 目的—行动 | "To verify the accuracy of the proposed method, a cantilever beam … is investigated." [J26] | 为检验……，本节考察…… | 为检验单胞元的能量误差在装配后能否给出准确的柔度和局部灵敏度，本节考察由一个学习被测胞元和一个精确相邻胞元组成的两胞元装配。 |
| 困难—对策 | "In order to circumvent the above difficulty, we propose …" [H22] | 然而，……，因而……；为此，本文…… | 然而，切割胞元的主自由度数目大且随几何变化，每个主自由度对应一个网络输出的方案不能直接适用。 |
| 原因—结果 | "Since ML is only carried out on a coarse-resolution element …, once established, they can be used …" [H23] | 由于……，……可（须）…… | 由于每次设计迭代均改变全部胞元的几何，上述分解须在每次迭代中重新进行。 |
| 结果—解释 | "This is due to the fact that the linear deformation assumption … overestimates the stiffness of substructures" [H24] | ……，其原因在于…… | M1 的误差仅由 13.5% 降至 4.69%，其原因在于其误差有相当大的部分位于光滑区间以下的模态中（第 5.4 节）。 |
| 让步—缺口 | "Despite the fact that …, there still exist some challenging issues …" [H23]（不仿其 "remarkable"） | 尽管……，但……仍…… | 尽管 NICE 在各切割程度分组中的平均能量误差均约为 0.1% 或更低，但从未切割胞元到重度切割胞元，其误差仍增大约七倍。 |
| 成对数值 | "The corresponding R2 values for PIML, PIML-symm, and PIML-proj are 0.9901, 0.9940, and 0.9972" [J26] | A 与 B 的……分别为……和…… | NICE 与基础网络的平均能量误差分别为 0.074% 和 6.89%。 |
| 分号并列 | "The classical FEM analysis takes 12.39 s; the PIML-OFEM analysis takes 2.64 s …" [G26b] | ……；…… | 对于两个八胞元点阵，NICE 的一次分析分别耗时 79 s 和 105 s；整体点阵直接求解分别耗时 868 s 和 1,042 s。 |
| 后置结果 | "…, limiting the accuracy and generalization of the machine learning models." [J26] | ……，从而（因而）…… | 网络参数作用于单元局部节点位置、特征通道和网格层级，而不对应具体的自由度，因而约 \(6\times10^5\) 个参数的同一网络可用于主自由度数目各不相同的胞元。 |
| 范围限定 | "only linear displacement interpolation along the boundary of substructure is explored" [H23] | 本文仅考虑……；……仅在……条件下成立 | 与整体点阵细尺度迭代求解器的比较仅在两个八胞元点阵上进行。 |
| 局限—补救 | "This issue can be alleviated by increasing the number of substructures, as shown in Table 2." [H24] | 该限制可通过……消除 | 在预条件子中再增加一层或采用迭代粗求解，是消除该限制的自然途径。 |

### 4.4 连接词

| 关系 | 优先使用 | 限用或避免 |
|---|---|---|
| 因果 | 由于、因而、故、从而、因此（每段至多两次） | 因为、所以、于是 |
| 递进 | 此外、同时、进一步、并且 | 而且；句首的"还" |
| 转折 | 但、然而、而 | 却、不过、可是 |
| 对比 | 与……相比、相应地、反之、而 | 反过来 |
| 条件 | 在……条件下、当……时、若……则 | "只要……就"改为"只要……即" |
| 例举 | 例如、以……为例 | 比如、像 |
| 列举 | 一是……二是……；（1）（2）（3）；首先……随后……最后…… | "第一项是……第二项是……"一类讲解式列举 |
| 小结 | 综上；由此可得（仅用于推导结论） | 总之；装饰性的"由此可见"（R5 已禁） |

现稿中文"因此"50 次、"因而"21 次；英文 "therefore" 51 次、"also" 44 次，而 "respectively" 仅 1 次。改写时应把一部分"因此""also"并入复句，成对数值改用"分别""respectively"。

### 4.5 定义、记号与方程

- 术语在首次出现时以判断句定义，例如"X 为……""将……称为 X"，每个术语只命名一次，此后不再解释。也可在式后的"式中"从句中定义。
- "即"只用于两种情况：数学等价（如"精确凝聚刚度矩阵，即 Schur 补"）和同一对象的符号对应。不用"即"解释普通词。全文的"，即"由 31 处减至 10 处以内。
- 括号只用于缩写展开、符号和单个数值，不在括号里解释概念。现稿中的"（即施加 ghost 罚项的单元面）""（即带有主自由度的节点）""（小型输出网络）"应改为定义句，或在不影响理解时删去。
- 方程以"……可表示为""……为""……满足"引入，式后写"式中，A、B 分别为……"。英文用 "… can be written as"，式后用 "where … are …, respectively"。
- 英文缩写可在摘要、引言和首次使用的章节中各展开一次（PIML 惯例）。中文首次写"切割有限元方法（CutFEM）"。

### 4.6 数字与结果表述

- 不改动任何数值、单位、范围、小数位和千分位写法，"至"与"–"的用法也按现稿保留。
- 结果句先写被测方法的数值，再写基准值，最后写相对误差或倍数。只有原文给出全部数值时才这样写。
- 成对数值写"分别为……和……"，多组数值以分号并列。
- 倍数和百分数沿用现稿写法，如"为……的 N 倍""降低 N%""快约十倍"。
- 定性比较只用"一致""接近""相近""吻合"，并紧随数值。不用"高度吻合""显著""大幅"。
- 原文给出原因时，在数值之后紧接一句原因，用"其原因在于……"或"这是由于……"。

### 4.7 段落组织

- 段首句给出本段的论点、目的或研究对象（R5 规则 5 保留）。方法段依次写上一节的结论、"然而"引出的困难和"为此"给出的处理。算例段依次写目的、设置与图、结果、与基准比较、原因、限定。
- 每段 3–7 句，一段只处理一个问题。
- 图表引用写"如图 N 所示""图 N 给出了……""表 N 列出了……"。每句至多一个括号交叉引用（R5 规则 4）。
- 节末不写口号式总结。需要小结时，用一句有数值支持的判断。

### 4.8 口语词与讲解腔：清单与替换

行号指 `MANUSCRIPT_CN.md` 的行号。

| 现稿用词 | 次数与位置 | 替换 | 示例 |
|---|---|---|---|
| 做法 | 5（L19、L178、L232 两处、L514） | 方法、方案、处理方式 | "两种直接做法均不可行"改为"两种直接方法均不可行" |
| 有的 | 1（L19） | 部分、一类 | "有的模型直接将节点位移映射为节点力"改为"一类模型直接将……" |
| 却 | 5（L11、L444、L456、L516、L621） | 而、但、则 | "却只包含"改为"而仅包含"；"却不能满足"改为"但不能满足" |
| 多得多 | 1（L38） | 远多于 | "该方案的网络需要预测的输出远多于前者" |
| 得不偿失 | 1（L38） | 计算代价超过所节省的代价 | 叙述 [G26a] 的结论时用平实说法 |
| 反过来 | 1（L611） | 反之、另一方面 | — |
| 那部分 | 1（L212） | ……的部分、……分量 | "固定步数的修正所不能恢复的内部位移分量" |
| 很 | 4（L19、L71、L462、L500） | 较、极、相当、大 | "很大一部分"改为"相当大的部分"；"很少材料"改为"少量材料"。程度须与 EN 一致 |
| 合在一起 | 1（L607）；L182 的"耦合在一起"可保留 | 合并处理 | 见示例 6.4 |
| 间接疑问：有多、哪些、为何、如何、什么、何处 | 有多 1（L607）；哪些 2（L54、L607）；为何 2（L442、L506）；如何 4（L40、L128、L290、L442，L611 的"无论网络如何"可保留）；什么、何处（L40） | 改为名词短语："……的精度""……的位移模式""……的原因""……的方式（过程）""交换的内容" | "第三个方面是……为何会导致……"改为"……导致……的原因" |
| 因为 | 3（L180、L344、L551） | 由于、其原因在于 | — |
| 下面 | 1（L532） | 以下、本节 | "下面比较……"改为"本节比较……" |
| 可以看出 | 1（L611） | 由……可知、……表明 | — |
| 还 | 28 | 此外、亦、尚、仍 | 选择性替换；"此外还需要"改为"此外需要" |
| 都 | 24 | 均 | — |
| 只 | 46 | 仅（陈述句中）；"只有……才"保留 | — |
| 可以、需要 | 12、16 | 可、需（须） | — |
| "被"字句 | 52（含术语"被测胞元"） | 主动句或"经……""由……"；术语"被测胞元"保留 | "被切割平面穿过"改为"与切割平面相交" |
| ，即 / 即为 | 31 / 4 | 按第 4.5 节处理 | 见示例 6.2 |
| 称为、下称 | 9、1 | 每个术语只在定义处用一次"称为"；"下称"改为"以下称为" | — |
| 这些、这一 | 61、41 | 合并句子后减少回指；必要时用"上述""该""其" | — |
| 讲解式列举 | L442、L607 等 | "一是……；二是……"或"本节从……四个方面考察……" | 见示例 6.4 |

R5 已禁用的"把……当作""其实""一下""总是""直接给出"，现稿中已无。

### 4.9 自检清单

- 随机抽一段，30 字以下的句子不超过两成。
- 无"做法""有的""却""多得多""反过来""那部分""得不偿失"，也无"有多""为何"一类间接疑问。
- "，即"不超过 10 处；每个术语只用一次"称为"。
- 每段首句是论点、目的或研究对象。
- 数值、引用、式号、节号、表号与现稿一致（用脚本逐段比对）。
- EN 与 CN 的段落数一致；摘要不超过 250 词。
- 无与 PIML 的数值对比；无"首次"类声明；无 GPU 机时、数据集生成成本或离线训练成本。
- PIML 术语按第 3 节使用，概念不同之处没有照搬。

## 5. 禁止事项

1. **不抄录 PIML 的句子。** 本文件的英文短例只用于说明句式。改写稿中不得出现与任一 PIML 论文连续 8 词以上相同的字符串，通用术语组合（如 "the condensed stiffness matrix of the substructure"）除外。中文不得套用 [G26b] 中文摘要的原句。其在多篇论文之间复用的套句（停止准则、无量纲约定、硬件说明）同样不得照搬。
2. **不引入未经证实的夸饰词。**
   - 英文：novel、remarkable(ly)、significant(ly)、substantially、dramatically、drastically、seamlessly、elegantly、rigorously（作褒义时）、superior、excellent、outstanding、exceptional、truly、totally、undoubtedly、obviously、powerful、unprecedented、state-of-the-art、paradigm。不写 "fully demonstrates the effectiveness" 一类收束句。
   - 中文：新颖、新型（作褒义时）、显著、大幅、极大、优异、卓越、完美、突破、范式、高度吻合、充分证明、毋庸置疑、显然。
3. **遵守硬性约束（R5 第 2 节及本轮任务）。**
   - 不写与 PIML（Huang et al.、Guo et al.、Jiang et al.、Zhang et al. 的学习子结构）的任何数值对比。可以用文字描述其方法。
   - 不写任何 "first" 类声明，包括"首次""首个""率先""尚无文献""To the best of our knowledge"。
   - 不写 GPU 机时、数据集生成成本和离线训练成本；不写重复测量、机器负载、限频和共享主机。第 5.9 节"各路线均不包含数据生成和训练的一次性成本"一句保持原样，不扩写。
   - 摘要不超过 250 词。EN 摘要现为 250 词，润色不得增词。
   - 不增删、不改动任何数值、条件、限定语和结论；不改公式与 LaTeX 符号。式号、命题号、节号、表号和图号按现稿；引用链接逐字保留。
   - 保持构建脚本解析的 Markdown 结构（图块、表题行、引用链接），EN 与 CN 段落对齐。
   - 不改动 AI 使用声明、作者信息、CRediT、利益冲突声明和致谢。
4. **不照搬概念不同的术语（第 3.2 节）。** 不把主自由度写成边界自由度；不把子结构层级称为粗网格；不把精确静力凝聚称为标准子结构法；不把网络输出称为形函数；不用"问题无关"描述 NICE。
5. **不模仿其非母语错误**，如 "does can reduce""fairy good learning results""The reminder of the article""A complaint mechanism example""the existed PIML model"。
6. **不让自造术语回流。** R5 第 3.1 节已替换的词不得重新出现，例如延拓、分工、可改进性、试探场、槽位、潜空间、切割带。

## 6. 示例改写

以下四处按本规范改写。改写不改动任何数值、条件、引用和公式编号，核对结果附在本节末。

### 6.1 第 1 节第二段

**现稿（EN）**

> Substructuring lies between these two approaches. Each cell is treated as a superelement and statically condensed onto the retained (master) degrees of freedom through which it connects to its neighbours, supports and loads [Guyan (1965)](https://doi.org/10.2514/3.2874), [Irons (1965)](https://doi.org/10.2514/3.3027). The assembled system contains only the retained degrees of freedom. After the solve, the interior displacements of each cell are recovered and used to evaluate local design quantities such as the sensitivities. Static condensation is exact, but it requires a factorisation of the interior stiffness matrix for every cell geometry. For cut thin-walled cells, the retained degrees of freedom are those of all nodes on the cell faces and of all nodes of the elements intersected by the trimming plane, called cut-plane elements below. For the 80 validation geometries of this paper, a cell has 2,679 to 45,900 retained degrees of freedom, with a median of about \(2.4\times10^4\). Because every design iteration changes the geometry of every cell, these factorisations must be repeated at every iteration.

**改写（EN）**

> The substructure method lies between these two approaches. Each cell is treated as a substructure, or superelement, and statically condensed onto the retained (master) degrees of freedom (DOFs) through which it is connected to its neighbours, supports and loads [Guyan (1965)](https://doi.org/10.2514/3.2874), [Irons (1965)](https://doi.org/10.2514/3.3027). The assembled system therefore contains only the retained DOFs, and after it has been solved, the interior displacements of each cell are recovered to evaluate local design quantities such as the sensitivities. Static condensation is exact, but it requires a factorisation of the interior stiffness matrix for every cell geometry. For cut thin-walled cells, the retained DOFs comprise those of all nodes on the cell faces and those of all nodes of the elements intersected by the trimming plane, hereafter referred to as cut-plane elements; for the 80 validation geometries of this paper, a cell has 2,679 to 45,900 retained DOFs, with a median of about \(2.4\times10^4\). Since every design iteration changes the geometry of every cell, these factorisations have to be repeated at every iteration.

**现稿（CN）**

> 子结构法介于上述两类方法之间。该方法将每个胞元视为超单元，通过静力凝聚将其缩减到与相邻胞元、支撑和载荷相连的主自由度上 [Guyan (1965)](https://doi.org/10.2514/3.2874), [Irons (1965)](https://doi.org/10.2514/3.3027)。装配后的整体系统只含主自由度。求解后再恢复各胞元的内部位移，用于计算灵敏度等局部设计量。静力凝聚是精确的，但每种胞元几何都需分解一次内部刚度矩阵。对于切割薄壁胞元，主自由度包括胞元表面全部节点的自由度，以及与切割平面相交的单元（下称切割平面单元）全部节点的自由度。在本文的 80 个验证几何中，每个胞元的主自由度为 2,679 至 45,900 个，中位数约为 \(2.4\times10^4\)。由于每次设计迭代都改变所有胞元的几何，这些分解须在每次迭代中重做。

**改写（CN）**

> 子结构法介于上述两类方法之间。该方法将每个胞元视为一个子结构（超单元），通过静力凝聚将其缩减至与相邻胞元、支撑及载荷相连的主自由度上 [Guyan (1965)](https://doi.org/10.2514/3.2874), [Irons (1965)](https://doi.org/10.2514/3.3027)。由此装配的整体方程组仅含主自由度，求解后再由主自由度位移恢复各胞元的内部位移，用于计算灵敏度等局部设计量。静力凝聚是精确的，但对每种胞元几何均需分解一次内部刚度矩阵。对于切割薄壁胞元，主自由度由胞元表面全部节点的自由度和与切割平面相交的单元（以下称为切割平面单元）全部节点的自由度组成；在本文的 80 个验证几何中，单个胞元的主自由度为 2,679 至 45,900 个，中位数约为 \(2.4\times10^4\)。由于每次设计迭代均改变全部胞元的几何，上述分解须在每次迭代中重新进行。

**改动要点**

- 首次出现时写"子结构（超单元）"，与 PIML 的用语衔接（第 3 节第 1 项）。
- "装配后的整体系统只含主自由度。求解后再恢复……"两个短句合为一句，原因与后续步骤写入同一复句。
- 主自由度的定义与其数目合为一个分号复句。"下称"改为"以下称为"，"须……重做"改为"须……重新进行"。
- EN 在此处首次展开 "DOFs"，此后统一用缩写。

### 6.2 第 3.2 节前两段

**现稿（EN）**

> The network approximates the interior part of the exact recovery, that is, the map \(q\mapsto E_Iq=-A^{-1}K_{IP}q\). This approximation gives a displacement field and, through the strain energy of that field, the condensed stiffness on all retained DOFs (Section 3.1). A cell has 2,679 to 45,900 retained DOFs, and this set changes with the cut. Two direct approaches are therefore excluded. Forming the recovery matrix for every geometry costs one interior solve per retained DOF. A network with one output per retained DOF would have to change its output dimension with the cut.

> Three properties must hold exactly, because the relations of Sections 3.1 and 4 rest on them. The architecture provides all three by construction. The recovery is linear in \(q\) at fixed geometry. All nonlinearity is confined to a geometry branch, and every operation on the displacement features is linear. The recovery also reproduces the retained displacements and the rigid-body motions. The rigid-body motion is separated before the network and reconstructed after it, and the retained displacements are overwritten at the output (Eq. (9)). Finally, the corresponding nodal forces of Eq. (5) require the transpose of the recovery. This transpose is obtained by transposing the linear displacement path operation by operation (Appendix G.1). Section 5.2 verifies all three properties numerically.

**改写（EN）**

> The network approximates the interior part of the exact recovery, the map \(q\mapsto E_Iq=-A^{-1}K_{IP}q\). An approximation of this map gives both a displacement field and, through the strain energy of that field, the condensed stiffness on all retained DOFs (Section 3.1). A cell, however, has 2,679 to 45,900 retained DOFs, and this set changes with the cut, so that two direct approaches are excluded: forming the recovery matrix for every geometry costs one interior solve per retained DOF, and a network with one output per retained DOF would have to change its output dimension with the cut.

> As the relations of Sections 3.1 and 4 rest on three properties of the recovery, these properties must hold exactly, and the architecture provides all three by construction. (1) At fixed geometry, the recovery is linear in \(q\), since all nonlinearity is confined to a geometry branch and every operation on the displacement features is linear. (2) The recovery reproduces the retained displacements and the rigid-body motions, because the rigid-body motion is separated before the network and reconstructed after it, and the retained displacements are overwritten at the output (Eq. (9)). (3) The corresponding nodal forces of Eq. (5) require the transpose of the recovery, which is obtained by transposing the linear displacement path operation by operation (Appendix G.1). Section 5.2 verifies all three properties numerically.

**现稿（CN）**

> 网络所逼近的是精确位移恢复的内部部分，即映射 \(q\mapsto E_Iq=-A^{-1}K_{IP}q\)。该映射的近似既给出位移场，又通过该位移场的应变能给出全部主自由度上的凝聚刚度矩阵（第 3.1 节）。各胞元的主自由度在 2,679 至 45,900 个之间，且该集合随切割而变化。因此，两种直接做法均不可行。若对每个几何形成位移恢复矩阵，则每个主自由度都需要一次内部求解。若网络对每个主自由度各设一个输出，则网络的输出维数须随切割而改变。

> 以下三项性质必须精确成立，因为第 3.1 节和第 4 节的关系均以这些性质为前提。网络结构由构造保证这三项性质。其一，几何固定时，位移恢复关于 \(q\) 是线性的。非线性仅出现在几何分支中，作用于位移特征的运算均为线性运算。其二，位移恢复精确再现主自由度位移和刚体运动。刚体运动在进入网络之前分离，并在网络之后重构；主自由度位移则在输出端重新赋值（式 (9)）。其三，计算式 (5) 中对应的节点力需要位移恢复的转置。该转置由线性位移路径逐个运算转置得到（附录 G.1）。第 5.2 节以数值算例验证了这三项性质。

**改写（CN）**

> 网络用于逼近精确位移恢复的内部部分 \(q\mapsto E_Iq=-A^{-1}K_{IP}q\)；该映射的近似既给出位移场，又通过该位移场的应变能给出全部主自由度上的凝聚刚度矩阵（第 3.1 节）。然而，各胞元的主自由度为 2,679 至 45,900 个，且主自由度集合随切割而变化，因而以下两种直接方法均不可行：若对每个几何形成位移恢复矩阵，则每个主自由度均需一次内部求解；若网络为每个主自由度各设一个输出，则网络的输出维数须随切割而改变。

> 由于第 3.1 节和第 4 节的关系均以下列三项性质为前提，这些性质必须精确成立，网络结构由构造予以保证。（1）几何固定时，位移恢复关于 \(q\) 为线性：非线性仅出现在几何分支中，作用于位移特征的运算均为线性运算。（2）位移恢复精确再现主自由度位移和刚体运动：刚体运动在进入网络之前分离、在网络之后重构，主自由度位移则在输出端重新赋值（式 (9)）。（3）计算式 (5) 中对应的节点力需要位移恢复的转置，该转置由线性位移路径逐个运算转置得到（附录 G.1）。第 5.2 节以数值算例验证了上述三项性质。

**改动要点**

- 第一段按 PIML 方法节的动因段组织：先写目标，再以"然而"引出困难，最后用冒号引出两种不可行的方法。删去"，即映射"和"做法"。
- 第二段把"……必须精确成立，因为……"改为"由于……，……"的前置原因句。三项性质用（1）（2）（3）在行内列举，每项性质与其实现方式合为一句（[H23] 第 2.3 节和 [H22] 第 4.2 节的列举方式），不再用"其一""其二"接三个短句。
- "这三项性质"改为"上述三项性质"。

### 6.3 第 5.3 节第一段

**现稿（EN）**

> In every cut-severity group, the mean energy error of NICE is about 0.1% or less (Figure 5). Under traction loads, the group means of NICE are lower than those of the base network by factors of 71 to 117. The error of NICE still grows about sevenfold from uncut to heavily cut cells. Its group means rise from 0.016% for uncut to 0.109% for heavily cut cells, whereas those of the base network rise from 1.14% to 12.7% (Supplementary Table ST03b). Over all 80 geometries, the mean energy error is 0.074% for NICE and 6.89% for the base network. The largest single-geometry mean is 0.65% for NICE and 48.6% for the base network. Over the 5,120 sampled test displacements under traction loads, 99% of the energy errors of NICE lie below 0.69%, and the largest is 1.24% (Supplementary Table ST03).

**改写（EN）**

> Figure 5 shows the energy errors on the 80 validation geometries. In every cut-severity group, the mean energy error of NICE is about 0.1% or less, and under traction loads the group means of the base network are 71 to 117 times those of NICE. The error of NICE nevertheless grows about sevenfold from uncut to heavily cut cells: its group means rise from 0.016% to 0.109%, whereas those of the base network rise from 1.14% to 12.7% (Supplementary Table ST03b). Over all 80 geometries, the mean energy errors of NICE and the base network are 0.074% and 6.89%, and their largest single-geometry means are 0.65% and 48.6%, respectively. Under traction loads, 99% of the energy errors of NICE over the 5,120 sampled test displacements lie below 0.69%, and the largest is 1.24% (Supplementary Table ST03).

**现稿（CN）**

> 在各切割程度分组中，NICE 的平均能量误差均约为 0.1% 或更低（图 5）。在面力载荷下，基础网络各分组的均值为 NICE 的 71 至 117 倍。从未切割胞元到重度切割胞元，NICE 的误差仍增大约七倍。其分组均值由未切割胞元的 0.016% 增至重度切割胞元的 0.109%，基础网络则由 1.14% 增至 12.7%（补充表 ST03b）。在全部 80 个几何上，NICE 的平均能量误差为 0.074%，基础网络为 6.89%。单个几何平均误差的最大值，NICE 为 0.65%，基础网络为 48.6%。在面力载荷下的 5,120 个采样测试位移中，NICE 的能量误差有 99% 低于 0.69%，最大值为 1.24%（补充表 ST03）。

**改写（CN）**

> 图 5 给出了 80 个验证几何上的能量误差。在各切割程度分组中，NICE 的平均能量误差均约为 0.1% 或更低；在面力载荷下，基础网络各分组的均值为 NICE 的 71 至 117 倍。尽管如此，从未切割胞元到重度切割胞元，NICE 的误差仍增大约七倍，其分组均值由 0.016% 增至 0.109%，而基础网络的分组均值由 1.14% 增至 12.7%（补充表 ST03b）。在全部 80 个几何上，NICE 与基础网络的平均能量误差分别为 0.074% 和 6.89%，单个几何平均误差的最大值分别为 0.65% 和 48.6%。在面力载荷下的 5,120 个采样测试位移中，NICE 的能量误差有 99% 低于 0.69%，最大值为 1.24%（补充表 ST03）。

**改动要点**

- 首句以图作主语，交代本段的研究对象（[H24]、[G26b] 的写法）。该句只复述图 5 题注的内容，不含新信息。
- "仍增大约七倍"与其后的分组均值合为一句，以"尽管如此"表示让步。
- NICE 与基础网络的两组成对数值各改为一句"分别为……和……"。
- 各分组中 NICE 平均误差约为 0.1% 或更低这一判断，原文没有限定载荷类别，改写中同样不加限定；"在面力载荷下"只限定倍数一句，与原文相同。

### 6.4 第 6.1 节第一段

**现稿（EN）**

> Using the complete set of retained DOFs, without reducing the interface, separates two choices that a reduced substructure model otherwise combines. The first is which displacement patterns neighbouring cells can exchange, and the second is how accurately the interior responds to each of them. The network and the correction affect only the second choice. The ablation of Section 5.7 shows the role of the first choice. Even with exact interior displacements, restricting the cell-face displacements changes the assembled response. A polynomial degree that gives an accurate compliance can still leave an appreciable error in the sensitivity. In the two-cell assemblies of the cut cells examined, restricting the cell-face displacements produced larger errors in the local sensitivity than the interior approximation of NICE (Sections 5.6 and 5.7). This comparison was made at a fixed cell size and without the refinement of the partition or the enrichment that methods with reduced boundary DOFs use. The retained DOFs of the cut-plane elements carry the support of the plate of Section 5.10 directly. The homogenised model needs a separate constraint on the plane of the cut for this support. Where the cut surfaces carry no loads or supports, these DOFs enlarge the assembled system (Section 6.3). The solution of the assembled system takes most of the time of a NICE analysis (Supplementary Table ST12d).

**改写（EN）**

> Using the complete set of retained DOFs, without reducing the interface, separates two choices that a reduced substructure model combines: the displacement patterns that neighbouring cells can exchange, and the accuracy with which the interior responds to each of them. The network and the correction affect only the second choice, and the ablation of Section 5.7 shows the role of the first. Even with exact interior displacements, restricting the cell-face displacements changes the assembled response, and a polynomial degree that gives an accurate compliance can still leave an appreciable error in the sensitivity. In the two-cell assemblies of the cut cells examined, restricting the cell-face displacements produced larger errors in the local sensitivity than the interior approximation of NICE (Sections 5.6 and 5.7). This comparison was made at a fixed cell size and without the refinement of the partition or the enrichment that methods with reduced boundary DOFs use. In addition, the retained DOFs of the cut-plane elements carry the support of the plate of Section 5.10 directly, whereas the homogenised model needs a separate constraint on the plane of the cut for this support. Where the cut surfaces carry no loads or supports, however, these DOFs enlarge the assembled system (Section 6.3), whose solution takes most of the time of a NICE analysis (Supplementary Table ST12d).

**现稿（CN）**

> 采用完整的主自由度集合而不缩减界面，可将约化子结构模型中原本合在一起的两项选择分开。第一项是相邻胞元之间可以传递哪些位移模式，第二项是胞元内部对每种模式的响应有多精确。网络和修正只影响第二项选择。第 5.7 节的消融算例说明了第一项选择的作用。即使内部位移精确，限制胞元表面位移也会改变装配后的响应。能给出准确柔度的多项式阶次，仍可能留下明显的灵敏度误差。在所考察切割胞元的两胞元装配中，限制胞元表面位移引起的局部灵敏度误差大于 NICE 内部近似引起的误差（第 5.6 和 5.7 节）。这一比较在胞元尺寸固定的条件下进行，未采用缩减边界自由度的方法所用的分区加密或富集。切割平面单元的主自由度直接承担第 5.10 节板的支撑。均匀化模型则须在切割平面上另加约束，才能表示该支撑。在切割面不承受载荷或支撑之处，这些自由度会增大装配系统的规模（第 6.3 节）。装配系统的求解占 NICE 分析的大部分时间（补充表 ST12d）。

**改写（CN）**

> 采用完整的主自由度集合而不缩减界面，可将约化子结构模型中合并处理的两项选择分离开来：一是相邻胞元之间可传递的位移模式，二是胞元内部对每种位移模式的响应精度。网络和修正仅影响后一项选择，第 5.7 节的消融算例则表明了前一项选择的作用。即使内部位移精确，限制胞元表面位移也会改变装配后的响应，而能够给出准确柔度的多项式阶次仍可能留下明显的灵敏度误差。在所考察切割胞元的两胞元装配中，限制胞元表面位移引起的局部灵敏度误差大于 NICE 内部近似引起的误差（第 5.6 和 5.7 节）；这一比较在胞元尺寸固定的条件下进行，未采用缩减边界自由度的方法所用的分区加密或富集。此外，切割平面单元的主自由度直接承担第 5.10 节板的支撑，而均匀化模型须在切割平面上另加约束方能表示该支撑。另一方面，在切割面不承受载荷或支撑之处，这些自由度增大了装配系统的规模（第 6.3 节），而装配系统的求解占 NICE 分析的大部分时间（补充表 ST12d）。

**改动要点**

- 删去讲解式列举"第一项是……第二项是……"，改为冒号后的"一是……，二是……"。"有多精确"改为"响应精度"，"原本合在一起"改为"合并处理"。
- 消融算例的两条结论合为一个复句；比较的前提条件紧随比较结论，内容不变。
- 切割平面单元自由度的利与弊分别以"此外""另一方面"引出，与均匀化模型的对比写入同一句。

### 6.5 核对结果

以脚本逐段比对现稿与改写稿。比对项为数值、引用链接、LaTeX 公式与符号，以及节、式、表、图、附录编号；列举序号（1）（2）（3）不计入数值。

| 段落 | 数值 | 引用链接 | 公式与符号 | 编号 | 比对结果 | 句数 | 平均句长 |
|---|---|---|---|---|---|---|---|
| 6.1 EN | 3 个 | 2 处 | 1 处 | 0 处 | 一致 | 8 → 6 | 20.4 → 28.2 词 |
| 6.1 CN | 3 个 | 2 处 | 1 处 | 0 处 | 一致 | 8 → 7 | 36.5 → 44.6 字 |
| 6.2 EN | 8 个 | 0 处 | 2 处 | 6 处 | 一致 | 15 → 8 | 14.0 → 27.8 词 |
| 6.2 CN | 8 个 | 0 处 | 2 处 | 6 处 | 一致 | 16 → 9 | 26.8 → 47.1 字 |
| 6.3 EN | 18 个 | 0 处 | 0 处 | 3 处 | 一致 | 7 → 5 | 20.0 → 27.2 词 |
| 6.3 CN | 18 个 | 0 处 | 0 处 | 3 处 | 一致 | 7 → 6 | 42.0 → 50.0 字 |
| 6.4 EN | 5 个 | 0 处 | 0 处 | 5 处 | 一致 | 12 → 7 | 18.3 → 31.0 词 |
| 6.4 CN | 5 个 | 0 处 | 0 处 | 5 处 | 一致 | 12 → 7 | 33.3 → 57.7 字 |

四处改写的比对项与现稿完全相同。改写后英文平均句长为 27.2–31.0 词，中文为 44.6–57.7 字，与第 4.2 节的目标（英文 22–30 词，中文 45–60 字）基本相符。6.4 的英文段略高于目标，主要来自首句和均匀化对比句两个 35 词以上的复句。
