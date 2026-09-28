# 方法命名与论文定位建议（子代理 B，2026-09-28）

本说明只提供建议，未改动稿件标题。定位改写已直接落在 `MANUSCRIPT_EN.md` 的摘要、第 1 节、第 7.1–7.3 节和第 8 节，见第 (c) 部分。

## (a) 方法名称 / 缩写候选

名称均避开 "PIML" 及 "problem-independent"、"shape function" 一类字样，突出"凝聚算子 + 学习初值 + 多重网格修正"。

| 候选 | 全称 | 一句话理由 |
|---|---|---|
| **LiCS** | Learned-initialised Corrected Schur complement | 直接点明对象是 Schur 补作用，学习只负责初值，修正保证可控；短、好读。 |
| **NICE** | Neural-Initialised Condensation with Equilibrium correction | 三个要素（神经初值、静力凝聚、平衡修正）全在名字里；缺点是常用词，检索时易混。 |
| **SCALE** | Schur Complement Approximation by Learned Extension | 强调"逼近凝聚算子"而非"学形函数"；但未体现修正，可配副标题 "with multilevel correction"。 |
| **LEMC** | Learned Extension with Multilevel Correction | 最中性、最贴近方法结构（Ê 与 𝒲），适合作为正文中的简称。 |
| **CoLEX** | Corrected Learned EXtension | 简短，突出 F = 𝒲Ê 这一核心构造；与算子形式 \(\widehat S=F^TKF\) 对应清楚。 |
| **FT-Cond**（备选） | Full-Trace learned Condensation | 突出与降维边界方法的区别（完整保留迹）；但"full trace"在切割带情形下需额外解释。 |

个人倾向：正文用 **LEMC** 或 **LiCS** 作简称；若需要较醒目的名字，可用 **LiCS**。

## (b) 备选标题

当前标题：*Learned static condensation for cut thin-walled TPMS cells with equilibrium correction*

1. *Learned interior extension with multilevel equilibrium correction: a controllable approximation of static condensation for cut thin-walled TPMS cells*
2. *Approximating the Schur complement of cut thin-walled TPMS cells with a learned initial field and multilevel correction: compliance and thickness sensitivity*
3. *Accurate compliance is not accurate sensitivity: learned static condensation with multilevel equilibrium correction for cut TPMS cells*（强调主要科学发现；风格上偏醒目，若期刊偏保守可用 1 或 2）

## (c) 本次所作定位改动（章节：旧要点 → 新要点）

| 章节 | 旧要点 | 新要点 |
|---|---|---|
| 摘要 | 从"学习子结构须复现内部变形与传力"出发，提出几何条件神经位移延拓，再附加平衡修正 | 从静力凝聚/Schur 补出发：在完整保留空间上逼近凝聚算子，不做边界降维；网络给初值，固定的几何相关多重网格修正（Chebyshev + 三线性 Galerkin 粗解）完成它；无矩阵、线性、保持保留值与刚体；\(\widehat S\) 对称半正定、以精确 \(S\) 为下界、从不显式形成；误差仅在内部，可由网络和修正预算两方面减小；主要发现"柔度准确不等于局部灵敏度准确"提前到结果之前。所有数字句保持原文 |
| 1 节第 1 段末句 | "理解学习场误差如何影响两种响应" | "降阶胞模型须同时逼近内部场与返回力，并在分析与设计所需量上理解其误差"（不预设"学习"） |
| 1 节第 2 段 | 静力凝聚暴露两种近似选择；本文固定保留坐标，问学习场如何被修正 | 静力凝聚作为起点：切割薄壁胞保留坐标数以万计，显式形成 \(S\) 需逐坐标内解、隐式作用需每个几何一次内部分解；两条降本路线（缩减保留坐标 vs 在完整保留空间上逼近内部延拓），本文走第二条，问如何使凝聚算子的逼近"可控" |
| 1 节第 3 段（新位置） | 原在后文：修正使学习场可数值修正，HINTS/DeepONet/GMT 为背景 | 多重网格作为第二个起点：固定系数与步数的修正是线性映射 𝒲，\(F=\mathcal W\widehat E\) 的能量形式自动对称半正定、\(\widehat S\succeq S\)、保留刚体核；精度由初值与修正预算决定；混合神经求解器（HINTS、DeepONet、GMT）作对照；学习的角色＝为修正提供几何条件初值 |
| 1 节 PIML 段 | PIML 作为"学习拓展上述选择"的主线，本文被"置于学习形函数与变分子结构方法之内" | PIML 作为"相关路线"如实介绍（引文与描述全部保留）；明确承认能量构造 \(N^TKN\) 与刚体约束为共有要素（不把等变性、刚体约束当创新）；指出少量边界坐标带来紧凑粗模型的优点，以及与内部学习精度无关的边界模型误差，其大小视应用而定：对本文切割胞，角点线性边界在精确胞算子下柔度误差 78–85%，高阶边界柔度收敛快于灵敏度（引第 6.8 节已有结果）；本文保留完整空间，误差只在内部 |
| 1 节 CNEE/Parish 段 | "这些先例把本文置于学习形函数与变分子结构方法之内" | 删去"置于其内"的说法，改为"这些模型共享变分结构；本文由同一修正延拓同时定义拼装算子和恢复场，从而可把剩余误差追踪到柔度和局部灵敏度" |
| 1 节构造段 | 在完整保留空间上构造几何条件神经延拓 | 强调算子形式：以万计的保留坐标下显式形函数矩阵不可行，延拓为无矩阵、对保留位移严格线性、几何条件（非线性几何分支 + 多层线性位移分支）的算子；\(\widehat S\) 只作用不形成；经修正训练（solver-in-the-loop）并在目标中加入灵敏度项 |
| 1 节分析段 | 关系"解释平衡修正的力学作用及全局与局部量的不同精度要求" | 明确主要科学发现：柔度准确不蕴含局部厚度灵敏度准确（参与度加权掩盖弱参与胞误差），并解释修正如何同时降低两者 |
| 1 节贡献 | (i) 几何条件延拓；(ii) 误差分析；(iii) 平衡修正 | (i) 框架：学习初值 + 固定几何相关多重网格修正，在完整保留空间上逼近凝聚算子，变分性质与单侧界，误差可由两方面减小；(ii) 无矩阵、线性、几何条件的延拓算子，经修正训练且目标含灵敏度；(iii) 误差分析（原文保留）；(iv) 发现"柔度准确 ≠ 灵敏度准确"并由修正算子将两者降到 1% 以下 |
| 7.1 节 | 固定保留空间与 Ritz 解释；Bernstein 对比说明边界限制与内部修正作用于不同部分 | 增加：框架把"交换哪些位移模式"与"内部响应多准确"两个决策分开；与少坐标边界方法（引 Huang 2023 角点线性）的关系：边界误差与内部学习精度无关、换取更小粗模型；本文误差在内部、可由初值与修正预算减小；对切割胞的局部灵敏度，边界限制误差大于修正算子的内部误差（6.6、6.8 节），而对更重视紧凑粗模型的结构，权衡可能不同（措辞保持对 PIML 的公平） |
| 7.2 节 | 全局功与局部设计误差分离的机理 | 末尾加一句：这一分离正是式 (8) 灵敏度项的动机——能量项按控制柔度的范数约束内部误差，灵敏度项按刚度导数加权 |
| 7.3 节 | 修正的能量判据与预测/修正互补 | 首段末加一句：修正使学习场成为可控凝聚算子的初值而非独立近似；变分性质不依赖网络精度，算子精度也不只受网络限制 |
| 8 节 | 以"内部平衡是学习与修正的共同判据"开篇 | 以"在完整保留空间上，用几何条件学习初值 + 固定多重网格修正逼近静力凝聚"开篇；无矩阵、线性、以 \(S\) 为下界、不形成；无边界降维故误差只在内部、两方面可减；第二段首句点明主要力学发现"柔度准确不蕴含局部厚度灵敏度准确"。其余数字与结论原样保留 |

## 需协调人留意的事项

- 定位中提到的 PIML 后续"三次 Bézier 边界描述"工作（CMAME 456 (2026) 118955，见 `docs/PRIOR_ART_SCREEN_20260926_CN.md`）目前**不在**稿件参考文献中。本次未新增该引用（作者与条目未在稿件中核实），正文只以"higher-degree boundary polynomials"和已引用的 PIML-OFEM 过采样基来泛指更丰富的边界描述。若要点名 Bézier 工作，需补引文。
- 第 3.1 节仍有"This choice connects the learned representation to local shape-function methods [Huang et al. (2023)]"一句，第 6.8 节首句称"the assumption on which boundary-interpolation substructures rest"。二者不在本次可改范围内，措辞尚属公允，是否调整由协调人决定。
- 标题未改，候选见 (b)。
