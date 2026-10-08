<!-- 2026-10-01 文献调研（4 路检索 + 引用核实 + 汇总，工作流 wf_935c0bcd-747）。"我的推导"部分需自行证明；UNVERIFIED 条目需读原文。 -->

# 后续方向 A / B 的文献定位报告

> 依据：本次运行中 4 个检索视角（映射点阵、学习/降阶子结构、非精确 DD、无数据训练）的结果，以及逐条 verify 记录。凡是 verify 没有覆盖、或 verify 判为 claim_ok=false 的内容，都标为 UNVERIFIED 或单独列出。下文中"我的推导"是我在已有理论框架上做的推论，**不是文献原文**，需要你自己严格证明。

---

## 0. 先更正背景里的两处事实

1. **arXiv:2607.22019 不是 Guo 等人的 Bézier 论文。** 它是同组的 *PIML-OFEM*（Guo Y. 等 10 人，2D，U-Net，重叠有限元，投 Acta Mechanica Sinica）。Bézier 论文只有期刊版：CMAME 456 (2026) 118955，doi:10.1016/j.cma.2026.118955。P1 的参考文献需要相应修改。
2. **"能量形式自动给出 S_hat ≥ S"不是 NICE 首创。** 任何延拓 E 都满足 Eᵀ K E ≥ S，这就是调和延拓的能量最小性。PIML 系列早就用 K_s = Nᵀ K N：PIML-OFEM 的 Eq.8 引用的是 Huang et al. JMPS 2024，**不是** EML 2023。SCRBE 的 RB-Galerkin、CNEE 第 5 节的 Rayleigh–Ritz 阶梯也都具备这一性质。另外，Cao & Song (arXiv:2608.05437) 已经在全场层面公开论证"离散能量损失 ≡ 刚度范数监督"。P1 和后续论文都不应把这两点写成新贡献。

---

## 1. 一句话结论

- **方向 A（全局 B 样条体映射单胞）：有空位，但很窄，抢发风险中高。** 目前没有人做"3D、cut TPMS 薄壁、以 Jacobian 场为条件、对任意输入有 S ≤ Ŝ 的学习凝聚算子"。但"映射点阵 + 拉回材料 + 快速胞元算子 + DD"已经被 Antolin/Hirschler 组做到 2D unfitted TPMS（Bonilla Moreno 2026）。该文结论里**明确把"3D 用神经网络代理"列为下一步**。郭旭组手里同时有"NURBS 分区坐标映射点阵"和"PIML 学习子结构"（Xu et al. 2025 CS）两块拼图。所以必须尽快占位，而且要正面对比经典的"查表/MDEIM + 精确凝聚"基线。
- **方向 B（NICE 作为 BDDC/FETI-DP 的非精确子域/延拓求解器）：理论空位清楚，风险中等。** 已有的 ROM-DD（Hirschler 2024、Bonilla 2026）和 ML-DD 工作都没有"确定性单侧谱序 + 条件数定理"。最接近的 Tu–Zhang 2025 只给出双侧、概率性的界。不过，界的**形状**是 Li–Widlund 2007 的直接推论，审稿人很可能说"只是个推论"。真正的新意必须落在三处：最坏情形 ε 的可计算/可认证、带回退的 certified inexact BDDC、3D CutFEM 上的实测。还有一点：把 B 写成"一般非均质结构"是在夸大，NICE 只对它训练过的几何族有效。

---

## 2. 方向 A：全局 B 样条体映射单胞

### 2.1 前人做到哪一步

先分清两种完全不同的"B 样条"：

| 类型 | 代表 | B 样条/Bézier 的作用 |
|---|---|---|
| **边界插值**（超单元内部的运动学降维） | Guo et al. CMAME 2026 (118955) | 每个面用 bicubic Bézier 曲面插值子结构的**边界位移**（56 个控制点，168 DOF），作为 DeepONet 的输入，监督 MSE 训练。不涉及几何映射、Jacobian 或映射胞元。 |
| **全局体映射**（几何） | Elber/Antolin 样条复合，郭旭组 NURBS-PCM | x = Φ(ξ) 把参考 tile 映射为弯曲、扭转、渐变的胞元。拉回后得到 Ĉ = j·(∂ξ/∂x)·C·(∂ξ/∂x)，与你的 C~ 同构。 |

方向 A 属于后者，与 Guo 2026 只是名字撞上。写作时要在引言里用一段话把两者分开。

按分析方式分四条线：

1. **全尺度 FE / IGA + DD/ROM（Antolin–Buffa–Hirschler–Bouclier 线），最强竞争者。**
   - Hirschler, Antolin, Buffa 2022 (Comput Mech)：把拉回张量场 Ĉ^(s)(ξ) 在宏观尺度做多项式投影，与参考 tile 的积分查找表收缩，实现快速组装，并提到可用于灵敏度。这是**投影近似**，一致性误差可控，不是"精确并入 Jacobian"。**没做**：凝聚、降阶、求解器、ML；只处理贴体参数化 tile。
   - Hirschler, Bouclier, Antolin, Buffa 2024 (IJNME)：Eq.(51)–(54) 写出拉回双线性形式和多项式系数 A^(s)。用 greedy 选主胞元，其余胞元的局部算子取主胞元的线性组合（系数可为负），放进 inexact FETI-DP。预条件子是完整的块 LDLᵀ 型；KR 块三角只是文中列出的对照选项，不是本文方法。外层用 GMRES。规模为 2D/3D、数百万 DOF、串行。**没做**：TPMS/水平集、SPD 或序保证、条件数分析、优化、ML。胞元差异大时（刹车踏板约 30 个主胞元）加速比会消失。
   - Bonilla Moreno, Guarino, Antolin，CMAME 463 (2027) 119304（2026 年在线）：2D 水平集（含 2D TPMS）、任意阶映射、unfitted p-FEM（每个胞元一个高阶单元）、α 稳定化。fast assembly 把映射贡献与 trimming 贡献分开，在此基础上再对 trimming 部分用 MDEIM 降阶。每个胞元是一个 BDDC 子域，用 PCG 求解；17000 胞元约 30 s（笔记本）；弱扩展到 131072 子域，迭代 15–18 次。作者承认 ROM 下"stability can be delicate"。v3 勘误承认印刷版式 (20) 的两个逆 Jacobian 因子写反了。**没做**：3D、条件数理论、SPD 或序保证、灵敏度、优化、ML。
   - Guillet, Hirschler, Jolivet, Bouclier 2025 (JCP)：matrix-free 两级方法，细层光滑子带 ROM 局部解；20 万胞元、1.5B DOF、4096 进程。后续有非线性超弹性版本（CMAME 461:119202, arXiv:2603.10741）。
2. **映射均匀化（郭旭组理论根基）**：Zhu et al. 2019 JMPS，映射加渐近分析，依赖尺度分离和线性化。胞元大或 Jacobian 剧变时没有误差保证。
3. **映射参数作为设计变量 + ML（郭旭组应用线）**：Xu et al. 2023 CS 中，NURBS-PCM 的扰动系数就是设计变量，所以**"映射控制点作为设计变量"已有先例，不能当新意**。Xu et al. 2025 CS 把 PIML 子结构代理用于 3D 弯曲/扭转胞元的梯度点阵优化；他们的网络是否把映射或 Jacobian 作为输入，**UNVERIFIED**，投稿前必须读全文。Zhang et al. 2024 EML 72:102237 学"等参子结构形状 + 材料 → 数值形函数"，是"学习映射子结构"的直接先例，但大概率是角点决定的双线性等参映射（UNVERIFIED）。
4. **学习型几何条件 Schur 补**：Jiang et al. 2026 CNEE (arXiv:2608.02036)。超网络输出 L(g)，K = LᵀL，保证 PSD；提出"正则零空间必须包含刚体模态"。**几何只是低维显式参数**（椭圆孔的 a、b、θ），不是 Jacobian 场；目标是逼近 S 本身，没有 Ŝ ≥ S；3D 只有标量扩散的 2×2×2 装配；问题很小（2D 不超过 2241 个未知量）。

### 2.2 最接近的 5 篇及与我们的差异

| 文献 | 已做到 | 没做到（NICE-A 的差异点） | 威胁 |
|---|---|---|---|
| Bonilla Moreno et al. 2026 CMAME 119304 | 映射 + 水平集 TPMS + unfitted + MDEIM + 胞元级 BDDC | 只有 2D；无序保证，靠 α 稳定化；无理论；无灵敏度；无 ML，但已公开计划用 NN 做 3D | 高 |
| Hirschler et al. 2024 IJNME | 拉回 Ĉ^(s) + 主胞元 ROM + inexact FETI-DP，有 3D | 只处理贴体 tile，不能做 TPMS/水平集；无 SPD 或序；无优化 | 高 |
| Xu et al. 2025 CS 119330 | NURBS-PCM 弯曲/扭转胞元 + PIML 代理 + 3D 优化 | 是否以 Jacobian 为条件 UNVERIFIED；无单侧界或校正；非 cut TPMS 薄壁 | 高（叙事几乎重合） |
| Jiang et al. 2026 CNEE | 几何参数化学习 Schur 补 + PSD + 刚体零空间原则 + 误差界 | 低维参数而非场；无 Ŝ ≥ S；无校正；规模极小；无 3D 弹性 | 中高 |
| Hirschler et al. 2022 Comput Mech | Ĉ 多项式系数 A^(s)，紧凑的 Jacobian 编码 | 只做组装，不做凝聚 | 中（审稿人必问"为什么不直接查表 + 精确凝聚"） |

### 2.3 NICE-A 可以防守的新意（按强弱排序）

1. **3D、ghost-penalty CutFEM、薄壁 TPMS、映射胞元的学习凝聚算子**，对任意输入保持 PSD 和 Ŝ ≥ S̃（S̃ 为映射后的精确 Schur 补），并用**精确映射算子**做平衡残差校正。这是目前最干净的空位。
2. **以 Jacobian 场为条件**。可以直接借用 Hirschler 2022 的多项式系数 A^(s) 作为低维输入，网络比较就和经典查表在同一编码上公平对比。
3. **零样本/OOD 的"符号安全性"**。需要说清楚：Ŝ ≥ S̃ 只保证"只会偏刚、柔度只会被低估"，**并不保证精度**。OOD 时 ε 可能很大，但下界依然成立，精度靠校正来恢复。只能这样表述，不能写成"零样本仍然准确"。
4. **对映射控制点的灵敏度**：沿 ∂Ŝ/∂P 经 J → C~ 的链式法则，与已有的厚度场灵敏度合并。经典 ROM 文献只说"容易求导"，没有实现，也没有误差量化。
5. **失去尺度分离时，映射均匀化（Zhu 2019）误差曲线的定量标定**。附带贡献，成本低，审稿人也爱看。

不能主张的点：映射几何本身、映射控制点作为设计变量、能量形式的 SPD 和上界、拉回公式。

### 2.4 我识别出的三个技术陷阱（需要自己核实）

1. **刚体模态在映射下可能不再精确。** 离散刚度核为 {u_h : sym(∇_ξ u_h · J⁻¹) = 0} = {a + W Φ(ξ)}。如果 Φ 在胞元上是三次多项式，而背景网格是 Q2，转动 WΦ 不在离散空间里，K~ 的核就只剩 3 个平移。这会影响 P1 的"再现刚体运动"性质、两级 PCG 的刚体粗空间，以及方向 B 要求的公共零空间。建议用**等参化映射** Φ_h = I_{Q2} Φ（在 32³ 背景网格节点上插值），这样转动是精确的；Φ 与 Φ_h 之间的几何误差单独评估。这与 CNEE 的"零空间原则"正好构成显式区分点。
2. **水平集定义在参考坐标还是物理坐标。** 在参考坐标定义（样条复合思路）时，cut 几何对所有胞元相同，只有 C~ 变化，固定的几何专属多层校正可以全局复用。这是最省事、也最能体现 NICE 结构优势的设定。但这样**物理壁厚会随法向拉伸而变化**，厚度设计变量的意义也随之改变。在物理坐标定义时，需要同时以 J 和 cut 为条件，正好与 Bonilla 的"映射/trimming 分离"形成对照。建议先做前者。
3. **校正的 Chebyshev 区间和 ghost penalty 尺度都要按 C~ 重新估计。** 拉回后 K~_II 的谱和各向异性都变了。P1 的"区间覆盖谱时校正不增大能量误差"在映射下需要逐胞元重估 λ_max（几步幂迭代即可），或者从 J 的奇异值推出一个先验界。后者需要证明，因为 sym(G J⁻¹) 与 sym(G) 之间的 Korn 型比值并不平凡。ghost penalty 参数也应按局部 C~ 缩放，否则病态程度会随畸变增大。

### 2.5 必须先做的验证实验：零样本扭转测试

**目的**：用最低成本判断两件事。一是 J 条件化是否必要；二是 S ≤ Ŝ 加精确算子校正，能否在 OOD 下"优雅退化"。这个实验不需要重新训练，2–4 周可完成。

**实验变体**
- **V0（真零样本）**：直接用 P1 训练好的、未映射的 NICE 预测内部位移，校正和能量评估都用映射后的精确 K~。
- **V1（J 条件化）**：在温和映射上训练，在强映射上测试，即外推。
- **基线**：精确凝聚（参考解）；Hirschler 式"多项式 Ĉ + 查表 + 精确凝聚"；Zhu 2019 式映射均匀化（只在大胞元比例下比较）。

**映射族**（每族单参数扫描，同时记录每胞元的 κ(J) = σ_max/σ_min、det J 范围、L_cell·‖∇J‖/‖J‖）
1. 仿射（均匀拉伸 1:0.5–1:2，剪切 0–0.3）：每个胞元内 J 为常数，Hirschler ROM 在此情形精确，作为 sanity 对照。
2. **扭转**：绕 z 轴 θ(z) = τz，每胞元扭转角 0°, 2°, 5°, 10°, 20°, 30°。
3. 弯曲（圆柱映射）：R / L_cell = ∞, 20, 10, 5, 3。
4. 渐变缩放：相邻胞元尺寸比 1.0–1.5。
5. 随机三次 B 样条控制点扰动：幅度 δ / L_cell = 0–0.3，保证 det J > 0。

**指标**
- 每胞元，在力驱动方向和邻胞诱导方向上：能量误差 q^T(Ŝ−S̃)q / q^T S̃ q。
- **最坏情形 ε**：用广义 Lanczos 求 λ_max(S̃⁻¹Ŝ) − 1；同时数值检查 λ_min ≥ 1 − 1e-10，作为保证的 sanity check。
- PSD 检查；平移和转动的再现误差（等参化前后对比）。
- 达到 0.1% 所需的校正循环数；Chebyshev 区间实际覆盖率。
- 点阵级（8 / 24 / 110 胞元）柔度误差；PCG 迭代数。
- 厚度和控制点灵敏度相对有限差分的误差；时间，对比整体 Cholesky。

**成功判据（建议）**
- (a) 所有测试中 λ_min(S̃⁻¹Ŝ) ≥ 1（数值容差内），否则保证本身有实现错误。
- (b) 分布内：能量误差 ≤ 0.1%，柔度误差 ≤ 0.05%（与 P1 同级）。
- (c) OOD：误差随畸变单调、平滑增长；扭转 20°/胞元时，校正后点阵柔度误差 ≤ 1%，且校正循环数不超过分布内的 2 倍。
- (d) PCG 迭代数增长 ≤ 20%。
- (e) 在 8 胞元以上，速度至少是整体 Cholesky 的 5 倍。
- **决策规则**：如果 V0 已经满足 (b)–(c)，J 条件化就缺乏理由，论文主线应改成"精确映射算子校正足以吸收映射"，这本身也是一个干净的结果。如果 V0 在 5–10° 就失效，而 V1 在 (c) 上明显更好，J 条件化才成立。

---

## 3. 方向 B：NICE 作为 BDDC/FETI-DP 的非精确子域/延拓求解器

### 3.1 已有理论（定理形式）

- **Li & Widlund 2007**（CMAME 196(8):1415–1428）。
  - Theorem 2：1 ≤ λ(M_k⁻¹A) ≤ Φ_k(H,h)。精确 Dirichlet 延拓时 Φ₂ = C(1+log(H/h))²。trivial 延拓（只取顶点 primal）时 Φ₁ = C(H/h)(1+log(H/h))；2D 加上边平均 primal 时为 C·H/h。
  - Theorem 3：在多重网格近似 K_II 和 K~、收敛因子 η* < 1 且与 H、h 无关时，c ≤ λ ≤ C·Φ₂。
  - Theorem 3 之后的正文（不是带编号的 remark）：子域矩阵可以换成谱等价且零空间相同的 Â^(i)。
- **Dohrmann 2007**（NLAA 14(2):149–168）：近似 BDDC 中，部分近似求解器必须满足"子结构零空间"性质。定理具体形式 UNVERIFIED。
- **Klawonn & Rheinbach 2007**（IJNME 69(2):284–307）：鞍点形式的 inexact FETI-DP。只要局部和粗近似求解器足够好，就得到与精确 FETI-DP 同等质量的界；常数如何依赖谱等价常数 UNVERIFIED。
- **Haase, Langer, Meyer 1991**（Computing 47）：近似 Dirichlet DD，条件数由近似调和延拓与界面预条件的谱等价常数控制。具体常数形式 UNVERIFIED。
- **Tu & Zhang 2025**（arXiv:2510.05993）：唯一一篇"代理 Schur 补 + BDDC + 条件数定理"。Lemma 5.4 为双侧界 |uᵀ(S̃ − S̃_F)u| ≤ γ‖u‖²；Theorem 5.11 的条件数界在截断趋于无穷时以概率 1 成立，常数含 (1−γ) 型因子，γ ≥ 1 时失效。
- 抽象 Schwarz 框架（Toselli–Widlund 2005，本轮未 verify）：κ ≤ C₀²·ω·(ρ(E)+1)。

### 3.2 ML 粗空间与学习型子域求解器的现状

- **学粗空间，子域保持精确**：Heinlein/Klawonn/Lanser/Weber 2021 SISC（预测哪些边/面需要求解特征值问题），Klawonn–Lanser–Weber-Hamacher 2026 arXiv:2607.06261（回归粗基、分类粗基个数，2D，监督）。鲁棒性完全继承自 adaptive DD 理论。
- **学界面参数**：Taghibakhshi et al. NeurIPS 2022 / MG-GNN ICML 2023，无监督，只做标量 Poisson。
- **学局部解，无谱保证**：Dimola–Vitullo–Zunino 2026（扰动不动点，误差底与代理误差成正比）；Secchi et al. 2026 NEST（3³ 体素 patch + Schwarz，无收敛理论）。
- **学粗求解**：Melchers–Dolean–Abdelmalik 2026 只保证对称化，无定理。本轮 verify 指出原检索对其谱结论的描述有误，见第 6 节。
- **学 SPD 预条件**：NeuralIF (TMLR 2024) 用 LLᵀ 只保证 M > 0，与精确算子之间没有谱序。

结论：**至今没有一篇工作把学习型子域延拓/Schur 近似，以确定性单侧谱序和公共零空间的形式，放进 BDDC/FETI-DP 并给出条件数界。**

### 3.3 单侧谱序能推出什么（我的推导，需要严格证明）

**假设**：每个子域 i 满足
- (H1) S_i ≤ Ŝ_i ≤ (1+ε_i) S_i，记 ε = max_i ε_i；
- (H2) ker Ŝ_i = ker S_i（浮动子域为刚体模态）；
- (H3) primal 约束与缩放 D 固定。若用 deluxe 缩放，D 依赖 Ŝ，需要单独处理。

**辅助引理（Schur 补单调性）**：若 Â ≥ A ≥ 0，则它们对同一块的 Schur 补保持 Schur(Â) ≥ Schur(A)；上界 (1+ε) 同理。所以把 P1 的保留坐标（面节点 + cut 带）再凝聚到箱面接口上，序和 ε 都不变。这一点让 NICE 能直接作为"面接口子域"使用。

- **命题 1（算子模式：求解扰动问题 Ŝu = g，BDDC 用 Ŝ 构造）**
  κ ≤ (1+ε)·Φ_S，其中 Φ_S = sup ‖E_D w‖²_{S̃} / ‖w‖²_{S̃}。
  证明链：‖E_D w‖²_Ŝ ≤ (1+ε)‖E_D w‖²_S ≤ (1+ε)Φ_S‖w‖²_S ≤ (1+ε)Φ_S‖w‖²_Ŝ。
  同时有柔度夹逼：fᵀŜ⁻¹f ≤ fᵀS⁻¹f ≤ (1+ε)fᵀŜ⁻¹f，这正是 P1 柔度误差的来源。
- **命题 2（预条件模式：精确 S 问题，预条件子用 Ŝ 构造）**
  1/(1+ε) ≤ λ ≤ (1+ε)Φ_S，所以 κ ≤ (1+ε)²Φ_S。
- **命题 3（Li–Widlund M3 型：在完整 K 上做 PCG，NICE 作非精确 Dirichlet 延拓，内部用多层 W 近似 K_II⁻¹）**
  能量形式给出 ‖Eu‖²_K = uᵀŜu ≤ (1+ε)‖Hu‖²_K，恰好满足 HLM 和 Li–Widlund 所需的延拓稳定性。猜想 κ ≤ C(ε, γ_W)·Φ₂，其中 γ_W 是 W 与 K_II 的谱等价常数。**这条最实用**：只需要 K 的矩阵-向量乘，不需要任何精确局部解。但界的具体形式需要证明。
- **命题 4（FETI-DP）**
  Ŝ ≥ S 推出 F̂ = B Ŝ̃⁻¹ Bᵀ ≤ F，序方向翻转；Dirichlet 预条件 B_D Ŝ B_Dᵀ 则变大。若算子和预条件都用 Ŝ，由 Mandel–Dohrmann–Tezaur 的谱等价性，得到与命题 1 相同的谱。这一点尚无人分析过。
- **命题 5（adaptive 约束）**
  在显式、廉价的 Ŝ 上解 deluxe/adaptive 广义特征值问题，得到 Ŝ 范数下的阈值 τ，转回 S 范数时只乘 (1+ε)。这可以与 Klawonn 组的学习粗空间组合成"学习约束 + 学习延拓"。

### 3.4 需要证明或测量的东西（也是主要风险）

1. **ε 必须是最坏情形 λ_max(S⁻¹Ŝ) − 1，而不是 P1 报告的平均能量误差 0.07%。** 高频边界模式上的最坏 ε 很可能远大于 0.07%。这需要立刻用 Lanczos 实测。实测之前，"κ 只放大 (1+0.0007) 倍"这类说法一律不能写。
2. **ε 的可计算性。** 由 Pythagoras 恒等式 qᵀ(Ŝ−S)q = ‖(E−H)q‖²_K = rᵀK_II⁻¹r，有残差型估计 rᵀW r。但要成为严格上界，还需要 W 的谱下界；P1 的 Chebyshev 区间如果是可证的，也许能提供。否则它只是估计，不是证书，"certified"一词要慎用。
3. **Φ_S 对 cut 薄壁子域是否仍是 C(1+log)²。** 经典界假设子域形状正则。TPMS 薄壁子域可能使 Φ 依赖壁厚与 h 的比值。Badia–Verdugo 2018 的 cut-cell BDDC 是相关背景（本轮未 verify）。这一项可能只能给数值证据。
4. **映射胞元下的公共零空间（H2）**，见 2.4 节第 1 点。
5. **网络的边际价值。** 在预条件模式下，ε = 1 也只让 κ 乘 4，NICE 的 0.07% 精度大部分是多余的。审稿人一定会问："几次多重网格 V-cycle 做近似调和延拓不就够了吗？"所以必须做消融：只用 W（从零初值或离散调和初值出发）对比网络加 W，比较墙钟时间和迭代数。另一个对照对象是 AMG-inexact BDDC（Klawonn–Lanser–Rheinbach 2018，本轮未 verify）。
6. **适用范围。** "一般非均质结构"是夸大。NICE 只对训练过的几何族有效，B 应限定为"胞元式结构（点阵/模块化），胞元即子域"。

### 3.5 最接近的工作与差异

| 文献 | 差异 |
|---|---|
| Hirschler et al. 2024 | ROM 局部算子 + inexact FETI-DP + GMRES。无序、无 SPD 保证、无定理，只适用贴体 tile。 |
| Bonilla Moreno et al. 2026 | 近似发生在**组装层**，凝聚是精确的，所以没有延拓层面的近似。靠 α 稳定化，无理论，只有 2D。结论里已计划"ROM 刚度作近似子域逆用于 Krylov 预条件"，与 B 直接撞车。 |
| Tu & Zhang 2025 | 有定理，但是双侧、概率性的，γ ≥ 1 时失效；PC 代理，不是 NN；标量问题。NICE 的确定性单侧序可以写成对它的明确改进。 |
| Li & Widlund 2007 / HLM 1991 / Dohrmann 2007 | 提供了界的模板，也构成"只是推论"的风险来源。必须明确：NICE 的贡献是 ε 很小、对任意输入在构造上成立（下界）、零空间保持、ε 可后验估计。 |
| Dimola 2026 / NEST 2026 | 神经局部求解器嵌入 DD，但没有 Krylov 条件数界，也没有序。 |

---

## 4. 无数据训练对两条线的作用与已有精度

**已有精度水平**（任务难度不同，不能直接横向比优劣）

| 工作 | 是否用标签 | 问题 | 精度 |
|---|---|---|---|
| Huang et al. JMPS 2024 | 无标签（最小势能） | 3D 体素子结构，24 个角点 DOF，可跨实例复用 | 柔度误差相对 EMsFEM 6.2%，位移误差 6.33%；优化设计柔度偏低 4.7%（只观察到单侧偏差，未给出界）；训练约 2–3 天（CPU 台式机） |
| Eshaghi et al. VINO, CMAME 2025 | 无标签 | 以 1D/2D 为主，含（超）弹性 | 具体误差未核实 |
| Jiang et al. CNEE 2026 | **监督** | 2D 弹性；3D 只做标量扩散 | 2D 弹性 0.20%；3D 标量 0.23%；一次性成本约 78 s |
| P1（NICE） | 监督 | 3D cut 薄壁弹性 | 能量误差约 0.07%；标签成本约 42 GPU-h |

截至目前，没有任何无标签方法在 3D 薄壁或 cut 弹性子结构族上报告过 < 0.1% 的能量误差，也没有报告过逐胞灵敏度误差。逐实例的无标签方法（如 Yadav 2026 变分 GNN，位移误差约 0.5%）训练成本比单次 FE 求解高一个数量级以上；该文本轮未 verify。

**理论现状**：Cao & Song 2026 已证明离散能量损失与刚度范数监督的极小点和梯度处处相同。所以"min Σ qᵀŜq 是无标签 Ritz 训练"在原理上已不新。42 GPU-h 真正花在三处：
- 物理方向 q 的生成（力驱动、邻胞诱导，需要精确解）；
- 对数比损失里用 qᵀSq 做归一化；
- 灵敏度标签。

仓库内部试点（2026-09-28，**非文献**，转引自 data_free 视角）：只用免求解方向（macro、grf）时，consistent 力类误差 1.86%（有标签 CTRL 为 1.98%），但节点力类误差 8–12%（CTRL 为 2.7–3.4%）。**瓶颈在方向分布，不在标签。**

**对方向 A 的作用：大。** 有了 Jacobian 场条件，输入空间变成高维，监督标签成本随之暴涨；而拉回能量只需用 C~ 组装，不需要额外求解。CNEE 那种"几十秒监督"的成本优势只在低维几何参数下成立。所以"无标签 + J 条件化"有真实动机。待解决的是免精确解的方向生成器，候选包括：
- 用 NICE-PCG 在伪点阵上的解做自举（self-training）；
- 以校正后的场作为 teacher；
- Krylov 随机方向。注意：FCG-NO 的训练其实依赖参考解，这点见第 6 节。

归一化可以用 qᵀK_PP q，或多次校正后的能量代替 qᵀSq。

**对方向 B 的作用：有限，而且目标不一致。** 能量迹最小化 Σ qᵀŜq 降低的是**平均** ε，而条件数界需要**最大** ε。要直接控制最大 ε，需要 Rayleigh 商的 minimax 或 log-sum-exp 型损失，而这又需要 S。B 更适合继续使用现有的 P1 网络，把无标签训练放到 A 之后。

---

## 5. 建议的论文路线

**第 0 步（2–4 周，决定后续一切）**
- 2.5 节的零样本扭转测试（V0）。
- 用 Lanczos 测 P1 现有网络在若干几何上的最坏情形 ε。
- 检查映射下的刚体模态，比较等参化前后。

**论文 P-B（建议先发预印本，占位 Bonilla 的公开计划）**
- 题目草案：*One-sided spectral bounds for learned inexact extensions in BDDC and FETI-DP, with application to 3D CutFEM lattices*
- 目标期刊：SIAM J. Sci. Comput.（偏理论），或 CMAME（偏应用）。
- 核心贡献：
  - 命题 1–5 和 Schur 单调性引理的严格证明；
  - 最坏 ε 的测量与残差型估计；
  - 超阈值子域回退到精确 Dirichlet 解的认证式回退；
  - 3D CutFEM 点阵，10²–10³ 胞元；
  - 与 Hirschler 式 ROM、AMG-inexact BDDC、纯多重网格延拓的对比。
- 依赖：只依赖 P1 的现有网络，可以立即开始。
- 主要风险：
  - 被认为是 Li–Widlund 的推论；
  - 最坏 ε 远大于平均值，导致叙事降级为"预条件模式"；
  - cut 子域上的 Φ 只能给数值证据；
  - 网络相对纯多重网格延拓的墙钟优势不明显。

**论文 P-A（主论文，你最感兴趣的方向）**
- 题目草案：*Certified neural static condensation for spline-mapped TPMS lattices: Jacobian-conditioned extensions with exact-operator correction*
- 目标期刊：CMAME。
- 核心贡献：
  - 拉回 CutFEM 公式，含等参化映射以保持刚体模态，ghost penalty 按 C~ 缩放；
  - J 条件化的 NICE，输入可用 Hirschler 2022 的多项式系数 A^(s)；
  - 零样本/OOD 研究，配合 S ≤ Ŝ 与校正；
  - 对控制点和厚度的灵敏度，并做有限差分验证；
  - 基线对比：精确凝聚、查表 + 精确凝聚、映射均匀化；
  - 弯曲/扭转点阵的厚度加控制点联合优化。
- 依赖：第 0 步的结果。P-B 的理论可以作为 A 中求解器部分的支撑，但不是必需。
- 主要风险：
  - Xu et al. 2025 或 Antolin 组先发 3D 版本；
  - V0 已经足够好，J 条件化的必要性不成立。这时应坦然把主线改成"精确映射算子校正"；
  - 物理壁厚随映射变化，导致设计变量语义复杂。

**论文 P-DF（第三篇，依赖 P-A 的架构）**
- 题目草案：*Label-free Ritz training of Jacobian-conditioned neural condensation with self-generated load directions*
- 目标期刊：JCP，或 CMAME / IJNME。
- 核心贡献：
  - 免精确解的方向生成器（自举、校正 teacher）；
  - 免标签的归一化与误差估计；
  - 在同一几何族上做"标签 GPU-h、训练 GPU-h、精度"的同口径对比；
  - 无标签灵敏度一致性损失。
- 主要风险：
  - 方向生成器达不到 0.1% 级精度（试点中节点力类误差 8–12%）；
  - 原理层面已被 Cao & Song 2026 和 Huang 2024 覆盖，新意只能落在"方向分布 + 证书 + 规模"。

**（可选）应用论文**：映射 TPMS 点阵的设计与实验验证，投 Struct. Multidiscip. Optim. 或增材制造类期刊，放在最后。

**顺序建议**：第 0 步 → P-B 预印本（约 3 个月），同时并行推进 P-A → P-A 投稿 → P-DF。如果资源只够做一篇，选 P-A：它社区更大，也最直接对上 Bonilla 公开的"3D NN"计划。但要接受它比 P-B 慢 3–6 个月。

---

## 6. 参考文献表

### 6.1 已核实（verify 中 exists 且 claim_ok）

1. G. Bonilla Moreno, G. Guarino, P. Antolin (2026 在线；卷期 2027). A ROM-based BDDC solver for unfitted p-FEM level-set-based two-dimensional lattice structures. *Comput. Methods Appl. Mech. Engrg.* 463, 119304. doi:10.1016/j.cma.2026.119304; arXiv:2604.09113.
2. T. Hirschler, R. Bouclier, P. Antolin, A. Buffa (2024). Reduced order modeling based inexact FETI-DP solver for lattice structures. *Int. J. Numer. Methods Eng.* 125(8), e7419. doi:10.1002/nme.7419; arXiv:2308.11371.（注：预条件子为完整块 LDLᵀ 型，不是 KR 块三角。）
3. T. Hirschler, P. Antolin, A. Buffa (2022). Fast and multiscale formation of isogeometric matrices of microstructured geometric models. *Comput. Mech.* 69(2), 439–466. doi:10.1007/s00466-021-02098-y; arXiv:2107.09568.（注：Jacobian 为多项式投影近似，不是精确并入。）
4. C. Guillet, T. Hirschler, P. Jolivet, R. Bouclier (2025). Multilevel matrix-free method for high-performance isogeometric analysis of lattice structures. *J. Comput. Phys.* 537, 114136. doi:10.1016/j.jcp.2025.114136.
5. C. Guillet, T. Hirschler, P. Jolivet, P. Antolin, R. Bouclier (2026). Efficient fine-scale simulation of nonlinear hyperelastic lattice structures. *Comput. Methods Appl. Mech. Engrg.* 461, 119202; arXiv:2603.10741.（只在第 4 条的 verify 附注中确认存在；DOI 未单独核对。）
6. W. Xu, C. Liu, Y. Guo, M. Huang, X. Guo (2025). Problem-Independent Machine Learning (PIML) enhanced 3D lattice composite structures optimization via moving morphable components approach. *Compos. Struct.* 369, 119330. doi:10.1016/j.compstruct.2025.119330.（其 PIML 如何处理映射：UNVERIFIED。）
7. W. Xu, C. Liu, Y. Guo, Z. Du, W. Zhang, X. Guo (2023). Graded infill lattice structures design based on the moving morphable component method and partitioned coordinate mapping technique. *Compos. Struct.* 326, 117613. doi:10.1016/j.compstruct.2023.117613.
8. Y. Guo, C. Liu, Z. Du, Y. Jia, C. Jiang, X. Guo, C. Shen (2026). High-generalization AI-enhanced mechanical analysis and topology optimization via cubic Bézier interpolation of substructure boundary displacements. *Comput. Methods Appl. Mech. Engrg.* 456, 118955. doi:10.1016/j.cma.2026.118955.（注：data_free 视角称"Full Node Prediction 比 Bézier 慢 7.3–7.5×"，verify 判定为错误：7.3–7.5× 是 FNP 相对直接 FEM 的加速比，FNP 实际比 Bézier 慢约 7.2× 和 14.7×。）
9. Y. Zhu, S. Li, Z. Du, C. Liu, X. Guo, W. Zhang (2019). A novel asymptotic-analysis-based homogenisation approach towards fast design of infill graded microstructures. *J. Mech. Phys. Solids* 124, 612–633. doi:10.1016/j.jmps.2018.11.008.
10. C. Liu, W. Xu, W. Huo, Y. Guo, X. Guo (2026 卷期；2025 在线). Surface lattice structure design via computational conformal mapping and structural optimization. *Comput. Methods Appl. Mech. Engrg.* 451, 118680. doi:10.1016/j.cma.2025.118680.（是否用 PIML 分析：UNVERIFIED。）
11. L. Zhang, M. Huang, C. Liu, Z. Du, T. Cui, X. Guo (2024). Problem-independent machine learning-enhanced structural topology optimization of complex design domains based on isoparametric elements. *Extreme Mech. Lett.* 72, 102237. doi:10.1016/j.eml.2024.102237.
12. M. Huang, C. Liu, Y. Guo, L. Zhang, Z. Du, X. Guo (2024). A mechanics-based data-free Problem Independent Machine Learning (PIML) model for large-scale structural analysis and design optimization. *J. Mech. Phys. Solids* 193, 105893. doi:10.1016/j.jmps.2024.105893.（注：约 3% 的误差指线性边界子结构法在 10×5×5 子结构下的误差，不是"m=10 的边界插值误差"。）
13. Y. Guo, C. Liu, Z. Du, J. Liu, J. Feng, X. Zhang, Y. Li, T. Yang, C. Shen, X. Guo (2026). PIML-OFEM: A new large-scale structural analysis method based on problem-independent machine learning and overlapping finite element technique. arXiv:2607.22019.
14. H. Jiang, J. Zhan, C. Zhang, F. Wang (2026). Convex Neural Energy Elements: Monolithic finite-element assembly of geometry-parameterized neural operators with stability and error guarantees. arXiv:2608.02036.（注：Rayleigh–Ritz 阶梯在正文第 5 节，不在附录。）
15. J. Li, O. B. Widlund (2007). On the use of inexact subdomain solvers for BDDC algorithms. *Comput. Methods Appl. Mech. Engrg.* 196(8), 1415–1428. doi:10.1016/j.cma.2006.03.011.（预印本 NYU TR2005-871。）
16. C. R. Dohrmann (2007). An approximate BDDC preconditioner. *Numer. Linear Algebra Appl.* 14(2), 149–168. doi:10.1002/nla.514.（定理具体形式 UNVERIFIED。）
17. A. Klawonn, O. Rheinbach (2007). Inexact FETI-DP methods. *Int. J. Numer. Methods Eng.* 69(2), 284–307. doi:10.1002/nme.1758.
18. G. Haase, U. Langer, A. Meyer (1991). The approximate Dirichlet domain decomposition method. Part I: An algebraic approach; Part II: Applications to 2nd-order elliptic B.V.P.s. *Computing* 47, 137–151; 153–167. doi:10.1007/BF02253431; doi:10.1007/BF02253432.
19. X. Tu, J. Zhang (2025). Stochastic BDDC algorithms. arXiv:2510.05993.
20. N. Dimola, P. Vitullo, P. Zunino (2026). Multiscale mixed-dimensional simulation via domain decomposition and non-intrusive neural model order reduction. arXiv:2607.15171.
21. P. Secchi, D. S. Balint, M. Maurizi (2026). Neural-Schwarz tiling for geometry-universal PDE solving at scale. arXiv:2605.12343.
22. R. Cao, X. Song (2026). Discrete energy as an exact label-free training objective for finite-element surrogates. arXiv:2608.05437.
23. M. S. Eshaghi, C. Anitescu, M. Thombre, Y. Wang, X. Zhuang, T. Rabczuk (2025). Variational physics-informed neural operator (VINO) for solving partial differential equations. *Comput. Methods Appl. Mech. Engrg.* 437, 117785. doi:10.1016/j.cma.2025.117785; arXiv:2411.06587.
24. P. Häusner, O. Öktem, J. Sjölund (2024). Neural incomplete factorization: learning preconditioners for the conjugate gradient method. *Transactions on Machine Learning Research*. arXiv:2305.16368.
25. A. Taghibakhshi, N. Nytko, T. U. Zaman, S. MacLachlan, L. Olson, M. West (2022). Learning interface conditions in domain decomposition solvers. *NeurIPS* 35. arXiv:2205.09833. 后续：MG-GNN: Multigrid graph neural networks for learning multilevel domain decomposition methods, *ICML 2023*, PMLR 202:33381–33395, arXiv:2301.11378.
26. A. Heinlein, A. Klawonn, M. Lanser, J. Weber (2021). Combining machine learning and adaptive coarse spaces—A hybrid approach for robust FETI-DP methods in three dimensions. *SIAM J. Sci. Comput.* 43(5), S816–S838. doi:10.1137/20M1344913.
27. A. Klawonn, M. Lanser, J. Weber-Hamacher (2026). Learning adaptive coarse spaces using transferable neural network models for linear and nonlinear overlapping domain decomposition methods. arXiv:2607.06261.

### 6.2 存在，但 verify 判定原检索的描述有误（claim_ok=false）

- M. Huang, T. Cui, C. Liu, Z. Du, J. Zhang, C. He, X. Guo (2023). A Problem-Independent Machine Learning (PIML) enhanced substructure-based approach for large-scale structural analysis and topology optimization of linear elastic structures. *Extreme Mech. Lett.* 63, 102041. doi:10.1016/j.eml.2023.102041。
  **错误**：PIML-OFEM 中的 K_s = NᵀKN 引的是 JMPS 2024，刚体补全引的是 CMAME 2026 Bézier 论文，都不能归给本文。
- H. Melchers, V. Dolean, M. Abdelmalik (2026). When can a neural operator replace a coarse solve? Architectural principles for two-level preconditioning. arXiv:2605.19867。
  **错误**：并非"其它架构都导致 PCG 崩溃"，积分型的 RINO 也能收敛；"谱聚集在 [1,2] 附近"是 Helmholtz k=5 时的描述，扩散问题的谱在 [0.8, 2.2]。
- A. Rudikov, V. Fanaskov, E. Muravleva, Y. M. Laevsky, I. Oseledets (2024). Neural operators meet conjugate gradients: The FCG-NO method for efficient PDE solving. *ICML 2024*, PMLR 235:42766–42782. arXiv:2402.05598。
  **错误**：训练**需要**参考解 A⁻¹f，不是无标签方法。

### 6.3 检索中出现、本轮 verify 未覆盖的条目（UNVERIFIED，引用前须自行核对）

- 几何与映射：
  - P. Antolin et al. (2019) *Comput.-Aided Des.* 115:23–33, doi:10.1016/j.cad.2019.05.020；
  - G. Elber (2017) LNCS MMCS, doi:10.1007/978-3-319-67885-6_6；
  - F. Massarwi et al. (2018) *CAD* 102:148–159；
  - Q. Hong et al. (2024) *Comput. Graph. Forum* 43:e15224, arXiv:2408.14068；
  - J. Feng et al. (2018) *CMAME* 336:333–352；
  - J. Wu, W. Wang, X. Gao (2021) *IEEE TVCG* 27(1):43–56。
- 映射均匀化与多尺度优化：
  - C. Liu et al. (2017) *J. Appl. Mech.* 84(8):081008；
  - S. Li, Y. Zhu, X. Guo (2022) *CMAME* 391:114589；Li–Zhu–Guo (2024) *SMO* 67:24；
  - S. Zhou, R. Tao, Q. Sun (2025) *CMAME* 442:118023；
  - P. Geoffroy-Donders, G. Allaire, O. Pantz (2020) *JCP* 401:108994；G. Allaire et al. (2019) *CAMWA* 78:2197–2229；
  - Y. Wang, H. Xu, D. Pasini (2017) *CMAME* 316:568–585；
  - R. Rubio et al. (2026) arXiv:2609.20053。
- 降阶与学习子结构：
  - E. Parish et al. (2024) *Comput. Mech.* 74(6):1357–1381；
  - L. Gaynutdinova et al. (2025) *EAAI* 154:110906；
  - M. Chasapi, P. Antolin, A. Buffa (2024) LNCSE, doi:10.1007/978-3-031-55060-7_4；
  - D. B. P. Huynh, D. J. Knezevic, A. T. Patera (2013) *ESAIM: M2AN* 47(1):213–251；
  - K. Smetana, A. T. Patera (2016) *SISC* 38(5):A3318–A3356；
  - S. McBane, Y. Choi (2021) *CMAME* 381:113813（卷号与文章号未核实）；
  - W. Ouyang et al. (2026) NOEM, *Nat. Comput. Sci.* 6:417–429；
  - M. Yin et al. (2024) DIMON, *Nat. Comput. Sci.* 4:928–940；
  - C. Wu et al. (2026) DFENN, *JMPS* 215:106703。
- DD 与预条件：
  - A. Toselli, O. B. Widlund (2005) Springer SCM 34；
  - A. Klawonn, M. Lanser, J. Weber (2024) *JCP* 496:112587；
  - A. Heinlein et al. (2019) *SISC* 41(6):A3887–A3912；
  - A. Klawonn, M. Lanser, O. Rheinbach (2018) *ETNA* 49:244–273 及 DD24 LNCSE 125；
  - Q. Sun, X. Xu, H. Yi (2024) *SISC*，arXiv:2207.10358；
  - A. Kopaničáková, G. E. Karniadakis (2025) *SISC* 47:C151–C181；
  - S. Badia, F. Verdugo (2018) *J. Comput. Appl. Math.* 344:740–759；
  - I. Luz et al. (2020) ICML；
  - Y. Li et al. (2023) ICML, PMLR 202；
  - Mandel–Dohrmann–Tezaur 的 BDDC/FETI-DP 谱等价（本轮未检索到具体条目）。
- 无数据训练：
  - A. R. Yadav et al. (2026) arXiv:2609.10983；
  - H. Li et al. (2026) *JMPS* 210（DOI 未核实）；
  - A. Dean, B. Bahtiri (2026) arXiv:2607.23299；
  - W. E, B. Yu (2018) *Commun. Math. Stat.* 6:1–12；
  - J. Xu, L. T. Zikatanov (2004) *Comput. Visual. Sci.* 7:121–127。