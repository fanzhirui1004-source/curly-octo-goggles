# Guo Xu / Liu Chang 组 PIML 系列与 NICE 的对照核查（R1 用）

> 用途：为 R1 修改稿与答复信提供可核对的事实。依据是六篇论文的全文抽取文本（加上 PIML-OFEM 预印本作背景），只引用短句，并标注页码、节号或表号。页码按期刊版面页。
> 下文"我们的数据"全部取自 `MANUSCRIPT_EN.md`、`SUPPLEMENTARY_EN.md`。本文件不修改稿件。
>
> **重要更正**：任务清单中的 CAD 2026 (S0010448526000084) **不是** Guo/Liu 组的工作。作者是 Wei Chen 和 Ming Li（浙江大学 CAD&CG 国家重点实验室），题为 *Conditioned Numerical Shape Functions on Unfitted Reduced Coarse Elements*，CAD 193 (2026) 104038。该文不含机器学习，只在引言和结论中把 PIML 当作可能的加速手段引用（p.3、p.12）。下文单列为"相关的非本组工作"。

---

## 0. 文献对照（本文件使用的简称）

| 简称 | 文献 | 我们稿件中的引用键 |
|---|---|---|
| **EML23** | Huang, Cui, Liu, Du, Zhang, He, Guo, *Extreme Mech. Lett.* 63 (2023) 102041 | Huang et al. (2023) |
| **JMPS24** | Huang, Liu, Guo, Zhang, Du, Guo, *J. Mech. Phys. Solids* 193 (2024) 105893（data-free PIML） | Huang et al. (2024) |
| **CMAME26** | Guo, Liu, Du, Jia, Jiang, Guo, Shen, *CMAME* 456 (2026) 118955（Bézier PIML） | Guo et al. (2026a) |
| **CS26** | Jiang, Liu, Guo, Du, Zhang, Guo, *Compos. Struct.* 397 (2026) 120865（等变 PIML，2026-09-04 在线） | 未引用 |
| **IJMS26** | Zhang, Liu, Guo, Jiang, Guo, *Int. J. Mech. Sci.* 328 (2026) 112007（PITL 迁移学习，2026-08-13 在线） | 未引用 |
| **CAD26** | Chen, Li, *Computer-Aided Design* 193 (2026) 104038（非本组；rMsFEM，无 ML） | 未引用 |
| **OFEM** | Guo et al., PIML-OFEM, arXiv:2607.22019v1（仅作背景） | Guo et al. (2026b) |
| （未提供全文） | Huang et al., *EML* 56 (2022) 101887：PIML 原始论文（2D） | 未引用 |
| （未提供全文） | Xu, Liu, Guo, Huang, Guo, *Compos. Struct.* 369 (2025) 119330 | Xu et al. (2025) |
| （未提供全文） | Zhang, Huang, Liu, Du, Cui, Guo, *EML* 72 (2024) 102237：等参 PIML（2D） | 未引用 |

---

## 1. 逐篇卡片

### 1.1 EML23：3D PIML（监督学习，线性边界）

- **问题**：线弹性结构分析与 SIMP 拓扑优化（柔度最小化和 3D 柔顺机构），体素网格，材料为随机或 SIMP 密度。
- **子结构**：规则立方体，m×m×m 个三线性六面体，m = 5 或 10。
- **边界表示**：主要使用线性边界假设，只保留 8 个角点，即 **24 个 DOF/子结构**。文中还构建了 m = 5 的 2D 全边界网络。
- **网络**：普通前馈网络（FNN），输入 m³ 个单元杨氏模量，输出 N_j2 去掉刚体约束后的独立部分（线性假设下为 N n_ji × 18）。m = 5 时输出 3456 个量，m = 10 时输出 39,366 个量（p.6）。m = 5 用 4 个 FNN，每个 15 层隐藏层（p.7）。另有一个 FNN 直接预测 K̃（上三角）。
- **训练**：监督 MSE，最后阶段加一项一致性损失，即由 N 算出的 K̃ 与网络预测的 K̃ 之差（p.6）。样本由 [0,1] 均匀随机模量生成，对每个样本求解局部方程得到标签。m = 5 用 400,000 个样本（p.7）；m = 10 时"约 3 h 生成 10⁵ 个样本"（p.4）。
- **刚体处理**：用 Eq. (13)–(15) 补全列，即"completion"，以约束刚体平移和转动。
- **K 的两种算法**：一是网络直接预测 K̃；二是用 Eq. (17) 由 K̃ = NᵀKN 计算，保证自洽（p.7）。MBB 和亿级算例采用第二种（p.8、p.10）。
- **精度**：柔顺机构输出位移相对 EMsFEM 的误差为 7.56% 和 7.65%（p.10）。文中没有给出相对全尺度 FEM 的位移或柔度误差表。
- **最大算例**：短悬臂梁，**1.024×10⁹ 个细单元**（160×80×80 个子结构，m = 10，约 3×10⁹ DOF），137 次迭代共 9–10 天，每次迭代 5677.8 s。全程串行，粗网格方程用迭代法求解（p.11）。硬件记为"laptop"，但配置是 Xeon Gold 6256 加 512 GB 内存（p.8）。
- **加速比**：MBB 算例中 N_F = 1,647,750 时为 472×，N_F = 11,718,750 时为 328×。基准只统计经典 SIMP 前 10 次迭代的平均时间（p.9）。摘要写的是"10⁴–10⁵ times"（p.1），与正文数字不一致。
- **灵敏度**：文中没有公式；柔顺机构只提到需要伴随场（p.10）。
- **作者所述局限**：线性边界会高估刚度，子结构数量少时尤其明显（p.11）；网络直接预测的 K 与 N 不自洽，因为子结构应变能之和不等于细网格应变能（p.10）。展望中提到可与 IGA、FCM、CutFEM 结合，例如按切割模式学习 cut element 的刚度（p.12）。

### 1.2 JMPS24：data-free PIML（能量训练）

- **问题**：同上，含柔度最小化、3D 扭转盒和柔顺机构。
- **子结构**：规则立方体，m = 10，即 1000 个细单元。
- **边界表示**：线性边界，**24 个角点 DOF**，其中独立列为 18 列（App. B）。
- **网络**：非堆叠 DeepONet。branch 输入单元密度，trunk 输入内部节点坐标，两者做 Hadamard 积，输出该点的多尺度形函数（p.7–8）。
- **训练（本系列唯一的 data-free / 能量训练）**：以最小势能作损失（Eq. 13–15）。构造一个 10×10×5 个子结构的"pseudo structure"，损失为所有细单元应变能之和。刚体约束作为硬约束（p.8–9）。不需要形函数标签：共随机生成 2000 次密度场，得到 10⁶ 个子结构密度，并预存 **2178 个正交归一的边界位移基**，每次迭代更换基和密度（p.9）。训练 40–50 epochs，在台式机上**约 2–3 天**；监督版训练时间相近，但生成数据要额外花很多时间（p.10）。作者称监督方案每个样本约 1 s、需要约 10⁶ 个样本，而且 m = 10 时需要 9 个 ANN 预测约 40,000 个分量（p.6）。
- **精度**：
  - 随机材料悬臂梁，20×10×10 个子结构：PIML 与 EMsFEM（同一线性边界下的精确解）柔度相差 **6.20%**（83.89 对 89.46）；位移误差相对 EMsFEM 为 6.27%，相对细网格为 **6.33%**（p.11）。
  - 50×25×25 个子结构：优化设计的柔度相对全尺度偏低 **4.7%**（321.81 对 337.76），分析加速约 87×（p.12）。
  - 只看边界假设本身（精确内部，Table 2，p.5）：m = 10、10×5×5 个子结构时位移误差为 2.92%；m = 5、20×10×10 时为 0.98%；子结构最少时为 11.6–11.9%。
  - 作者观察到 EMsFEM 与 PIML 的柔度都小于全尺度结果（p.11）。这是单侧性，但只是观察，没有给出界。
- **最大算例**：扭转盒，1.8×10⁸ 个细单元，每次迭代 1175.86 s；1.152×10⁷ 个单元时 PIML 为 49.07 s，全尺度为 11,761.28 s，即 >230×（Table 3，p.14）。正文把这两个数字写反了（p.14）。硬件为 Xeon Gold 6256、512 GB，Matlab 2020b（p.4 脚注、p.10）。Table 1 中 25 万单元（约 0.8 M DOF）的全尺度分析在 Matlab 中耗时约 1320 s（p.4），可见基准偏慢。
- **灵敏度**：Eq. (18)–(19)，p.10，原文为"For ease of implementation, the sensitivity results are calculated following the topology optimization with full-scale analysis"。做法是用恢复出的细单元位移计算 −3ρ²(E₀−E_min)u_eᵀk_eu_e，**忽略 ∂N/∂ρ**，与我们的场基估计 s̃ 同类。柔顺机构用 K_ML 解伴随方程。
- **作者所述局限**（p.15–16）：(1) 材料不连通（如 QR 图案）时精度显著下降；(2) 复杂几何难以用规则立方体离散；(3) 超大规模问题串行实现仍费时费内存。

### 1.3 CMAME26：Bézier 边界 + DeepONet（监督学习）

- **问题**：随机模量场分析；SIMP/ITDF 拓扑优化，算例有 MBB、悬臂梁、桥和飞翼。
- **子结构**：5³ 或 10³ 个体素。
- **边界表示**（p.8）：每个面用一个 bicubic Bézier 曲面，共 **56 个控制点，即 168 DOF**（8 个顶点、24 个边内点、24 个面内点）。对照方案有线性（8 点，24 DOF）、二次（26 点，78 DOF），以及 **"Full Node Prediction"**（保留全部边界节点：5³ 为 152 个节点即 456 DOF，10³ 为 602 个节点即 1806 DOF）。
- **网络**：DeepONet。branch 为 200 个神经元，trunk 为 [200, 100, 80, 60]，激活函数 tanh。输入为单元模量加查询节点坐标，输出 N̂(x) ∈ ℝ^{3×168}，其中网络实际只输出 3×162，其余 6 列由刚体条件补全（Eq. 30–31）。实现为 TensorFlow 2.10，p.11–12。Remark 3（p.12–13）说明 branch 为固定维 MLP，所谓分辨率无关只针对输出侧。
- **训练**：**监督 MSE**，标签是精确的数值形函数（p.11）。5³ 约 10⁵ 个样本，10³ 约 10⁶ 个样本（p.11）。数据量收敛研究：5³ 超过 10⁵ 后收益递减（Fig. 8，p.11–12）。5³ 测试集 MSE 为 1.846×10⁻⁵（1000 个测试样本，p.12）。**没有给出训练耗时**。
- **精度**（误差为 ‖U_PIML − U_FEM‖_L2 / ‖U_FEM‖，参考为全尺度 FEM；随机模量悬臂梁 80×40×40）：见第 3(a) 节 Table 1 摘录；桥型算例见 Table 2（p.15）。集中载荷算例见第 3(b) 节。
- **最大算例**：MBB 720×120×120，即 10,368,000 个单元（利用对称只算 1/4，p.17–18）；飞翼包围盒 540×600×140，设计域约 1038 万个单元（p.20、p.22）。这两个算例**都没有给出每次迭代的时间，也没有细网格复核的柔度误差**。
- **速度**：悬臂梁 200×100×100（2 M 单元，约 6.15 M DOF）每次迭代分析时间为线性 PIML 20.09 s、Bézier PIML 125.07 s 或 129.40 s、线性 PIML（小过滤半径）170.46 s；直接 FE 为 1262.39 s（p.19），即 Bézier 约 10×、线性约 63×。另引 Ma et al.（SSRN 5003535）的并行线性 PIML：100 亿 DOF（3.456×10⁹ 个单元），6750 核，每次迭代 42 s（p.25）。
- **硬件**：i9-13900K、128 GB（p.14）；"only one CPU in purely serial mode"（p.25）。**直接 FEM 所用求解器没有说明**。
- **灵敏度**：文中**没有给出**任何灵敏度公式，只写 SIMP 用 OC、ITDF 用 MMA（p.14）。
- **作者所述局限**：集中载荷下任何边界假设都会导致响应过刚（p.17）；高阶插值使每次迭代比线性慢（p.19）；并行只实现了线性版本（p.26）；branch 与网格绑定（Remark 3）。

### 1.4 CS26：等变 PIML（对称解耦 + 投影校正）

- **问题**：随机材料分析、BCC 梯度点阵的 MMC 类优化，以及 3D SIMP 悬臂梁。
- **子结构**：s³，s = 5 或 10。
- **边界表示**：以线性边界为主（24 DOF）；§4.4 推广到二次（PIML-Q）和 Bézier（PIML-BS）。
- **网络**：**单个 FNN**。利用 O_h 群（48 个对称操作）把形函数分解为 vertex、edge、diag、face 四类基本单元（Eq. 27）。先把密度场变换到代表位置，再预测，最后逆变换；s = 5 时每个形函数要调用网络 48 次，其中两组 24 次取平均（p.5–6）。参数量从 1,278,736 降到 278,692（s = 5），从 12,093,246 降到 2,121,707（s = 10）（Table 1，p.6）。
- **刚体约束**：用**投影校正** N̂ = ÑP + N_r（Eq. 29–31）代替列补全。作者证明列补全会把误差放大约 √(n_o − n_r)·‖D¹D²⁻¹‖ 倍（Eq. 34–35），实测约 1.99–2.21 倍（p.8）。投影本身只把预测误差再降 0.25–0.33%（p.8）。另用 Galerkin 形式 K = NᵀKN 恢复对称性（Eq. 25）。
- **训练**：监督 MSE（§2.3 写的是形函数 MSE 加凝聚刚度 MSE，§3.3 只写形函数 MSE）。s = 5 和 s = 10 **各用 10,000 个样本**，按 8:1:1 划分，训练 3000 epochs，PyTorch（p.6）。R² 分别为 0.9972（s = 5）和 0.9931（s = 10）（p.7）。训练耗时未给出。
- **精度**：Table 3（p.11）中的**参考解是"Sub"，即同一线性边界下的精确子结构法，不是细网格 FEM**：
  - s = 5：PIML 为 2.30%，PIML-proj 为 0.46%；
  - s = 10：PIML 为 19.88%，PIML-symm 为 9.07%，PIML-proj 为 7.01%；PIML-symm 的 u_y 分量最大相对误差达 111.40%。
  - Table 6（p.15）把优化结果放回细网格重算：柔度误差为 PIML-L **9.17%**、PIML-Q **2.79%**、PIML-BS **3.14%**，预测值都偏低。PIML-BS 设计的真实柔度 2.1076，比直接 FEM 优化结果 2.0982 差 0.45%。
- **速度与基准**：半域 200×50×100（1 M 单元），**PCG + incomplete Cholesky** 每次迭代 646.73 s，PIML-BS 13.89 s，约 46×（p.15）。硬件为 i7-13700、32 GB（p.6）。
- **灵敏度**：没有公式。作者只作定性表述，称 PIML-proj 给出"more accurate sensitivity estimation"（p.13）。
- **作者所述局限**：方法依赖立方体对称性，不规则或过渡子结构需要重新定义基本单元（p.15）。

### 1.5 IJMS26：PITL，从规则子结构到等参子结构的迁移学习

- **问题**：复杂 3D 域（支架、发动机支架）上的柔度最小化、柔顺机构和 p-norm 应力最小化。
- **子结构**：等参六面体，内部 5³ 或 10³ 个细单元。
- **边界表示**：线性，8 个导出节点，即 **24 DOF**（p.6）。
- **网络**：冻结一个在规则子结构上预训练的 DeepONet 型源模型（trunk 输入三阶多项式扩展坐标，branch 1 输入 125 个密度），再增加一个 branch 2，输入 8 个导出点坐标（21 维），与源模型输出做 Hadamard 调制（Eq. 13，Table 1，p.5–7）。网络只预测 N₂′ ∈ ℝ^{3n_r×18}，其余 6 列由 Eq. (20) 补全（p.6）。
- **训练**：**监督 MSE**（Eq. 14）。500,000 个等参样本，由六个半空间求交生成凸六面体，10 epochs（p.4、p.6）。源模型用 400,000 个样本。成本见 Table 2（p.8）：源模型 2.85 h，目标适配 3.18 h，**首次使用合计 6.03 h，比从头训练的 3.31 h 更贵**。作者据此承认迁移不降低首次成本，只降低以后再适配的成本（p.7–8）。收敛所需迭代数为 5000，从头训练需 16,000（p.6–7）。每 10⁵ 个样本约 15 min。
- **精度**：
  - 支架（约 2.5 M 单元），参考为同一线性边界下的子结构法：柔度误差 1.2%（5³）和 5.9%（10³），全场位移误差 1.5% 和 6.5%（p.11）。
  - 柔顺机构输出位移误差为 0.5–1.7%（p.12）。
  - **App. C（p.20）用 30 万单元中等规模算例对照全尺度 FEA**：柔度 PITL 456.44、子结构法 474.36、FEA 575.34，即 PITL **偏低 20.7%**，子结构法偏低 17.6%。位移幅值误差 24.8% 和 21.5%。前 1% 高应变能密度区的平均值为 0.0024、0.0027、0.0046，前 1% 高 von Mises 应力区为 0.0613、0.0668、0.0928。
  - 应力算例（p.13）：σ_PN 在 PITL 为 0.00913，在子结构参考为 0.00945（3.5%），**细网格重算为 0.04820，约为 PITL 的 5.3 倍**。作者因此把 PITL 定位为"warm start"，随后在细网格上再优化 31 步。
- **速度与基准**：OptiStruct 32 核并行每次迭代 8223.03 s，PITL 串行为 63.54 s（5³）或 116.90 s（10³），即 129× 和 70×。总时间 93.65 h 对 1.90 h 和 2.99 h，即约 50× 和 31×（p.12）。每次迭代的时间中，组装占 60%，求解只占 13.3%。App. C 中全尺度 FEA 重分析 30 万单元（约 1 M DOF）耗时 **1986.2 s、峰值 137.9 GB**，PITL 为 3.9 s、4.0 GB（p.20）；FEA 的求解器未说明。硬件为 Xeon Gold 6246R、512 GB（p.10）。
- **灵敏度**（p.9，App. A）：柔度和柔顺机构用 Eq. (24)、(27) 的场基公式，**忽略 ∂N/∂ρ**。应力问题 Eq. (30)–(31) 含 T3 项，其中 **∂N/∂ρ 通过网络反向传播得到**，这是本系列唯一计入学习形函数设计导数的地方。应力灵敏度每次迭代 287.7 s，细网格为 1745.3 s，加速有限（p.14）。
- **作者所述局限**（p.15–16）：线性边界压制高阶边界模态，会抹平应力集中；局部量（应变能密度、应力）与 FEA 差距明显；最终设计仍需全尺度 FEA 验证。

### 1.6 CAD26（非本组）：unfitted 缩减粗单元上的条件数控制数值形函数

- **问题**：嵌入规则背景网格（unfitted）的复杂实体线弹性与热传导分析，**不含 ML**。
- **子结构**：粗单元内部为 10³ 个细单元（默认），PCB 算例也用 5³、8³ 和 25³。
- **边界表示**：Bézier 粗单元，p = 3，每面 (p+1)² 个节点。对 cut element 删除病态非顶点节点，并重写顶点插值以保持单位分解（Def. 7，Eq. 28–30，p.6）。给出条件数估计 κ₂(K) = O(1/(Hhε²))（Theorem 2，p.5）。
- **形函数计算**：精确静力凝聚，每个粗单元先做一次 Cholesky 分解再解多个右端项（p.7）。实现为 C++、Eigen、TBB 并行和 AMGCL（p.8）。
- **精度**：L2 位移误差为 1.1×10⁻² 和 2.9×10⁻³，约为 FCM 的 1/5–1/3（Table 2，p.9）；热传导 r_T 为 5.3×10⁻² 到 5.7×10⁻³（p.12）。
- **最大算例**：PCB 热传导，1500×1000×300 细网格，**1.83 亿个细节点**，253K 个粗节点，523 s。600×400×120 网格上 FEM 需 133.3 s，rMsFEM 需 11.0–20.8 s（p.12）。硬件为 i7-11700、64 GB。
- **局限**：只能强施加平面 Dirichlet 边界，Nitsche 留待后续；粗单元数多时形函数计算成本显著，建议用 NN（引 PIML）加速（p.12–13）。

---

## 2. 横向对照表（系列与 NICE）

| 维度 | EML23 / JMPS24 | CMAME26 | CS26 | IJMS26 | **NICE（本文）** |
|---|---|---|---|---|---|
| **精度：NN 自身误差**（参考为同一边界表示下的精确凝聚） | JMPS：柔度 6.20%，位移 6.27%（相对 EMsFEM，m = 10） | Full-node 算例可视为只有 NN 误差：位移 L2 误差 0.84%（5³）/3.46%（10³） | 位移（相对 Sub）：0.46%（s = 5）/7.01%（s = 10）；基线 PIML 为 2.30%/19.88% | 柔度 1.2%/5.9%，位移 1.5%/6.5%（相对 Sub） | 单胞能量误差均值 0.074%（最大几何均值 0.65%）；两胞柔度 ≤0.056%，灵敏度 ≤0.67%；八胞点阵柔度 ≤0.015%（随机载荷 ≤0.069%），灵敏度 ≤0.14% |
| **精度：相对细网格 FEM 的总误差** | JMPS 优化设计柔度 −4.7%；位移 6.33% | 位移 L2 误差 1.3–13.6%（Table 1/2）；集中载荷时加载点位移 63%（线性）/15%（Bézier），即柔度误差 | 柔度 9.17%（L）/2.79%（Q）/3.14%（BS） | App. C：柔度 −20.7%，位移 24.8%，应力 p-norm 约 5.3 倍低估 | 参考即离散 CutFEM 模型（Q2，n = 32），因此误差等于上一行。重胞参考自身的离散误差 ≳1%（§6.2） |
| **边界表示（DOF/子结构）** | 24（角点线性） | 168（Bézier）；对照 24/78/456/1806 | 24（另有 Q、BS 变体） | 24（等参线性） | 完整保留迹：box 面加 cut band，**1.7–2.6×10⁴** |
| **几何表示** | 体素模量或密度场（规则立方体） | 同左 | 同左，利用 O_h 对称 | 等参六面体（8 个节点坐标）加密度 | 隐式 Schwarz-P 薄壁（8 个角点厚度参数加一个平面切割），CutFEM + ghost penalty，Q2 |
| **通用性（问题无关性）** | 与 BC、载荷、域形无关，但与 m、单元类型、ν = 0.3 绑定 | 同左，且与插值方案绑定 | 同左，且依赖立方对称 | 可用于任意凸六面体子结构，不同目标函数无需重训 | 与 BC、载荷、邻胞无关（cut band 保留，支持切面载荷），但与 Schwarz-P 族、n = 32、ν、γ 绑定；输入空间比随机模量场小得多 |
| **训练数据/成本** | EML：监督，4e5 个样本；JMPS：**data-free 能量训练**，1e6 个密度、2178 个边界基，CPU 2–3 天 | 监督 MSE，1e5（5³）/1e6（10³）个样本；耗时未报 | 监督，1e4 个样本，3000 epochs | 监督，4e5 + 5e5 个样本，首次共 6.03 h | 能量对数比损失加灵敏度项（需精确 S 归一化和灵敏度标签）；691 个几何，数据生成约 42 GPU-h、占 1.05 TB，训练 4.6 h + 3.5 h，总计约 60 GPU-h |
| **已演示规模** | EML：1.024×10⁹ 个单元（串行，9–10 天）；JMPS：1.8×10⁸ 个单元 | 约 1.04×10⁷ 个单元；另引并行线性 PIML 的 10¹⁰ DOF | 3.6 M 单元（点阵），1 M 单元（半悬臂） | 2.5 M 单元 | 八胞点阵：总 DOF 1.83–2.11 M，自由保留 DOF 0.14 M |
| **加速比与基准** | JMPS：>230×，对比 Matlab 全尺度（Xeon 6256，512 GB）；EML：328–472×，对比经典 SIMP 前 10 步 | 分析 17–722×；TO 迭代 7.4–63×；直接 FEM 求解器未说明，串行单 CPU，128 GB | 46×，对比 PCG + IC | 70–129×，对比 OptiStruct 32 核；30 万单元 FEA 1986 s/138 GB | 四胞：38 s 对 242–256 s（MKL PARDISO Cholesky，16 核，约 6.3–6.7×）；八胞：81–110 s 对 >342/>380 s（仅符号分析）[PENDING E1] |
| **优化演示** | SIMP 柔度、柔顺机构、扭转盒 | SIMP/ITDF，MBB、悬臂、桥、飞翼 | 梯度点阵、SIMP 悬臂 | 柔度、柔顺机构、应力 | 尚无（§6.11 占位） |
| **灵敏度** | 场基（JMPS Eq. 18），忽略 ∂N/∂ρ | 未说明 | 未说明 | 场基；应力问题用反向传播计入 ∂N/∂ρ | 场基 s̃，并推导完整导数 Eq. (14) [PENDING E3]；**逐胞报告灵敏度误差** |
| **理论** | 刚体不变性；观察到柔度偏小，但无界 | Bézier 插值的平移/转动不变性证明；Remark 2 称收敛分析"forthcoming" | 列补全误差放大的统计估计（Eq. 34–35） | 刚体不变性与秩充分 | Ritz 恒等式 Ŝ − S = HᵀAH ⪰ 0（单侧界）、参与度界 Eq. (12)、灵敏度界（App. J.4）、校正单调不增（Eq. 17） |
| **是否有校正/平滑** | 无，只有刚体补全 | 无；建议在荷载点附近保留全部边界节点并做精确凝聚（p.17） | 投影校正（rank-6），48 次对称平均 | 无；应力问题转到细网格继续优化（warm start） | 固定的、逐几何的 Chebyshev + Q1(17) Galerkin + Chebyshev 校正 W，并参与训练 |

说明：
1. 系列论文的输入是 m³ 维随机模量场，含近零模量和不连通材料；我们的输入是低维参数族（8 个厚度参数加一个切面）。**精度上的差距不能单独解读为方法优劣**，任务难度不同。
2. 系列论文的误差参考是细网格 FEM 或同一边界假设下的精确子结构；我们的参考是同一离散模型的精确凝聚。比较 NN 自身误差时，应使用表中第一行。

---

## 3. 可用于稿件和答复信的具体发现

### (a) CMAME26 的 "Full Node Prediction"：与我们保留完整迹最接近的做法

- **Table 1（p.14）**：悬臂梁 2×1×1，80×40×40 网格，模量在 [0,1] 均匀随机，均布载荷。误差为位移 L2 相对误差，参考为直接 FEM，**直接 FEM 耗时 307.15 s**。

| 子结构 | 方案 | 控制点/子结构 | 时间 (s) | 误差 | 加速 |
|---|---|---:|---:|---:|---:|
| 5³ | Linear | 8 | 0.87 | 7.76% | ~352× |
| 5³ | Quadratic | 26 | 1.94 | 2.37% | ~158× |
| 5³ | Cubic Bézier | 56 | 5.64 | 1.95% | ~54× |
| 5³ | **Full Node** | 152 | **40.76** | **0.84%** | ~7.5× |
| 10³ | Linear | 8 | 0.81 | 13.64% | ~379× |
| 10³ | Quadratic | 26 | 1.56 | 5.23% | ~196× |
| 10³ | Cubic Bézier | 56 | 2.87 | 2.22% | ~106× |
| 10³ | **Full Node** | 602 | **42.14** | **3.46%\*** | ~7.3× |

  星号 \* 在抽取文本中没有脚注说明。
- **作者对 10³ 全节点方案精度下降的解释**（p.14–15）：原文为"reduced predictive capability of the corresponding neural network when all boundary degrees of freedom are retained"，即网络输出变量增多后预测能力下降。也就是说，退化的原因是**网络误差**，不是边界模型误差。
- **作者对全节点方案成本的论证**（§2.2，p.5）：保留全部边界节点会在全局矩阵中形成稠密块、增大带宽，并称经验上当子结构约为 20×20×20 或更大时，"the additional cost resulting from reduced sparsity can outweigh the savings in the linear-solve phase"。
- **Table 2（p.15）**：桥型设计 240×60×60，直接 FEM 1472.64 s。全节点方案耗时 >1000 s，因此作者"no longer counted as an effective acceleration method"，未列入表中。
- **对我们的意义**：
  1. 这一系列已经试过"保留完整边界并学习内部"，结论是太慢，而且在 10³ 时网络误差反而更大（3.46%）。**我们的定位应主动承认这一点**，不能写成"系列从未考虑"。
  2. 两者差异可以量化：他们的全节点方案每个子结构保留 456–1806 DOF，内部只有 192–2187 个 DOF，没有校正，NN 误差为 0.84–3.46%（位移）；我们每胞保留 1.7–2.6×10⁴ DOF，内部 DOF 在 10⁵ 量级，经校正后能量误差为 0.074%。另外，NICE-post 说明同一网络加上校正后，M1 的误差从 13.5% 降到 0.19%（§6.5）。**"输出变量增多导致网络精度下降"正是平衡校正 W 要解决的失效模式**，这个论点可以直接用在答复信里。
  3. 他们的成本论证只针对体素子结构。我们的对应论证是：薄壁 cut 胞的内部远大于边界（单胞总 DOF 约 0.07–0.40 M，保留 DOF 为 1.7–2.6×10⁴，内部与保留之比约 3–15；例如 M1 的内部 DOF 为 165,927，见 §6.5 和 ST17），而且需要 cut band 来承担切面虚功。
  4. **可比性提醒**：他们的误差是随机模量场上的全场位移 L2 误差，参考为细网格 FEM；我们的是能量和柔度误差，参考为离散精确凝聚。在正文中不宜把两组数字并列排比，答复信中可以引用，但要写明口径。

### (b) 柔度与位移误差

- **"63.1% → 15.8%"（结论，p.25）**：原句称柔度预测误差从 63.1%（线性插值）降到 15.8%（Bézier）。对应 §4.1（p.15–17）的集中载荷悬臂梁：网格 200×100×100，子结构 5³，模量随机。正文给出的是**加载点竖向位移误差 63%（线性）和 15%（Bézier）**，加载点同时也是最大位移点。单位集中力下加载点位移等于柔度，因此按柔度解读是成立的。同一算例的全场位移 L2 误差为 **8.42%（线性）和 5.63%（Bézier）**（p.17）。作者承认在集中载荷下"any algorithm that assumes the substructure boundary displacements will inevitably lead to an overly stiff numerical response"（p.17）。
- **CMAME26 的优化设计全部没有细网格复核**。文中报告的柔度（1978.6、1883.1、809.8、1154.2、23.96、20.29）都是 PIML 自身的数值（p.18、p.22）。
- **系列中所有经细网格复核的柔度误差都偏小（偏刚）**：JMPS 为 −4.7%（p.12）；CS26 为 −9.17%、−2.79%、−3.14%（Table 6，p.15）；IJMS App. C 为 −20.7%（p.20）。这与 Ritz 恒等式预测的单侧性一致，因为他们用 NᵀKN、边界行精确。**答复信可以指出：我们的界同样适用于他们的 NᵀKN 变体（边界受限时下界为 LᵀSL），而我们首次给出了显式界并做了定量验证。**
- **仅由边界假设造成的误差**（内部精确），JMPS Table 2（p.5）：m = 10、10×5×5 个子结构时为 2.92%，4×2×2 时为 11.9%；m = 5、20×10×10 时为 0.98%。**这是用网格细化控制边界误差的直接证据**，可以支撑我们 §1 中"refining the partition"的说法。
- 系列中**没有任何一篇报告灵敏度误差**（相对精确灵敏度的误差或余弦）。**我们逐胞的八角点灵敏度误差是这条线上首次报告的灵敏度精度**；IJMS 的应力 p-norm 约 5 倍偏差说明，局部量误差确实可能远大于全局量。

### (c) 直接 FEM 基准与硬件

| 论文 | 基准 | 硬件 | 是否说明基准求解器 |
|---|---|---|---|
| EML23 | 经典 SIMP 全尺度 FEA，只计前 10 步平均（p.9） | Xeon Gold 6256、512 GB，文中称"laptop" | 否 |
| JMPS24 | Matlab 2020b 全尺度（Table 1：0.25 M 单元约 1320 s；Table 3：11.52 M 单元 11,761 s） | 同上 | 否，只注明 Matlab |
| CMAME26 | "Direct FEM (Benchmark)"：0.41 M DOF 307 s，2.69 M DOF 1473 s，6.15 M DOF 1262 s | i9-13900K、128 GB，**串行单 CPU**（p.14、p.25） | **否** |
| CS26 | PCG + incomplete Cholesky（半域 1 M 单元，646.73 s/迭代） | i7-13700、32 GB | 是 |
| IJMS26 | OptiStruct 32 核（8223 s/迭代）；App. C 全尺度 FEA：30 万单元 1986 s、137.9 GB | Xeon Gold 6246R、512 GB | 只注明 OptiStruct |

- CMAME26 基准的**时间与规模不单调**：2.69 M DOF 需 1473 s，而 6.15 M DOF 只需 1262 s，求解器又未说明，因此无法复核。0.41 M DOF 用 307 s，对现代稀疏直接法（PARDISO/CHOLMOD）来说偏慢。答复信里可以**客观地**说明："the PIML studies compare against a full-scale FE solve whose solver is not always specified (serial, single CPU in Guo et al. 2026a); Jiang et al. (2026) use PCG with incomplete Cholesky; Zhang et al. (2026) use a 32-core commercial solver."我们的基准是 MKL PARDISO Cholesky（16 核），至少不弱于系列的基准。**这支持 CONSOLIDATED 中已决定的 §6.10 基准说明句**（"PIML studies compare against full-scale FEM"），建议把句子改得更具体。
- 系列报告的加速多为 10²×，我们的四胞加速约 6.5×。**不宜在稿中比较加速比**，因为基准强度、精度层级和规模都不同。

### (d) 灵敏度的计算方式

- JMPS24 Eq. (18)–(19)（p.10）：**场基估计**，即用恢复的细单元位移计算 −∂E/∂ρ·u_eᵀk_eu_e；原文称"for ease of implementation"按全尺度的公式计算，**忽略 ∂N/∂ρ**。这与我们的 s̃_c（§4.3）完全对应。
- IJMS26 Eq. (24)、(27)：同样是场基估计。Eq. (30)–(31) 对应力问题加入 T3 项，**用反向传播求 ∂N/∂ρ**（App. A）。可见作者清楚存在设计依赖项，但在柔度问题中省略了它。**对于学习得到的 N，柔度导数中的 ∂N 项并不为零**：它正是我们 Eq. (14) 中的 −2(F_{I,c}q̂)ᵀr_I。**我们的 Eq. (14) 和 App. J.5 把这一省略所带来的误差量化了，这是相对于该系列的一项明确增量。**
- CMAME26 和 CS26 没有给出灵敏度公式；CS26 只作定性表述（"more accurate sensitivity estimation"，p.13）。EML23 也没有给出公式。

### (e) 是否有 energy-trained/data-free 方法，是否有校正/平滑

- **JMPS24 是 data-free 的能量训练**：最小势能损失（Eq. 13–15），在 pseudo structure 上用随机正交边界基训练，不需要标签（p.8–10）。作者还称其泛化优于监督 MSE，且"over-fitting is seldom observed"（p.10）。我们的 Eq. (8) 同样是 Ritz 型能量损失。其能量项对 θ 的梯度与 log(qᵀŜq) 相同（qᵀSq 只起方向加权作用），所以能量项本身近乎无标签；但力学方向（force、face、support、glued 等）的生成需要精确求解，灵敏度项还需要精确灵敏度标签，因此整体**不是 data-free**（数据生成约 42 GPU-h）。CONSOLIDATED 的 I-33（R2-17、R7-06）已要求在 §3.3 补充这段对比，data-free 试点正在进行，可参考第 5 节第 1 条。
- 本系列后续论文（CMAME26、CS26、IJMS26）**都回到了监督 MSE**；PIML-OFEM 也是监督学习，损失为 Smooth-L1 加刚度保真项。
- **没有任何一篇对学习到的内部场做平衡残差的迭代校正（平滑、粗网格校正或多重网格）。** 已有的"校正"都是代数约束或流程层面的处理：
  - 刚体列补全：EML23 Eq. 13–15、JMPS24 Eq. 11、CMAME26 Eq. 30–31、IJMS26 Eq. 20；
  - CS26 的 rank-6 投影 N̂ = ÑP + N_r，外加 48 次对称平均；
  - Galerkin 对称化 NᵀKN（CS26 Eq. 25；EML23 Eq. 17）；
  - 损失层面的一致性约束（EML23 的第二类损失；OFEM 的 L_K）；
  - IJMS26 的"PITL warm start 后转细网格继续优化"，以及每 5 步做一次细网格复核（p.13）。
- CMAME26 Remark 1（p.12）提出可以把同一 ML 方法用于"learn projection operators in multigrid methods"，但没有实施。**这是我们"学习初场加多重网格校正"的最近呼应，可在引言中顺带提及**，以免审稿人认为我们在关键点上忽视了这条线索。

### 其他可引用事实

- EML23 摘要中的"10⁴–10⁵ times"与正文的 328–472×（p.1 对 p.9）不一致；JMPS24 Table 3 与正文数字写反（p.14）。**引用时应以表格为准**，不要转引摘要数字。
- CMAME26 引言称在 >10⁷ DOF 问题上有"2–3 orders of magnitude speedups"（p.3），但表中 TO 迭代的加速只有 7.4–63×（p.19），分析加速的最大值 722× 对应的是线性插值、误差 13.53%（p.15）。结论改称"one to two orders"（p.25）。
- IJMS26 自己承认首次离线成本并未降低（6.03 h 对 3.31 h，p.8）。
- CS26 与我们都使用 48 个立方对称操作做数据增强，但他们通过规范化加平均实现了**严格等变**（等变残差 3.25×10⁻¹⁶，Table 2，p.8）。我们在 §7.5 中说明网络不等变，只在恒等朝向下评估，CS26 给出了现成的补救办法。

---

## 4. 稿件中引用或描述这些工作的句子：逐句核查

原句以英文给出，句段可能截断。结论分三类：accurate（准确）、imprecise（大体正确但不严密或有缺漏）、inaccurate（错误）。

| # | 位置 | 原句（摘要） | 结论 | 依据 | 建议改写 |
|---|---|---|---|---|---|
| 1 | MS §1 ¶5（l.19） | "Problem-independent machine learning (PIML) predicts substructure shape functions, imposes rigid-motion constraints and forms the stiffness as NᵀKN [Huang et al. (2023)]." | **imprecise** | 内容属实（EML23 Eq. 13–15、Eq. 17）。但 PIML 首次提出于 Huang et al. (2022, EML 56, 2D)；且 EML23 还提供了由网络直接预测 K̃ 的另一种算法，NᵀKN 只是自洽选项（p.7） | "Problem-independent machine learning (PIML), introduced for two-dimensional substructures [Huang et al. (2022)] and extended to three dimensions [Huang et al. (2023)], predicts substructure shape functions, imposes rigid-motion constraints and, in its self-consistent form, evaluates the stiffness as NᵀKN." |
| 2 | 同段 | "Its data-free variant trains the shape functions by minimum potential energy, extending the displacement of a three-dimensional substructure from 24 corner degrees of freedom [Huang et al. (2024)];" | **accurate** | JMPS24 Eq. 13–15；24 个角点 DOF 见 App. B | 可补充"without labelled shape functions, on a pseudo-structure of random density fields"，以便与我们需要标签的训练形成对比 |
| 3 | 同段 | "applications include lattice optimisation [Xu et al. (2025)], Bézier-enriched boundaries [Guo et al. (2026a)] and overlapping local bases [Guo et al. (2026b)]." | **imprecise** | Guo 2026a 和 2026b 是**方法扩展**，不是应用；Guo 2026b 为二维，用 oversampled 基加重叠单位分解。Xu 2025 没有全文，但 CS26 p.10 和 IJMS26 p.2 的描述与之一致。另外遗漏了 CS26（等变）和 IJMS26 / Zhang 2024（复杂域、等参） | "It has been extended to Bézier-enriched boundaries [Guo et al. (2026a)], oversampled bases joined by an overlapping partition of unity [Guo et al. (2026b), two-dimensional], symmetry-equivariant shape functions [Jiang, C. et al. (2026)] and isoparametric substructures for complex domains [Zhang et al. (2026)], and applied to lattice optimisation [Xu et al. (2025)]." |
| 4 | 同段 | "Describing each substructure by a few boundary coordinates gives coarse models small enough for very large structures, at the price of a boundary model error independent of the learned interior, which this line of work controls by refining the partition, enriching the interpolation or oversampling." | **accurate**（可补强） | 网格细化：JMPS Table 2，p.5，Fig. 8d；插值增强：CMAME26；oversampling：OFEM。"independent of the learned interior"有 IJMS App. C 支持：PITL 24.8% 对 Sub 21.5% | 可在句末加"— or by retaining all boundary nodes near concentrated loads [Guo et al. (2026a)]"（CMAME p.17） |
| 5 | 同段 | "The present framework makes the opposite trade: it keeps the complete retained space, so that its approximation error lies in the interior extension, …, at the price of tens of thousands of coarse coordinates per cell." | **imprecise**（有被审稿人质疑的风险） | CMAME26 Table 1（p.14）已经测试过保留全部边界节点并用 DeepONet 学习内部，5³ 误差 0.84%，10³ 误差 3.46%，约 7.3–7.5×，随后以成本为由放弃（p.5、p.15）。"opposite trade"容易被读成该系列从未尝试 | "Guo et al. (2026a) also evaluated retaining every boundary node with a learned interior for 5³ and 10³ voxel substructures and found it one to two orders of magnitude slower than Bézier interpolation and, at 10³, less accurate because of network error (their Table 1). The present framework makes this trade for cut thin-walled cells: it keeps the complete retained space, … and a fixed equilibrium correction reduces the interior error that limits a learned extension on a large retained space." |
| 6 | MS §1 ¶7（l.25） | "Any admissible, rigid-motion-reproducing extension evaluated in the energy form, as in … learned shape-function substructures [Huang et al. (2023)], yields a symmetric positive semidefinite condensed stiffness with the rigid kernel, bounded below by the Schur complement with a quadratic excess" | **imprecise** | 在 PIML 中，边界行由插值矩阵 L 精确给出，因此下界是**受限迹上的** Schur 补 LᵀSL，而非全迹 S；而且只有用 NᵀKN（EML23 Eq. 17）时才成立，EML23 中直接预测 K̃ 的做法不满足（其柔顺机构算例就用了该做法，p.10） | "… learned shape-function substructures when the stiffness is evaluated as NᵀKN [Huang et al. (2023)], yields … bounded below by the Schur complement on the retained (possibly interpolated) trace — LᵀSL for an interpolated boundary — with a quadratic excess" |
| 7 | MS §6.8（l.442） | "…it is not a model of reduced-boundary substructure methods, which control this error by partition refinement, boundary enrichment or oversampling (Supplementary Note S4)." | **accurate** | 同 #4 | 无需修改；可加上引用 [Huang et al. (2024)], [Guo et al. (2026a,b)] |
| 8 | MS §7.1（l.483） | "Substructure methods that describe the boundary by a few coordinates [Huang et al. (2023)], [Guo et al. (2026a)] accept this boundary error, which they control by partition refinement, boundary enrichment or oversampling [Guo et al. (2026b)], in exchange for coarse models small enough for very large structures." | **accurate**（引用分配可更精确） | 网格细化的证据在 JMPS24（Table 2，Fig. 8d），而不是 EML23；boundary enrichment 对应 2026a；oversampling 对应 2026b | "… accept this boundary error, which they control by partition refinement [Huang et al. (2024)], boundary enrichment [Guo et al. (2026a)] or oversampling [Guo et al. (2026b)], …"，并可加入 #5 中关于 Full Node 的一句 |
| 9 | MS §7.1 | "For the local sensitivity of the cut cells examined, the boundary restriction produces larger errors than the interior approximation of NICE (…for single cells at fixed size); where a compact coarse model matters more, the balance can differ." | **accurate** | 只比较我们自己的消融（ST14），没有对 PIML 下结论 | 可补充"none of these studies reports sensitivity errors against exact sensitivities"（见第 3(b) 节），以强调我们的增量 |
| 10 | SUPP S4 首段（l.539） | "…reduced-boundary substructure methods, which control the boundary error by refining the partition, enriching the boundary interpolation or oversampling overlapping local bases [Guo et al. (2026a)], [Guo et al. (2026b)]; boundary-space reduction is also used by learned shape-function substructures [Huang et al. (2023)], [Huang et al. (2024)]." | **imprecise** | (i) Guo 2026a 本身就是学习形函数子结构（DeepONet），此句把它与"learned shape-function substructures"对立起来；(ii) "refining the partition"应归于 Huang 2024；(iii) OFEM 的准确描述是在扩展域上构造 oversampled 基，再用重叠单位分解拼接 | "It is not a model or a reproduction of reduced-boundary learned substructures [Huang et al. (2023)], [Huang et al. (2024)], [Guo et al. (2026a)], [Guo et al. (2026b)], which control the boundary error by refining the partition, enriching the boundary interpolation (cubic Bézier: 56 control points per substructure) or oversampling local bases joined by an overlapping partition of unity." |
| 11 | SUPP S4.2 | "One cell per substructure, the fine-scale consistent face tractions used throughout, and no oversampling." | **accurate** | — | 可加："By contrast, Guo et al. (2026a) use 5³–10³-voxel substructures and report 1.95–2.22% displacement error with cubic Bézier faces under a distributed load (their Table 1)"，用来说明 r = 3 的 38.9% 不能与之对比 |
| 12 | SUPP ST14 表注 | 数据文件名 `piml4_all_*.json` 与 `piml4_interface_*.json`（"historical file names"） | 不属于事实错误，但容易误读 | 文件名暗示这是 PIML 复现 | 归档时改名为 `bernstein_all_*`，或在表注中写明"not a PIML reproduction" |
| 13 | MS 参考文献 | Guo 2026a、Huang 2023、Huang 2024、Guo 2026b 四条的作者、题目、卷号 | **accurate** | 已与全文首页核对 | 若新增 CS26，需要与现有的 "Jiang et al. (2026)"（H. Jiang, CNEE, arXiv）区分，建议写作 "Jiang, C. et al. (2026)" 和 "Jiang, H. et al. (2026)" |
| — | APPENDICES | 没有句子引用本系列（只有 l.546 指向 S4） | — | — | — |
| — | 同名文献 | "Zhang et al. (2024)"（E. Zhang, *Nat. Mach. Intell.*）、"Xu (1992)"、"Jiang et al. (2026)"（H. Jiang, arXiv）都**不属于本系列** | 未核对（未提供全文） | — | — |

**建议新增的引用**：Huang et al. (2022, EML 56) 作为 PIML 的起源；Jiang, C. et al. (2026, CS 397) 可在 §7.5 讨论等变性时引用；Zhang et al. (2026, IJMS 328) 可在讨论几何泛化或迁移时引用；Chen & Li (2026, CAD 193) 可在 cut/unfitted 粗单元与数值形函数的相关工作中引用，它与我们的 cut 胞最接近。

---

## 5. 对下一篇论文有用的技术（按可操作性排序）

1. **data-free 能量训练（JMPS24）→ 去掉 42 GPU-h 的精确解数据（与 I-33 的试点一致）。** 由 Ritz 恒等式可知 qᵀŜq ≥ qᵀSq，因此直接最小化 Σ_j q_jᵀŜq_j 就是无标签的 Ritz 训练；方向 q_j 按 q_jᵀK_PPq_j = 1 或按 harmonic 延拓的能量归一化即可，不需要 S。需要去掉的是依赖精确求解的力学方向，只保留免求解的方向（macro、grf）。这与 JMPS 的 2178 个正交边界基是同一思路，对我们来说可以换成多项式或多尺度方向，加上 pseudo-lattice 中邻胞诱导的方向。灵敏度项可以改为无标签的残差项 ‖r_I‖²_{A⁻¹} 的代理，例如 Eq. (10) 的 D⁻¹ 加权形式，或者用 NICE 校正后的场作为 teacher 做自蒸馏。预期收益：离线成本从约 60 GPU-h 降到训练本身的约 10 GPU-h，并可以直接扩展到 n = 48/64、其他 TPMS 族。风险：JMPS 报告固定边界基时容易陷入局部极小（p.9），需要逐步更换方向。
2. **迁移学习（IJMS26 PITL）→ 新 TPMS 族或新分辨率。** 冻结几何编码器和线性位移支路，只微调系数头，或像 IJMS 那样加一个几何调制支路，用于 gyroid、diamond 或 n = 48。IJMS 的数据可作为预期：收敛迭代从 16,000 降到 5000，但**首次总成本不降**（p.8）。应当如实报告"源模型成本加每族适配成本"。
3. **严格立方对称等变（CS26）。** 把胞变换到规范朝向（以切面法向和厚度梯度为准则），预测后逆变换，或对 48 个像取平均，从而得到严格等变，补上 §7.5 的局限。CS26 的等变残差为 3×10⁻¹⁶，网络参数减少约 80%。由于 NICE 的前端要做 48 次前向，更现实的做法是规范化加一次前向。
4. **规模：混合边界表示与 Bézier 粗空间。** (i) CMAME26 p.17 的思路是在荷载点附近保留全部边界节点，其余用 Bézier。对我们来说，可以在 cut、加载或设计敏感的胞保留完整迹，在远场的未切割胞之间用 Bernstein r = 5–8 的界面。按 ST14b，只限制界面、r = 5 时，柔度误差为 0.01–0.07%，但邻载灵敏度误差为 2.2–3.4%，因此只适合对灵敏度不敏感的区域。(ii) 更稳妥的做法是**把 Bézier/Bernstein 面模态当作 PCG 的粗空间**（GDSW 风格），离散本身保持完整迹。现有两层预消除器是三线性宏顶点乘刚体模态，迭代数为 114–165；加入 r = 2–3 的面模态有望明显减少迭代，精度不受影响。(iii) 并行：Ma et al. 在 6750 核上以每次迭代 42 s 求解 100 亿 DOF，说明子结构级并行可行。我们的逐胞前端和校正天然可以多 GPU 并行，下一篇应报告多 GPU 下 ≥27–125 胞的可扩展性。
5. **完整导数与伴随（IJMS26 App. A）。** IJMS 已演示用反向传播求 ∂N/∂ρ。我们的 Eq. (14) 中 −2(F_{I,c}q̂)ᵀr_I 项可以用 autograd 穿过网络和校正（W 在固定几何下是线性的）得到，从而直接完成 E3，并给优化器提供与代理目标一致的梯度。
6. **应力或局部量作为差异化卖点。** IJMS26 在线性边界下把 p-norm 应力低估约 5 倍（p.13），CMAME26 在集中载荷下位移误差达 15%。我们保留完整迹，并在 §6.4 中说明位移误差会在薄壁中放大为能量误差，因此应当补做**应力恢复精度**（如 von Mises 的 top-1% 区域）实验，这是 PIML 线路的公认短板。
7. **廉价训练的实证设计（CMAME26 Fig. 8）。** 对训练几何数（148、305、591）和方向数做数据量收敛曲线，把"为什么是 591 个几何"量化，同时回应 R2 关于端到端成本的问题。
8. **unfitted 条件数（CAD26）。** 若以后在 cut 面上限制边界表示，CAD26 的"删除病态非顶点并重写顶点插值以保持单位分解"可以直接借用；Theorem 2 的 κ = O(1/(Hhε²)) 也能用来论证我们在 cut band 上保留完整迹以避免小切割病态的合理性。

---

## 附：本文件的数字来源一览（便于复核）

- CMAME26：p.5（§2.2 带宽论证）、p.8（56 个控制点）、p.11–12（训练与样本）、p.14（Table 1，硬件）、p.14–15（全节点精度退化的解释）、p.15（Table 2，>1000 s）、p.15–17（63%/15%，8.42%/5.63%）、p.19（悬臂梁每迭代时间）、p.20（桥 >1.2×10⁷ DOF）、p.22（飞翼）、p.25（63.1%→15.8%，串行，100 亿 DOF）。
- JMPS24：p.4–5（Table 1/2，硬件）、p.6（监督方案成本）、p.8–10（能量损失，pseudo structure，2–3 天）、p.10（灵敏度 Eq. 18）、p.11–12（6.20%、6.33%、4.7%、87×）、p.14（Table 3）、p.15–16（局限）。
- EML23：p.1（摘要）、p.4（3 h/10⁵ 个样本）、p.6–7（网络与 400k 个样本）、p.8（硬件）、p.9（472×/328×）、p.10（7.56%/7.65%）、p.11（10⁹ 算例）、p.12（CutFEM 展望）。
- CS26：p.2（系列综述）、p.5–6（对称解耦）、p.6（训练与硬件）、p.7（R²）、p.8（Table 2，误差放大）、p.11（Table 3）、p.13（Table 4）、p.14–15（Table 5/6，PCG 基准）。
- IJMS26：p.4–7（样本、网络、训练）、p.8（Table 2 成本）、p.9（灵敏度）、p.10–12（支架算例与 OptiStruct 对比）、p.13–14（应力算例）、p.15–16（局限）、p.17–18（App. A）、p.20（App. C）。
- CAD26：p.5（Theorem 2）、p.6（reduced element）、p.8（实现与硬件）、p.9（Table 2）、p.12（PCB 算例）。
