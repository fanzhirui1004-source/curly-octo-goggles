# 计算力学中英文术语库

本文按数学对象选择用词，保持 trace（迹）、interface（界面）和 retained coordinates（保留坐标）的区别。42项术语用于主稿、附录和图注。本轮同步当前公式、图表与章节编号，并补充固定离散载荷、设计载荷导数以及粗系数列与机械保留坐标的区别。

标题：**Learned static condensation for cut thin-walled TPMS cells with equilibrium correction**。

## 术语、定义和替换边界

### T01 · cell（胞）

**定义：**由隐式几何定义的一个TPMS实体单元；在组装中可担当子结构。
**使用：**几何、样本、厚度和切割描述。
**替换：**module（指几何时）
**区分：**cell（胞）说明几何对象；substructure（子结构）说明力学分解角色。

### T02 · substructure（子结构）

**定义：**具有局部K、保留集合P和可独立消去内部I的离散力学组成部分。
**使用：**静力凝聚、局部刚度和全局组装。
**替换：**module（指局部力学系统时）
**区分：**不因使用子结构一词就意味着端口或边界已经降阶。

### T03 · retained degrees of freedom（保留自由度）

**定义：**P内保留于Schur系统的位移系数，包括盒面及切割带；q=J_P u。
**使用：**全文P、p及组装输入输出。
**替换：**interface DOFs（泛指整个P时）；boundary DOFs（泛指整个P时）
**区分：**部分切割带系数的节点离开实体表面；P不必是迹函数的最小独立基。

### T04 · full retained space（完整保留空间）

**定义：**本文选定P的全部坐标，未再施加边界插值或端口子空间约束。
**使用：**标题后的首次定义、摘要、结论和算法对象。
**替换：**full trace（泛指P而未定义时）；full interface（泛指P时）
**区分：**完整相对于选定的离散P，不代表连续体无离散误差或训练覆盖所有方向。

### T05 · interface（界面）

**定义：**两个子结构共同连接并通过位移兼容/力平衡交换作用的区域。
**使用：**公共盒面、邻胞耦合、共享自由度。
**替换：**保留该精确术语，不做机械替换。
**区分：**不与外边界、整胞边界或切割带互换。

### T06 · boundary（边界）

**定义：**实体或胞的几何边界，可包含外边界与公共界面。
**使用：**边界条件、表面载荷、盒面和切面。
**替换：**保留该精确术语，不做机械替换。
**区分：**几何边界不是P内所有节点的集合。

### T07 · displacement trace（位移迹）

**定义：**位移场限制到指定表面后得到的函数。
**使用：**盒面多项式限制、Q2表面载荷及迹范数。
**替换：**保留该精确术语，不做机械替换。
**区分：**保留系数表示迹，但可同时包含离面体场信息；same trace文中按已定义的same retained vector使用。

### T08 · retained cut-band coordinates（切割带保留坐标）

**定义：**具有正面积切面片的活动切割单元所保留的全部Q2节点位移系数。
**使用：**P构造、载荷虚功及胞内私有坐标。
**替换：**cut-surface nodes（指这些系数时）
**区分：**节点不必位于切平面；非盒面坐标不自动成为相邻胞共享坐标。

### T09 · static condensation（静力凝聚）

**定义：**固定保留位移、通过内部静力平衡消去I并获得S的操作。
**使用：**标题、方法定位与式(2)。
**替换：**full-trace condensation（作为未定义类别时）
**区分：**经典消元原理不是本文首创；本文研究学习延拓和内部修正的具体构造。

### T10 · condensed stiffness operator（凝聚刚度算子）

**定义：**作用于保留位移并返回功共轭保留力的线性算子；精确为S，近似为F^T K F。
**使用：**力学输出、作用算法、组装。
**替换：**readout（作为力学对象名）
**区分：**近似算子是变分构造；F未平衡时不等于仅提取(KFq)_P。

### T11 · equilibrium displacement extension（平衡位移延拓）

**定义：**E_P=I、E_I=-A^{-1}K_IP，对给定q求唯一内部平衡场。
**使用：**精确参照E、教师场及能量最小化。
**替换：**exact lifting（不明确平衡时）
**区分：**equilibrium（平衡）为本文三维弹性离散对象；不需要引入易混淆的标量调和含义。

### T12 · admissible displacement extension（可容许位移延拓）

**定义：**满足J_P F=I的延拓；固定几何时关于q线性，但未必满足内部平衡。
**使用：**学习场和修正场的F。
**替换：**displacement path（指F时）
**区分：**lifting（提升）可作数学近义词，但本稿优先extension（延拓），不凭lifting推断平衡。

### T13 · rigid-body reconstruction（刚体运动重构）

**定义：**用R和R_P从保留刚体系数重建平移与转动，确保F R_P=R。
**使用：**式(B.3)、式(5)、图2–3与网络外围确定性操作。
**替换：**rigid lifting；rigid handling
**区分：**刚体再现与K只有六个刚体零模态是两个条件。

### T14 · transpose of the complete extension（完整延拓的转置）

**定义：**在当前欧氏系数配对下对网络、刚体处理及内部修正完整转置得到F^T。
**使用：**式(7)、反向作用算法、附录E。
**替换：**complete adjoint（未指明离散配对时）
**区分：**此处不是另求一个连续PDE伴随问题；设计导数的伴随法仍可称adjoint method（伴随法）。

### T15 · stabilised discrete energy（稳定化离散能量）

**定义：**二次泛函1/2 u^T K u，包含材料弹性能与规定的幽灵罚项。
**使用：**变分定义、能量超额与误差恒等式。
**替换：**physical strain energy（当K含幽灵罚时）；discrete elastic energy（易误认为仅物理项时）
**区分：**physical strain energy（物理应变能）只能指材料弹性积分对应部分。

### T16 · compliance（柔度）

**定义：**固定力载荷下C=f_g^T U；平衡时等于u^T K u，为离散能量的两倍。
**使用：**全局响应与导数，式(A.2)、式(11)。
**替换：**保留该精确术语，不做机械替换。
**区分：**与位移范数、1/2 u^T K u及局部能量参与率区分。

### T17 · relative directional energy excess（相对方向能量超额）

**定义：**epsilon(q)=q^T(Shat-S)q/(q^T S q)，参考能量非零。
**使用：**单个保留方向上的精度及其统计。
**替换：**operator error（仅指方向样本统计时）
**区分：**无额外1/2；均值与百分位不能替代全方向上确界。

### T18 · relative operator error（相对算子误差）

**定义：**刚体补空间上归一化误差算子的谱范数，等价于最大广义Rayleigh商。
**使用：**理论最坏方向界与Ritz搜索解释。
**替换：**maximum error（实为几何方向均值最大值时）
**区分：**Ritz搜索达到的商是上确界的下界；足够方向覆盖需另证。

### T19 · maximum of geometry-level direction means（几何方向均值的最大值）

**定义：**先对每个几何的一类方向取均值，再在几何间取最大。
**使用：**验证分布、ST01及图5。
**替换：**worst-case operator error（指该统计时）
**区分：**不是单个最坏方向，也不按每个几何的方向数量加权。

### T20 · field-based sensitivity estimate（基于场的灵敏度估计）

**定义：**用重构场代入-sum uhat^T K_,c uhat，未加入延拓随设计变化的链式项。
**使用：**所有所报厚度灵敏度误差。
**替换：**surrogate gradient（指该量时）
**区分：**不等于代理柔度完整导数；在精确平衡且固定载荷、映射时等于参考柔度导数。本文固定基准设计的离散载荷。

### T21 · complete design derivative of surrogate compliance（代理柔度的完整设计导数）

**定义：**式(14)含刚度导数项和残差加权F_,c项；涉及多胞时累加受影响胞。
**使用：**设计一致性分析和优化推论。
**替换：**sensitivity（未区分场估计时）
**区分：**小函数值误差不自动给出小导数；共享全局设计还需参数映射链式法则。

### T22 · corner thickness parameter（角点厚度参数）

**定义：**tau_c通过非负Q1形函数定义隐式带半宽tau(x)。
**使用：**几何、八角点敏感度和嵌套增厚论证。
**替换：**wall thickness（直接称tau_c为物理长度时）
**区分：**隐式场非距离函数，tau不是逐点物理壁厚；薄壁三维实体也不是壳单元模型。

### T23 · ghost-penalty stabilisation（幽灵惩罚稳定化）

**定义：**规定K中的跨背景单元面惩罚项，控制小切割区导致的不良离散性质。
**使用：**参照离散与A、S的定义。
**替换：**equilibrium correction（指幽灵罚时）
**区分：**它定义参照K；内部平衡修正只改善F，不改变该K。

### T24 · equilibrium correction（平衡修正）

**定义：**固定保留值，利用内部残差r_I=(Ku)_I更新内部位移。
**使用：**平滑、粗空间校正及组合循环。
**替换：**wrapper（作为方法名）
**区分：**修正作用于F的内部场；并非边界空间降阶或任意刚度稳定化。

### T25 · interior coarse-space correction（内部粗空间校正）

**定义：**以独立内部粗基V求解Galerkin方程并校正内部场；精确解给出A正交投影。
**使用：**式(16)–(17)、方法图及ST04。
**替换：**port reduction（指此修正时）
**区分：**网络潜在层级、内部修正粗空间和全局预条件器粗空间是三个不同对象。相关生成列的数量不是独立维数；PU实例另核对秩与求解精度。

### T26 · Chebyshev smoothing（Chebyshev平滑）

**定义：**用固定多项式p_k(D^{-1}A)衰减内部误差，固定步数和几何系数。
**使用：**谱诊断与式(D.1)–(D.2)。
**替换：**保留该精确术语，不做机械替换。
**区分：**估计上端点不自动保证谱包含；固定步数普通CG一般仍依赖右端而不构成线性映射。

### T27 · Jacobi-scaled interior mode（Jacobi缩放内部模态）

**定义：**广义本征问题A v=lambda diag(A) v的模态，保留位移固定为零。
**使用：**低模态累计能量图。
**替换：**bending mode（仅凭小lambda命名时）；soft Schur response（未证时）
**区分：**和未缩放物理软模态、自由胞凝聚响应、式(B.7)的Rayleigh商不同。

### T28 · balanced two-level preconditioner（平衡两层预条件器）

**定义：**理想对称形式Q+(I-Q A_g) B(I-A_g Q)，本实现细层B取组装K_PP的逆。
**使用：**表4b、ST09及全局PCG成本。
**替换：**BNN-form（首次出现且未定义时）
**区分：**保留原始代码标识BNN以便溯源；不得据名称宣称使用了局部Neumann Schur逆。

### T29 · consistent nodal traction load（一致节点牵引载荷）

**定义：**通过Q2迹形函数对表面牵引积分得到节点力，从而保存离散虚功。
**使用：**面载荷和切面载荷。
**替换：**保留该精确术语，不做机械替换。
**区分：**一致积分不等于合力、合矩为零；self-equilibrated traction（自平衡牵引）另指后者。几何变化后重新积分牵引会产生载荷导数，区别于本文固定基准离散载荷的灵敏度。

### T30 · Boolean assembly map（布尔组装映射）

**定义：**B_m从自由全局向量提取该胞保留坐标，并包含齐次支承消元。
**使用：**式(3)与共享坐标。
**替换：**gather（正文力学定义时）
**区分：**代码中的gather/scatter张量操作可保留在实现附录；不能改变实际排列或局部坐标约定。

### T31 · six-load accuracy criterion（六载荷精度准则）

**定义：**六个规定面载荷中柔度误差最大值和两胞灵敏度误差最大值均不超过3%。
**使用：**通过数量、表3及失败行。
**替换：**gate（科学正文）；certification（指3%准则时）
**区分：**灵敏度准则在两个胞和六个载荷间取最大；仅目标胞的匹配载荷值另列。切面牵引不属于六载荷集合。

### T32 · reduced retained degrees of freedom（降阶后的保留自由度）

**定义：**Bernstein盒面坐标加未受限切割带坐标组成的Galerkin未知量。
**使用：**ST07、6.7节和图10。
**替换：**controlled DOFs
**区分：**这些是代数广义坐标；不是由FE/Gmsh节点产生的几何控制节点。

### T33 · field-only / solution-only replacement（仅替换重构场／仅替换保留解）

**定义：**分别用Fq和E qhat评估局部响应，完整场为F qhat。
**使用：**图8、ST05、式(H.2)。
**替换：**nonlinear replacement diagnostics（易误认为非线性力学）
**区分：**力学仍线性；灵敏度是位移的二次泛函，各范数不构成可加分量。

### T34 · exact discrete reference（精确离散参照）

**定义：**对固定K、P、I和组装映射求平衡所得的参照；数值求解精度另述。
**使用：**所有exact与误差分母。
**替换：**ground truth（暗示连续体或实验真值时）
**区分：**未声称连续体收敛、实验验证或改变离散后仍同一参照。

### T35 · verification（数值核验）

**定义：**检验实现或代数与指定离散/恒等式相符。
**使用：**残差、算子对称、矩阵例子。
**替换：**保留该精确术语，不做机械替换。
**区分：**80 unseen geometries的model validation（模型验证）指冻结模型在未训练几何的评价；不能改称实验验证。

### T36 · energy participation（能量参与率）

**定义：**在同一精确组装载荷下w_m=q_m^T S_m q_m/C。
**使用：**式(12)及匹配载荷解释。
**替换：**保留该精确术语，不做机械替换。
**区分：**不是材料体积分数、几何保留体积分层或单胞误差。

### T37 · recomputed residual（重算残差）

**定义：**使用该次运行实际算子重新计算f-A_run U；区别递推Krylov残差。
**使用：**表4、ST09、式(18)。
**替换：**true residual（未说明参照算子时）
**区分：**重算残差仍不是相对精确参照算子的响应误差；作用/能量不一致另有omega项。

### T38 · fixed discrete specification（固定离散定义）

**定义：**明确K、P、I、刚体场、装配及导数对象的确定性数学设定。
**使用：**科学正文首次对象说明。
**替换：**contract；guardrail；pipeline
**区分：**原始元数据字段名、源文件名及神经网络coefficient gates（系数门控）保留原文。

### T39 · predictor S8; fixed-weight correction of predictor B（预测器S8；预测器B的固定权重修正）

**定义：**S8使用另行继续训练的权重并作八步评价修正；B固定权重修正从同一B预测场出发只改变数值修正。
**使用：**摘要、模型表、固定权重图和跨模型分布。
**替换：**A2_tail8 / A2 -> S8；v2L1 -> B；c_oh -> P0；A0_ctrl / A0 -> C；c_ctrl -> D
**区分：**S8不能称B加八步；也不能由S8与B的差值单独识别八步修正的因果效果。模型标签使用正文直立字形，不改数学C、D、B_m。

### T40 · coefficient columns（系数列）

**定义：**粗空间生成矩阵经过支撑和对角能量筛选后剩余的列；其数量包含可能的线性相关。
**使用：**PU粗表示、ST04及S03B。
**替换：**coarse DOFs（尚未证实列独立的PU计数）
**区分：**独立空间维数是矩阵的秩。Independent basis（独立基）与generating family（生成族）分别使用；不能把删去小对角列等同于揭示秩。这里的剩余列不指P中的机械保留坐标。

### T41 · geometry embedding width（几何嵌入宽度）

**定义：**网络几何分支隐含特征的分量数，当前冻结结构为64。
**使用：**网络结构及方法图。
**替换：**64 coarse channels（误指位移分支时）
**区分：**几何分支对几何非线性，用于产生系数；这个宽度不等于位移通道数或力学自由度数。

### T42 · displacement feature channels（位移特征通道）

**定义：**线性位移传播分支的隐含分量，当前细/粗潜在层均为32。
**使用：**网络结构及方法图。
**替换：**保留该精确术语，不做机械替换。
**区分：**通道是潜在表示维度；机械输入输出仍是同一保留坐标，不能用通道数表示保留空间降阶。

## 期刊用语与证据范围

四个出版商页面用于核对常用词及对象区分；读取范围为摘要或页面可见引言，不作为整篇全文结构对照的完成证明。本文P、K与F的实际定义优先于词频。

- [A static condensation reduced basis element method: Complex problems](https://doi.org/10.1016/j.cma.2013.02.013)，CMAME，2013。读取：Publisher abstract；用途：Static condensation; distinction between interior approximation and interface port representation。
- [A component-based hybrid reduced basis/finite element method for solid mechanics with local nonlinearities](https://doi.org/10.1016/j.cma.2017.09.014)，CMAME，2018。读取：Publisher abstract and visible introductory excerpts；用途：Component interiors, interface coupling, and port reduction terminology。
- [Mechanical and corrosion behavior of sheet-based 316L TPMS structures](https://doi.org/10.1016/j.ijmecsci.2023.108439)，IJMS，2023。读取：Publisher abstract；用途：Sheet-based TPMS is a geometry designation。
- [Design, mechanical properties and energy absorption capability of graded-thickness triply periodic minimal surface structures fabricated by selective laser melting](https://doi.org/10.1016/j.ijmecsci.2021.106586)，IJMS，2021。读取：Publisher abstract；用途：Graded-thickness describes geometry and does not prescribe a shell discretization。
