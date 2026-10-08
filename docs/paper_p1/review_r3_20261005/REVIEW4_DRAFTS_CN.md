# P1 第四轮审读：问题清单与中英改稿（2026-10-08）

四路审读：防御型写作（按作者认可的判据）、数字一致性、中英文对齐、第 5 节可读性。行号为当前 `MANUSCRIPT_EN.md` / `MANUSCRIPT_CN.md` 的行号（两版逐行对齐）。**未改动任何源文件**；每项给出 EN + CN 改稿，待作者确认后写入。

> **执行记录（2026-10-08，作者确认后已写入中英文源文件）**
> - 已写入：A1（结论改为"不超过 0.27% 和 1.49%"）、A4、B1、B2、B6、B7、B9（短版）、B10、B11、B13、B14、B15（删"代价"句，线性句压成一句）、B16、B17（删 6.1 节第 4 段）、B18；C1 仅第 494 行（拆句、保留数字）与第 587 行；C2、C3、C4、C5 全部。
> - 写入时的修正：B2 用 "addresses" 代替有歧义的 "meets"；B14 保留"装配求解占 NICE 分析大部分时间"；C4 的迭代次数出处为表 ST20；C5 第 458 行写明低于 2% 的是 U1、U2、H1 三个胞元（M2 为 3.64%）；C5 第 538 行不写"另一个单精度部分"（校正平时为双精度），保留"相符"而非"归因"。
> - 不改：A2、A3、B3、B4、B5、B8，C1 第 492、498、536、538 行，D 组第 583 行末句。B12 已被第 5.6 节能量份额段的重写取代。

标记：**必改** = 数字不一致或错字；**建议** = 采纳审读意见；**可选** = 价值不大，作者定；**不采纳** = 审读提出但我不同意，附理由。

---

## A. 数字一致性（自查）

核对了摘要、引言、表 3、第 5 节各小节、第 6 节、结论与补充表 ST03、ST04、ST08b、ST09、ST12、ST14、ST17、ST20、ST21 之间的全部头条数字。一致的不列；不一致的如下。

**A1（必改）结论第 3 段，l.641。** "compliance and sensitivity errors below 0.28% and 1.5% in every two-cell configuration"。表 3 与 5.6 节写的是 ≤ 0.27% / ≤ 1.49%（ST08b 的 W1：0.2706% / 1.485%）。两处口径不同。
- EN：…below **0.3% and 1.5%** in every two-cell configuration.
- CN：……在每个两胞元配置中柔度与灵敏度误差分别低于 **0.3% 和 1.5%**。

**A2（可选）5.9 节 l.561 与结论 l.641。** "79 and 105 s" / "79–105 s"，表 6 为 79.2 / 105.5 s；105.5 四舍五入应为 106。
- EN：79 and **106** s；79–**106** s。CN：79 s 和 **106** s；79–**106** s。
- 不改也说得过去（截断），列出供参考。

**A3（可选）5.3 节 l.436 与结论 l.637。** "0.0965%"，ST03 表中为 0.097（表取三位小数）。属于位数差异，不是错误；建议保留 0.0965%。

**A4（必改）SUPPLEMENTARY_CN.md l.633。** "对于 算例 A" 多一个空格 → "对于算例 A"。

**中英文对齐：** 三对文件逐行比对，无实质错位；数字、引用、图表号、式号全部一致。

---

## B. 防御性重复（按判据逐句核查）

总体结论：全文几乎没有模糊对冲（只有 l.434 一处 "essentially / slightly"），预先回应也都放在质疑产生的位置、写得具体，应保留。问题在**重复**：同一句保证在引言、第 4 节、6.1 节和结论里反复出现（"without retraining" 10 处，"forms no matrix" 7 处，"changes with the cut" 8 处，"any SPSD stiffness with PD interior block" 5 处，"5.4 to 265 times" 4 处）。6.1 节大半是第 1、4、5 节和 6.3 节的复述，是主要清理对象。

**B1（建议）l.29，引言定位段。** "although that space changes with the cut" 在引言出现 6 次，此处删去。
- EN：It keeps the complete retained space, represents the interior by a geometry-conditioned network, and improves the approximation inside the condensed operator that is assembled.
- CN：它保留完整保留空间，以几何条件化网络表示内部，并在被装配的凝聚算子内部改进近似。

**B2（建议）l.42，1.1 节末。** 上一轮加的两句，"much larger" 可以给数。
- EN：NICE meets both obstacles. The interior of a thin-walled cell is several times larger than its retained set (Figure 1b retains 9,674 of 72,631 nodes), so that condensation still removes most unknowns; and the size of the network does not grow with the number of retained degrees of freedom, while the equilibrium correction reduces the error that the network leaves (Section 5.5).
- CN：NICE 对这两个问题都有相应的处理。薄壁胞元的内部比其保留集大数倍（图 1b 中 72,631 个节点保留 9,674 个），因此凝聚仍能消去大部分未知量；网络规模不随保留自由度的个数增长，平衡校正又减小了网络留下的误差（第 5.5 节）。

**B3（建议）l.54，第 2 节开头末句。** 通用性声明在 l.35、54、123、619、643 共 5 处；此处改为指向假设 (A1)。
- EN：…, which Sections 3 and 4 must accommodate; their relations use only assumption (A1) on the resulting stiffness (Section 3).
- CN：……这正是第 3、4 节必须处理的；它们的关系只用到所得刚度满足的假设 (A1)（第 3 节）。

**B4（建议）l.210，3.3 节。** (D) 在第 3 节内第三次陈述；缩成指向。
- EN：Both derivatives hold under (D).
- CN：两种导数都只在 (D) 下成立。

**B5（建议）l.249，4.2 节开头。** 与引言第 6 段"三个障碍"重复；缩短。
- EN：The retained set has 2,679 to 45,900 DOFs and changes with the cut, which rules out a per-geometry extension matrix and a network with one output per retained DOF (Section 1).
- CN：保留集包含 2,679 至 45,900 个自由度，且随切割而变化，这排除了逐几何构造延拓矩阵以及对每个保留自由度各设一个输出的网络（第 1 节）。

**B6（可选）l.271，4.2 节。** "权重作用于模板槽位、通道和网格层级"在 l.253 的设计选择清单里刚说过。
- EN：Because the weights do not act on individual DOFs, the same network, of about \(6\times10^5\) parameters (Table ST15), serves every retained set.
- CN：由于权重不作用于单个自由度，同一个网络（约 \(6\times10^5\) 个参数；表 ST15）适用于所有保留集。

**B7（建议，不删、并入注 1）l.330 → l.332。** 审读建议删掉"改变光滑次数会改变多项式，因此误差的减小不一定关于 \(k\) 单调"，理由是没人会质疑。我不同意删：命题 4 对每个 \(k\) 分别成立，读者容易误推出"多做几步一定更好"，这句是保证的边界，应保留，但它属于注 1 的内容。改法：l.330 删去该句；注 1 在"(Appendix D)."之后插入：
- EN：Changing the smoothing degree changes the polynomial, so the ordering holds for each \(k\) separately and the reduction need not be monotone in \(k\).
- CN：改变光滑次数会改变多项式，因此序关系对每个 \(k\) 分别成立，误差的减小不一定关于 \(k\) 单调。

**B8（建议）l.424 末句删除。** "All comparisons that follow are made against this discrete reference (Section 2.1)." 与 2.1 节 l.71 重复。CN 删"以下所有比较均以该离散参考解为基准（第 2.1 节）。"

**B9（建议）l.434 末句。** 全文唯一的模糊对冲（"essentially unchanged"、"slightly higher"），ST03 有数。数字已核：九类平均值相差至多 0.005 个百分点；一致面力下重度切割分层 0.126% 对 0.109%；五个最大几何平均 0.287–0.651%。
- EN：On the 60 geometries outside checkpoint selection, NICE's class means lie within 0.005 percentage points of those over all 80 (0.077% against 0.074% under consistent tractions), its cut-stratum means under consistent tractions are higher (0.126% against 0.109% in the heavily cut stratum), and the five largest geometry means, 0.29–0.65%, all belong to these 60 geometries (Supplementary Table ST03).
- CN：在检查点选择之外的 60 个几何上，NICE 各类别的平均值与全部 80 个几何上的平均值相差不超过 0.005 个百分点（一致面力下为 0.077% 对 0.074%），其一致面力下的切割分层平均值更高（重度切割分层为 0.126% 对 0.109%），且五个最大的几何平均值（0.29–0.65%）均出自这 60 个几何（补充表 ST03）。

**B10（建议）l.456。** "holds exactly in every direction with nonzero interior error and nonzero exact interior field"：非零条件只是比值的定义条件，附录 B.3 已写。
- EN：Then \(\varepsilon=\delta^2\kappa\) identically (Appendix B.3; values in Supplementary Table ST04).
- CN：则 \(\varepsilon=\delta^2\kappa\) 恒成立（附录 B.3；数值见补充表 ST04）。

**B11（建议）l.466 末句。** "eight times the smoothing work" 两句连说。
- EN：The learned field thus supplies what smoothing and the coarse space do not reach.
- CN：由此可见，学习得到的场提供了光滑与粗空间无法达到的那部分内部平衡。

**B12（建议）l.516 末句。** 前半句复述命题 3(b)。
- EN：The relative sensitivity error of a cell is, in addition, divided by its own reference value, which is small for a cell that carries little of the assembled energy (Proposition 3(b), Appendix H.1).
- CN：此外，胞元的相对灵敏度误差还要除以其自身的参考值，而对于承担装配能量很少的胞元，该参考值很小（命题 3(b)，附录 H.1）。

**B13（建议）l.577 末句。** 0.0051 两句前刚说过，"yet…still" 是重复的安抚。
- EN：Both errors grow as the design approaches the bounds (Table ST17).
- CN：两类误差均随设计趋近界限而增大（表 ST17）。

**B14（建议）6.1 节第 1 段 l.613，切割带两句。** "without retraining" 第四次出现；37–50% 的代价在 6.3 节是指定位置。
- EN（替换 "The retained cut band keeps … (Table ST12d)." 三句）：The retained cut band carries directly the support for which the homogenised model needs a separate constraint on the plane of the cut (Section 5.10); where the cut surfaces carry nothing, it enlarges the assembled system (Section 6.3).
- CN（替换"保留的切割带使……（表 ST12d）。"三句）：保留的切割带直接承担了均匀化模型需要在切割平面上另行施加约束才能表示的支撑（第 5.10 节）；在切割面不承载之处，它会增大装配系统（第 6.3 节）。

**B15（建议）6.1 节第 2 段 l.615，删两句。**
- 删 "Exact linearity in the retained displacement gives superposition; … returns the work-conjugate force."（与贡献 (ii) 和 4.2 节近乎逐字重复）。
- 删 "The price of this generality is that … (Section 6.3)."（与 6.3 节重复）。
- CN 对应删"对保留位移的精确线性给出叠加性；……其转置给出功共轭力。"和"这种通用性的代价是……（第 6.3 节）。"

**B16（建议）6.1 节第 3 段 l.617。**
- "The learned field supplies what neither reaches at a comparable budget: under the same correction, a harmonic start leaves 5.4 to 265 times … in only one of six cells." → "The learned field supplies what neither reaches at a comparable budget (Table 4)."
- 删 ", so an existing network need not be retrained to benefit from the correction"。
- CN："学习场提供了二者在相当预算下都无法达到的部分：……弥合了与校正后学习场的差距。" → "学习场提供了二者在相当预算下都无法达到的部分（表 4）。"；删"，因此现有网络无需重新训练即可从校正中获益"。

**B17（建议，二选一）6.1 节第 4 段 l.619。** 与结论 l.643 近乎逐字重复（上一轮挪到这里的）。建议删 6.1 的这一段，保留结论的。若作者想留在 6.1，则改结论。

**B18（建议）6.2 节 l.623。** 审读建议删 BDDC/FETI-DP 对比句和其后一句。我建议只删后一句（与 5.9 节重复），保留对比句（1.1 节 l.48 的 "(Section 6.2)" 指针指向它）：
- 删 "Its output is a condensed operator per cell with a quantified and reducible error; whole-lattice iterative solvers converge to the exact discrete solution but repeat the fine-scale setup and assembly of every cell (Section 5.9)."
- CN 删"其输出是每个胞元的一个凝聚算子，其误差可量化且可降低；整体点阵迭代求解器收敛到精确离散解，但要重复每个胞元的细尺度设置与装配（第 5.9 节）。"

**不采纳的审读意见（附理由）：**
- l.21 末句"NICE is trained and checked … (Section 5.6)"：摘要、引言、贡献、结论各一次是正常结构，且 5.6 指针是上一轮刚按编辑建议加的。保留。
- l.253 "Weights act on stencil slots…"：这是未做消融的设计选择清单的一项，删了清单不完整。保留（改 l.271，见 B6）。
- l.407 "the sixteen cells … entered no selection step"：5.1 是数据卫生的集中披露处，5.8 是读者需要的地方，各一句。保留。
- l.436 "a deterministic harmonic start leaves 5.4 to 265 times…"：上一轮按编辑建议特意加到 5.3 的头条，保留。
- l.583 末句"约四分之一……刚度较低的设计"：可读性审读把它当作该段的结论句，删了读者要自己从十几个数字里拼结论。保留。

---

## C. 第 5 节可读性（轻度）

审读判断：5.1–5.8 已不像审计记录；5.9 和 5.10 仍是，表 6、表 7 的每个数字在正文里又念了一遍，内存段没有结论句。按作者的规则只做轻度处理：**已进表 3 的数字从正文删去；5.9 与 5.10 的三段补结论句、把中间数字指向表；几处难读的句子改写。**

**C1（建议）表 3 的重复句。**
- l.492：…in all fourteen configurations of the seven selection cells, with the largest compliance error on M1/x and the largest sensitivity error, taken over both cells, on U1/y (Table 3).
  CN：……在七个选择胞元的全部十四种配置中均低于两条 3% 线，最大柔度误差出现在 M1/x，在两个胞元上取最大的最大灵敏度误差出现在 U1/y（表 3）。
- l.494（同时修掉 30 个词的插入语）：Nine further validation cells outside checkpoint selection were assembled in the same way: the five geometries with NICE's largest single-cell errors and one randomly chosen cell from each cut stratum. All eighteen configurations stay below both 3% lines, with the maxima on the geometry with the largest single-cell error (Table 3; Supplementary Table ST08b). The five largest-error geometries account for all compliance errors above 0.02%; the four stratum cells stay below 0.011% and 0.22%.（原末句"All held-out configurations stay below both 3% lines."并入，删去。）
  CN：另有九个检查点选择之外的验证胞元以同样方式装配：NICE 单胞误差最大的五个几何，以及每个切割分层中随机选取的一个胞元。全部十八种配置均低于两条 3% 线，最大值出现在单胞误差最大的几何上（表 3；补充表 ST08b）。所有高于 0.02% 的柔度误差均来自这五个最大误差几何；四个分层胞元分别保持在 0.011% 和 0.22% 以下。
- l.498：Under tractions on the target cut face, NICE stays below 0.2% in compliance and sensitivity (Table 3), whereas the Uncorrected continuation exceeds the 3% line in four of its seven configurations, up to 14.1% in sensitivity (Table ST09).
  CN：在目标胞元切割面的面力下，NICE 的柔度与灵敏度误差均低于 0.2%（表 3）；而未校正延续在其七种配置中有四种超过 3% 线，灵敏度误差最高达 14.1%（表 ST09）。
- l.536 前两句：Under the face loads both lattices keep the compliance error and every cell's sensitivity error below the two-cell maxima of Section 5.6 (Table 3); the assembled retained solutions differ from the exact ones by 0.13% and 0.12% (relative Euclidean norm over all six loads). They are below because Eq. (7) bounds the lattice compliance error by …
  CN：在面载荷下，两个点阵的柔度误差和各胞元的灵敏度误差均低于第 5.6 节两胞元装配的最大值（表 3）；装配保留解与精确解分别相差 0.13% 和 0.12%（全部六个载荷上的相对欧几里得范数）。之所以更低，是因为式 (7) 以……
- l.538 末句：Aggregated over the shared corner parameters, the field-based sensitivities give the lattice gradient with cosines above 0.999999 to the exact one (Table 3).
  CN：在共享角点参数上汇总后，场基灵敏度给出的点阵梯度与精确梯度的余弦值高于 0.999999（表 3）。
- l.587 末句：Exact checks of the 24- and 51-cell plates at their first design iteration give the compliance and gradient errors of Table 3, with the exact sign in every gradient component.
  CN：对 24 和 51 胞元板第一次设计迭代的精确核查给出表 3 所列的柔度与梯度误差，且所有梯度分量符号一致。
- **l.581 的 0.011% / 0.030% / 0.029% / 0.30% 保留**：摘要和引言的 "0.03% / 0.3%" 头条只有这里给出精确来源（表 3 该行是五个设计的范围）。

**C2（建议）5.9 节计时段 l.561。** 保留八胞元的对比和"五至八倍"，四胞元数字和线程数研究指向 ST12b。
- EN：For the two eight-cell lattices, NICE takes 79 and 105 s per analysis; the whole-lattice direct solution takes 868 and 1,042 s, about ten times as long, and conventional exact condensation longer still (Table 6). The four-cell lattices give similar ratios, and the learned compliance agrees with the direct solution within 0.022%. More threads help the direct solution little, because cell setup and assembly do not speed up (32-thread and single-thread times in Supplementary Table ST12b). In route (a), cell setup and stiffness assembly on the CPU take 41 to 53% of the time; with them excluded, and with NICE's cell preparation on the GPU excluded as well, which besides setup and assembly contains the network encoding and the correction setup, NICE remains five to eight times faster (phase times in Table ST12b). For the eight-cell lattices, the timed compliance errors include the algebraic error of the \(10^{-6}\) solve (Eq. (8)) and exceed the operator errors of Section 5.8 by up to 0.005 percentage points.
- CN：对于两个八胞元点阵，NICE 每次分析分别耗时 79 s 和 105 s；整体点阵直接求解耗时 868 s 和 1,042 s，约为其十倍，常规精确凝聚更慢（表 6）。四胞元点阵的比值相近，学习柔度与直接求解结果的吻合在 0.022% 以内。增加线程对直接求解帮助不大，因为胞元设置与装配并不随之加速（32 线程与单线程耗时见补充表 ST12b）。在路线 (a) 中，CPU 上的胞元设置与刚度装配占总时间的 41 至 53%；扣除这两部分，并同时扣除 NICE 在 GPU 上的胞元准备（除设置与装配外，还包括网络编码与校正设置）后，NICE 仍快五至八倍（各阶段耗时见表 ST12b）。对于八胞元点阵，计时所得的柔度误差包含 \(10^{-6}\) 求解的代数误差（式 (8)），比第 5.8 节的算子误差至多高 0.005 个百分点。
- （若采纳 A2，105 改 106。）

**C3（建议）5.9 节内存段 l.563。** 补结论句，删四胞元的因子大小。
- EN：Memory follows the same pattern. The direct solution peaks at 71 and 86 GiB for the eight-cell lattices, set by its Cholesky factor, which grows slightly faster than the number of DOFs; conventional exact condensation needs 40 and 42 GiB, most of it for the condensed cell matrices held until their elimination (Supplementary Table ST12). NICE peaks at 9.2 GiB of GPU memory and keeps 4.7 GiB of CPU memory after cell preparation. Per cell, NICE's stored state takes 0.25 to 1.37 GiB, against 0.60 to 9.4 GiB for the interior Cholesky factor of conventional exact condensation (Supplementary Note S3, Table ST13).
- CN：内存也是同样的格局。八胞元点阵上直接求解的进程峰值为 71 GiB 和 86 GiB，由其 Cholesky 因子决定，该因子的增长略快于自由度数；常规精确凝聚需要 40 GiB 和 42 GiB，其中大部分用于存放在被消去之前一直保留的凝聚胞元矩阵（补充表 ST12）。NICE 的 GPU 内存峰值为 9.2 GiB，胞元准备后占用 4.7 GiB 的 CPU 内存。按单个胞元计，NICE 存储的状态占用 0.25 至 1.37 GiB，而常规精确凝聚的内部 Cholesky 因子占用 0.60 至 9.4 GiB（补充说明 S3，表 ST13）。

**C4（建议）5.10 节规模段 l.587。** 保留 110 胞元、3270 万、每胞元 10.1–10.9 s；中间板的数字指向表 7 和 ST20。
- EN：To examine how the cost grows with the size of the lattice, the design iteration was timed on plates with the proportions and cut of this plate and 24, 51, 88 and 110 cells, clamped along the uncut long side and loaded in plane on the opposite face (Figure 15c, Table ST20). The largest has 32.7 million DOFs in the cut finite-element model and 1.62 million free retained DOFs; each plate was run over its first four design iterations, with the learned substructures beyond 4 GiB of GPU memory streamed from CPU memory. The time of a design iteration grew in proportion to the number of cells, at 10.1 to 10.9 s per cell (19.4 min for 110 cells), with 138 to 173 conjugate-gradient iterations (Table 7). CPU memory, which holds the streamed state of the learned cells, grew by about 0.7 GiB per cell; with the learned cells held on the GPU within the fixed 4 GiB, the GPU memory grew with the assembled retained system, whose stiffness \(\mathbb K_{PP}\) the preconditioner factorises directly (Table ST20). Exact checks of the 24- and 51-cell plates at their first design iteration give the compliance and gradient errors of Table 3, with the exact sign in every gradient component.
- CN：为考察成本随点阵规模的增长情况，在与该板比例和切割方式相同、分别含 24、51、88 和 110 个胞元的板上对设计迭代进行计时；这些板沿未切割的长边固支，并在相对面上施加面内载荷（图 15c，表 ST20）。最大的板在切割有限元模型中含 3270 万个自由度，自由保留自由度为 162 万个；每块板运行其前四次设计迭代，超出 4 GiB GPU 内存的学习子结构从 CPU 内存流式传输。一次设计迭代的耗时与胞元数成正比，每胞元 10.1 至 10.9 s（110 胞元为 19.4 min），共轭梯度迭代次数为 138 至 173（表 7）。CPU 内存存放学习胞元的流式状态，每增加一个胞元约增长 0.7 GiB；在学习胞元驻留于 GPU 上固定的 4 GiB 内存中的情况下，GPU 内存随装配保留系统增长，预条件子直接分解该系统的刚度 \(\mathbb K_{PP}\)（表 ST20）。对 24 和 51 胞元板第一次设计迭代的精确核查给出表 3 所列的柔度与梯度误差，且所有梯度分量符号一致。

**C5（建议）难读的句子。**
- l.444：在 "…terms of Eq. (9)." 后分段，下一段以 "In the spectrum, the base network's error occupies…" 开头。CN 同样分段，以"在谱上，基础网络的误差……"开头。
- l.458："Elsewhere it is not negligible, contributing 28–43% … and 24–72% under nodal forces in all six cells" 中 "elsewhere" 与 "all six cells" 冲突。
  EN：In the other cells, whose energy errors under consistent tractions are below 2%, it contributes 28–43%; under nodal forces it contributes 24–72% in all six cells. Being first order in the field error, it dominates as that error tends to zero (Table ST05, Figure S03).
  CN：在其余胞元中（其一致面力下的能量误差低于 2%），线性项占 28–43%；在节点力下，它在全部六个胞元中占 24–72%。由于线性项关于场误差是一阶的，当该误差趋于零时，它将占主导（表 ST05，图 S03）。
- l.538："consistent with rounding … so the correction is not its source" 的指代不清。
  EN：…stagnates at \(3.6\times10^{-4}\) (block) and \(3.2\times10^{-3}\) (layer). This stagnation is attributed to rounding in the single-precision network: running the correction, the only other single-precision component, in single instead of double precision changes the stagnation level by less than 1% in the block, so the correction is not its source.
  CN：……停滞于 \(3.6\times10^{-4}\)（块体）和 \(3.2\times10^{-3}\)（层）。这一停滞归因于单精度网络中的舍入误差：将校正（另一个以单精度运行的部分）由双精度改为单精度，块体中的停滞水平改变不到 1%，因此校正并非其来源。
- l.542："The exact routes solve six loads but do not evaluate sensitivities … (Supplementary Table ST12b)." 出现在三条路线定义之前；移到 l.550 段（"All routes include cell setup…"）之后。CN 同。
- l.581：两个均匀设计先命名再比较。
  EN：Two uniform designs of the same volume serve as references: in the first the ten loaded-face parameters are kept at 0.40 and the other 64 take the common value that meets \(V^*\); in the second all 74 parameters are uniform. Against the first, the NICE design lowers the exact compliance by 32.1%, from 115.36 to 78.29, and against the second by 30.9%; the homogenisation design lowers it by 30.5% against the first. NICE reproduces both uniform designs to within 0.019% in compliance and 0.064% in gradient (Table ST21).
  CN：以两个同体积的均匀设计作为参照：第一个保持加载面的 10 个参数为 0.40、其余 64 个取满足 \(V^*\) 的共同值；第二个 74 个参数全部均匀。相对第一个，NICE 设计将精确柔度降低 32.1%，由 115.36 降至 78.29；相对第二个降低 30.9%；均匀化设计相对第一个降低 30.5%。NICE 对这两个均匀设计的柔度误差在 0.019% 以内，梯度误差在 0.064% 以内（表 ST21）。
- l.583："The fine-scale model clamps the whole cut band…" 前加引语。
  EN：The underestimate is not an artefact of the support: the fine-scale model clamps the whole cut band, …
  CN：这一低估并非支撑方式造成的：细观模型固支整个切割带，……

**可选、未起草：** l.500（表 5 段，12 个数字全在表 5 里）、l.466（表 4 段的 28–1100%、0.08–17%、0.004–0.19% 三个范围都在表 4 的列里）也可以同样处理；按轻度原则本轮不动。

---

## D. 两份审读的分歧（供作者定）

- l.466 末句：防御型审读要删尾巴，可读性审读保留全句。我取前者（B11）。
- l.583 末句：防御型审读要删，可读性审读视为结论句。我取后者（保留）。
- l.577 的 0.0051：防御型审读删第二处，可读性审读删第一处。我取前者（B13），第一处是主结果。

---

## E. 建议的执行顺序

1. A1、A4（必改）。
2. B 组（B1–B5、B7–B18，B6 可选）。
3. C 组（C1–C5）。
4. 全部写入后核对：摘要词数（现 250）、中英文行数一致、表 3 与正文引用、LaTeX 重建。
