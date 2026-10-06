# P1 第二轮拔高：五项改稿（EN + CN，2026-10-06）

依据独立编辑冷读的五条建议。以下为逐字改稿，确认后原样写入中英文源稿。新引用 Falgout & Vassilevski (2004) 已经 Crossref 核实：SIAM J. Numer. Anal. 42(4), 1669–1693，doi:10.1137/S0036142903429742。其中的理想插值 \(P_\star=(I-S(S^TAS)^{-1}S^TA)R^T\)，取 \(R\) 为保留自由度的选择、\(S\) 为内部自由度的选择时，正是 \(E=[\,-A^{-1}K_{IP};\,I\,]\)，且 \(E^TKE=S\)（Schur 补）。

---

## 1. 多重网格视角（引言第 5 段、第 4 节开头）

**引言第 5 段**，插在 "exact static condensation is the case in which the extension is exact." 之后：

- EN: In the language of algebraic multigrid, the exact extension \(E\) is the ideal interpolation from the retained to all degrees of freedom and \(S=E^TKE\) is its Galerkin coarse operator [Falgout & Vassilevski (2004)](https://doi.org/10.1137/S0036142903429742); NICE learns an approximation of this interpolation and smooths it, and its condensed stiffness is the Galerkin operator of the result.
- CN: 用代数多重网格的语言来说，精确延拓 \(E\) 是从保留自由度到全部自由度的理想插值，\(S=E^TKE\) 是它的 Galerkin 粗算子 [Falgout & Vassilevski (2004)](https://doi.org/10.1137/S0036142903429742)；NICE 学习这一插值的近似并对其做光滑，其凝聚刚度就是所得插值的 Galerkin 算子。

**第 4 节开头第一句**改为：

- EN: NICE is defined by one object, the corrected extension \(F\), which maps a retained displacement to a displacement of the whole cell; its energy \(\widehat S=F^TKF\) is the condensed stiffness, the Galerkin operator of \(F\), and exact static condensation is the case \(F=E\), the ideal interpolation.
- CN: NICE 由一个对象定义，即校正后的延拓 \(F\)：它把保留位移映射为整个胞元的位移；其能量 \(\widehat S=F^TKF\) 就是凝聚刚度，即 \(F\) 的 Galerkin 算子，精确静力凝聚即 \(F=E\)（理想插值）的情形。

参考文献表加一条（EN/CN 相同）：

> Falgout, R. D., Vassilevski, P. S. (2004). [On generalizing the algebraic multigrid framework](https://doi.org/10.1137/S0036142903429742). *SIAM Journal on Numerical Analysis*, 42(4), 1669–1693.

---

## 2. 头条换成读者关心的基准

**5.3 标题**
- EN: Single cells: about 0.1% mean energy error or less in every cut stratum
- CN: 单胞：各切割分层的平均能量误差均约为 0.1% 或更低

依据：一致面力下分层平均最高 0.109%（重度切割），节点力下最高 0.090%，其他载荷类别同量级。

**5.3 第三段末**加一句：
- EN: Under the same correction, a deterministic harmonic start leaves 5.4 to 265 times the error of NICE (Section 5.5).
- CN: 在相同校正下，确定性调和初值留下的误差是 NICE 的 5.4 至 265 倍（第 5.5 节）。

**5.8 标题**
- EN: Lattices with every cell learned: the errors do not accumulate
- CN: 所有胞元均为学习胞元的点阵：误差不累积

依据：点阵误差低于两胞元装配的最大误差，原因是式 (7) 的份额加权（5.8 第二段已有）。

**5.9 开头第一句**换成与硬件无关的事实：
- EN: NICE solves the assembled problem on the retained DOFs only, 139,002 and 143,685 for the eight-cell lattices instead of the 1.83 and 2.11 million DOFs of the whole-lattice model, and prepares every cell by a network evaluation and a fixed correction instead of an interior factorisation. A lattice analysis with sensitivities therefore takes less time and memory than the whole-lattice direct solution and conventional exact condensation, and less time than whole-lattice conjugate gradients with algebraic multigrid; at the scale of Section 5.10, a design iteration of a 110-cell plate, whose cut finite element model has 32.7 million DOFs and whose assembled retained system has 1.6 million, takes 19 min, about 11 s per cell.
- CN: NICE 只在保留自由度上求解装配问题，两个八胞元点阵分别为 139,002 和 143,685 个，而整体点阵模型为 183 万和 211 万个；每个胞元由一次网络求值和固定校正准备，而不做内部分解。因此，含灵敏度的点阵分析所需的时间和内存均少于整体点阵直接求解和常规精确凝聚，所需时间也少于以代数多重网格预条件的整体点阵共轭梯度法；在第 5.10 节的规模上，110 胞元板的切割有限元模型有 3270 万个自由度、装配保留系统有 160 万个，一次设计迭代耗时 19 min，约每胞元 11 s。

依据：表 ST20，110 胞元 1,164 s / 110 = 10.6 s；32.70 M / 1.618 M。

相应地，6.2 节去掉重复的数字："…solves the assembled problem only on the retained DOFs, 139,002 and 143,685 for the eight-cell lattices instead of 1.83 and 2.11 million." 改为 "…solves the assembled problem only on the retained DOFs (Section 5.9)."；CN "…只在保留自由度上求解装配问题（第 5.9 节）。"

---

## 3. 理论的形式包装

**命题 3** 改为 (a)(b) 两款：

- EN:

  **Proposition 3 (sensitivity error).** Under (A1), (A2) and (D), at the same retained displacement:

  (a) the field-based estimate differs from the exact sensitivity by a term linear and a term quadratic in the interior error,

  \[ \boxed{\widetilde s_c-s_c=-2d^TK_{,c}u-d^TK_{,c}d,\qquad u=Eq,\quad d=Fq-Eq;} \tag{9} \]

  (b) with \(\mathcal E=d^TKd\) the energy of the interior error, \(\|\widetilde{\boldsymbol s}-\boldsymbol s\|_2\le2L\sqrt{\mathcal E}+Q\mathcal E\), where the constants \(L\) and \(Q\) of Eq. (H.4) depend on the cell, the direction and the stiffness derivatives, so that the energy error controls the relative sensitivity error only at the order \(\sqrt\varepsilon\) (Appendix H.1).

  其后一段删去与 (b) 重复的一句，变为：Interior equilibrium sets \((Ku)_I=0\) but generally leaves \((K_{,c}u)_I\ne0\), so the linear cross term survives, and a decrease of the energy error need not decrease the sensitivity error monotonically. Because thickening a corner …（以下不变）

- CN:

  **命题 3（灵敏度误差）。** 在 (A1)、(A2) 和 (D) 下，在相同的保留位移处：

  (a) 场基估计与精确灵敏度之差由一个关于内部误差的线性项和一个二次项组成，

  \[ \boxed{\widetilde s_c-s_c=-2d^TK_{,c}u-d^TK_{,c}d,\qquad u=Eq,\quad d=Fq-Eq;} \tag{9} \]

  (b) 记 \(\mathcal E=d^TKd\) 为内部误差的能量，则 \(\|\widetilde{\boldsymbol s}-\boldsymbol s\|_2\le2L\sqrt{\mathcal E}+Q\mathcal E\)，其中式 (H.4) 的常数 \(L\) 与 \(Q\) 依赖于胞元、方向和刚度导数；因此能量误差只能以 \(\sqrt\varepsilon\) 的阶控制相对灵敏度误差（附录 H.1）。

  其后一段：内部平衡使 \((Ku)_I=0\)，但一般不使 \((K_{,c}u)_I=0\)，因此线性交叉项保留下来；能量误差减小时，灵敏度误差也不一定单调减小。由于加厚角点……（以下不变）

**命题 4 之后加 Remark**（并从其后一段删去 "The ordering does not extend to the sensitivity error of Eq. (9) (Appendix D.1), and"，改为 "Changing the smoothing degree changes the polynomial, so the reduction need not be monotone in \(k\)."）：

- EN: **Remark 1.** The guarantee rests on both conditions and covers the energy only. If the spectrum of \(D^{-1}A\) extends above \(a+b\), the Chebyshev polynomial amplifies those modes and the cycle can increase the energy error (Appendix D). For an approximate coarse inverse \(G\), the correction stays nonexpansive when \(2G-GA_cG\succeq0\) (Eq. (D.6)), which the nonnegative shift satisfies and an arbitrary approximation need not. And the ordering does not reach the sensitivity: in the example of Appendix D.1, a projection that halves the energy of the interior error changes the sensitivity discrepancy of Eq. (9) from zero to \(2t-t^2\), against a reference sensitivity of \(-1\).
- CN: **注 1。** 这一保证依赖两个条件，且只涉及能量。若 \(D^{-1}A\) 的谱超出 \(a+b\)，Chebyshev 多项式会放大这些模态，循环可能增大能量误差（附录 D）。对于近似粗逆 \(G\)，只要 \(2G-GA_cG\succeq0\)（式 (D.6)），校正就保持非扩张；非负平移满足这一条件，任意近似则不一定。此外，排序不涉及灵敏度：在附录 D.1 的例子中，一个使内部误差能量减半的投影，使式 (9) 的灵敏度偏差从零变为 \(2t-t^2\)，而参考灵敏度为 \(-1\)。

依据：附录 D（"eigenvalues above \(a+b\) are amplified"）、式 (D.6) 及 D.1 的反例，均为原文已有结果，不引入新结论。

---

## 4. 一个论点，四处呼应

论点句：**网络只需准确，不必保持结构。**

- **引言第 2 段末**加：
  - EN: The network then has to be accurate, but it does not have to preserve structure.
  - CN: 于是网络只需准确，而不必自己保持结构。
- **引言第 5 段**删去同义句 "With this division the network has to be accurate, but it does not have to preserve structure."（CN："在这种分工下，网络只需要准确，而无需自己保持结构："改为直接以"只要它给出的延拓……"起句）。
- **摘要**在分工句后加一句，并在别处省 4 词，总数正好 250：
  - 加：EN "The network must be accurate, not structure-preserving." / CN "网络只需准确，不必保持结构。"
  - 省：首句 "the cells cut by the boundary can carry its supports and loads" → "its cut cells can carry the supports and loads"；第二句 ", whereas homogenisation" → "; homogenisation"。CN 相应为"其切割胞元可能承受支撑与载荷"、"；均匀化"。
- **结论第 1 段首句**末尾：
  - EN: …and a fixed correction the improvability, so that the network has to be accurate but not structure-preserving.
  - CN: ……固定校正提供可改进性，因而网络只需准确，不必保持结构。
- **引言第 4 段**（柔度与灵敏度那段）末尾 "…checked on compliance and local sensitivity separately" 后加 "(Section 5.6)" / "（第 5.6 节）"。编辑原建议指向图 11，但图 11 在引言里先引会打乱图号顺序，所以改指第 5.6 节。

---

## 5. 第 5 节"装配精度一览"表

放在第 5 节开头那段之后、5.1 之前，成为新的**表 3**；原表 3–6 依次改为表 4–7（正文、附录、补充材料中英文一并改号，补充表 ST 不动）。第 5 节开头段末加：

- EN: Table 3 collects the accuracy against exact condensation in every assembled example.
- CN: 表 3 汇总了各装配算例相对于精确凝聚的精度。

**EN**

**Table 3. Accuracy of NICE against exact condensation in the assembled examples.** Compliance error \(|\widehat C/C-1|\); sensitivity error: largest relative error \(e_s\) of a cell's eight-component thickness-sensitivity vector; gradient error: relative error of the lattice gradient over the shared vertex parameters. Maxima or ranges over the loads and designs listed. In the two-cell assemblies only the target cell is learned; in all other examples every cell is.

| Example | Section | Cells (cut) | Loads or designs | Compliance error (%) | Sensitivity or gradient error (%) |
| --- | --- | --- | --- | ---: | --- |
| Two-cell assemblies, seven selection cells | 5.6 | 2, 14 configurations | six face loads each | ≤ 0.056 | sensitivity ≤ 0.67 |
| Two-cell assemblies, cut-face tractions | 5.6 | 2, 7 configurations | three cut-face tractions each | ≤ 0.10 | sensitivity ≤ 0.16 |
| Two-cell assemblies, nine held-out cells | 5.6 | 2, 18 configurations | six face loads each | ≤ 0.27 | sensitivity ≤ 1.49 |
| Lattices with every cell learned | 5.8 | 8 (4), 8 (3) | three face loads; three random loads | ≤ 0.015; ≤ 0.069 | sensitivity ≤ 0.14; ≤ 0.45; gradient 0.04–0.09; 0.22–0.28 |
| Case A, thickness optimisation | 5.10 | 8 (4) | designs at iterations 0, 12, 23 | 0.011–0.028 | gradient 0.069–0.33 |
| Plate clamped through its cut band | 5.10 | 24 (8) | start, final, homogenisation and two uniform designs | 0.011–0.032 | gradient 0.029–0.32 |
| Plates of the scale study | 5.10 | 24 (8), 51 (12) | first design | 0.013, 0.011 | gradient 0.045, 0.047 |

**CN**

**表 3. 各装配算例中 NICE 相对于精确凝聚的精度。** 柔度误差 \(|\widehat C/C-1|\)；灵敏度误差：胞元八分量厚度灵敏度向量的最大相对误差 \(e_s\)；梯度误差：点阵梯度在共享顶点参数上的相对误差。表中为所列载荷或设计上的最大值或范围。两胞元装配中只有目标胞元是学习胞元，其余算例中所有胞元均为学习胞元。

| 算例 | 节 | 胞元（切割） | 载荷或设计 | 柔度误差 (%) | 灵敏度或梯度误差 (%) |
| --- | --- | --- | --- | ---: | --- |
| 两胞元装配，七个选择胞元 | 5.6 | 2，14 个配置 | 每个配置六个面载荷 | ≤ 0.056 | 灵敏度 ≤ 0.67 |
| 两胞元装配，切割面面力 | 5.6 | 2，7 个配置 | 每个配置三个切割面面力 | ≤ 0.10 | 灵敏度 ≤ 0.16 |
| 两胞元装配，九个留出胞元 | 5.6 | 2，18 个配置 | 每个配置六个面载荷 | ≤ 0.27 | 灵敏度 ≤ 1.49 |
| 所有胞元均为学习胞元的点阵 | 5.8 | 8 (4)，8 (3) | 三个面载荷；三个随机载荷 | ≤ 0.015；≤ 0.069 | 灵敏度 ≤ 0.14；≤ 0.45；梯度 0.04–0.09；0.22–0.28 |
| 算例 A，厚度优化 | 5.10 | 8 (4) | 第 0、12、23 次迭代的设计 | 0.011–0.028 | 梯度 0.069–0.33 |
| 经切割带固支的板 | 5.10 | 24 (8) | 初始、最终、均匀化及两个均匀设计 | 0.011–0.032 | 梯度 0.029–0.32 |
| 规模算例的板 | 5.10 | 24 (8)，51 (12) | 初始设计 | 0.013，0.011 | 梯度 0.045，0.047 |

**有了表 3，正文证据句随之减数**（只删已进表的数字，不压缩叙述）：

- 5.8 第二段首句
  - EN: Both lattices keep the compliance error below 0.015% under the face loads and every cell's sensitivity error below 0.14% (Table 3); the assembled retained solutions differ from the exact ones by 0.13% and 0.12% (relative Euclidean norm over all six loads).
  - CN: 两个点阵在面载荷下的柔度误差均低于 0.015%，各胞元的灵敏度误差均低于 0.14%（表 3）；装配保留解与精确解分别相差 0.13% 和 0.12%（六个载荷上的相对欧几里得范数）。
- 5.8 末句
  - EN: Aggregated over the shared corner parameters, the field-based sensitivities give the lattice gradient to within 0.09% under the face loads, with cosines above 0.999999 (Table 3).
  - CN: 在共享角点参数上汇总后，场基灵敏度给出的点阵梯度在面载荷下误差在 0.09% 以内，余弦值高于 0.999999（表 3）。
- 5.10 算例 A
  - EN: At the three checked designs the NICE compliance lies at most 0.028% below the exact one, the sign given by Eq. (7), and the lattice gradient has errors of at most 0.33% with the exact sign in every component (Table 3, Table ST17).
  - CN: 在三个经校核的设计上，NICE 柔度至多比精确值低 0.028%，符号与式 (7) 一致；点阵梯度误差至多 0.33%，且每个分量的符号均与精确值一致（表 3，表 ST17）。
- 5.10 板
  - EN: Checked with exact condensation, the NICE compliance lies 0.011% below the exact one at the uniform start and 0.030% below it at the final design, and the lattice gradient has errors of 0.029% and 0.30% with the exact sign in all 64 components (Table 3, Table ST21).
  - CN: 经精确凝聚校核，NICE 柔度在均匀初始设计处比精确值低 0.011%，在最终设计处低 0.030%；点阵梯度误差为 0.029% 和 0.30%，且全部 64 个分量的符号均与精确值一致（表 3，表 ST21）。

被删去的余弦、第 95 百分位数等细项仍在表 ST17、ST21 中。
