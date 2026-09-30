# 建议修改稿：S1–S6（不依赖新结果；待作者审阅后再并入正文）

2026-10-01 夜间整理。对应 SUBMISSION_READINESS_CN.md 第 (3) 节“建议完成”的 S1–S6。每条给出位置、现文、建议文字和数字来源；正文尚未改动。数字均已按记录复算。

---

## S1. 前处理的硬件不对称（I-06）

**位置**：MANUSCRIPT_EN.md §6.10，"For the eight-cell lattices, NICE takes 81 and 110 s …" 一句之后。

**建议加一句**：

> Cell setup and stiffness assembly run on the host in route (a) and on the GPU in route (c), and take 41–53% of the direct solution's time with 16 threads; without them, the direct solution takes 165 to 610 s and NICE 33 to 99 s, still five to eight times less.

**来源**（SUPPLEMENTARY_EN.md 表 ST17b、ST17d）：

| 格架 | (a) 16 线程总时间 | 其中胞建立+组装 | 占比 | 扣除后 | (c) 总时间 | 前端 | 扣除后 | 比值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2×2×1, z=0 | 484.4 | 229.9 | 47% | 254.5 | 38.65 | 5.47 | 33.2 | 7.7 |
| 2×2×1, z=1 | 349.2 | 184.2 | 53% | 165.0 | 37.74 | 5.14 | 32.6 | 5.1 |
| 2×2×2 | 867.8 | 379.5 | 44% | 488.3 | 81.24 | 9.73 | 71.5 | 6.8 |
| 3×3×1 | 1,041.6 | 431.3 | 41% | 610.3 | 110.12 | 11.03 | 99.1 | 6.2 |

---

## S2. 回本点（I-02）

**位置**：§7.4 末尾，或 §6.10 末尾（"none includes the one-off cost of data generation and training" 之后）。

**建议加一句**：

> Measured against the offline cost of about 60 GPU-hours (Section 6.1), each eight-cell design iteration saves 787 to 932 s against the parallel whole-lattice direct solution, so the offline cost is recovered after about 230 to 270 such iterations, about ten optimisations of the size of Section 6.11, if GPU and host time are counted alike; the trained operator applies only to the cell family and discretisation of its training (Section 7.5).

**来源**：60 GPU-h = 216,000 s（FACTS_R1.md，§6.1）；节省 = 868 − 81 = 787 s，1,042 − 110 = 932 s（表 5）；216,000/932 = 232，216,000/787 = 274。对路线 (b)：950 − 81 = 869，1,218 − 110 = 1,108 → 195–249 次。算例 A 用了 24 次迭代，所以约 10 次优化。

**注意**：GPU 时间和 CPU 时间不是同一种资源，句中已写 "if … counted alike"。要不要写，作者决定。

---

## S3. 为什么主机只用 16 线程（I-36）

**位置**：§6.10 (a) 的括号 "(MKL PARDISO on one AMD EPYC 9654 host with 16 threads, and with one thread for comparison with serial solvers)" 之后，或放进 Note S5。

**建议加一句**（依据同一台主机上 32 线程的记录）：

> With 32 threads the factorisation is 1.2 to 1.8 times faster but the whole direct solution only 2 to 22% faster, since cell setup and assembly do not speed up (Supplementary Note S5).

**来源**：`docs/data/newmachine_20260924/r1_cpu_results/cpu120/R1/cpu/host/lat2t32_*_chol.json` 与 `lat2_*_chol_rep1.json` 的 `runs[0]` 各阶段时间：

| 格架 | 分解 16 → 32 线程 (s) | 总时间 16 → 32 线程 (s) |
| --- | --- | --- |
| 2×2×1, z=0 | 193.1 → 105.5（1.8×） | 484.4 → 378.5（−22%） |
| 2×2×1, z=1 | 116.9 → 87.4（1.3×） | 349.2 → 341.1（−2%） |
| 2×2×2 | 368.2 → 275.7（1.3×） | 867.8 → 823.4（−5%） |
| 3×3×1 | 481.5 → 395.1（1.2×） | 1,041.6 → 955.5（−8%） |

**注意**：
- 32 线程的记录里有一个 `throttled_s` 字段（0.2–0.8 s）。按硬规则，论文不提它。
- 32 线程的记录要先按 PATH_MAP 的规则复制到 evidence/，才能在补充材料里引用。这一步还没做，等作者决定采用后再做。

---

## S4. 代理误差沿路径增长（I-20）

**位置**：DRAFT_6_11_EN.md 算例 A 段末（并稿时一起改）。

**建议加一句**：

> Both errors grow along the path as the design approaches the bounds, yet the exact compliance of the NICE design is the lower of the two final designs.

**来源**：代理误差 −0.011% → −0.028%，梯度误差 0.069% → 0.33%（`A_checks`）；终设计的精确柔度 24.110504（NICE）对 24.110806（孪生），见 `A_final_designs`。

---

## S5. 两处“设计迭代”口径不同（§6.10 与 §6.11）

**位置**：DRAFT_6_11_EN.md 表 6 表注，或 S9.1。

**建议加一句**：

> A NICE design iteration of case A takes 141 s here against 81 s in Table 5 for the same block, because it regenerates every cell's geometry (14 s), integrates the moments and their derivatives without the fused kernels (front end 20.7 s against 9.7 s; sensitivities and volume gradient 46.4 s against 3.7 s) and solves for one load instead of three.

**来源**：`A.phase_means`（FACTS_6_11.json）；`evidence/d5_off.json`。PCG 时间两边分别是 52.7 s（1 个载荷）和 60.7 s（3 个载荷），两者都有热启动与否的差别，建议只写前两项。

---

## S6. 其余一两句的修改

**a. §7.1：删掉跨硬件的加速比并列。**
- 现文："… whereas NICE's directional energy error is about 0.07% and its assembled compliance errors are 0.01–0.27% against exact condensation, at 9 to 13 times the speed of the parallel and 56 to 74 times that of the serial whole-lattice direct solution (Section 6.10)."
- 建议：删去 ", at 9 to 13 times … (Section 6.10)"，同时删掉前半句 Guo 的 "at 7 to 380 times the speed of their serial direct finite element solution"。只比误差，不比速度，因为两边的硬件和基线都不同。

**b. §7.2：给 "7 to 290 times" 加上范围。**
- 现文："a harmonic start leaves 7 to 290 times its error"
- 建议："a harmonic start leaves 7 to 290 times its error on the five detailed cells"

**c. §1：说明为什么不用参数化降阶或气泡基。**
- 位置：第三段 "First, component reduced-basis methods …" 句后。
- 建议加 1–2 句：

  > Such offline bases presuppose a common discrete space across the parameter family. A cut changes the active elements and the retained set from one geometry to the next, so the cells share neither a discrete space nor an affine parameter dependence; NICE therefore conditions a single network on each cell's discrete geometry instead.

**d. I-32：说明设计比较都在同一离散模型内。**
- 位置：DRAFT_6_11_EN.md 均匀化段末，或 S9.3。
- 建议加一句：

  > All design comparisons are made within one discrete model; the designs reach the lower bound τ = 0.18 (V^H = 0.103, Table ST23b), where the reference's discretisation error is largest (Section 6.2).

- 来源：ST23b（τ 0.18–0.70 对应 V^H 0.103–0.400）。§6.2 中 n = 32 与 64 相差 0.28%/0.44% 这一对数字，并稿前需复核它对应的具体胞。

---

## 不在本稿内的事项

- 摘要的三处修改（成本限定、"cannot increase" 的前提、§6.11 句）：等板核对结果出来后一并改。
- 正文 §7.5 的更新：并入 §6.11 时一起改（优化证据的范围、局部最优、规模上精度未核对、单卡上限由 K_PP 分解决定）。
