# 2026-10-01 夜间工作记录（作者睡觉期间，按授权自主推进）

## 已决定（作者 09-30 深夜）

- 规模展示保留，上限写 110 胞。51 胞用旧机上已有的 2 次分析（方案 A），不重跑。
- 标题等板核对结果再定：通过就用 D8，否则用 B 或 C。
- AI 声明只留占位，由作者最后重写。
- 孪生运行的计时保留现写法，不做改动。
- CPU 机上跑最小集核对；不依赖结果的写作 1–7 项开始做。

## CPU 机：板设计精确核对（最小集）

- 机器：32 vCPU、240 GiB（cgroup 限额）；数据盘 /root/autodl-tmp/xc。
- 环境：miniconda Python 3.12.3、NumPy 2.3.2、SciPy 1.18.1、PyTorch 2.8.0（只用 CPU）、MKL 2026.1、pypardiso 0.4.7；自检 ALL PASS。
- 代码：旧部署包 `OPL_DEPLOY/src_v2.tar`，用仓库镜像覆盖。缺的三个模块 box_encode、encode_r1、element_moments 与旧服务器副本逐字节相同。
- 输入包：`/root/autodl-fs/XC_PACK/plates_exact.tar`（1.5 GB，10 个设计），逐文件 md5 校验通过。
- 运行链 `xc_chain.sh`：
  1. B1:0 与 B2:0（共用 T），只算柔度；
  2. B1:22 与 B2:29，带梯度；
  3. Hom-z:0 与 X-z:11，只算柔度。
- 容器不允许 numactl，已去掉该前缀，这不影响结果。

| 设计 | 精确柔度 | NICE 柔度 | 代理误差 | 精确 PCG / 残差 | 状态 |
| --- | ---: | ---: | ---: | --- | --- |
| B1:0 | 94.353153 | 94.340658 | −0.0132% | 243 / 9.8e-11 | 完成（12.3 min） |
| B2:0 | 1,468.052207 | 1,467.743385 | −0.0210% | 252 / 1.9e-10 | 完成（1.7 min，复用 T） |
| B1:22 | | | | | 进行中（第 2 步，02:42 开始） |
| B2:29 | | | | | 排队 |
| Hom-z:0 | | | | | 排队（第 3 步） |
| X-z:11 | | | | | 排队 |

**初步读法**：
- 两个起点设计的代理误差都与算例 A 同量级（A 为 0.011–0.028%），符号为负，与 Eq. (12) 一致。
- 均匀化在起点的预测误差，改用精确柔度作参照后为 −26.68%（y）和 −36.60%（z）：69.177/94.3532 − 1，930.693/1468.052 − 1。与 NICE 参照的 −26.67%/−36.59% 只在第四位有效数字上不同。

## 写作（已提交）

1. MANUSCRIPT §6.2 Gershgorin 倍数的基准改为 power-iteration endpoint；§6.1 约 60 GPU-h 补上"三个早期训练阶段（7.5 GPU-h）"。
2. 补充材料中的内部路径改为 evidence/ 路径：主机记录已复制到 evidence/host_direct、host_direct_1thread、host_condensed，映射见 PATH_MAP_EVIDENCE_CN.md；ST04 的列名 fastnet-trainlib 改为 "deployed vs training"。
3. DRAFT_6_11 与 DRAFT_S9 的小错已按审计意见修改（0.452 注明是中间设计，+1e-4/+1e-3，只引 ST21，Eq.(12) 的措辞，Hom，S9.6 删去"两次评估"，孪生运行注明同一 GPU）。
4. Svanberg (1987) 已按 Crossref 核实，加入 bib。
5. 规模展示已填入：§6.11 末段、表 6 Scale 行、图 12(e)、S9.6 与 ST26。
   - DOF 数：从各胞几何的网格节点只读去重计算，见 scale_dofs.json。
   - GPU 内存改用 nvidia-smi 的 30 s 采样。它包含 cuDSS 分解；PyTorch 的统计不含这部分，只有 13–20 GiB。
   - 110 胞时显存用满（31.4/31.4 GiB），与 135 胞失败一致。
6. 投稿包：
   - Highlights 候选（submission/HIGHLIGHTS.md，每条 ≤85 字符，附来源）；
   - cover letter 初稿（submission/COVER_LETTER_DRAFT.md）；
   - endmatter 加上 AI 声明占位，去掉重复的数据可用性占位（改用正文不编号的 "Code and data availability"）；
   - 补充材料 PDF 构建脚本 latex/build_supp_tex.py，53 页，无错误；
   - main.pdf 已重建，超页的图已修好。
7. 建议稿 PROPOSED_EDITS_S1_S6_CN.md 只写了建议，正文未改：前处理占比、回本点、32 线程扩展、误差增长、两处迭代口径、跨硬件加速比、新颖性句、I-32。

## GPU 机

- 新机的 src_v2、S4 仍为空，同步进行中（数据盘已用 2.0T）。没有在 GPU 机上运行任何计算。
- 只读取回了规模记录（SCALE 目录），并在服务器上建了 /root/autodl-tmp/exact_tools（工具与拼合的代码，未改动原数据）。
