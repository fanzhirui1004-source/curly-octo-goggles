<!-- 2026-09-30 投稿准备度盘点（五路只读审查 + 汇总）。主会话抽查属实：M7 两处（MANUSCRIPT_EN.md:358 Gershgorin 倍数基准、:344 约 60 GPU-h 的构成），Svanberg 不在 bib，梯度范数 0.45184 > 0.45，补充材料 :765/:776/:796/:800 含内部路径，LaTeX 构建早于 08a6a33，孪生运行 469 s/步 与 I-22 冲突。 -->

# NICE 论文投稿计划（汇总五份审计，已逐条核对原文件）

## (1) 一句话结论

现在还不能"整理整理就投"。§1–§6.10 的科学内容和数字基本可以投：抽查约 60 个关键数字，只有 §6.2 一处实质错误。但 §6.11 仍是占位，它依赖的两项结果也还没完成：板设计的精确核对，以及规模展示的存档。最短的路径是四步：
1. 数据同步。
2. 大内存机上做板核对的最小集，约 1 天。
3. 规模数据存档后写入，或者整段删除。
4. 并稿，做投稿包，约 2–3 天。

大内存机 D1 到位的话，最早 D5 可以投。如果核对做不成，就按 M2 的退路删掉板上的设计比较，D3 也能投，但 §6.11 的分量会明显变弱，不建议。

---

## (2) 投稿前必须完成（按依赖顺序）

**M1 数据同步与存档（其他各项的前提）**
- **做什么**：在新机器上确认旧服务器的数据齐全。具体包括：
  - plateB1/B2、hevalH_y/z、xstartH_y/z 的 body/ 和 packets/（10 个核对设计，共 240 个胞）；
  - optA（回归测试要用）；
  - 完整的 src_v2（仓库镜像缺 box_encode、encode_r1、element_moments 三个模块）；
  - 24/88/110/135 胞规模运行的 history.jsonl、meta.json、日志，含 135 胞的报错栈；plateS24r4 和 plateS51r4 如果有也要；
  - 几何生成失败胞的日志 body/<case>.log.attempt*。

  然后用 `pack_exact_inputs.sh --dry-run` 做 INCOMPLETE/DANGLING/MISSING_MODULES 检查。runbook 里的服务器路径（`O=/root/autodl-tmp/OPL/S1/V2/R1/OPT`）要改成新位置，SPECS 里加上 `optA:23`。规模记录和日志摘录分别存进 `results/X6_opt/scale` 和 `results/X6_opt`。
- **为什么**：
  - runbook §4.1 假设在旧服务器上打包（EXACT_CHECK_RUNBOOK_CN.md:105-137），§8 说明镜像缺模块（:303-366）；
  - `results/X6_opt/scale/` 目前只有存储试验的文件，QUEUE_STATUS_CN.md:360 写着"结果尚未取回"；
  - 按 I-59 的规则，没有存档来源的数字不能写进论文。
- **资源和时间**：作者负责同步，外加脚本，约 0.5 天。

**M2 板设计的精确核对（大内存机）**
- **做什么**，按顺序：
  1. 回归测试：`optA:23`。已知 C_exact = 24.11050，g_exact 在 `results/X6_opt/optA/optA/check_023.json`。
  2. `plateB1:0` 和 `plateB2:0` 一起跑。两者共用 T，同时当作流程试点，约 0.5–1.5 h。用它的 phases.json 重新估算后面的时间。
  3. `plateB1:22`、`plateB2:29`，带梯度。
  4. `hevalH_z:0`、`xstartH_z:11`，加 `--no-sens`。
  5. 有余力再跑 `plateB1:11`、`plateB2:14`、`hevalH_y:0`、`xstartH_y:11`；如果采纳 S7 的等体积基准，也一并加进来。

  结果按 runbook §7（:279-302）的规则写入，只报精度，不报租用机的硬件、时间和内存。
- **为什么**：
  - 板上所有降幅和设计排序（1.07%、0.34%、0.87%、0.060%）目前都只是 NICE 值（review_r1/DRAFT_6_11_EN.md:22,26,38-43,50）。
  - B2 的重算残差是 3.5e-3–6.7e-3，而算例 A 只有 5.3e-5–3.2e-4（FACTS_6_11.md）。
  - B2 终设计的 74 个顶点里，38 个在 τ=0.18、10 个在 0.69（B1 分别为 28 和 4；plates/plateB2/history.jsonl 末条）。再加上出平面弯曲和 8% 的碎片胞，算例 A 的 0.03% 不能直接外推到板上。
  - 审稿 M3/M7/M9 要的正是这项核对。
- **资源**：≥192 GB 内存（推荐 256 GB），≥32 核（64 核更好），不需要 GPU，数据盘 ≥200 GB（用 `--delete-T`）。最小集约 6–15 h，10 个设计全做 13–30 h（runbook §5，:244-262），另加环境搭建 2–4 h。
- **退路**（机器或数据出问题时，约 2 h）：
  - 删掉全部 [TBD-EXACT] 和 ST27；
  - 表 6 板各行的核对列写 "Not checked"；
  - 删掉 0.34% 和 0.06% 两条排序；
  - "optimisations are local" 前加 "in NICE compliance"；
  - §7.5 写明经过验证的规模仍 ≤8 胞；
  - 均匀化误差改写成"相对离散模型至少 26.7%/36.6%"。由 Eq.(12) Ĉ≤C，这个说法不依赖核对；求解的代数误差 Ūᵀρ/C ≤3.1e-7，不影响结论。

**M3 规模展示：存档后按统一口径写入，或整段删除**
- **做什么**（依赖 M1）：
  - ST26 每一行注明驻留预算、压缩存储、稀疏粗空间这几项设置，并给出 PCG 次数、重算残差、Ūᵀρ/C，以及 GPU 和主机峰值（写明计量口径）。
  - 24 胞这一点，用 4 GB 预算的 plateS24r4。没有的话，逐行注明预算：协调人给的 24 GB/5 GB/约 9 min 与 B1 的记录（24.3/5.0 GiB、574 s、`resident_gb 22.0`，plates/plateB1/meta.json）一致，属于大预算，和 88/110 胞的 4 GB 不在同一口径。
  - 135 胞写成"全局 K_PP 在 GPU 上的直接分解超出显存"。
  - 时间标注为"优化实现，含几何生成，未用表 5 的融合核，不可与表 5 比较"；完整优化时间明确写成外推（23–30 步 × 43 min ≈ 16–22 h）。
  - 摘要、贡献 (iii) 和 §8 不能写 "optimised"：每个规模只做了 3 次 MMA 更新（DRAFT_6_11_EN.md:58,62,66,70）。改写成"a design iteration of a 110-cell plate took about 43 min … accuracy not verified at this size"。
- **前提**：
  - 正文要先交代 K_PP 在 GPU 上用 cuDSS（nvmath DirectSolver）分解（teacher.py:68-80、lat_precond.py:86-90）；§6.9 和 S6.1 目前没写设备，也没写用的库。
  - 补软件版本：PyTorch 2.8.0+cu128（见 meta.json），CUDA 和 cuDSS/nvmath 的版本需另查。
  - GPU 峰值来自 `torch.cuda.max_memory_allocated`（opt_design.py:628），cuDSS 的因子内存可能不在其中（未证实）。110 胞报 20 GB，135 胞却失败在 cuDSS 上，所以要在 RTX 5090 上做一次短测（NVML 或 `mem_get_info`），或者在表注里写明口径。
- **依据**：DRAFT_6_11_EN.md:30,44,50；DRAFT_S9_EN.md:241-247；figures/F13_optimisation.png 的 (e) 面板上直接印着 "[TBD-SCALE…]"；QUEUE_STATUS_CN.md:346-360。
- **资源**：写作 0.5–1 天；GPU 短测不到 1 h，必须在 RTX 5090 上做。
- **删除版**（约 2 h）：删掉 §6.11 末段、表 6 的 Scale 行、图 12(e)、ST26，以及摘要、贡献、§8 里的 [TBD-SCALE]；§7.5 加一句"更大规模未测"。

**M4 精确孪生运行的计时写法（需要作者决定）**
- **事实**：
  - 孪生运行用 `make_T_gpu.py`，在同一块 RTX 5090 上做稠密精确凝聚（opt_design.py:25-27, 429-445）。optAx/meta.json 记录为 `exact:true, sens:fd, warm:false, exact_tol 1e-10`。
  - 每步 469 s，其中稠密凝聚 123.5 s（DRAFT_S9_EN.md:59,115），草稿里没写设备。
  - 这与 I-22 已删除的"同 GPU 精确计时"冲突（REVIEW_CHECKLIST_R1_CN.md:100）。读者还会拿它和表 5 路线 (b) 的 950–1,218 s 对照，追问表 5 的基线为什么不用 GPU 精确凝聚。
- **推荐做法**：
  - 表 6 孪生行的时间列写"—（verification run）"；
  - S9 写明"稠密精确凝聚在同一 GPU 上逐胞构造，灵敏度用中心差分，未做性能优化，不作为成本基线"；
  - 删掉 ST22c 中孪生运行的分阶段时间。
- **时间**：约 1 h。

**M5 §6.11 和 S9 并稿，以及必须修改的表述**（依赖 M2–M4）
- **a. 清占位，做联动更新**：
  - 清掉 MANUSCRIPT_EN.md:473-477（含 HTML 注释）、:7、:25、:509、:513 的 [Zenodo DOI]。
  - §6 引言（:326）加上优化。
  - §7.5（:499）的 "assembled evidence comprises … two eight-cell lattices" 和 "An optimiser must keep…" 两句要更新，补上：优化证据的范围、结果是局部最优、板精度已核对或未核对、规模上的精度未验证、单卡规模受 K_PP 分解限制。
- **b. 优化收益的参照**：
  - 现在写的"降 2.0%/7.7%/8.3%，材料少 20%"是相对体积超出 25% 的起点（DRAFT_6_11_EN.md:20,22）。
  - 已有记录里，k=3 是自由顶点均匀 τ=0.3235、V/V*=1.025 的设计，柔度 129.69 和 2,119.25。终设计比它分别低 32.8% 和 36.5%（plates/*/history.jsonl）。加一句或表 6 加一列即可。
- **c. 均匀化对比要写得平衡**：
  - 拟用的摘要和 §8 句只写了 27%/37% 的预测误差（:58,70）。
  - 同一批数据还显示：Hom-y 只比 B1 差 1.07%，Hom-z 比 B2 低 0.34%，X-z 比 B2 低 0.87%（FACTS_6_11.md）。
  - 建议写成"均匀化给出相近布局，但绝对柔度低估至少 27%/37%；NICE 在其基础上精修并核对"。
  - "since the single layer offers no separation of scales" 改为推测语气；S9 自己写的是 "not analysed further"（DRAFT_S9_EN.md:128）。
- **d. 排序措辞**：X-y 比 B1 高 0.060%，而算例 A 的代理误差是 0.011–0.028%，两者不是"同量级"。按 M2 的结果决定写法，没核对就写"无法区分"。
- **e. 小错误**：
  - "largest values reached 0.450 and 0.452"：实际梯度范数最大 0.4518（算例 A，k=14），超过 0.45 的限值，需要注明是 MMA 的可行性容差。
  - "scaled by 1±10⁻⁴ or 1±10⁻³"：实际只用过 +1e-4 和 +1e-3。
  - "(Tables ST21 and ST25)" 改为只引 ST21。
  - 表 6 注中的 "H:" 改为 "Hom"。
  - "consistent with Eq. (12)" 改为"符号与 Eq.(12) 一致，量级与 ST17e 计时路线的误差相同"。1e-6 路线超出 Eq.(12) 的上界，见 SUPPLEMENTARY_EN.md:735。
  - 失败原因原引 Appendix A.1，改引 S9.1。
- **f. ST22b 的 "KKT residual, volume multiplier only" 列**（DRAFT_S9_EN.md:96,102）：要么删掉，要么用全部起作用约束的 NNLS 乘子重算。重算需要 2–4 h 脚本，不用 GPU，g_exact 已经有了。
- **g. 硬规则清理**：
  - DRAFT_S9_EN.md:239 "differences between two evaluations with unpacked storage" 属于重复运行，删掉；
  - :25、:47 "earlier cell-local scheme … tried first" 和 :49 "arguments from iteration 16 onward" 压缩成一句；
  - :49、:262 的内部路径改成归档包里的中性路径。
- **h. 文献**：Svanberg (1987) 目前既不在 references_verified.bib，也不在参考文献列表里，要加入；均匀化文献加 1–2 条，由作者核实书目信息。
- **i. 编号**：
  - 表 6 注改为 "ST21–ST27"；
  - ST26、ST27 改成 "### Table" 标题格式；
  - Figure S06 补上图片行，并移到附图区；
  - 图文件按图号重命名（现在 F13 对应图 12，S06_reference_verification 对应图 S01，容易混）。
- **时间**：写作 1–1.5 天。

**M6 摘要**（依赖 M2、M3）
- **现状**：去掉占位后 253 词（MANUSCRIPT_EN.md:7），再加 §6.11 的句子（25–51 词）会超过 250 词。
- **删减**：去掉单线程时间和 "(0.077% …)" 这个括号，约 40 词。
- **成本句**：加上 "excluding one-off data generation and training"。I-01 在清单里标为已完成，但摘要里其实没有这句，§8（:507）里有。
- **"the correction cannot increase its error"**：加前提 b≥λmax。§6.2 自己写的是 "verified per geometry, not guaranteed a priori"（:358）。
- **§6.11 句**：用"exact compliances within 0.002%"代替"0.0051 in every corner parameter"。板的句子等核对完成后再加，或者只写均匀化"至少 27%/37%"。
- **核对上限**：按 CMAME Guide 核对摘要字数和关键词个数（现有 7 个，:9）。
- **时间**：1–2 h。

**M7 正文里已有的两处事实错误**（10 min，可以马上改）
- MANUSCRIPT_EN.md:358 写 "the Gershgorin bound exceeds it by factors of 4.7 to 19.4"，这里的 it 指 λmax。但 4.7–19.4 是相对 operational endpoint 的倍数（APPENDICES_EN.md:217）；按 ST04，M1 相对 λmax 是 99.5/4.979 = 19.98，已超出这个范围（SUPPLEMENTARY_EN.md:185-186）。应改成 "exceeds the operational endpoint by 4.7–19.4"。
- MANUSCRIPT_EN.md:344 写 42+5.0+4.8 ≈ 52 GPU-h，却说 "about 60"。要恢复 FACTS_R1.md:195 原句中的 "with the three earlier training stages of its lineage (Table ST01)"，以及 "This cost recurs for another cell family…"。

**M8 标题**（作者决定，见 (5)；MANUSCRIPT_EN.md:1-3，D8）。改完要同步 main.tex 和 PDF 元数据。

**M9 投稿包**
- **LaTeX 构建已过时**：main.tex 和 main.pdf 生成于 8f72c63（09-29 06:50），早于 E13 写入（08a6a33，08:20）；main.tex 里 "one-corner sweep" 出现 0 次。
  - 需要重建；
  - 修掉 main.log:643 的 "Float too large for page by 108pt"，以及 :828 的 18 pt、:580 的 35 pt overfull；
  - 表 6 可能要横页，build_tex.py 目前只把表 5 放横页。
- **作者与声明**：latex/authors.tex 和 latex/endmatter.tex 还是红色占位。编号节 "Code and data availability"（MANUSCRIPT_EN.md:511-513）和 endmatter 里的 "Data availability" 重复，合并成一个。
- **Highlights 和 cover letter**：仓库里都没有。Highlights 写 3–5 条，每条 ≤85 字符（需核对期刊要求）。
- **补充材料 PDF**：build_tex.py 只读正文和附录（:100-101），需要另写补充材料构建脚本。SUPPLEMENTARY_EN.md:882-883 的标题前缺一个空行。
- **生成式 AI 使用声明**：按 Elsevier/CMAME 现行政策核对（本审计没有联网）。
- **时间**：1–1.5 天，另需作者输入。

**M10 补充材料正文中的内部路径和机器名**（投稿前，2–3 h）
- SUPPLEMENTARY_EN.md:765、:776、:796 以及全文约 49 个反引号路径中，有 `newmachine_20260924/…/cpu120/…/host`、`host1`、`hostcond` 等，暗示用过多台机器，违反"一台 CPU"的硬规则。附录 F.2 也有 3 个路径。
- 统一改成归档包内的中性相对路径。
- ST04 的列名 'fastnet-trainlib' 等是代码里的内部名称（:179），改成文字描述。

---

## (3) 建议完成

写作类（都用现有数据，合计约 1 天）：
- **S1 前处理的硬件不对称（I-06）**：清单标为完成，但审稿要点没有落实。§6.10 加一句：扣除胞建立和组装后，16 线程直接解需要 165–610 s，NICE 需要 33–99 s，约快 5–8 倍（含前处理的全程是 9–13 倍）；前处理占主机路线时间的 41–53%（ST17b/d，SUPPLEMENTARY_EN.md:750-785）。
- **S2 回本点（I-02）**：60 GPU-h ≈ 216,000 s。每个八胞迭代比 16 线程直接解省 787–932 s，约 230–275 次迭代回本（相当于约 10 次算例 A 规模的优化）；对路线 (b) 约 195–250 次。要注明两边硬件不同，算子只适用于训练范围内的族。
- **S3 主机只用 16 线程的理由（I-36）**：写一句不涉及主机共享的理由，可引用 1→16 线程分解加速 8–13 倍作为扩展性参考。
- **S4 代理误差沿路径增长（I-20）**：代理误差从 0.011% 增到 0.028%，梯度误差从 0.069% 增到 0.33%，正文要正面写出。同时写明两个终设计中，NICE 设计的精确柔度反而更低（24.110504 对 24.110806）。
- **S5 "设计迭代"的口径**：§6.10 是 81–110 s，§6.11 是每步 141 s。差别来自几何生成（14 s）、灵敏度（46.4 s 对 3.7 s）和载荷数（1 对 3），正文加一句说明。
- **S6 其余一两句的修改**：
  - 新颖性：§1 加 2–3 句，说明为什么不用参数化缩减气泡基。切割使活动单元和保留集随几何变化，各几何之间没有公共离散空间，也没有仿射分解。
  - §7.1（:483）：删掉 NICE 跨硬件加速比和 Guo 同硬件加速比的并列。
  - §7.2（:487）："7 to 290 times" 加上 "on five cells"。
  - I-32：引用 ST23b 的 τ→V^H 映射（0.18–0.70 对应 0.103–0.400），并说明所有设计比较都在同一离散模型内部；设计贴着下界，而参考解的离散误差正在这里最大（§6.2：n=32 与 64 相差 0.28%/0.44%）。

少量计算（GPU 项必须在 RTX 5090 上做）：
- **S7 严格等体积基准**：两种载荷各跑一次 V=V* 的均匀设计 NICE 分析，每个约 10 min，并加入 M2 的核对清单。
- **S8 B2 续跑，检验"局部最优"**：B2 在 k=25–26 时 dx 仍在移动限制 0.0255 上，k=29 时 dx=0.0005，按 xtol 停止。重置渐近线后续跑约 10 步，≤2 GPU-h。结果决定写"局部最优"还是"停止准则偏早"。
- **S9 宏观网格加密**：从 m=6 加密到 12，在起点复算一次，不到 1 h CPU，用来排除宏观离散误差对 27%/37% 的贡献。
- **S10 完整 KKT**：见 M5f。时间允许就算，否则删列。

呈现与存档：
- **S11 篇幅（I-37）**：按不含表格、图注和行间公式的口径，正文现在 11,360 词，并稿后约 1.2 万词，超过 R5 的 9.5–11k 窗口。可以从这几处压 600–1,000 词：§6.6 的逐个罗列、表 4（移到补充材料）、§7.1 与 Guo 的数值对照；§6.11 的细节留在 S9。
- **S12 I-59**：图 7 层掩码（:380）的来源，重算后存档，或者改成定性表述。
- **S13 图 12 加终设计的 TPMS 三维渲染**（M10/I-30）：如果删掉规模展示，正好放在 (e) 的位置，需要 3–6 h。
- **S14 图件技术要求**：F08 只有 PNG，190 mm 宽下约 284 dpi；F12_field_error_M1.pdf 有 8.5 MB；fig_opt 里有 6 pt 小字。
- **S15 派生文档和清单更新**：FIGURES.md、README_CN.md、PAPER_OVERVIEW_CN.md 已过时。清单里要修正：I-01（摘要缺限定）、I-06、I-36（各有一个审稿要点没落实）、I-37（9,800 词已过时）、I-13/I-30，补上 I-07，I-29 的图注说法也要改。
- **S16 发布归档清理**（接收前必须做；如果审稿期间提供访问包，则投稿前做）：
  - evidence/ 118 个文件中有 93 个含 `/root/autodl-tmp`；
  - X6_opt 下的 meta.json 和日志含 `CLAUDE_TAKEOVER_20260923` 目录名；
  - figures_src/figstyle.py:1 有 AI 工具名；另有 HANDOFF_FROM_CODEX_20260927.md；
  - review_r1/results/R1_ACC_SUMMARY.json、.txt 和 PREREG_R1ACC.json 含 H2 的配置 y（`fresh_val_2010_d0_v0/y`）；evidence/ 和三份文稿里没有；
  - 约 20 幅图中只有 7 个生成脚本（F01/F03/F04/F06/F07/F08/F09/F11/S02A/S02B/S03/S04 在仓库里找不到脚本）；没有 LICENSE。

  快照不带 .git 和 review_r1/。用 `grep -riE 'claude|codex|autodl|newmachine|cpu120|rep[0-9]'` 验收。需要 1–3 人日。

可选（不建议为投稿专门去做）：
- 未校正网络的优化对照（N1）：结果可能与 NICE 没有差别，反而削弱贡献 (ii)。
- 分布外零样本测试。
- 未旋转板方向的对照：两个方向 NICE 柔度之差是误差的免费下界。
- 重复修正循环下界 Ĉ₂−Ĉ₁。
- fp64 网络重跑 B2。
- 88 胞完整优化，约 18 GPU-h。
- graphical abstract。
- 推荐审稿人名单。
- 格式统一（Eq. 引用写法、e 记法、'teacher' 等）。

---

## (4) 写进局限说明，或留到正式审稿再答

首次投稿没有回复信，所以 RESPONSE_MAP 里准备"在回复信中解释"的点，第一轮审稿人看不到。重要的用一句话写进 §7.5，其余留到正式审稿时再答。
- **写进 §7.5**：
  - 没有和整体迭代求解器、AMG、BDDC、FETI-DP 比较（I-11，已写）；
  - 只有一个训练种子（已写）；
  - 没有做等成本的精度–成本比较（I-12）；
  - 更大规模上的精度未核对，单卡规模上限由全局 K_PP 的直接分解决定；
  - 均匀化误差的来源未分析；
  - 最优性以孪生运行为依据（如果不算完整 KKT）；
  - 方法依赖精确解数据：在 "require new training" 后加半句 "with exact reference solves for the mechanically generated directions and sensitivity labels"。
- **留到正式审稿**：
  - I-31 非光滑性：E13 扫描、逐步的离散开关日志、孪生结果一致，已经足够，准备好答辩要点即可；
  - I-24 贴体参考和 γ 条件数；
  - 分布外（多切割、一般法向、其他 TPMS 族）；
  - 单种子的离散程度；
  - D5 中 92 胞的结果留给 P2。

---

## (5) 需要作者拍板的决定（附推荐）

1. **最大规模报 110 胞，还是再试约 120 胞**：推荐报 110 胞，不补跑。110 胞跑通、135 胞失败，上限已经夹住了；换了机器的数据也不能和旧机混在一张表里。
2. **51 胞和 24 胞这两个点**：51 胞删掉。24 胞用 plateS24r4；没有这份记录时，只有在同型硬件（RTX 5090 + 90 GiB 主机）上才用 4 GB 预算重跑（约 40 min），否则逐行注明驻留预算。
3. **规模展示是否保留**：只有 M1 能取回完整记录才保留，否则整段删除。
4. **孪生运行的计时**：推荐表 6 写"—（verification run）"，S9 写明设备和性质，不给分阶段时间（M4）。
5. **标题**：两位审计的意见相反。推荐按条件决定：板的精确核对通过，就用 D8 候选 "Neural-initialised static condensation with equilibrium correction for the analysis and thickness design of cut thin-walled TPMS lattices"；做不成，就用不带 "design" 的方法标题。另外，I-05 已把范围收窄为 Schwarz-P-type，标题也可以考虑体现这一点。
6. **摘要**：用短版的 §6.11 句，删掉单线程数字和 0.077% 括号，成本句加限定，"cannot increase" 加前提；板句等核对完成后再定。
7. **均匀化文献**：一条经典周期均匀化文献，再加一条用均匀化做梯度 TPMS 设计的文献，都要先核实书目信息。
8. **规模时间的标注**：用 M3 的写法，完整优化时间明确标为外推。
9. **16 线程的理由**：见 S3。推荐写成"与一块 GPU 可比的计算预算"，并引用线程扩展数据。
10. **接近硬规则的措辞**：
    - S9.6 "two evaluations with unpacked storage"：删；
    - "earlier cell-local scheme"：压缩成一句；
    - §6.1 "interrupted … resumed"（MANUSCRIPT_EN.md:344）：推荐保留。这是成本核算，而且数字偏保守。如果按严格口径理解硬规则，就删掉。
11. **生成式 AI 声明**：先读 Elsevier/CMAME 政策原文。如果政策要求写出工具名，它与"不写模型标识"的内部规则冲突。推荐以期刊政策为准，这是合规问题。
12. **代码与数据可用性**：推荐写 "will be deposited at Zenodo upon acceptance"，或者现在就预留 DOI。只有做完 S16 的清理，才提供审稿访问链接。
13. **P1 和 P2 的边界（D5）**：P1 只写 88/110 胞的时间、内存和失败上限，不写精度结论。请确认这不影响 P2 的规模叙事。
14. **是否做 S8（B2 续跑）和 S10（KKT）**：推荐都做，总成本约 2 GPU-h 加半天脚本。它们直接加强"局部最优"和 M3 这两点。

---

## (6) 时间线（假设 D1 大内存机到位；D0 约为 10-01）

- **D0（机器到位前）**
  - 作者：完成 M1 的同步和完整性检查；定下 (5) 中的 1、4、5、11 项。
  - 打包 plates_exact，含 `optA:23`，如有也含等体积设计。
  - 写作：不依赖结果的部分先做，即 M7、M10、M5e/g/h/i、S1–S6、M9 中的作者块、声明、Highlights 和 cover letter 草稿，以及补充材料构建脚本。
  - GPU（只在 RTX 5090 上）：S7 两次分析、M3 的内存短测、S8 续跑，合计约 3 h。
- **D1**
  - 租机，按 runbook §3 搭环境，传包并校验 md5，跑 self-test。
  - 回归 `optA:23`，然后跑 B1:0+B2:0 试点，用 phases.json 外推时间，启动最小集，过夜运行。
  - 同时：规模记录存档，扩展 facts_6_11.py，写 ST26 和图 12(e)（或者删除）。
- **D2**
  - 最小集完成（B1:22、B2:29、Hom-z、X-z），有余力继续跑其余 4 个设计。
  - 取回结果，facts_6_11.py 生成 ST27，按 runbook §7 填写 [TBD-EXACT]，重出图 12 和表 6。
- **D3**
  - M5 并稿，M6 摘要，M8 标题。
  - 写 KKT 脚本，或者删掉那一列。
  - S11 压缩篇幅。
- **D4**
  - 构建 main.pdf 和 supplementary.pdf，修溢出，完成 M9。
  - 全文 grep 验收：`Placeholder|TBD|textcolor{red}`、机器名和路径、fp16/bf16/TF32、H2 的配置 y。
- **D5**
  - 重跑 facts 和 tables 脚本，逐一核对数字；更新清单（S15）；作者确认后投稿。
- **风险缓冲**：如果板上核对出的误差明显大于算例 A（例如 >0.1%，或者排序符号反了），加 1–2 天改写 §6.11 的结论。如果机器或数据不可用，走 M2 和 M3 的退路，D3 可投。

---

## 附：审计意见的核对与取舍

- **135 胞失败在哪个阶段**：两种说法不矛盾。18 GB 预算那一轮失败在前处理的显存不足（QUEUE_STATUS_CN.md:356，提交 a908462）；4 GB 那一轮失败在 cuDSS 分解 K_PP（提交 91f90b5 和协调人的说法）。论文以最终预算那次的存档日志为准。
- **扣除前处理后的比值**：复算结果是 5.1–7.7 倍。有一份审计写"6–8 倍"，下限偏高，统一用"约 5–8 倍"。
- **字数**：各审计口径不同，分别报了 10.7k 和 12.3k。本次口径是 11,360 词，结论一样：并稿后超出窗口。
- **GPU 峰值是否含 cuDSS 因子**：没有证实，只是可能，所以列为 M3 里的短测，不当作已知错误。
- **几条降级或改动的意见**：
  - "图 12 图注必然超页"：不构建无法确认，降为重建后检查。
  - "12 幅图无脚本"：在仓库里属实，但可用性声明承诺的是录用时归档，所以降为接收前事项。
  - "相对精确模型至少低估 26.7%/36.6%"：确认成立，理由见 M2 退路。
  - 审计指出的草稿小错（0.4518 超限、ε 的符号、Eq.(12) 的措辞、ST17e）都已逐条核实为真。
- **H2 配置 y**：三份文稿里确认没有痕迹，只在 review_r1/results 的内审文件里有，按 S16 从归档中排除。