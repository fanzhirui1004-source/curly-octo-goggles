# P1 论文接手文档（2026-10-01）

本文档给接手的人（或新会话）用：读完即可知道论文现状、文件在哪、怎么重建、还剩什么、哪些规则不能碰。

## 1. 一句话现状

论文 P1（NICE：neural-initialised static condensation with equilibrium correction，目标期刊 CMAME）的科学内容已全部完成并写入正文。R1 内审修订已完成，§6.11 厚度优化（含板设计的精确核对和规模展示）已并入正文和补充材料。正文与补充材料的 PDF 都能编译通过，无错误。

剩下的主要是作者本人的事项（作者信息、声明、通读）和发布前的归档清理，见第 6 节。

- **标题**：Neural-initialised static condensation with equilibrium correction for the analysis and thickness design of cut thin-walled TPMS lattices
- **摘要**：246 词（CMAME 上限 250）。
- **分支**：仓库 `fanzhirui1004-source/curly-octo-goggles`，分支 `claude/wizardly-euler-3m9cwx`。

## 2. 文件地图（`docs/paper_p1/`）

| 路径 | 内容 |
| --- | --- |
| `MANUSCRIPT_EN.md` | 正文（编辑源）：§1–§8、Code and data availability、References |
| `APPENDICES_EN.md` | 附录 A–J |
| `SUPPLEMENTARY_EN.md` | 补充材料：R1/R2、表 ST01–ST27、Note S1–S9、图 S01–S06 |
| `references_verified.bib` | 已核实的参考文献库（正文用作者-年份链接，构建脚本自带参考文献列表） |
| `figures/` | 全部图（主图 1–12，补充图 S01–S06）。**注意文件名与图号不一致**：`F13_optimisation.*` 是图 12，`F12_*` 是图 7，`S06_reference_verification` 是图 S01 |
| `figures_src/` | 部分图的生成脚本，如 `fig_opt.py`（图 12、图 S06）；`figstyle.py` 是统一样式 |
| `FIGURES.md` | 图目录，由 `latex/gen_figures_md.py` 生成 |
| `latex/` | elsarticle 构建：`build_tex.py`（正文+附录 → `main.tex`）、`build_supp_tex.py`（补充材料 → `supp.tex`）、`preamble.tex`、`authors.tex`（作者占位）、`endmatter.tex`（CRediT、利益冲突、致谢、AI 声明占位）、`gen_supp_tables.py`（部分补充表的生成） |
| `evidence/` | 论文引用的结果记录（发布时随代码归档）。`evidence/opt/` 是 §6.11 全部记录：算例 A、板、均匀化、规模（`scale/`）、精确核对（`exact_plates/`） |
| `submission/` | `HIGHLIGHTS.md`（候选 8 条，推荐 1、2、4、5、7）、`COVER_LETTER_DRAFT.md`（初稿） |
| `review_r1/` | 内审过程文件，**不随论文发布**，见下表 |

`review_r1/` 中常用的文件：

| 文件 | 内容 |
| --- | --- |
| `REVIEW_CHECKLIST_R1_CN.md` | 审稿意见逐条状态 |
| `SUBMISSION_READINESS_CN.md` | 09-30 的投稿准备度盘点（必须 / 建议 / 局限 / 决定 / 时间线），其中大部分 must 项已完成 |
| `NIGHT_LOG_20261001_CN.md` | 10-01 夜间工作记录：核对过程、数字、机器状态 |
| `PROPOSED_EDITS_S1_S6_CN.md` | S1–S6 建议稿，已全部并入正文 |
| `PATH_MAP_EVIDENCE_CN.md` | evidence 中各文件与原始工作目录的对照 |
| `EXACT_CHECK_RUNBOOK_CN.md` | 大内存 CPU 机精确核对手册（已用过一次） |
| `DRAFT_6_11_EN.md`、`DRAFT_S9_EN.md` | 已并入正文，仅作历史；**今后改 MANUSCRIPT/SUPPLEMENTARY** |
| `results/X6_opt/` | §6.11 结果与事实脚本：`facts_6_11.py` → `FACTS_6_11.json`；`tables_s9.py`；`scale/scale_summary.py`；`exact_plates/exact_plates_facts.py` → `EXACT_PLATES.json` |
| `GUO_SERIES_COMPARISON_CN.md` | 与 Guo 等系列工作的逐项对比 |

代码镜像在 `docs/data/newmachine_20260924/src_v2_wip/`，其中：
- `opt_design.py`、`mma.py`、`homog_*.py`、`lat_scale.py`、`stream_ops.py`：优化与规模展示；
- `exact_check_cpu.py`、`pack_exact_inputs.sh`：精确核对；
- `r1_chains/`：各运行链脚本。

镜像里**缺** `box_encode.py`、`encode_r1.py`、`element_moments.py` 三个模块。服务器的 `src_v2` 里有，已核对与旧服务器逐字节相同。

## 3. 怎样重建

```bash
cd docs/paper_p1
python3 figures_src/fig_opt.py                 # 图 12 与图 S06（读 review_r1/results/X6_opt）
python3 latex/gen_figures_md.py                # FIGURES.md
cd latex
python3 build_tex.py && pdflatex main.tex && pdflatex main.tex          # 正文 PDF（当前 93 页）
python3 build_supp_tex.py && pdflatex supp.tex && pdflatex supp.tex     # 补充材料 PDF（当前 79 页）
```

- 需要 pandoc、pypandoc 和 TeX Live。
- 构建脚本会自动做这些事：把表 5、表 6 放到横页；把很长的图注改用小字号；"Code and data availability" 设为不编号节；末尾按顺序插入声明。
- 事实与表格脚本：`review_r1/results/X6_opt/facts_6_11.py`、`tables_s9.py`、`exact_plates/exact_plates_facts.py`、`scale/scale_summary.py`。论文中的每个数字都能追溯到这些脚本或 evidence 中的记录。

## 4. 关键数字速查（均已核实）

**单胞与格架：**
- 80 个验证几何的平均方向能量误差 0.074%。
- 两胞装配：开发胞的柔度误差 < 0.06%、灵敏度误差 < 0.7%；权重选择之外的胞分别 < 0.28% 和 < 1.5%。
- 两个八胞格架：柔度误差 ≤ 0.015%，灵敏度误差 ≤ 0.15%。

**成本：**
- 八胞设计迭代：NICE 81–110 s（RTX 5090），整格架 Cholesky 868–1,042 s（16 线程，AMD EPYC 9654）。
- 离线成本约 60 GPU-h，其中数据 42 GPU-h；回本约 230–270 次八胞迭代。

**§6.11 厚度优化：**
- 算例 A：与精确孪生运行的终设计相差 ≤ 0.0051。精确核对：代理误差 0.011–0.028%，梯度误差 0.069–0.33%。
- 板 B1/B2：精确核对代理误差 0.013–0.038%，终设计梯度误差 0.22% 和 0.32%，64 个分量符号全对。
- 均匀化模型低估初始柔度 26.7% 和 36.6%。
- 弯曲方向：X-z 比 B2 低 0.87%，Hom-z 比 B2 低 0.34%，精确模型中同样成立。
- 规模展示：24、51、88、110 胞（切割有限元模型 6.5–32.7 百万自由度，凝聚为 0.39–1.62 百万自由保留坐标），每步 9、20、35、43 min；110 胞时显存用满；135 胞时 K_PP 的 cuDSS 分解超出显存。

## 5. 硬规则（不能违反）

- 论文中不出现任何 AI 模型标识；不提重复运行、负载、降频、共享主机。
- 论文只写一种 CPU（AMD EPYC 9654）和一种 GPU（RTX 5090）。精确核对所用租用机的硬件和时间不写入论文。
- H2 胞在配置 y 下的结果在论文中不留任何痕迹（只存在于 `review_r1/results` 的内审文件中，发布时排除）。
- 不用 fp16、bf16、TF32。
- 冻结目录不改；Codex 目录只读；`dataset_independent_20260910` 永不删除。
- 不要直接删除文件，先列出来给作者确认；不猜测、不尝试凭据；不开新的付费资源；大动作先讨论；GPU 生产运行前先告知作者。
- 提交只推到分支 `claude/wizardly-euler-3m9cwx`。提交信息末尾加两行 trailer：
  - `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  - `Claude-Session: <会话链接>`
- 与作者交流用中文。

## 6. 剩余事项

**作者本人：**
1. `latex/authors.tex`：作者、单位、通讯作者。
2. `latex/endmatter.tex`：CRediT、利益冲突、致谢与基金、生成式 AI 声明（Elsevier 要求写工具名称和用途，作者自行撰写）。
3. 通读正文与补充材料；在 Highlights 中选 5 条；定稿 cover letter。

**发布或录用前（SUBMISSION_READINESS_CN.md 的 S16）：**
- evidence/ 中的 JSON 含服务器绝对路径和机器名，`evidence/opt/` 的 meta 和日志含旧目录名，需要清理；
- `figures_src/figstyle.py` 第 1 行有工具名；
- 约 12 幅图在仓库中没有生成脚本；
- 没有 LICENSE；
- 打包快照时不带 `.git` 和 `review_r1/`。验收命令：`grep -riE 'claude|codex|autodl|newmachine|cpu120|rep[0-9]'`。

**可选的增强（投稿时不是必需）：**
- 正文约 1.2 万词，可压缩 600–1,000 词（§6.6 的逐个罗列、表 4 移入补充材料）。
- 等体积均匀设计基准；B2 续跑约 10 步，检验"局部最优"；宏观网格 m=6 → 12 的加密核对；完整 KKT（全部起作用约束的乘子）。
- 图件技术要求：F08 只有 PNG（约 284 dpi）；`F12_field_error_M1.pdf` 有 8.5 MB。
- main.log 里还剩两处 18–19 pt 的 overfull，不影响阅读。

## 7. 机器与数据

| 机器 | 状态 | 关键路径 |
| --- | --- | --- |
| GPU 机（RTX 5090，25 vCPU，90 GiB；AutoDL bjb2） | 数据已同步完。10-01 起没有运行任何计算 | `/root/autodl-tmp/OPL/src_v2`（运行代码）；`/root/autodl-tmp/OPL/S1/V2/R1/OPT`（优化运行，651 GB）；`/root/autodl-tmp/OPL/S1/V2/R1/SCALE`（规模运行）；`/root/autodl-tmp/exact_tools/`（核对工具与拼合代码）；`/root/autodl-tmp/OPL/P1_PAPER_20261001/`（本次上传的论文材料，见第 9 节） |
| 大内存 CPU 机（32 vCPU，240 GiB） | 核对已完成，**可释放**（关机或退租由作者操作） | `/root/autodl-tmp/xc/`（核对工作目录与 T 缓存） |
| AutoDL 文件存储 `/root/autodl-fs`（同区域共享） | 持久 | `XC_PACK/plates_exact.tar`（核对输入）、`XC_PACK/exact_results.tgz`（核对全部结果，md5 6fdb6436…）、`OPL_DEPLOY/`；`P1_PAPER_20261001.tgz`（本次上传的论文材料） |

访问方式：通过 Jupyter kernel API 访问（主机地址和 token 由作者提供，不写入仓库）。服务器上没有 git 凭据，代码同步走打包上传或文件存储。

## 8. 进行中与后续方向

- **文献调研**（10-01 启动，后台运行）：
  - (A) 全局 B 样条体映射的扭转或共形单胞，以及有没有映射单胞的学习型凝聚；
  - (B) NICE 作为 BDDC/FETI-DP 的非精确子域或延拓求解器，单侧谱序 S ≤ Ŝ 能推出的条件数界；
  - 无数据（能量或 Ritz 型）训练。
  
  结果会整理到 `review_r1/` 或新目录中，由作者决定。
- **方向 A 的思路**：映射单胞等价于参考单胞配上各向异性材料 C̃ = det J · J⁻¹CJ⁻ᵀ（指标形式）。几何与切割流程可以复用，只需让单元刚度支持逐点材料张量。第一步是零样本扭转测试，在 GPU 机上跑，跑之前须经作者同意。
- **方向 B**：理论性更强，面向 SISC/CMAME。
- 早先列出的 P2–P10 方向见会话记录。
