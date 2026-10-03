# 方向 A 交接（样条/等参映射 TPMS 单胞上的 NICE），2026-10-01

本文给接手方向 A 的新会话。P1 论文由另一个会话继续改，两边共用同一台 GPU 机器（见 §2.4）。

## 0. 先读这些（按顺序）

1. `docs/followup/A/STEP0_PLAN_CN.md`：第 0 步的问题、离散设定、代码、单元测试、实验设计、判据（§6）、三次运行记录（§9）和全量结果与建议（§10）。
2. `docs/followup/ROADMAP_20261001_CN.md`：A 在整体路线中的位置、竞品（郭旭组 Xu 2025 弯扭胞元 PIML、Antolin 组映射 TPMS + BDDC）、"最小可发表结果"的设想。
3. `docs/followup/LIT_SURVEY_AB_20261001_CN.md` §2.4–2.5：映射单胞的三个陷阱（刚体模态、参考坐标水平集、积分权）。
4. P1 论文 `docs/paper_p1/MANUSCRIPT_EN.md` 的 §2–§4：NICE 的构造（学习延拓 Ê、校正 W = Chebyshev 8 / Q1(17) Galerkin 粗修正 / Chebyshev 8、能量读出 Ŝ = FᵀKF）。
5. `docs/paper_p1/HANDOVER_20261001_CN.md`：P1 整体背景（只需浏览）。

**这些文件都在分支 `claude/wizardly-euler-3m9cwx` 上**（还没合到 main）。新会话克隆的是自己的分支，先执行：
`git fetch origin claude/wizardly-euler-3m9cwx && git checkout origin/claude/wizardly-euler-3m9cwx -- docs/followup docs/data/newmachine_20260924/src_v2_wip`
（或者把自己的分支建在它上面）。P1 论文目录 `docs/paper_p1/` 归另一个会话，A 会话只读，不改。

## 1. 现状一句话

P1 的网络（检查点 A3，即论文里的 NICE）不重训，直接用在等参映射胞上（16 个验证胞 × 24 个映射，384 对全部成功）：
- 结构在所有映射下都成立；
- 共转拉回（V0R）让以转动为主的映射保持 P1 精度量级；
- 各向异性拉伸是唯一明显的失效方向；
- 平均值掩盖了最坏方向，最坏方向随映射涨得很快。

数字见 STEP0_PLAN §10 与 `full16/SUMMARY.md`。

| 映射 | V0（直接用） | V0R（共转拉回） | 最坏方向 μ−1 中位数（下界） |
| --- | ---: | ---: | ---: |
| 恒等 | 0.072% | 0.072% | 0.23% |
| 刚体旋转 30° | 3.72% | 0.074% | 0.3% |
| 扭转 10° / 30° 每胞 | 0.144 / 0.771% | 0.089 / 0.273% | 0.85 / 18% |
| 弯曲 R/L = 5 / 3 | 0.128 / 0.226% | 0.097 / 0.141% | 0.27 / 0.41% |
| 剪切 0.3 | 0.906% | 0.481% | 34% |
| 拉伸 x×2 / x×0.5（cond J = 2） | 2.66 / 3.87% | 同左 | 26 / 215% |

表中误差均为一致面力（force_c）方向的 16 胞平均能量误差。只做校正、不用网络时为 350–930%。完整胞对映射几乎不敏感，误差来自切割胞（P1 里本来就难的 2003、2007、2012、2015）。物理刚体基分离在所有映射下都使刚体延拓能量保持在 10⁻¹² 量级；照 P1 的参考坐标分离（V0ref）时，旋转 30° 下达到 0.14，即刚体陷阱是真实存在的。

## 2. 机器与访问

### 2.1 机器

- AutoDL 租用的 GPU 机器，北京 B 区（bjb2），一张 RTX 5090（32 GB）。作者付费租用：**不要新租、不要释放、不要关机**，这些都由作者决定。
- 访问方式：通过机器上的 JupyterLab 网关执行命令。主机名和 token 由作者在开启会话的消息里提供，环境变量名分别为 `JHOST`、`JTOK`。
- **token 绝不能写进仓库里的任何文件**，只放在环境变量或 scratchpad 里。

### 2.2 工具（仓库内 `docs/followup/A/tools/`，均不含 token）

- `jrun3.py`：通过 Jupyter kernel 的 websocket 在服务器上执行 shell 命令。
  - 依赖 `pip install websocket-client`。
  - 使用 `HTTPS_PROXY` 和 `/root/.ccr/ca-bundle.crt`；这两样 Claude Code 云环境里已有。
- `c3n.sh <script.sh>`：把本地脚本 base64 后发到服务器执行。用法：
  `export JHOST=… JTOK=…; JRUN_NEW_KERNEL=1 bash c3n.sh my.sh`。
  务必带 `JRUN_NEW_KERNEL=1`，否则可能排在一个卡住的 kernel 后面。
- `rr.sh`：c3n.sh 加连接错误重试。
- `upload_example.sh`：上传文件的模式（文件 base64 嵌在远程脚本里，上传前备份旧文件，上传后打印 md5 核对）。
- `status_full16.sh`：远程查看 a0_eval 运行状态的例子。

使用要点：
- 每次调用用 `timeout 300` 包住；单次输出不要太大（大结果先 gzip + base64，再在本地解码）。
- 长任务在服务器上用 `setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &` 启动，本地用后台轮询脚本每 10 分钟查一次。不要在前台 sleep。
- 本地工作目录在并行的 Bash 调用之间会被重置，一律用绝对路径。

### 2.3 服务器上的目录与环境

- `/root/autodl-tmp/OPL/src_v2/`：P1 的冻结代码，不要改。
- `/root/autodl-tmp/OPL/A0/src/`：A 的代码。
  - 新文件：`mapped_cell.py`、`a0_eval.py`、`a0_maps.json`、`a0_summary.py`、`t_mapped.py`、`t_mapped_quick.py`。
  - 这些文件与仓库镜像 `docs/data/newmachine_20260924/src_v2_wip/` 下的同名文件一致，上传记录在 `runs/full16/CODE_MD5.txt`。
- `/root/autodl-tmp/OPL/A0/runs/full16/`：全量结果。
  - `RESULTS.jsonl`：每 (胞, 映射) 一行；续跑时只追加，取最后一条。
  - `SUMMARY.md` / `SUMMARY.json`、`COMPACT.json`、`CASES.txt`、`CODE_MD5.txt`。
  - `data/`：映射后的方向库与网络输入。
- `/root/autodl-tmp/OPL/A0/logs/`：运行日志。`run_rest.sh` 每个胞一个进程，支持续跑，结束时写 ALLDONE。
- 检查点：`/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt`。
- 胞体：`/root/autodl-tmp/OPL/S3/body/<case>/`。
- 几何参数包：`/root/autodl-tmp/OPL/S3/packets/<case>/FRESH_CONTEXT.json`。
- 运行环境变量：
  `PYTHONPATH=/root/autodl-tmp/OPL/A0/src:/root/autodl-tmp/OPL/src_v2 OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S3/packets OPL_CONV_FP32=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- 软件环境：
  - PyTorch 2.8.0+cu128；scipy 1.18.1；nvmath-python 1.0.0 + nvidia-cudss-cu12 0.8.0.10。
  - cuDSS 是用 `--no-deps` 装的，并钉住 `cuda-bindings==12.9.*`。**不要直接 `pip install nvidia-cudss-cu12`**：它会把 cuBLAS 升到 12.9，弄坏 PyTorch cu128。
- 显存：完整胞（约 40 万自由度）同时持有 Neumann 和内部两个 cuDSS 因子时超过 31 GiB。a0_eval.py 现在一次只持有一个因子，出错时也会释放。新脚本要保持这个做法。

### 2.4 与 P1 会话共用 GPU

- P1 会话目前只改文字，不在 GPU 上跑东西；如需跑，会经作者协调。
- A 会话启动任何 GPU 任务前，先查 `nvidia-smi` 和 `pgrep -af python`，确认没有别人的进程。
- 作者的规矩：生产性的 GPU 运行先告诉作者（预计时长、内容），作者同意后再跑。

## 3. 代码要点

- `mapped_cell.py`：
  - 映射族 `make_map(spec)`：identity、rotation、scale、affine、twist、bend、grade、bezier。
  - Q2 基函数（含 Hessian）。
  - 正权积分块：复用 P1 的 Kuhn 四面体切割；满子立方用 3 点 Gauss，整单元全满用 5 点 Gauss。
  - 45 组加权矩 × 固定模板：A = det J · J⁻¹ C J⁻ᵀ，只变换导数指标，位移仍是笛卡尔物理分量。
  - 物理坐标的 ghost penalty（物理梯度、完整等参 Hessian、Nanson 面积元、h_x = h·det^{1/3}）；物理刚体基。
  - `class MappedCell(teacher.Cell)`。
  - **未实现**：`lean`、`assemble_deploy`、`dmoments`、`energy_density`、`sens`、`sens2`（直接抛 NotImplementedError）。也就是说，映射胞上还没有厚度灵敏度，也没有部署路径。
- `a0_eval.py`：
  - 映射后的方向库：一致面力用 Nanson 因子。
  - 网络输入数据由映射后的 K̃ 写出。
  - 四个变体：
    - V0：网络在参考网格上用，物理刚体分离；
    - V0R：每个节点用 ∂x/∂X 的极分解转动 R 做共转拉回；
    - V0ref：照 P1 的参考坐标刚体分离；
    - Conly：只做刚体部分加校正。
  - 指标：各类能量误差均值、p90、最大、最小；V0/V0R 的最坏方向 μ（block power，rtol 1e-2，最多 25 步，只是下界）；物理刚体模态的延拓能量；映射统计。
- 单元测试 `t_mapped.py`（**在服务器 GPU 上跑，不要在本地 CPU 上跑**，这是作者的要求）。已通过的项：
  - 恒等映射与 P1 的 K、矩、ghost 逐项一致（1e-14）；
  - 旋转、缩放、仿射映射下的变换规律（1e-14）；
  - 扭转、弯曲、梯度、Bézier 映射下的刚体核（3.5e-16）；
  - 单元刚度半正定。

## 4. 待讨论、待决定的事（作者）

1. **目标几何里有没有明显拉伸**（单元长宽比偏离 1 超过约 1.3）。有，就要做"拉伸条件化"：网络输入极分解的右伸长张量 U（每节点 6 个数），只用含拉伸的映射生成数据。没有，主线就是"物理刚体分离 + 共转拉回 + 校正"，不需要新数据。
2. **是否先出短文或预印本**（零样本 + 共转拉回 + 刚体陷阱 + 映射 CutFEM 正确性），用来抢在竞品前面占位。
3. **下一步跑哪些**，每项都要作者点头才上 GPU：

| 步骤 | 内容 | 成本 | 回答什么 |
| --- | --- | --- | --- |
| 0b 校正预算扫描 | 拉伸、剪切 0.3、扭转 30° × 16 胞 × 1/2/4 个校正循环 | 约 2–3 GPU 小时，基本只改参数 | 不重训，靠校正能把分布外误差降多少（可改进性） |
| 0d 最坏方向分析 | 胞 2012_d0_v2、2014_d0_v1（恒等与 twist30），迭代到收敛，看方向的空间分布、所在节点、一致面力能否激发 | 约 1 GPU 小时 | P1 要不要在 6.4 提一句；也是方向 B 的第一步 |
| 0c 映射格架 | 弯曲或扭转的 2×2×2 格架，全学习，与精确映射凝聚比柔度 | 代码 1–2 天（映射胞装配；若要灵敏度还要实现 dmoments/sens），运行约 2 GPU 小时 | 平均好、最坏差，装配后哪个起作用 |
| 拉伸条件化 | 含拉伸映射的数据生成 + 再训练 | 数据生成可能需要第二台机器（此前讨论过：到数据生成时再考虑租同区 5090，由作者决定），数天 | 仅在第 1 项答"需要"时做 |

**特别提醒：** 恒等映射就是 P1 的 NICE 本身。P1 在 2012_d0_v2 上有能量误差 ≥ 44.5% 的方向，2014_d0_v1 上 ≥ 8.8%，而 P1 正文报告的采样方向最大值是 1.24%。0d 的结论要回报给作者，再由作者转给 P1 会话。

## 5. 作者的长期规矩（与 A 相关的部分）

- 用中文回复。
- 删除任何文件之前先列出来，等作者确认（"别直接删，先列出来"）。
- 不猜、不试凭据。
- 不碰 westb 那台 CPU 主机。
- 不新租任何付费资源。
- 不用 fp16/bf16/TF32。
- 新行为放在默认关闭的开关或新脚本里，不改原有路径；冻结的目录（`src_v2`、P1 的 evidence）不改；Codex 的目录只读；`dataset_independent_20260910` 不删。
- 进程只在作者同意时杀（自己启动、出了错的运行除外）。
- 大动作之前先讨论。
- 提交信息末尾按会话系统提示给出的署名行写（Co-Authored-By 与本会话自己的 Claude-Session 链接）。
- 不在提交、文档、论文里写模型名。
- 只推送到自己会话指定的分支。
- token 不写进仓库。
