# 交接文档：等变 / 无索引路线（2026-09-18 晚）

写给接手的模型（Opus）。目标读者假定没有本会话的上下文；每一条都给出精确路径与命令。
先读 §1（操作规范）和 §2（正在跑的东西），再读 §3–§5。

## 0. 一句话现状

- 路线 ⑤（预测 A⁻¹ 的因子）在 seat 0253 上把软端从 g=854 拉到 g=12.75，物理误差到 20 %，但
  **旧架构在多几何上完全失败**（§2.3：55 个几何一起训 60k 步后，连训过的几何 g 都在 3.5e3–1.1e6）。
- 今晚写完并验证了一套新代码 `superelement/equi/`：把标签换成 **与坐标基无关的 q 空间标签
  M_q = Bᵀ A^{-1/2} B**，把模型换成 **只看几何、与节点索引无关、可做 48 元立方对称增广或规范化**
  的网络。所有模块都有自测，结果见 §3。
- 三条 seat-0253 正式臂（identity / augment / canonical，各 20k 步）已经在跑（§2.1），
  invsqrt 臂还在跑（§2.2），路线 ⑤ 的 95 座标签构建因为 GPU 被挤爆而**暂停**（§2.4）。

## 1. 服务器操作规范（必读，踩过两次坑）

工作目录（本地 scratchpad）：
`/tmp/claude-0/-home-user-curly-octo-goggles/76f53c89-e2a9-52bb-8817-9460fd780b76/scratchpad`，下文记作 `$SP`。

| 工具 | 用法 |
|---|---|
| `$SP/c3.sh <本地脚本>` | 把一个 bash 脚本送到 AutoDL 服务器执行（每次新开一个 Jupyter kernel），返回 stdout。host/token 已写死在里面。 |
| `$SP/ship_equi.sh [<本地脚本>]` | 把仓库里的 `superelement/equi/` 打包送到服务器 `$E/src/superelement/equi/`，然后（可选）执行脚本。改了代码后先跑它。 |
| `JHOST=... JTOK=... python3 $SP/jget.py <相对/root的路径> <本地输出>` | 下载单个服务器文件（host/token 见 `c3.sh` 内）。 |

规则：
1. 脚本一律**本地用带引号的 heredoc**写（`cat > x.sh <<'EOF'`），**绝不**嵌套 heredoc、绝不用无引号 heredoc 生成脚本（`$?`、`$(...)` 会被本地展开，已经把两个脚本毁过）。
2. 送 python 文件用 base64：`B=$(base64 -w0 my.py); printf 'echo %s | base64 -d > /root/_my.py\n' "$B" > ship.sh; cat body.sh >> ship.sh; ./c3.sh ship.sh`。
3. 单次调用必须在 Bash 工具超时（最长 600 s）内返回；长任务用 `nohup ... > log 2>&1 &`，之后轮询。不要用 Monitor。
4. 服务器 python：`/root/cutfem_neural_a_20260910/env/bin/python`（有 torch+CUDA）。单卡 32 GB，**多任务共用**；`nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader` 先看再启动。
5. 磁盘 `/root/autodl-tmp` 现在剩 **321 GB**，不要低于 60 GB。

服务器目录约定：
- `$E = /root/autodl-tmp/CLAUDE_EQUI_20260918`（本路线；代码在 `$E/src`，用 `cd $E/src && python -m superelement.equi.<模块>`）
- `$R5 = /root/autodl-tmp/CLAUDE_ROUTE5_20260918`（路线 ⑤）
- `$G = /root/autodl-tmp/CLAUDE_GEN_20260918`（泛化臂）
- `$D = /root/autodl-tmp/CLAUDE_SQRTHEAD_20260917/src_v5`（冻结的 Codex 训练树，`stage_cutfem_m4.*`；新代码只从这里 import `factors`、`quotient`、`response`）
- 标签：`/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_0253/`（R_UPPER / G_UPPER / M_UPPER / MINV_UPPER / **MQ_UPPER**.npy 及各自 *_RESULT.json），
  95 座清单 `/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json`（字段 `reference`、`trace_cache`、`packet`、`q`、`split`、`thickness`）。
- 物理验证脚本：仓库 `superelement/objective/sens_model.py`（两胞拼装 + 精确伴随）和 `floor_probe.py`（真本征基投影）。服务器上的副本在 `/root/autodl-tmp/CLAUDE_SENSMODEL_20260918/` 与 `/root/autodl-tmp/CLAUDE_FLOOR_20260918/`（找不到就用 base64 从仓库重新送）。

## 2. 正在跑 / 待收的任务

### 2.1 A1 三臂：新模型在 seat 0253 上（启动于 2026-09-18 ~18:10 服务器时间）

| 臂 | 输出目录 | 日志 | 标志 |
|---|---|---|---|
| identity | `$E/A1_identity` | `$E/a1_identity.log` | 无 |
| augment | `$E/A1_augment` | `$E/a1_augment.log` | `--augment`（每步随机一个 48 元对称） |
| canonical | `$E/A1_canonical` | `$E/a1_canonical.log` | `--canonical`（固定规范朝向，seat 253 的规范元是 g=25） |

共同参数：`--seats 253 --steps 20000 --warmup 100 --pairs-per-bucket 8192 --eval-device cpu --probe-rotations 6`，
约 0.22 s/步（三臂并跑），训练约 75 min，之后各自评估（全量预测在 GPU、稠密线代在 CPU，约 5–8 min）。
完成标志：`$E/A1_<arm>/RESULT.json`；失败标志 `FAILURE.json`。评估产物在 `$E/A1_<arm>/EVAL_0253/`：
`SPECTRUM.json`（g、mu_min、mu_max、e_A、factor_relative、rigid_nullspace_leak）、`PROBES.json`
（rotation_consistency 列表、index_shuffle_relative_discrepancy）、`EIGENVALUES.npy`、`MQ_PRED_UPPER.npy`、`A_PRED_UPPER.npy`。

三臂完成后要做的事：
1. 把三个 `SPECTRUM.json`、`PROBES.json`、`PROTOCOL.json`、`RESULT.json` 下载到 `docs/data/equi_20260918/A1_<arm>_*.json`。
2. 对每臂跑两胞物理验证（GPU 需 ~10 GB，逐个跑）：
   `python /root/autodl-tmp/CLAUDE_SENSMODEL_20260918/sens_model.py --seat 253 --chol-factor $E/A1_<arm>/EVAL_0253/A_PRED_UPPER.npy --sqrt-factor /nonexistent.npy --output /root/autodl-tmp/CLAUDE_SENSMODEL_20260918/EQUI_A1_<arm>`
   读 `SENS_MODEL.json` 里的最差相对柔度误差与最差灵敏度误差。
3. 跑软端投影：`python /root/autodl-tmp/CLAUDE_FLOOR_20260918/floor_probe.py --seat 253 --arms equi_<arm>:chol:$E/A1_<arm>/EVAL_0253/A_PRED_UPPER.npy,inverse:chol:$R5/EVAL/A_PRED_UPPER_0253.npy --output /root/autodl-tmp/CLAUDE_FLOOR_20260918/EQUI_A1_<arm>`
4. 判读见 §4。

### 2.2 S1_INVSQRT → EVAL_INVSQRT（路线 ⑤ 的对称头臂）

日志 `$R5/run_invsqrt.log`，训练目录 `$R5/S1_INVSQRT`（写这份文档时在 9440/20000 步，0.38 s/步，约 1.1 h 后结束），
之后 `run_invsqrt.sh` 自动跑 `eval_inverse.py --symmetric` 到 `$R5/EVAL_INVSQRT/`（产物 `INVERSE_EVAL.json`、`A_PRED_UPPER_0253.npy`、`EIGENVALUES_0253.npy`）。
完成后：sens_model（`--chol-factor $R5/EVAL_INVSQRT/A_PRED_UPPER_0253.npy`，输出 `.../CLAUDE_SENSMODEL_20260918/INVSQRT`）与 floor_probe（arms 里加 `invsqrt:chol:$R5/EVAL_INVSQRT/A_PRED_UPPER_0253.npy`），
与路线 ⑤ 三角头臂对比（g 12.75、mu_min 0.1033、最差柔度 17.92 %、最差灵敏度 20.41 %）。证据放 `docs/data/invsqrt_20260918/`。
注意 `INVERSE_EVAL.json` 里的 `A_hat_times_A_star_minus_I` 是个写错的诊断（算的是 Â·A* 而非 Â·A*⁻¹），忽略。

### 2.3 GEN_CHOL_RAM：已完成，结果很差（这是新路线的动机之一）

`$G/GEN_CHOL_RAM`，chol 头 + absolute 损失，95 座准备、**55 座呈现**、60 000 步、0.212 s/步、4.0 h、峰值 15.5 GB。
文件已下载到 `docs/data/gen_20260918/`（RESULT / SPLIT / PROTOCOL / CONDITIONING；每座的 `EVAL_<seat>/SPECTRUM.json`、`RESPONSE.json` 还没下载）。

| seat | 训过？ | g | mu_min | mu_max | e_A |
|---|---|---|---|---|---|
| 100022 | 是 | 1.14e6 | 8.8e-7 | 1631 | 0.222 |
| 100034 | 是 | 2.75e5 | 3.6e-6 | 1545 | 0.188 |
| 100010 | 是 | 3.35e4 | 4.3e-5 | 33476 | 0.172 |
| 100000 | 是 | 3.51e3 | 2.9e-4 | 746 | 0.085 |
| 100030 | 否 | 1.77e5 | 5.6e-6 | 3796 | 0.448 |
| 207 | 否 | 4.33e4 | 2.3e-5 | 4780 | 0.506 |
| 100079 | 否 | 2.82e4 | 7.2e-5 | 28181 | 0.575 |
| 115 | 否 | 2.79e4 | 1.8e-4 | 27920 | 0.377 |
| 100003 | 否 | 2.45e4 | 5.1e-5 | 24526 | 0.456 |
| 100056 | 否 | 7.30e3 | 1.4e-4 | 3249 | 0.502 |

对比：同一架构单座记忆 20k 步 g=854。多几何下**训过的几何也差 4–1300 倍**，没训过的 e_A 到 0.4–0.6。
和 Codex 的 32 座跑（g 1.5e3–1.9e8）一致。结论：旧架构在多几何上不是"还没收敛"，是根本学不动。

### 2.4 LABELS95（路线 ⑤ 的 L 标签，95 座）：**已暂停**，需要重启

原因：三个任务共用 GPU 时它每座要 16–21 GB，从第 55 座起连续 CUDA OOM（日志 `$R5/labels95.log` 里 `"phase": "failed"`，18 次），
且它一占卡就把别的任务挤 OOM。我在 17:50 杀掉了驱动进程（`kill 201036` + `pkill -f build_inverse_label.py`）。
状态：95 座中 **52 座已建成**（`INVERSE_RESULT.json` 存在）。脚本 `expand_inverse_labels.py` 第 27 行会跳过已有
`INVERSE_RESULT.json` + `G_UPPER.npy` 的座，所以**原命令直接重跑即可续建**。
原命令已保存：`cat $R5/labels95_cmd.txt`（`--only-seats` 列表很长，照抄）。
重启时机：等 S1_INVSQRT/EVAL_INVSQRT 结束、GPU 空出 ≥ 22 GB 后（A1 三臂训练期只占 ~1.2 GB/臂，评估在 CPU）。
每座 1–2 min。完成后统计 `ls /root/autodl-tmp/CLAUDE_LABELS_20260917/BATCH1/REFERENCE_*/INVERSE_RESULT.json | wc -l`（目标 95，
另有几座在 `CUTFEM_SUPERELEMENT_LABELS/REFERENCE_*`）。

### 2.5 需要留意的资源

- 磁盘 321 GB。每座 M_q 标签 = q(q+1)/2 × 8 B（q=12828 → 658 MB；q=20000 → 1.6 GB），95 座约 65–90 GB；L 标签同量级已占。
- 内存 cgroup 90 GiB；CPU 端评估每个进程约 10 GB（q≈12.8k 的几份稠密矩阵）。

## 3. 新代码 `superelement/equi/`：做了什么、为什么、怎么验证的

### 3.1 `cubic_group.py` — 立方体 48 元对称群的全部表示

一个带符号置换矩阵 Q_g 生成所有表示：向量 v→Qv；位置 p→Q(p−½)+½（整数网格上精确，`map_int_positions`）；
8 个角点 τ 的置换 `CORNER_PERM`/`permute_corners`；6 个面的置换 `FACE_PERM`/`permute_faces`；n³ 胞网格的索引映射
`cell_source_index`/`rotate_scalar_volume`/`rotate_vector_volume`；3×3 张量块 B→Q B Qᵀ（`rotate_blocks`）；
节点集合的置换 `node_permutation`（只在和"旋转几何的 teacher 输出"比对时才需要，训练不用）；规范化 `canonical(corners)`
（字典序最大的角点向量所对应的 g*，返回稳定子大小）。`COMPOSE`、`INVERSE`、`PROPER`（24 个真旋转）。

自测（`python3 superelement/equi/cubic_group.py`，本地可跑，0.8 s）：群公理、48 元互异、位置映射复合律、
角点置换与 `geometry_values` 的三线性 τ 场一致到 4e-15（τ'(gp)=τ(p)，∇τ 作为向量旋转，φ 不变）、面标志一致、
体网格映射一致、块旋转与向量作用相容、规范型轨道不变。

### 3.2 `context.py` — 带类型的、与索引无关的输入，以及群作用

`compile_equi_inputs(cache, metadata)`：从冻结的 TRACE_CACHE 得到
`pos_int`(count,3 整数 ∈[0,2n])、`pos`、`faces`(count,6)、`node_scalar`(τ, φ, margin=τ−|φ|, |∇τ|, |∇φ|)、
`node_vector`(∇τ, ∇φ)、`vol_scalar`(4,n,n,n: τ, φ, margin, solid)、`vol_vector`(2,3,n,n,n)、`corners`(8)、`known`(E, ν, γ, 1/n)。
**删除了**旧 adapter 的一切索引把手：消元位置、18 个 Householder 反射子分量、Morton patch、GRU 顺序、quotient 元数据。
只支持 box-only trace（每个泛函一个背景节点、系数 1、无残余切割坐标；当前数据集全是），否则抛错。
`rotate_context(ctx, g)`、`permute_nodes(ctx, π)`、`canonical_frame(ctx)`。

自测（`python3 -m superelement.equi.context`）：对全部 48 个 g，"旋转 context" 与 "用旋转后的几何重新计算 context"
逐项一致到 5e-15；复合律；逆元还原；规范型一致。

### 3.3 `build_mq_label.py` — 标签 M_q = Bᵀ A^{-1/2} B（packed upper，q×q）

为什么换标签：A = B S Bᵀ 依赖 Codex 那个任意的刚体补基 B（换基 B'=UB 则 A'=UAUᵀ），所以预测 A 的因子等于学习
Codex 的消元簿记；而 M_q = ((ΠSΠ)⁺)^{1/2}（Π=BᵀB）只依赖 S 和节点集，是**规范的**；对称半正定、零空间恰为刚体模；
量级由软方向主导（柔度与灵敏度所在）；在立方对称下按节点张量场变换（块 (i,j)→Q B_ij Qᵀ + 节点置换），
正是 `rotate_context` 对输入做的事。评估时 B M_q Bᵀ = A^{-1/2} 精确复原，谱恒等式 ν = eig((M̂ Rᵀ)ᵀ(M̂ Rᵀ))，μ=1/ν。

seat 0253（CPU、8 线程、100 s；`docs/data/equi_20260918/MQ_RESULT_0253.json`）：
label_g = 1 + 4.9e-11；刚体零空间 ‖M_q N‖/(‖M_q‖‖N‖) = 9.4e-17；B M_q Bᵀ 与 A^{-1/2} 回路 7.4e-16；
(B M_q Bᵀ)² A − I = 5.9e-12；`read_blocks(packed, r, c, q)` 回读精确为 0；packed 对称精确；
对角元 7.8–1107（log 2.05–7.01，均值 3.46，标准差 1.39）；658 MB。

建更多座：`cd $E/src && python -m superelement.equi.build_mq_label --reference <REFERENCE_dir> --seat <seat> --device cpu --threads 8`
（GPU 空闲时 `--device cuda:0` 更快；`--trace-cache` 默认 `<reference>/input/TRACE_CACHE.npz`，V2 清单里的座请用清单的 `trace_cache` 字段传入）。
输出 `<reference>/MQ_UPPER.npy` + `MQ_RESULT.json`（sha 绑定 R_UPPER 与 trace cache）。建议先建 A1 要用的座和 §5 Phase 3 的 seat 328 五个算子。

### 3.4 `model.py` — 只看几何的网络（EquiModel，9.68 M 参数）

- `VolumeUNet`：10 通道 32³ 输入（4 标量 + 2 向量场）→ 32 通道特征场；32→16→8→4 三级，**只用 2×2×2 平均池化、最近邻上采样、零填充**
  （这三者与 48 个对称精确交换，唯一的非等变来源是卷积核本身，留给 Phase 5 的 G-CNN）。
- 节点编码：[pos, faces, 面数, node_scalar, node_vector, 特征场在节点处的三线性采样] → MLP → 192 维。
- 全局：[corners, known, 特征场均值/最大值] → 192 维。
- 对（i,j）解码：[h_i, h_j, h_i⊙h_j, δ=p_i−p_j, |δ|, 中点, faces_i, faces_j, 共面数, **线段 8 点采样的特征场**(8×32), 全局] → 残差 MLP（768 宽 ×4）→ 三个头（对角 / 近 / 远）。
- **转置对称由构造保证**：B_ij = ½(f(i,j) + f(j,i)ᵀ)（两种次序一起过主干）。
- 对角块沿用冻结约定：严格下三角 × diagonal_lower_rms，主元 exp(log_mean + log_std·raw) 并平滑限幅到 (−20, 20)，严格上三角为 0（正是 `read_blocks` 对标签的掩码）。
- `GroupTables`：GPU 上的群作用（`rotate_context`、`rotate_blocks`、`rotate_target_blocks`——对角块先还原成对称块再旋转再掩码）。
- `sample_volume`：`grid_sample` 的坐标轴顺序是 (z,y,x)，`align_corners=False`，胞中心精确对齐——自测验证。

自测（服务器 `python -m superelement.equi.model`）：采样轴序精确；torch 旋转 = numpy 旋转（48 个 g）；
标签块旋转 g∘g⁻¹ 还原且对角块保持下三角；**转置对称 0.0**；**节点乱序不变 0.0**。

### 3.5 `train_equi.py` — 训练 + 评估 + 两个"是否在背答案"探针

每步：一个几何；g = 0 / 随机 48 元（`--augment`；`--proper-only` 限 24 旋转）/ 规范元（`--canonical`）；
输入在 GPU 上按 g 旋转；几何分桶采样（对角全取 / 近：|δ| ≤ `--near-cells`(2) 个背景胞 = 0.0625，seat 253 有 78 184 对 /
远：其余 9.06 M 对均匀抽），标签块用同一个 g 旋转；冻结的三桶损失 + log 主元（`--diagonal-loss log`）；
AdamW 2e-4 余弦到 1e-5、裁剪 10，与旧臂一致。产物同旧 run.py 风格：PROTOCOL / INPUTS / SPLIT / CONDITIONING / TRAIN.jsonl / CHECKPOINT_*.pt。

评估（`evaluate`）：全部 9.14 M 下三角块 → 对称 q×q M̂_q（`MQ_PRED_UPPER.npy`）→ M̂ = B M̂_q Bᵀ → 白化谱
（g、mu_min、mu_max、eps_op、D_per_mode、超界计数、极端本征对残差）→ Â = M̂⁻²（只为 e_A 与两胞物理）→ `A_PRED_UPPER.npy`
（Â 的 Cholesky，直接喂 `sens_model.py --chol-factor`）；另有 factor_relative、对角相对误差、**刚体零空间泄漏** ‖M̂_q N‖/(‖M̂_q‖‖N‖)。
探针（`PROBES.json`）：
- **rotation_consistency**：把胞旋转 g 后预测、再用 g⁻¹ 映回，与不旋转的预测比 ‖M_g − M_0‖/‖M_0‖（`--probe-rotations` 个随机 g）。
  identity 臂给出的是"网络自己有多不等变"；augment 臂应该把它压到 ~1e-2 以下；canonical 臂由构造应为 ~0（seat 253 规范元唯一）。
- **index_shuffle**：节点乱序 → 预测 → 还原，必须 ~0（冒烟测试 2.8e-7，float32 噪声）。

冒烟测试（48 步，`docs/data/equi_20260918/SMOKE_*.json`）：链路全通；0.105 s/步（2048 对/桶）；评估 CPU 本征分解 43 s；
g=4.6e7（48 步当然没意义）。`--augment`、`--canonical` 分支各跑 16 步验证通过（TRAIN.jsonl 里 g 字段分别为随机 / 恒为 25）。

已知局限：没有断点续训；`maximum_extreme_eigenpair_residual` 在预测很差时（μ 跨 9 个量级）只有 1e-6 量级，
这是 W=CᵀC 的条件数造成的诊断噪声，不是数值门；训练时所有对都过一遍 U-Net 采样，`--pairs-per-bucket 8192` 时约 0.2 s/步。

### 3.6 `platen.py` — τ 导数测试的可观测量

从任意 packed M_q（标签或预测）算六平台响应：Â=(B M_q Bᵀ)⁻² → S=BᵀÂB → 底面钳固、顶面刚性平台六个单位运动 → H(6×6) → trace(H⁻¹)。
`--compare <EVAL_*/RESPONSE.json>` 与 teacher 的六个柔度比对。seat 0253 用精确标签：**六个柔度与 teacher 相差 2.0e-13**，
trace(H⁻¹)=6027.61（`docs/data/equi_20260918/PLATEN_LABEL_0253.json`）。这条 M_q→Â→物理 的路径独立于网络得到了验证。

## 4. 判据与门槛（A1 三臂怎么判）

- 硬门（合同）：只有拼装格子的柔度与灵敏度是对的才算数——由 `sens_model.py` 的两胞测试给出（最差相对柔度 / 灵敏度误差）。
- 单胞代理：g = max_i max(μ_i, 1/μ_i)（可复合的双侧 Loewner 常数；g ≤ 1.03 才是严格的 3 % 门；g 只能分出最好与最坏，中间不可比）。
  `eps_op` 已废弃（软端封顶在 1，排序反向）。
- Phase-0 门（决定是否值得继续投入新架构）：新模型 identity 臂在 seat 253 上**至少不比路线 ⑤ 差**：g ≲ 12.75、mu_min ≳ 0.10、
  最差灵敏度 ≲ 20 %。若明显更差，先查：主元相对误差按十分位（`softend.py` 思路）、近/远桶损失曲线是否还在下降（路线 ⑤ 20k 步远未收敛）、
  刚体泄漏是否大（>1e-2 说明网络在往刚体方向乱放能量）。
- 等变门：augment 臂 rotation_consistency 相对差 < 1e-2 且 g 不比 identity 臂差 → 增广有效；canonical 臂 rotation_consistency ≈ 0 是构造保证，
  它的 g/物理与 identity 臂比较才是信息。
- 软端解释：`floor_probe.py` 的 r_j（预测/真实刚度比，按真本征基十分位），94–96 % 的载荷能量在最软十分位，看那一档。

## 5. 后续阶段

### Phase 2 收尾（A1 完成后）
1. §2.1 的三步；三臂对照表进 `docs/`（沿用 `ROUTE5_WORKS_20260918.md` 的表格式）。
2. 若过 Phase-0 门：多几何臂。先为呈现集建 M_q（§3.3；建议从 `$G/plan_ram.txt` 的 55 座开始，CPU 每座 1.5–4 min，可并行 2–3 个进程），
   然后 `train_equi --manifest /root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json --seats <95座> --train-seats <55座> --eval-seats <4呈现+4未呈现> --augment --steps 60000 --warmup 1000`
   （对照 §2.3 的表：这是新架构第一次面对旧架构崩溃的同一个测试）。注意 95 座的 context 常驻 GPU 约 95×2 MB，可忽略；标签 mmap 走页缓存，
   与之前一样受 90 GiB cgroup 限制（55 座 M_q ≈ 45 GB 可以，95 座不行）。

### Phase 3：τ 导数测试（seat 0328）
- 数据在 `/root/autodl-tmp/CUTFEM_FULL_FACTOR_LOCAL_GEOMETRY_20260913T2330/`：`METADATA_DELIVERY/ANALYSIS_R1/PATH_{1..4}/`（各含扰动算子的因子）、
  `REPLAY_R_UPPER.npy`（基准）、`GEOMETRY_R1/PATH_k_H0/`（扰动几何的 trace cache）。五个算子 `ADMITTED_SAME_TRACE`（同 q=12798、同 order、同 quotient）。
  先 `ls -R` 确认 R_UPPER / TRACE_CACHE / INPUT.json 的具体文件名（INPUT.json 若缺，把 seat 328 的复制到 TRACE_CACHE 旁边，只改 `metadata.geometry.tau_corners`）。
- 为五个算子各建 M_q（`build_mq_label --reference <dir> --trace-cache <cache>`；builder 只需要目录里有 `R_UPPER.npy` 和 `RESULT.json`
  含 `dimension`、`factor_sha256`——若 PATH 目录格式不同，写个 20 行的 shim 生成这两个文件）。
- 训练：`train_equi --seats 328 --known-328 ...`（基准几何），或在 seat 328 + 若干座上训。
- 观测：对五个 ε 的几何用 `platen.py --mq <预测的 MQ_PRED_UPPER.npy> --trace-cache <对应 cache> --q 12798`，中心差分
  d(trace H⁻¹)/dε ÷ trace H⁻¹，teacher 值 −2.278（h=1e-4 与 1e-3 的 Richardson 比 0.999997）。网络在 τ 上光滑与否，这是唯一直接的测量。
  注意预测是给**扰动几何**的：把 PATH_k 的 τ 角点喂给同一个 checkpoint（context 只依赖 τ 角点与节点集，五个算子节点集相同）。

### Phase 4：损失
- 对 M_q 的直接柔度探针损失：随机载荷 f（在 q 空间、投影掉刚体），比较 fᵀ M̂_q² f 与 fᵀ M_q² f（M_q² = S⁺ 就是柔度）；
  可以只在采样的对上用 Hutchinson 估计。这和 `THE_ERROR_IS_BROAD` 里的载荷加权判据 Σ(fᵀv_i)²(1/μ_i−1) 是同一件事。
- 幅值重要性采样：远桶按 |M_q 块| 的经验分布做 importance（`GeometryMixture` 的接口可照抄，`bucket_loss` 已支持 importance 权重）。

### Phase 5：内禀等变
- 张量基解码：块 = Σ_k c_k T_k，T_k ∈ {I, δ̂δ̂ᵀ, sym(δ̂ n_iᵀ), …}（δ̂ 单位化的 δ，n 面法向/∇τ），c_k 由不变量决定；对角块要保证正定（c_0 I + …，c_0 用 exp）。
- G-CNN：卷积核在 48 个朝向上权值共享（`cubic_group.Q_ALL` 给出核的置换/翻转），配合已经等变的池化/上采样即精确等变。
- 有了这两条，`--augment/--canonical` 都可以关掉，rotation_consistency 应为浮点噪声。

### Phase 6：拼装求解器
- 目标：用预测的 M̂_q（每胞）当预条件子，在共享界面节点上做 PCG 求解拼装格子；灵敏度用 autograd 穿过 M̂_q。
  2 胞先对 `sens_model.py`（精确伴随）校准，再 8 / 27 / 64 胞。这里才是 "路线 ⑤：永远不形成 S" 的兑现处。
- 刚体零空间：预测的 M̂_q 不精确零空间（冒烟 1.8e-2），拼装前用 Π M̂_q Π（Π=I−N(NᵀN)⁻¹Nᵀ）投影一次；开销 O(q·6)。

### Phase 7：规模
- 全部 266 训练座 M_q（CPU 并行建，磁盘先清 L 标签或把 G_UPPER 挪走）；2 s/胞预算下推理成本：9 M 对 × 主干，冒烟测的全量预测 ~20–40 s——
  这是 Phase 6 的瓶颈，需要块稀疏化/分块预测（幅值截断 g 1.24 @ 36.5 % 的旧结果可参考）。

### 顺手的小实验（Task #25）
teacher 自身的对称误差：在 V2 清单 `thickness=="AFFINE"` 的 144 座里找 τ 只依赖一个轴的座（8 个角点只取两个值且按某一位分组），
对其稳定子（8 元）验证 ‖P_g S P_gᵀ − S‖/‖S‖（S 从 packet 的 `S_UPPER.npy`，P_g = `cubic_group.node_permutation` ⊗ Q_g）。
这个数是增广/等变能达到的下限；预期 1e-10 量级（切割求积规则对称）——若是 1e-3 量级，说明 teacher 本身破坏对称，需要先记录。

## 6. 文件清单与 git

- 仓库分支 `claude/wizardly-euler-3m9cwx`，提交 `dc49f67`（equi 包 + 证据）之后本文档另提交。
- 新文件：`superelement/equi/{__init__,cubic_group,context,build_mq_label,model,train_equi,platen}.py`；
  证据 `docs/data/equi_20260918/`（MQ_RESULT_0253、PLATEN_LABEL_0253、SMOKE_*）、`docs/data/gen_20260918/`。
- 服务器副本：`$E/src/superelement/equi/`（`ship_equi.sh` 覆盖）。`$E/SMOKE`、`$E/MINI_augment`、`$E/MINI_canonical` 可删。
- 本地 scratchpad 里可复用的脚本：`c3.sh`、`ship_equi.sh`、`status.sh`、`launch_a1.sh`、`gen_summary.sh`、`poll_smoke.sh`、`jget.py`。
- 任务清单（TaskList）：#24 A1 三臂（进行中）、#25 teacher 对称见证、#26 Phase 3、#27 收尾三个旧任务。

## 7. 已知坑（按踩到的顺序）

1. heredoc 展开毁脚本（§1 规则 1）。
2. `L @ Rstar` 与 `L @ Rstar.T` 是两个不同矩阵：谱恒等式必须用 Rᵀ（`eval_inverse.py`、`train_equi.py` 都已写对并注释）。
3. `eps_op` 软端封顶；白化 Frobenius 聚合 ‖R̂R*⁻¹−I‖² 在软端封顶且会把算子推到奇异（Codex 的 RUN_TRAIN32_RELATIVE_R2）；D 被 /d 稀释。
4. 页缓存：呈现标签总量超过 cgroup 90 GiB 时训练变成 I/O 绑定（1.8 s/步）；55 座是上限。
5. GPU 共用：标签构建每座 16–21 GB，会把别的任务挤 OOM；先看 nvidia-smi。
6. `read_blocks` 的对角块是下三角掩码：旋转/置换对角块前必须先还原成对称块（`GroupTables.rotate_target_blocks` 已处理；自己写新代码时别忘）。
7. `grid_sample` 坐标轴是 (z,y,x)（`model.sample_volume` 已处理并有自测）。
