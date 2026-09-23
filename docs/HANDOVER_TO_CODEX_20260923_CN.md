# 给 Codex：Claude 接续工作的交接（2026-09-23）

写于 2026-09-23 约 12:00 北京时间。回应你的 `CODEX_HANDOVER_20260923/HANDOVER_TO_CLAUDE_MECHANICS_LEARNING_20260923_CN.md`。
详细数据见 `docs/NIGHT_REPORT_20260923_CN.md`（逐项实验）和 `docs/MORNING_REPORT_20260923_CN.md`（结论与待决事项），
证据与源码副本在 `docs/data/takeover_20260923/`。服务器工作根 `T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923`。

**当前没有任何在跑的任务**（GPU 0 MiB，CPU 上无我的进程）。你交接时在跑的中度完整谱 PID 218807 已正常结束，结果见第 2 节。

---

## 0. 一句话

你建议的架构“几何 → 局部能量数值 → 确定性装配与跨尺度平衡 → 完整边界响应”已经端到端跑通，
而且**“几何 → 局部能量数值”这一格目前由确定性几何积分填上，不是网络**：
每个单元在子网格上把三个水平集线性化，对裁出的多面体精确积分 125 个四次张量矩，再乘固定弹性模板得到单元刚度。
配合冻结代码生成的 P 和 GP，第二批 31 个有教师标签的几何（完全不用教师产物）全部在 ±3% 以内。
平衡部分也从“有限层 Chebyshev”换成了 GPU 稀疏 Cholesky（cuDSS），查询精确、毫秒级。

这改变了“网络该学什么”的问题。第 5 节是我对学习路线的判断，以及它的证据边界。你可以不同意。

## 1. 对你交接文中各项建议的逐条回应

| 你的建议 | 我做了什么 | 结果与状态 |
|---|---|---|
| 中度强耦合配对的完整谱（PID 218807） | 等它跑完并核对 | **[1.0000, 1.00133]，D/d 2.1e-10，通过**。重度 [1, 1.00154]，FULL（32 层）[1, 1.0223]；`CROSS_CASE_04_PAIR0p9/FULLSPEC_*` |
| 把局部耦合变成因果证据（只联合两对 vs 拆开两对） | **没做** | 主线改用直接分解后，平滑器不再在关键路径上。若仍想为旧网络的失败写机理，这组对照依然有价值 |
| 非负能量分解复核误差位置 | **没做** | 同上 |
| 同一困难方向送进旧 E1/E2 | **没做** | 这是把新发现接回旧学习问题的桥，仍未搭 |
| 固定规则是否推广到未参与调试的母场 | 做了 | 配对平滑器 + 64 层在 6 个开发样本上**不普遍够用**（dev0000 1.033，dev0006 1.071）；编码时 Lanczos 自适应定深度（容差 1e-6）后全部通过。后来整条平衡换成直接分解，这个问题不复存在 |
| 走到真正的几何学习 | 先检查了“学局部数值”的可行性，然后找到了确定性替代 | 见第 3、5 节 |
| 局部参数化先能重建教师能量 | 做了 | 125 个矩乘固定模板重建全部单元刚度，最大误差 6.9e-13（存档 V 是 Gram 的因子：Gram = VᵀV） |
| 分开“局部数值误差”和“平衡计算误差” | 做了 | 教师单元 + 直接分解：见证 1 + 1e-11（平衡精确）；几何单元：误差全部来自单元积分 |
| 独占 GPU 复测成本 | 做了 | 见第 2 节的时间表 |
| 灵敏度 | **没做** | 最重要的缺口，见第 6 节 |

## 2. 主线结果（全部“有限见证”，除非注明“全谱”）

**链路**：几何参数 →（冻结代码）P、GP →（GPU）单元矩、组装、PᵀKP →（GPU）cuDSS 规划与分解 → 查询 S·q = Dq + Cᵀz。

| 结论 | 证据 | 位置 |
|---|---|---|
| 第二批 31 个有标签几何，不用教师产物，全部在 ±3% 以内 | 有效分辨率 8：见证 1.0016–1.0158，中位数约 1.0050 | `DIRECT_01`、`DIRECT_02`、`DIRECT_04` |
| 分辨率加倍，误差约缩小 4 倍 | 最差 4 个在有效分辨率 16 下为 1.0020–1.0039 | `DIRECT_03` |
| **全谱**：中度 train13，几何单元，PARDISO 精确凝聚 | 有效分辨率 8：[0.99923, 1.0064]，D/d 1.5e-7；**有效分辨率 16：[0.9999984, 1.0021]，D/d 9.6e-9** | `ELEMENT_TOL_03/` |
| **全谱**：重度 train3，子网格 s=4 | [0.9959, 1.0107]，D/d 8.9e-7 | `ELEMENT_TOL_03/train3_S4_all` |
| P、GP 从几何重算与生产机逐位一致 | 配精确单元时 7 个探针反力与教师差 < 1e-12（31 个样本） | `COVER_INPUTS/<case>/RESULT.json` |
| 第二批 G 阶段本机重算 | 36/36 与生产机矩逐位对账 MATCH | `COVER_G/runs/*_G_VERIFY.json` |
| 第一批几何单元（教师 P/GP） | 重度 1.0024，中度 1.0064，FULL 1.0034，train6_d1_v1 1.0042，train6_full 1.0044，dev0000 1.0056 | `DIRECT_01`、`DIRECT_02` |

**时间（独占 GPU，RTX 5090，FP64）**：编码 0.9–18 s（单元积分 0.4–2.4 s，PᵀKP 0.02–1.5 s，cuDSS 规划 0.02–3 s，分解 0.01–5.8 s）；
查询 1 列 0.5–40 ms，64 列 1–324 ms。**编码时间不含 P 和 GP**：冻结的 compile_trace 1–216 s（93% 在精确有理数逐行消元），GP 2–31 s（选面 10 s）。

三种查询方式在教师单元下的对比（只比平衡求解）：

| 样本 | Chebyshev（约 186 层）1 / 64 列 | CPU PARDISO 分解；回代 1 / 64 列 | GPU cuDSS 规划 + 分解；查询 1 / 64 列 |
|---|---|---|---|
| 重度 | 0.42 / 3.5 s | 1.5 s；0.05 / 0.45 s | 0.4 + 0.03 s；2 / 17 ms |
| 中度 | 1.7 / 32 s | 9.0 s；0.52 / 7.1 s | 3.0 + 1.5 s；16 / 133 ms |
| FULL | 2.8 / 54 s | 17 s；0.88 / 12.6 s | 4.9 + 3.1 s；28 / 227 ms |

## 3. 单元级的关键实验（为什么没有训练局部数值网络）

在重度 train3 上做精确凝聚，对单元刚度施加不同形式的扰动，看整体 μ：

| 单元误差的形式 | 1% | 3% | 10% |
|---|---|---|---|
| 能量意义（谱）K^½(I+εE)K^½ | [0.995, 1.005] | [0.983, 1.015] | [0.948, 1.052] |
| 逐条目相对 K∘(1+εZ) | [-25, 4.7]，非正定 | [-1026, 15] | — |
| 125 个单项式矩逐个相对误差 1e-4 | μ ∈ [0.867, 1.145] | | |
| 固定 GLL 网格非负权重拟合（p=8） | 整体 [0.862, 1.389]；只有 41% 的单元能精确表示 | | |

对照：确定性多面体积分在有效分辨率 8 时，单元能量偏差中位数 0.05%，整体见证 ≤ 1.016，二阶收敛。
第一批 111 个、第二批 21 个样本的单元级普查：大单元（体积分数 ≥1e-3）里单个单元的最大偏差可达 18.8%，
但整体误差都在 1.6% 以内，因为决定整体的是占绝大多数、偏差 < 1% 的单元。

## 4. 如何复现（服务器）

```
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923
# 1) 第二批：本机重算 G 阶段（冻结树的私有副本 COVER_G/src；冻结树本身不写）
bash $T/xcase_src_11/run_cover_g.sh <workers> <case> ...   # 之后 verify_cover_g.py 对账
# 2) 第二批：冻结代码生成 P、GP，并用精确单元做 PARDISO 探针核对 → COVER_INPUTS/<case>/
bash $T/xcase_src_11/run_cover_blocks.sh <case>
# 3) GPU 编码 + 冻结见证协议（最新版在 xcase_src_19）
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $T/xcase_src_19
/root/cutfem_neural_a_20260910/env/bin/python -u encode_gpu.py --solver cudss --gpu-trace \
    --case <case> --cover-inputs $T/COVER_INPUTS/<case> --elements polyref --s 4 --levels 1 --output <dir>
#   第一批：去掉 --cover-inputs（用 MECHANICS_NETWORK 资产）；train6 与开发样本加 --asset-root $T/ASSETS_DEV
#   有效分辨率 = s·2^levels（s=4, levels=1 → 8；levels=2 → 16）；--elements teacher 用教师 A/C/D 只测平衡
#   --solver chebyshev 是旧的层级 + 自适应 Chebyshev 路径
# 4) 汇总
/root/cutfem_neural_a_20260910/env/bin/python $T/xcase_src_18/summarize_direct.py $T/DIRECT_01 $T/DIRECT_02
```

`encode_gpu.py` 的 `--solver cudss` 用 `DirectModel` 实现了冻结评估器需要的接口（`A C D Ub Ui`、`forward`、`energy`、
`extension`、`extension_adjoint`），见证、7 个原始探针和数值恒等式都走你的冻结 `evaluate_mechanics` / `evaluate_chebyshev_strain`，没有改动。

## 5. 对学习路线的判断（可以不同意）

1. **局部数值**：“几何 → 局部能量”已经有一个确定性、半正定、刚体零空间精确、二阶收敛、GPU 上 1–3 秒的实现。
   网络要在这里有价值，必须在同样的能量范数精度下更快或更可微，而目前积分本身已经不是瓶颈。
2. **证据边界要说清**：第 3 节是**随机扰动**实验，不是训练出来的网络。结构化的输出（例如预测子网格积分权重、
   保证半正定的因子，并按能量范数训练）可能比随机扰动好得多。所以严格地说，结论是
   “逐条目或逐矩回归风险很大，而确定性基线已经很强”，而不是“网络不可能”。
3. 如果用户仍要网络，我认为有信息量的位置是：**以确定性积分为基线、学习它的修正**
   （比如用低分辨率积分加网络修正，去逼近高分辨率积分），目标是速度，精度由能量范数下的界来保证。
   另一个位置是替代慢的 Python 环节（compile_trace、GP），但那些是确定性代码，重写比学习更合适。
4. 你提的“计算结构是否把需要协同的自由度拆开了”：在迭代求解器里，这是真实的机理（配对平滑器让中度从 12.6 变为全谱 [1, 1.0013]）；
   直接分解下它不再影响精度。它对旧网络学习差距有多大解释力，需要第 1 节里没做的那座桥（把困难方向送进 E1/E2）。

## 6. 没做完的事（按我认为的优先级）

1. **设计灵敏度**（最重要）：对 τ 的 8 个角点值、切面法向和偏移的导数，要求 3% 以内。
   线性化积分对 τ 是分段光滑的，导数精度未知。建议用伴随：d(qᵀSq)/dθ = uᵀ(dK/dθ)u，dK/dθ 由矩对几何参数的导数给出；
   与教师的有限差分或已有的灵敏度标签对比。
2. **compile_trace 提速**（待用户决定）：(a) 用编译语言按原规则重写精确消元，坐标逐位不变；(b) 换成浮点可算的坐标约定，需要一次坐标迁移（冻结代码里有 `exact_coordinate_migration`）。
3. GP：我自己的自由度散射已向量化（2.7 s → 0.1 s，逐位相同，`xcase_src_18/gp_time.py` 中验证），尚未并入 `cover_blocks.py`；冻结选面的区间判定 10 s 未动。
4. **更多全谱**：第二批 31 个只有有限见证；全谱只做了 train13（两档分辨率）和重度。至少应对最差的 0020、0017 各补一次全谱。
5. 第一批普查中 5 个样本失败未查；第二批 5 个开发母场（dev0000/0002/0003/0005/0007 的 cover01_r1）没有教师标签，没有做端到端。
6. cuDSS 限制：规划时固定右端列数，编码器按 8 列一块求解；规划（重排序）在 CPU 上，只依赖稀疏结构，可以缓存或提速。

## 7. 等用户拍板的事

1. 是否正式放弃神经网络路线（我建议放弃，或改成第 5.3 节的“学习修正”）。
2. compile_trace 选 (a) 还是 (b)。
3. 精度档位：有效分辨率 8（最差 1.6%）还是 16（最差 0.4%，单元积分约乘 4）。
4. 第二批改作验证集。

## 8. 环境与坑

- cuDSS 装在我自己的目录 `$T/pylib_cudss`（nvmath-python 1.0.0、cuda-core 1.2.0、cuda-bindings 12.9.8、nvidia-cudss-cu12 0.8.0.10），
  靠 `PYTHONPATH` 和 `LD_LIBRARY_PATH` 引入，**没有装进你的 env**。
- cuDSS 用自己的 cudaMalloc，**分解前要 `torch.cuda.empty_cache()`**，否则 PyTorch 缓存的空闲显存它拿不到（ALLOC_FAILED）。
- 冻结评估器的能量梯度检查会自动求导穿过 `torch.sparse.mm`，torch 会为大稀疏块建转置副本，大样本显存不足；
  `DirectModel` 用存好的转置写了自定义反向（`_SpMM`）。autograd 反向在另一个线程里，调 cuDSS 前要 `cuda.core.Device(0).set_current()`。
- 多面体单元时，数值检查中只有 `full_variational` 一项不通过，Chebyshev 路径也一样：它用教师内部场验证“能量差 = 误差的 A 范数”，前提是模型的 A 就是教师的 A，换了单元必然不成立。
- G 阶段重算需要 `CUTFEM_MEMORY_POLICY=container_headroom_1g` 和 `runtime_environment()` 导出的库路径（含 r13_pardiso_v1/lib），见 `run_cover_g.sh`。
- PARDISO：`CUTFEM_EXECUTION_CONFIG=$MN/NATIVE_EXECUTION_CONFIG.json`，`CUTFEM_RUNTIME_ROOT=/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime`。
- 同一时间只跑一个大 GPU 任务：夜里几次显存不足都是并发造成的。

## 9. 继续有效的约束

- 两棵冻结树不改：`/root/autodl-tmp/CUTFEM_INGEST_R38/source_independent_6624dc8_20260910`（HEAD 6624dc86…）和 `/root/autodl-tmp/CLAUDE_SQRTHEAD_20260917/src_v5`；我用的是私有副本 `COVER_G/src`。
- `dataset_independent_20260910`（1.3 TB）不删。不碰独立 CPU 生产主机，不读封存测试集，不新增付费资源。
- 门槛不放宽：工作 [0.9, 1.1]，目标 [0.97, 1.03] 且 D/d < 1e-4；不用 jitter、伪逆、谱裁剪、删方向。
- 切面承载蒙皮：P 保留全部切面几何泛函，没有做自由切面凝聚，也没有抬高保留体积下限（第二批保留体积最小到 0.008）。
- 仓库只在分支 `claude/wizardly-euler-3m9cwx` 上开发和推送。远程脚本在本地写好后 base64 传送，不嵌套 heredoc。
