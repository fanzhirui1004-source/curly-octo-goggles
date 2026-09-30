# 板算例设计的精确核对：大内存 CPU 机操作手册

本手册用于 Section 6.11 中板算例（B1、B2、均匀化设计 H_y/H_z 及其 NICE 续算 XH_y/XH_z）的精确核对。核对在一台租用的大内存 CPU 机上完成，不需要 GPU。所用脚本都是新增文件，原有脚本不改：

- `docs/data/newmachine_20260924/src_v2_wip/exact_check_cpu.py`：核对程序，在 CPU 上复现 `opt_design.py --check`。
- `docs/data/newmachine_20260924/src_v2_wip/pack_exact_inputs.sh`：在 5090 服务器上打包所需输入，只读原始数据。

核对内容与 `opt_design.py --check`（算例 A 的 k = 0/12/23 核对）相同。每个设计生成一个 `check_exact_KKK.json`，键名与 `check_KKK.json` 相同，另加一个 `cpu` 记录。核对内容如下：

1. 逐胞构造 fp64 稠密精确凝聚矩阵 T。
2. 在主机上组装格架，用与原运行相同的预条件（`bnn:kpp:q1r`）从零初值做 PCG，递归相对残差收敛到 1e-10，得到精确柔度。
3. 用内部因子回代得到精确场 u = E q，再求 8 个角点的精确灵敏度（`dmoments` 中心差分，与 `analyse_exact` 相同）。
4. 按顶点聚合，并与运行记录中的 NICE 值比较，得到代理柔度误差、梯度相对误差、余弦、逐分量误差分位数、符号一致率，以及只含体积乘子的 KKT 值。

---

## 1. 核对哪些设计

"mid" 的定义是最后一步编号整除 2（`last // 2`）。下表中的数据取自运行记录 `results/X6_opt/plates/*/history.jsonl`。T 的大小按记录中的保留自由度数 n 计算，每个 T 为 8n² 字节。"去重后"一栏：k = 0 一行已按服务器上的文件核对；其余各行是按离散开关特征估计的不同胞数。实际数目一律由脚本按输入的指纹判定。

| 运行:k | 含义 | NICE 柔度 | NICE PCG 次数（1e-6） | 保留自由度/胞 | T 合计 | 去重后（估计） | 最大单胞 T |
| --- | --- | --- | --- | --- | --- | --- | --- |
| plateB1:0 | B1 起点（均匀 τ = 0.40） | 94.340658 | 169（冷启动） | 15,654–30,573 | 86.4 GiB | 7 个胞，30.7 GiB | 6.96 GiB |
| plateB2:0 | B2 起点，几何与 B1:0 相同 | 1,467.743385 | 183（冷启动） | 同上 | 与 B1:0 共用 T | — | — |
| plateB1:11 | B1 中间设计 | 95.347693 | 221 | 9,048–32,196 | 66.1 GiB | 24 个胞 | 7.72 GiB |
| plateB1:22 | B1 最终设计 | 87.088895 | 177 | 9,048–34,260 | 68.0 GiB | 24 个胞 | 8.75 GiB |
| plateB2:14 | B2 中间设计 | 1,504.775265 | 282 | 9,048–26,496 | 60.8 GiB | 约 22 个胞 | 5.23 GiB |
| plateB2:29 | B2 最终设计 | 1,346.353540 | 200 | 9,048–30,384 | 63.0 GiB | 约 20 个胞 | 6.88 GiB |
| hevalH_y:0 | H_y 在细尺度上的 NICE 评估 | 88.019342 | 245（冷启动） | 9,048–36,159 | 72.1 GiB | 约 23 个胞 | 9.74 GiB |
| hevalH_z:0 | H_z 在细尺度上的 NICE 评估 | 1,341.783699 | 292（冷启动） | 9,048–30,384 | 61.5 GiB | 约 18 个胞 | 6.88 GiB |
| xstartH_y:11 | XH_y 最终设计 | 87.141332 | 162 | 9,048–34,260 | 69.2 GiB | 24 个胞 | 8.75 GiB |
| xstartH_z:11 | XH_z 最终设计 | 1,334.696860 | 261 | 9,048–30,384 | 61.5 GiB | 约 18 个胞 | 6.88 GiB |

说明：

- B1:0 与 B2:0 的几何逐字节相同，两个运行目录下 body 文件的 md5 已在服务器上核对。k = 0 时 16 个未切胞（19,872 个保留自由度，T 为 2.94 GiB）完全相同。8 个切胞的 body 文件两两相同（保留体积 0.917/0.333/0.667/0.083 各一对）。其中 0.917 和 0.333 两对的切平面偏移也完全相同；0.667 和 0.083 两对的偏移只在末位不同（例如 0.27735009811261513 与 0.2773500981126147）。指纹包含偏移的全部位数，所以这 4 个切胞各自构造 T。k = 0 共 7 个不同的 T（30.7 GiB），两种载荷共用。同一次调用中先算 B1:0、再算 B2:0，T 会自动复用。
- XH_y/XH_z 的 k = 0 就是 H_y/H_z 本身，body 从 hevalH 拷贝而来，所以核对 hevalH_y:0 和 hevalH_z:0 即可。
- 所有设计都是 24 胞：16 个未切胞，8 个切胞。运行参数全部从各自的 meta.json 读取（clamp x,min；load x,max；B1/H_y/XH_y 的载荷列为 y，B2/H_z/XH_z 为 z；tmin 0.18；tmax 0.69；预条件 `bnn:kpp:q1r`），不需要在命令行上重复给出。
- 草稿表 ST27 中 H、XH 两行的梯度列是"—"。可以用 `--no-sens` 只算这 4 个设计的精确柔度，每个设计约省 1 小时。第 5 节给出两种情况的时间。

## 2. 租什么机器

| 项目 | 建议 | 依据 |
| --- | --- | --- |
| 内存 | ≥ 192 GB，推荐 256 GB | 求解阶段要同时装入一个设计的全部不同 T：k > 0 的设计为 61–72 GiB（hevalH_y 最大，72.1 GiB），另加粗空间和 K_PP 因子等工作内存（估计 10–20 GiB），合计约 90 GiB。构造 T 的进程每个峰值估计 40–60 GiB（ST18 中 G4 的 Schur 路线峰值为 42.7 GiB，板胞的保留自由度更多）。192 GB 可以并行 2 个 T 进程，256 GB 可以并行 3 个。 |
| CPU | ≥ 32 个物理核，64 核更好；x86-64，最好支持 AVX-512（MKL） | T 的构造、灵敏度（矩导数）和 PCG 都靠多核。PCG 每步读一遍全部 T（60–70 GiB），受内存带宽限制，双路机建议用 `numactl --interleave=all` 启动。 |
| 数据盘 | 使用 `--delete-T` 时 ≥ 200 GB；保留全部 T 时 ≥ 700 GB | 输入包约 1.5 GiB。T 全部保留至多约 555 GiB（9 组，k > 0 按未去重计）。加 `--delete-T` 后，同一时刻最多保留一个设计的 T（≤ 72 GiB），外加 k = 0 那组（30.7 GiB）和矩导数缓存（每个设计约 3 GB）。 |
| GPU | 不需要 | — |
| 容器限制 | 开机后查 `cat /sys/fs/cgroup/memory.max`（v2）或 `/sys/fs/cgroup/memory/memory.limit_in_bytes`（v1） | 容器里 `free` 显示的是宿主机内存。脚本按 cgroup 限额和匿名内存判断可用内存（与 `pardiso_direct.avail_gib` 相同）。 |

与此前 120 GiB 主机相同的型号（AMD EPYC 9654）最省事，但型号不影响精度。这台机器上测得的时间不写进论文（论文只点名 AMD EPYC 9654 和 RTX 5090），见第 7 节。

## 3. 软件环境

此前 120 GiB 主机 CPU 路线的版本，取自 `docs/data/newmachine_20260924/r1_cpu_results/cpu120/R1/cpu/host/env_host.json` 与 `env_host.txt`：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12.3（/root/miniconda3/bin/python） |
| NumPy / SciPy | 2.3.2 / 1.18.1 |
| PyTorch | 2.8.0（该机装的是 +cu128 版，在 `CUDA_VISIBLE_DEVICES=` 下只用 CPU；CPU 版 2.8.0 亦可） |
| MKL | 2026.1（pip：mkl 2026.1.0、intel-openmp 2026.1.2、tbb 2023.1.0、tcmlib 1.5.0、umf 1.1.0；目录 /root/autodl-tmp/mklenv/lib） |
| pypardiso | 0.4.7（目录 /root/autodl-tmp/pylib_cpu） |
| 环境变量 | 见 `src_v2_wip/r1_chains/env_cpu_westb.sh` |

新机器上的安装步骤：

```bash
# 方案 A：pip（与 120 GiB 主机版本一致）
python3.12 -m venv /root/xenv && source /root/xenv/bin/activate
pip install numpy==2.3.2 scipy==1.18.1
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install mkl==2026.1.0 intel-openmp==2026.1.2 pypardiso==0.4.7
M=$(python -c "import sys; print(sys.prefix)")/lib        # pip 的 mkl 把 libmkl_rt.so.3 放在 <venv>/lib
ls $M/libmkl_rt.so.3
# 方案 B：若 120 GiB 主机还在，整体拷贝 /root/miniconda3、/root/autodl-tmp/mklenv、/root/autodl-tmp/pylib_cpu，再照 env_cpu_westb.sh 设置
```

环境变量和后面命令用到的变量全部写进一个文件 `/root/autodl-tmp/xc/xc_env.sh`（相当于 `env_cpu_westb.sh`），每个新 shell 都先 `source` 它。tmux 或 nohup 新开的 shell 不继承当前 shell 里未 export 的变量；已经在运行的 tmux 服务器保留它启动时的环境，之后的 export 不会进入新的 tmux 窗口。所以不要依赖手工 export，一律在 tmux 窗口里 source 这个文件。

```bash
mkdir -p /root/autodl-tmp/xc
cat > /root/autodl-tmp/xc/xc_env.sh <<'EOF'
# source /root/autodl-tmp/xc/xc_env.sh    （每个新 shell、每个 tmux 窗口里先执行一次）
export W=/root/autodl-tmp/xc                                  # 数据盘目录
[ -f /root/xenv/bin/activate ] && source /root/xenv/bin/activate   # 方案 A 的 venv
export M=/root/xenv/lib                                       # 方案 A；方案 B 为 /root/autodl-tmp/mklenv/lib
export LD_LIBRARY_PATH=$M PYPARDISO_MKL_RT=$M/libmkl_rt.so.3
export OPL_DEV=cpu CUDA_VISIBLE_DEVICES= OPL_GP_CACHE=0       # 脚本对自身和子进程也会强制这三项
export PYTHONPATH=                                            # 方案 B 时设为 /root/autodl-tmp/pylib_cpu
export R=$W/plates_exact/runs WK=$W/plates_exact/exact_cpu IP=$W/plates_exact/extra/iparm_tuned.json
# 64 核、256 GB（其他机器见 4.4 节的表）：
export COMMON="--work $WK --t-route schur --iparm-file $IP --threads 64 --t-jobs 3 --t-threads 21 \
  --cell-jobs 8 --cell-threads 8 --sens-jobs 4 --sens-threads 16 --min-free-gib 64 --delete-T"
EOF
source /root/autodl-tmp/xc/xc_env.sh && echo "$COMMON"
```

- 线程数不用手工设置：主进程的线程数由 `--threads` 决定，各子进程的 `MKL_NUM_THREADS`/`OMP_NUM_THREADS` 由 `--t-threads`、`--cell-threads`、`--sens-threads` 决定。
- 若出现 "OMP: Error #15 … libiomp5 already initialized"（torch 的 OpenMP 与 MKL 的 libiomp5 冲突），加 `export KMP_DUPLICATE_LIB_OK=TRUE`。120 GiB 主机上同样的组合没有出现过这个问题。
- 不要直接运行 `make_T_cpu.py`。几个板运行都用了 `OPL_GP_CACHE=0`，body 里没有 `GP_UPPER.npz`，`teacher.Cell` 会调用 `box_encode.ghost_faces_gpu`，而它的设备固定为 cuda:0。本地已复现这一失败：`AssertionError: Torch not compiled with CUDA enabled`。`exact_check_cpu.py` 的子进程先像 `bench_cpu2.py` 那样设置 `box_encode.dev`，再调用原样的 `make_T_cpu.main()`。需要 `make_T_cpu.py --check` 时，用 `exact_check_cpu.py --make-t-check <body>:<case>` 代替。

## 4. 操作步骤

### 4.1 在 5090 服务器上打包（只读原始数据）

先从仓库把下面三个文件拷到服务器上的同一个目录 $T（例如 /root/autodl-tmp/exact_tools/）。后两个文件下面用 `--extra` 打进包里，所以必须事先放在 $T；否则打包会在 dry-run 之后报 `MISSING`。

| 仓库路径 | 拷到 |
| --- | --- |
| `docs/data/newmachine_20260924/src_v2_wip/pack_exact_inputs.sh` | `$T/pack_exact_inputs.sh` |
| `docs/data/newmachine_20260924/src_v2_wip/exact_check_cpu.py` | `$T/exact_check_cpu.py` |
| `docs/data/newmachine_20260924/r1_cpu_results/cpu120/R1/cpu/host/iparm_tuned.json` | `$T/iparm_tuned.json` |

```bash
O=/root/autodl-tmp/OPL/S1/V2/R1/OPT
T=/root/autodl-tmp/exact_tools
SPECS="$O/plateB1:0,mid,last $O/plateB2:0,mid,last $O/hevalH_y:0 $O/hevalH_z:0 $O/xstartH_y:last $O/xstartH_z:last"
bash $T/pack_exact_inputs.sh --dry-run $SPECS                   # 先看清单和大小，不写任何文件
bash $T/pack_exact_inputs.sh --out /root/autodl-tmp/exact_pack --name plates_exact \
     --extra $T/exact_check_cpu.py --extra $T/iparm_tuned.json $SPECS
```

- `--dry-run` 已在服务器上实跑过（只读）。10 个设计每个 264 个文件，约 143–169 MiB；加上 src_v2/*.py，全包 2,878 个文件，1.463 GiB。
- 包的结构是 `plates_exact/runs/<run>/{history.jsonl, meta.json, layout.json, body/GP_TEMPLATES_n32.npz, body/<case>/, packets/<case>/}`，再加 `plates_exact/src_v2/*.py`（与运行所用代码逐字节相同）、`extra/`、`MANIFEST.tsv`（字节数、md5、路径）和 `SPECS.json`（每个 k 对应的 case 列表、NICE 柔度、载荷设置）。
- 输出文件只有 `--out` 下的 `plates_exact.tar`、`.MANIFEST.tsv`、`.SPECS.json`、`.tar.md5`，打包时的符号链接临时目录结束后自动删除。`--out` 不能放在任何运行目录之内，脚本会拒绝这种设置。

### 4.2 传到新机器并校验

```bash
W=/root/autodl-tmp/xc                                  # 新机器上的数据盘目录
mkdir -p $W && cd $W
# 用 scp / rsync 把 plates_exact.tar 与 plates_exact.tar.md5 传到 $W
md5sum -c plates_exact.tar.md5
tar -xf plates_exact.tar
cd plates_exact && awk -F'\t' 'NR>1{print $2"  "substr($3, index($3,"/")+1)}' MANIFEST.tsv | md5sum -c --quiet && echo MANIFEST_OK
cp extra/exact_check_cpu.py src_v2/                   # 必须与 teacher.py 等放在同一目录
```

### 4.3 自检与计划

```bash
cd $W/plates_exact/src_v2
python3 exact_check_cpu.py --self-test                # 期望最后一行为 ALL PASS（约 10 s）
R=$W/plates_exact/runs; WK=$W/plates_exact/exact_cpu
python3 exact_check_cpu.py $R/plateB1:0,mid,last $R/plateB2:0,mid,last $R/hevalH_y:0 $R/hevalH_z:0 \
        $R/xstartH_y:last $R/xstartH_z:last --work $WK --dry-run
```

dry-run 对每个设计输出一行，包括胞数、按指纹去重后的不同胞数、需要新建的 T 个数和不同 T 的总 GiB，以及从 meta 读到的 clamp/load/载荷列。核对这些值与第 1 节一致：B2:0 的 `T_to_build` 应为 0，即与 B1:0 共用。

### 4.4 先跑 k = 0 这一对（验证整个流程，估计 0.5–1.5 小时）

T 路线的选择如下：

- 默认路线 `make_T_cpu` 原样调用 `make_T_cpu.py`：每 256 列做一次 pypardiso LU 内部回代。按 ST18 与 E1 记录估算，每个板胞要 6 分钟到 2 小时以上，一个设计要 10–30 小时（见第 5 节），所以不建议用于全部设计。
- `--t-route schur` 用 PARDISO Cholesky 的 Schur 补选项（iparm(36)）一次得到稠密 S，即 Table ST18 "Explicit S" 列的做法，每胞 11–166 s。脚本写入 T64.npy 之前，先用独立的内部 LU 回代（与 `make_T_cpu.py --check` 相同）抽 16 列核对，相对误差超过 1e-8 就拒绝写入。

本地已在一个真实板胞上比较两条路线（plateB1 k = 22 的最小切胞，9,048 个保留自由度）：两条路线的 T 相对差为 3.5e-16。建议全部设计都用 schur 路线。

下面的命令按 64 核、256 GB 的机器给出；32 核、192 GB 的机器把各 `--*-threads` 减半，`--t-jobs` 设为 2，`--sens-jobs` 设为 3。

```bash
cd $W/plates_exact/src_v2
IP=$W/plates_exact/extra/iparm_tuned.json
COMMON="--work $WK --t-route schur --iparm-file $IP --threads 64 --t-jobs 2 --t-threads 16 \
        --cell-jobs 8 --cell-threads 8 --sens-jobs 4 --sens-threads 16 --min-free-gib 64 --delete-T"
tmux new -s xc      # 或 nohup；没有 numactl 时 apt-get install -y numactl，或去掉 numactl 前缀
numactl --interleave=all python3 -u exact_check_cpu.py $R/plateB1:0 $R/plateB2:0 $COMMON 2>&1 | tee -a $WK.k0.log
```

跑完后检查（结果写在各运行目录下，如 `runs/plateB1/check_exact_000.json`）：

- `exact_true_residual` ≤ 1e-10；`cpu.energy_T_vs_K_rel_max` 约 1e-12，即 q^T T q 与 u^T K u 一致；`cpu.energy_sum_rel` 约 1e-10 以下。
- `cpu.V_exact_cells` 与 NICE 的 `V`（history 中的体积）相对差约 1e-12。两者都来自同一套零阶矩；若不一致，说明几何没有对上，应停下来排查。
- `surrogate_err` 为负且量级在 1e-4 附近（算例 A 的起点为 -0.0112%）。NICE 的凝聚刚度不小于精确 Schur 补（Ŝ ⪰ S），所以 NICE 柔度偏小。
- `logs/T_*.log` 中的 `verify_rel_err` 约 1e-14。可选的额外核对（`make_T_cpu.py --check`，32 列）：先不加 `--delete-T` 跑 `--phases cellinfo,T`，再执行 `python3 exact_check_cpu.py --make-t-check $R/plateB1/body:<case1>,<case2>`，期望 `check_rel` 约 1e-14。

### 4.5 其余 8 个设计

```bash
numactl --interleave=all python3 -u exact_check_cpu.py $R/plateB1:mid,last $R/plateB2:mid,last $COMMON 2>&1 | tee -a $WK.B.log
numactl --interleave=all python3 -u exact_check_cpu.py $R/hevalH_y:0 $R/hevalH_z:0 $R/xstartH_y:last $R/xstartH_z:last \
        $COMMON 2>&1 | tee -a $WK.H.log            # 只要柔度时加 --no-sens
```

- 每条 JSON 日志行都带 `event` 字段：`JOB_START`/`JOB_END` 记录每个子进程的秒数、退出码和峰值常驻内存；`SOLVED` 记录精确柔度、PCG 次数和残差；最后一行是 check 的摘要。各阶段的秒数和主进程峰值内存在 `$WK/<run>_kKKK/phases.json`，每个子进程的完整输出在 `$WK/logs/`。
- 断点续跑：直接重跑同一命令。已有的 cellinfo、T（有效性按形状、完整性和 BOX_NODES 判断）、solve.json 和 sens/*.json 会跳过；已有 `check_exact_KKK.json` 的设计整个跳过。`--force` 重做求解、灵敏度和 check。`--phases cellinfo,T` 只预先构造 T；`--phases check` 只按已有结果重写 JSON。
- 失败处理：子进程失败（例如被 OOM 杀掉，退出码 -9）会在该阶段末单独重试一次，若仍失败则整个运行停止，并给出日志位置。求解前若 T 总量加 `--solve-margin-gib`（默认 24）超过可用内存，脚本拒绝运行。可以换更大内存的机器，或确认余量足够后加 `--no-mem-check`。PARDISO 的 Schur 选项失败时，脚本自动依次试 iparm(24)=0 和 iparm(2)=2（与 bench_cpu2 相同）。K_PP 的 PARDISO 因子若出错，可加 `--fine factory` 改用 lat_precond 自带的 SuperLU/CHOLMOD（未在板上试过），或把 `--prec` 改为 `bnn:jac:q1r`（只用对角细层，PCG 次数会增加，未测）。
- 选项 `--sens ad` 用反向模式矩导数（NICE 运行所用的路线），与中心差分在一个真实切胞上相差 3.5e-10。它在 CPU 上并不更快（本地 115 s 对 121 s），所以保留默认的 `fd`。

## 5. 预计耗时（估计值）

| 阶段 | 每个设计（24 胞） | 依据 |
| --- | --- | --- |
| cellinfo（胞建立、组装、K_PP 三元组、6 个面的一致载荷权重） | 5–10 min（8 个并行） | E1 主机前端：setup 2–10 s，组装 9–37 s/胞（bench_cpu2，16 线程）；本地 4 核下一个 618 单元切胞需 18–38 s |
| T，schur 路线（含 16 列核对与写盘） | 0.5–1.5 h（2 个并行，各 16 线程） | ST18：G2/G1/G3/G4（6.7–40 万自由度，1.75–2.59 万保留自由度）为 11/48/76/166 s，峰值 8.8–42.7 GiB；板胞保留自由度 0.9–3.6 万、单元 618–16,256；核对用的 LU 因子 3–91 s（E1 luref）；本地一个小切胞 26 s（其中 Schur 5 s） |
| T，make_T_cpu 路线（仅供比较） | 10–30 h | E1 分块路线（Cholesky 直接调用）外推 120–1,544 s/胞，make_T_cpu 用 pypardiso LU，单次查询慢 3–4.6 倍（E1 luref 与 chol 的 64 列查询之比）；本地小切胞 98 s，而 schur 路线为 30 s |
| 求解（载入 T、K_PP 因子、q1r 粗空间、PCG 到 1e-10） | 5–20 min | PCG 次数估计 250–650：算例 A 的精确核对与 NICE 冷启动之比为 178/122 = 1.46，精确孪生从起点到终点又增 47%（178 → 262）；板上 NICE 冷启动为 169/183/245/292 次。每步读 61–72 GiB 的 T，受内存带宽限制，约 0.3–1.5 s |
| 灵敏度（fd：每胞 16 次矩积分） | 0.7–1.5 h（4 个并行，各 16 线程） | 本地 4 核：618 单元切胞一次矩积分 9.8 s，dmoments 120 s。E1 主机上未切胞（0.99–1.46 万单元）的组装（含一次矩积分）约 37 s，据此估一次矩积分 15–30 s，dmoments 每胞 4–8 min，另加胞建立与 LU 因子 1–2 min。k = 0 的 B2 复用 B1 的矩导数缓存 |
| check | < 1 min | — |

合计（schur 路线，64 核、256 GB）：

- 每个 k > 0 的设计约 1.5–3.5 小时。
- k = 0 这一对（7 个不同胞）约 0.5–1.5 小时。
- 全部 10 个核对约 13–30 小时；H、XH 四个设计用 `--no-sens` 时约 10–24 小时。
- 32 核机器按约 1.5 倍估计。

以上都是估计值，实际以第一对（k = 0）的 `phases.json` 为准再外推。

## 6. 带回结果

```bash
cd $W/plates_exact
tar -czf exact_results_$(date +%Y%m%d).tgz runs/*/check_exact_*.json \
    exact_cpu/*_k*/plan.json exact_cpu/*_k*/solve.json exact_cpu/*_k*/phases.json exact_cpu/*_k*/sens \
    exact_cpu/*_k*/q exact_cpu/cellinfo/*.json exact_cpu/env_*.json exact_cpu/tcache.json exact_cpu/jobs/*.out.json \
    exact_cpu/logs exact_cpu.*.log
md5sum exact_results_*.tgz > exact_results.md5
```

- 这个包只有几 MB，不含 T64.npy、矩导数缓存和 cellinfo 的 npz；q/ 下的端口向量每个设计约 4 MB，保留下来便于复查。
- 带回后放到仓库 `docs/paper_p1/review_r1/results/X6_opt/exact_plates/`（目录名可由协调人另定），再由 `facts_6_11.py` 读入 `check_exact_*.json`，生成事实表后再写论文。
- 核对结束后删除机器上的 T 和实例。

## 7. 结果对论文的决定（核对清单）

- [ ] **有效性**：每个设计 `exact_true_residual ≤ 1e-10`；`energy_T_vs_K_rel_max` 约 1e-12；`V_exact_cells` 与 NICE 的 V 一致；T 的 `verify_rel_err` 约 1e-14。任一项不满足时，该设计不用于论文，先排查。
- [ ] **B1、B2 起点/中间/最终设计（表 ST27、正文 [TBD-EXACT]）**：精确柔度、代理误差 `surrogate_err`、`grad_rel_err`、`grad_cos`、`comp_err_rel_to_max`（中位数/p95/最大）、`sign_agreement`、`exact_pcg` 与残差。
  - 若 6 个点的 |代理误差| 与算例 A 同量级（≤ 0.03%），梯度误差与算例 A 相当（≤ 0.33%），且符号全对，正文可写"板上同样成立"，并给出数值。
  - 若有明显更大者，例如终点 > 0.1%、或符号一致率 < 1：如实报告，找出对应的胞或顶点（`cpu.s_cell_exact` 与 `cpu.s_cell_nice` 逐胞比较；多半是 τ = 0.18 下界处的薄胞或 0.083 切胞），并在 6.11 中讨论，不改结论之外的文字。
  - 代理误差的符号应为负（NICE 柔度偏小）；若为正，先查 NICE 运行 1e-6 求解的代数误差（Eq. 18）。
- [ ] **KKT**：`kkt_exact` 只含体积乘子，而终点处梯度约束也起作用，所以它高估平稳性残差，不作为 KKT 残差写入论文（PLAN_6_11_CN.md 第 6 条）。
- [ ] **均匀化比较（DRAFT_S9_EN.md 表 ST23c、DRAFT_6_11_EN.md 第 20 行）**：用 B1:22、B2:29、hevalH_y/z:0、xstartH_y/z:11 的精确柔度重算以下差值。NICE 值为：H 相对 B，y 1.07%、z -0.339%；X 相对 H，-0.998%、-0.528%；X 相对 B，+0.0602%、-0.866%。
  - 精确差值与 NICE 差值同号时，排序成立，正文可以按精确值写。
  - X 相对 B（y）只有 0.06%，与代理误差同量级，必须看精确值：若精确差值变号或小于两者代理误差之和，就写"在代理误差范围内无法区分"，不写孰优孰劣。
- [ ] **均匀化在初始设计上的预测误差**（-26.7%/-36.6%，相对 NICE）：用 B1:0、B2:0 的精确柔度改为相对精确模型的值。按算例 A 的量级，改动应在第三位有效数字以下，但必须核实后再写。
- [ ] **体积**：精确 V 与 NICE V 相同（同一套矩），不另报。
- [ ] **表述规则**：
  - 论文只写精确核对的精度结果，不写这台机器上的时间。
  - 不写租机、机器来源或运行次数。
  - T 的构造方式如需说明，写成"dense exact condensed matrices by PARDISO's Schur-complement option, verified column-wise against interior solves"，与 Note S5 的写法一致。
  - 数字按手稿格式：千分位（1,346.4），百分数取合理有效位。

## 8. 本地已验证与未验证的部分

已验证（本地 4 核、15 GB 机器；代码与服务器 src_v2 逐字节相同，md5 已核对；另从服务器只读取来 3 个本地缺少的模块和 plateB1 k = 22 的一个最小切胞）：

- `python3 -m py_compile` 通过；`bash -n pack_exact_inputs.sh` 通过。
- `--self-test` 全部通过：
  - 参数解析与 last/mid。
  - 顶点聚合，与 `r1x3_common.aggregate` 原代码逐位一致。
  - 核对量，与 `opt_design.check()` 原代码在打乱格架顺序的玩具数据上逐位一致，键名集合相同。
  - 指纹、分块对称化、T 文件有效性（含截断文件）。
  - 并行作业池（含失败重试和子进程峰值内存）。
  - 合成格架上的主机求解（HostDenseOp、共享算子对象批量作用、Factory 的 `bnn:kpp:q1r`），与稠密直接解比较：柔度相对差 5e-15；PARDISO 细层与 SuperLU 细层的 PCG 次数相同（17 次）。
- 真实切胞（plate841_160_o022：20,475 自由度，9,048 保留自由度）上两胞拼成的假运行，全流程 cellinfo → T → solve → sens → check 跑通：
  - schur 路线的 T 与 SciPy 稠密 Schur 补相差 1.35e-15；与原样 `make_T_cpu.py`（经脚本前置处理）所得 T 相差 3.5e-16；16 列核对误差 4.8e-16。
  - 格架 PCG 到 1e-10 用 84 次，真实残差 8.3e-11。
  - 精确柔度与完全独立的整胞稀疏直接解（含内部自由度，按 lat_multi 的约束和一致载荷）相差 3.7e-13；两条 T 路线所得柔度相差 8.5e-13。
  - q^T T q 与 u^T K u 相差 2.7e-13；去重（两胞共用一个 T 和一个算子对象）、`--no-dedupe`、`--delete-T` 均按预期工作。
  - 反向模式与中心差分灵敏度相差 3.5e-10。
  - 三个设计分属两个运行目录、几何相同，一次调用处理：T 只构造一次，跨运行复用；矩导数在第一个设计算出，后两个设计直接读缓存；`--delete-T` 在最后一次使用后删除 T 和缓存；三个柔度相差约 1e-14。
  - `--make-t-check`（原样 `make_T_cpu.py --check`，32 列）对 schur 路线的 T 给出 `check_rel` 2.2e-16。直接运行 `make_T_cpu.py`（不经前置处理）在无 GPU 的机器上失败，失败原因见第 3 节。
- 打包脚本：
  - 在本地假运行上打包、解包，manifest 的 md5 全部通过，解包后的运行目录可直接交给 `exact_check_cpu.py`。
  - 打包前后运行目录的文件、大小和修改时间完全不变；`--out` 放在运行目录内会被拒绝。
  - 在服务器上以 `--dry-run` 对 10 个设计实跑（只读），全包 1.463 GiB。

未验证（只能在大内存机上验证）：

- 24 胞板格架的整体求解：内存、PCG 次数，以及 PARDISO 对 K_PP 的因子时间。
- 大胞（1.2–1.6 万单元、3–3.6 万保留自由度）Schur 路线的时间和峰值内存。
- 大胞 dmoments 的时间。
- 多进程并行时的内存守护阈值是否合适。
- 与 120 GiB 主机不同的 CPU 或 MKL 环境下，PARDISO 调用是否都能直接运行。
- 第 5 节的时间都是按上述记录外推的估计值。
