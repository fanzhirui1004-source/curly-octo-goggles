# R22 吞吐与审计整改（Claude，2026-09-07）

分支 `claude/r22-throughput`，基于 Codex 的 `f3f7565`。数值阶段、封存回执与失败保留规则不变；方法上的三处登记变化各有独立策略文件并写入记录哈希。

## 已落地

| 项 | 改动 | 验证 |
| --- | --- | --- |
| 1 配对门 | `PAIR_GATE_POLICY_R22.json`：严格门 1e-8/1e-9 继续报告，生产门 1e-6 决定通过；`evaluate_pair_gates` 单独可测 | tests/r22/test_r22_pair_gate.py；E2E 样本 |
| 2 测试口径 | `stage_cutfem_runtime.full_tests`：分别记录“收集数”和“实际执行通过数” | CLAUDE_R22_FULL_TESTS_01 |
| 3 中间数组归档 | `stage_cutfem_runtime.retire_sample`：打包封存后对 body/ghost/global_body 大数组逐文件 gzip、解压校验、回执、退役；流水线自动调用 | E2E PACKET_RETIRE |
| 4 旧大 JSON | 同一工具应用于 R12/R15/R20 的 BODY/QUADRATURE/PHYSICAL_BODY 大 JSON | CLAUDE_R22_RETIRE_JSON_02 含恢复探针 |
| 6 双车道 | `stage_cutfem_preproduction.pipeline`：几何车道与算子/打包车道并行，按 cgroup 用量加车道预算准入；`PRODUCTION_POLICY_R22.json` | 干跑 + 单样本 E2E |
| 7 审计抽样 | 流水线 `--pair-audit-every N`（默认 20）；`trial packet` 省略 `--response-run` 即生产模式 | E2E（audit=1） |
| 8 精确迹编译 | `stage_cutfem_full_interface/exact_numbers.py`：gmpy2.mpq 后端，运行时目录 `runtime/r22_gmpy2_v1` | P06 逐位一致：坐标哈希、P、逆矩阵；182 s → 58 s |
| 10 缓存 | `cosine_axis` 与应变 `coefficients` 按进程 lru_cache | tests/r22 逐位一致 |
| 11 候选阶梯 | `INTEGRATION_POLICY_R22.json`：从 8 阶起搜 | 90 单元比对：67 逐位一致，23 升为 8 阶，能量差 ≤4e-11 |
| 13 调度 | `ExecutionPool.map_unordered` + 按 owner 重组 | tests/r22；E2E |
| 17 包络阶梯 | 同一策略文件 `enclosure_confirmation: disabled`；记录 `DISABLED_BY_R22_POLICY` | 90 单元比对 |
| 18 带符号误差 | `qualification.compare` 增加带符号与仅物理项误差；报告加抵消说明 | 语法与单元测试 |
| 20 手写数字 | Codex 在 T01 已改为读取证据；本轮只补带符号说明 | — |
| 21 主机锁定 | `require_target()` 按 `EXECUTION_CONFIG.json` 目标校验，`CUTFEM_EXECUTION_TARGET` 可选目标；`require_westb` 保留为别名 | 单元测试 |
| 22 测试重名 | 10 个文件改名，5 处模块导入同步 | `pytest tests --collect-only` 无重名错误 |

## 未做或留待

- 5 贴体参考、19 新边界场复验：需要 Tet10 参考序列的小时级 64 核计算，与正在运行的 T04 队列冲突；命令见 REMAINING_R22.md。
- 9 导出后收尾：实测为打包校验 12 s、刚体检查与探针 12 s、封存 7 s，合计约 31 s，不是此前估计的 90 s；未改。
- 12 QR 归约：每切割单元占比约 12%，改为 Gram 累积会损失秩判定精度；未改。
- 14 内部单元快速路径、15 认证循环 C++ 化：收益小或工程量大；未改。
- 16 生产运行中 2.5 倍减速：`stage_cutfem_runtime.scaling_probe` 已就绪，需在队列空闲时运行。

## 已知的历史测试失败（与本轮无关）

`tests/implicit_macro_cut_whole_boundary/test_filtered_rank.py::test_scaled_1e_minus_12_sparse_full_rank_is_certified_without_scale_floor` 在 Codex 原树、原运行时组合下同样失败（PROPACK 对 1e-12 尺度矩阵不收敛，返回 ADAPTIVE_REQUIRED）。该文件随 T16 的完整 Git 部署首次进入服务器，此前从未在服务器执行过；`stage_zx_implicit.filtered_rank` 不在 CutFEM 生产链上。与 gmpy2 运行时无关。建议由 Codex 调整用例尺度或增加直接 SVD 回退，不放宽 CERTIFIED 门槛。
