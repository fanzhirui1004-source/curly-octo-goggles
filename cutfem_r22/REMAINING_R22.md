# 留待队列空闲后执行的项

以下命令在 `source/git_claude_r22` 下运行，均通过 runner 封存。

## 16 生产运行减速的对比测试（约 5 分钟）

```bash
cutfem_root=/root/autodl-tmp/CUTFEM_TRUE_IMPLICIT_GHOST_20260905_V1
cutfem_python=$cutfem_root/.venv/bin/python
cd $cutfem_root/source/git_claude_r22
"$cutfem_python" -m stage_cutfem_runtime.runner --run-id R22_SCALING_PROBE_01 --stage tests --threads 1 \
  --pss-gib 40 --disk-headroom-gib 8 --runtime r13_pardiso_v1 --runtime r14_fitted_v2 --runtime r17_algoim_v2 --runtime r22_gmpy2_v1 -- \
  "$cutfem_python" -m stage_cutfem_runtime.scaling_probe --geometry-run $cutfem_root/artifacts/R21_T04_P06_GEOMETRY --workers 1 16 32 64
```

## 5 四个参考候选的贴体参考（小时级）

使用 Codex 的 `stage_cutfem_preproduction.reference` 流程与 `R21_T04_PANEL/PANEL.json` 的 `reference_case_ids`，每个候选按 PROTOCOL 的表面预设（production、reference、fine）和体网格序列各跑一次，然后用 `stage_cutfem_preproduction.qualification` 汇总。此项未在本轮执行。

## 19 用未参与选参的边界场复验 gamma=0.0003

在 `stage_cutfem_assembly.experiment.nonaffine` 中登记第四个解析场（例如频率 4），对 n32 完整单胞重放 `R20_38_GP_G0003_N32_D0` 的输入并生成对应的贴体参考序列，再进入 qualification 比较。此项未在本轮执行。

## 生产运行示例（两车道，每 20 个样本审计一次）

```bash
"$cutfem_python" -m stage_cutfem_preproduction.pipeline --prefix PROD01 \
  --panel-run $cutfem_root/artifacts/<registered panel> --cases <case ids...>
```
