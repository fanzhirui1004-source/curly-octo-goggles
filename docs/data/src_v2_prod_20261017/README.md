# src_v2 as deployed on the GPU machine (snapshot of 2026-10-17)

Copy of `/root/autodl-tmp/OPL/src_v2` on the RTX 5090 machine at 02:58 CST on 17 October 2026 (Python and shell files;
`__pycache__`, `.npz` and `.pt` excluded). This is the code behind the FINAL3 timings of Table 5 (learned route,
`lat_scale.py`), the scale runs of ST20 (`opt_design.py`), the CPU routes (`lat_direct_cpu2.py`, `lat_cond_cpu.py`,
`bench_cpu2.py`) and the 2026-10-17 re-measurement (`docs/paper_p1/review_r1/results/X6_final/remeasure_20261017`).
`docs/data/newmachine_20260924/src_v2_wip` is an older, partial copy.

Run environment on the GPU machine: `/root/autodl-tmp/OPL/S1/env_gpu.sh` (gpuenv: PyTorch 2.8.0+cu128, CUDA 12.8,
NVIDIA cuDSS 0.8 through nvmath). Paths in the code refer to `/root/autodl-tmp/...` on that machine.
