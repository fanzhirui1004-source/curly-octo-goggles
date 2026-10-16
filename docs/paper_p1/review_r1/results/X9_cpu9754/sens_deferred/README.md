# Design sensitivities on the CPU routes (validated, deferred)

Scripts that add the sensitivity of the learned route (summed compliance of the three face loads, per cell the 8
corner derivatives, reverse-mode differentiation of the element moments, `moments_ad.cell_sens`) to Table 5 routes
(a) and (b). Not used for the paper: the author decided on 2026-10-17 to keep the exact routes without sensitivities
(as in the paper) and to revisit a CPU implementation together with the large-scale and speed / memory work.

- `cpu_sens.py`: sensitivity of one cell from its exact full field (light cell record taken before the route
  releases the cell). On the host: no gradient checkpointing, row batches of 256.
- `lat_direct_sens.py`: route (a) of `lat_direct_cpu2.py` plus the sensitivity stage (fields from the global solution).
- `lat_cond_sens.py`: route (b) of `lat_cond_cpu.py` plus interior recovery (K_II refactorised per cell by PARDISO)
  and the sensitivity stage.

## Validation (GPU machine's host CPU, 16 threads; not timing runs)

- One cell (hlat222_110, 102,786 DOFs): reverse mode vs central differences of the moments (`Cell.dmoments` +
  `Cell.sens`): relative difference 8.7e-11.
- hlat221a, route (a): compliances equal the earlier exact solution; NICE's compliance errors 0.015 / 0.012 /
  0.010 % (as Table 5); maximum relative residual 8.7e-11. NICE's per-cell sensitivities (FINAL3 command,
  `--save-sens`) differ from the exact ones by 2.6e-4 to 1.6e-3 per cell, 5.0e-4 overall.
- hlat221a, route (b): energy check sum_c u_c^T K_c u_c vs sum_k f_k^T U_k 7.8e-12; compliances vs (a) within 7e-12;
  interior residuals 4e-15 to 6e-15; sensitivities (a) vs (b) 5e-12 to 1.3e-11 per cell.

## Why deferred: cost of this implementation on the CPU

The moment code is written for the GPU (large batched tensors, autograd). On the host CPU it is dominated by overhead:

- hlat221a: route (a) 550 s, of which sensitivities 297 s (cell setup and assembly 151 s); route (b) 652 s, of which
  sensitivities 308 s and interior recovery 46 s (route (b) without them 298 s). NICE's whole sensitivity stage for
  this lattice takes about 1.8 s on the RTX 5090.
- One cell (hlat222_110): 45.5 / 45.5 / 42.7 s with checkpointing at batches 256 / 1024 / 4096, 35.1 / 36.6 / 38.2 s
  without; 33.3 s with 16 threads, 55.0 s with 4, 176.5 s with 1 (5.3x from 1 to 16 threads).

A CPU implementation (compiled per-element kernel for the moments and their derivatives, parallel over elements) would
be the fair choice if the CPU routes are to include sensitivities; the same applies to the moment integration in the
CPU routes' cell assembly.

`val/`: the validation records (`a_hlat221a.json`, `b_hlat221a.json`, `nice_sens_hlat221a.npz`).
