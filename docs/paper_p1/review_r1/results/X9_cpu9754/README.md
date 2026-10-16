# CPU cost runs on the AMD EPYC 9754 (2026-10-17)

All CPU-timed content of the paper, re-run on one machine so that it rests on a single CPU: Table 5 routes (a) and (b),
ST12b (16 threads, 32 threads, single thread), ST12c, ST12e (GAMG, BoomerAMG, BDDC local factorisations) and the
conventional columns of ST13. The runs use the scripts and commands of the earlier EPYC 9654 runs unchanged
(`lat_direct_cpu2.py`, `lat_cond_cpu.py`, `bench_cpu2.py`, `r1_host_summary.py`, `r1_cond_summary.py`, and
`lat_dump.py`, `build_K.py`, `petsc_solve.py`, `bddc_lb.py` of the iterative-solver comparison); no sensitivities are
computed on the CPU (see `sens_deferred/`).

## Machine and environment

- CPU: AMD EPYC 9754 128-Core Processor, one socket (256 threads, L3 256 MiB, one NUMA node); container quota 36 CPUs,
  memory limit 120 GiB (cgroup v2). Ubuntu 22.04.
- Software, the versions of the EPYC 9654 runs: Python 3.12.3, NumPy 2.3.2, SciPy 1.18.1, PyTorch 2.8.0+cu128
  (`cpuenv`), MKL 2026.1.0 (build 20260612), intel-openmp 2026.1.2, tbb 2023.1.0, tcmlib 1.5.0, umf 1.1.0 (`mklenv`),
  pypardiso 0.4.7 (`pylib_cpu`); PETSc 3.26 environment `ITERCMP/penv` as before. `scripts/build_env.sh` builds
  `cpuenv` and `mklenv`; `scripts/env_cpu9754.sh` sets the environment.
- PARDISO settings re-tuned on this CPU (`bench_cpu2.py --mode tune`, as before): iparm(2) = 3, iparm(24) = 0,
  iparm(25) = 0, the same choice as on the EPYC 9654.

## How the jobs run

`scripts/queue.sh` runs the jobs of `scripts/jobs/` one at a time in name order (never two timed runs together),
waits while a file transfer is in progress, and records start, end, exit code, seconds, load average and CPU
throttling. Order: environment record, tuning, ST13 (three repetitions), route (a) 16 threads, route (b) 16 threads,
the exports and runs of ST12e, route (a) 32 threads, route (a) single thread, summaries. Outputs on the machine:
`/root/autodl-tmp/OPL/S1/V2/R9754/{host, host1, hostcond, itercmp}`.

Inputs (code, lattice geometry, the four ST13 cells and their packets, environments) were copied from the GPU machine.

## Status: stopped (2026-10-17 03:55)

Stopped by the author's decision once the CPU sensitivities were deferred: the paper keeps the CPU-timed content of
the EPYC 9654 runs, and this round adds only the two GPU measurements (`X6_final/remeasure_20261017`). Completed before
the stop: environment record and self-tests, PARDISO tuning (same settings as on the EPYC 9654), the hlat222 export of
ST12e, and two of the three ST13 repetitions (`partial/`). For reference, ST13 conventional columns on the EPYC 9754
(G1-G4, two repetitions): cell preparation 29.4 / 11.1 / 48.7 / 48.8 s, analysis + factorisation 10.9 / 2.1 / 25.5 /
43.5 s, PARDISO factor 2.89 / 0.60 / 6.18 / 9.42 GiB (EPYC 9654: 27.3 / 10.9 / 45.0 / 47.0 s, 10.5 / 1.9 / 27.3 /
48.9 s, the same factors). The first attempts of the tuning and ST13 jobs (`partial/queue/*.failed1.log`) failed on four
missing packet directories, which were then copied.
