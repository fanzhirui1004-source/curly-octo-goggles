#!/bin/bash
# run_amd.sh <lattice> : on the AMD EPYC 9654 host (the machine type of Table 5), one after the other:
#   rebuild K from the cells (not timed), GAMG-PCG and BoomerAMG-PCG (16 MPI ranks, 6 loads, unpreconditioned relative
#   residual 1e-9), and the BDDC lower bound (PARDISO, 16 threads). Records machine state before each run.
L=$1; D=/root/autodl-tmp/ITERCMP; P=$D/penv/bin; S=$D/sys_$L; R=$D/res_$L
mkdir -p $R; cd $D
st() { echo "$(date +%T) $L $*" >> $D/run_amd.status; }
{ date; lscpu | grep -E 'Model name|^CPU\(s\)'; cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max; cat /proc/loadavg; } > $R/env.txt
[ -f $S/K_data.npy ] || { $P/python -u build_K.py $S > $R/build_K.log 2>&1; st "build_K rc=$?"; }
while pgrep -x rsync > /dev/null; do sleep 30; done          # no incoming transfer during timed runs
st "quiet: $(cat /proc/loadavg)"
export OMP_NUM_THREADS=1
cat /proc/loadavg >> $R/env.txt
$P/mpiexec -n 16 $P/python -u petsc_solve.py $S $R/gamg.json --method gamg --loads 6 --maxit 3000 \
  -- -ksp_converged_reason > $R/gamg.log 2>&1; st "gamg rc=$?"
cat /proc/loadavg >> $R/env.txt
$P/mpiexec -n 16 $P/python -u petsc_solve.py $S $R/hypre.json --method hypre --loads 6 --maxit 3000 \
  -- -ksp_converged_reason -pc_hypre_boomeramg_strong_threshold 0.5 -pc_hypre_boomeramg_coarsen_type HMIS \
  -pc_hypre_boomeramg_interp_type ext+i -pc_hypre_boomeramg_nodal_coarsen 6 -pc_hypre_boomeramg_vec_interp_variant 3 \
  > $R/hypre.log 2>&1; st "hypre rc=$?"
cat /proc/loadavg >> $R/env.txt
PYPARDISO_MKL_RT=$D/penv/lib/libmkl_rt.so.3 MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 \
  $P/python -u bddc_lb.py $S $R/bddc_lb.json > $R/bddc_lb.log 2>&1; st "bddc_lb rc=$?"
st DONE
