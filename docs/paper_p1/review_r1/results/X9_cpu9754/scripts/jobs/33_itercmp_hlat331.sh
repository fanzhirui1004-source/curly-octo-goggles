# ST12e on the EPYC 9754 (run_amd.sh of the 9654 runs, same commands): GAMG-PCG and BoomerAMG-PCG with 16 MPI ranks,
# six loads, unpreconditioned relative residual 1e-9; BDDC local-factorisation lower bound (PARDISO, 16 threads).
. /root/autodl-tmp/cpu9754/jobenv.sh
L=hlat331; D=/root/autodl-tmp/ITERCMP; P=$D/penv/bin; S=$D/sys_$L; R=$IT/res_$L
unset LD_LIBRARY_PATH PYTHONPATH PYPARDISO_MKL_RT
mkdir -p $R; cd $D; rc=0
{ date; lscpu | grep -E 'Model name|^CPU\(s\)'; cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max; cat /proc/loadavg; } > $R/env.txt
[ -f $S/K_data.npy ] || { $P/python -u build_K.py $S > $R/build_K.log 2>&1 || rc=1; }
export OMP_NUM_THREADS=1
cat /proc/loadavg >> $R/env.txt
$P/mpiexec -n 16 $P/python -u petsc_solve.py $S $R/gamg.json --method gamg --loads 6 --maxit 3000 \
  -- -ksp_converged_reason > $R/gamg.log 2>&1 || rc=1
cat /proc/loadavg >> $R/env.txt
$P/mpiexec -n 16 $P/python -u petsc_solve.py $S $R/hypre.json --method hypre --loads 6 --maxit 3000 \
  -- -ksp_converged_reason -pc_hypre_boomeramg_strong_threshold 0.5 -pc_hypre_boomeramg_coarsen_type HMIS \
  -pc_hypre_boomeramg_interp_type ext+i -pc_hypre_boomeramg_nodal_coarsen 6 -pc_hypre_boomeramg_vec_interp_variant 3 \
  > $R/hypre.log 2>&1 || rc=1
cat /proc/loadavg >> $R/env.txt
PYPARDISO_MKL_RT=$D/penv/lib/libmkl_rt.so.3 MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 \
  $P/python -u bddc_lb.py $S $R/bddc_lb.json > $R/bddc_lb.log 2>&1 || rc=1
exit $rc
