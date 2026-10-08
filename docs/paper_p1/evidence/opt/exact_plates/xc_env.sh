# source /root/autodl-tmp/xc/xc_env.sh   (32 vCPU, 240 GiB cgroup; big-memory CPU machine for the plate exact checks)
export W=/root/autodl-tmp/xc
export M=/root/miniconda3/lib
export LD_LIBRARY_PATH=$M PYPARDISO_MKL_RT=$M/libmkl_rt.so.3
export OPL_DEV=cpu CUDA_VISIBLE_DEVICES= OPL_GP_CACHE=0
export PYTHONPATH=
export R=$W/plates_exact/runs WK=$W/plates_exact/exact_cpu IP=$W/plates_exact/extra/iparm_tuned.json
export COMMON="--work $WK --t-route schur --iparm-file $IP --threads 32 --t-jobs 2 --t-threads 16 \
  --cell-jobs 4 --cell-threads 8 --sens-jobs 2 --sens-threads 16 --min-free-gib 48 --delete-T"
