#!/bin/bash
# Section 6.11: exact checks of case A (NICE run optA) at iterations 0, 12, 23 (opt_design --check: exact compliance,
# surrogate error, gradient error / cosine / sign agreement, KKT residual with the volume multiplier).
# Tag OPTCK; marker OPT_CHECKA_DONE_20260930 (EXIT trap).
ST=/root/autodl-tmp/OPL/S1/V2/chain_p1.status; R=/root/autodl-tmp/OPL/S1/V2/R1/OPT; TAG=OPTCK; RC=0
st() { echo "$(date +%T) $TAG $*" >> $ST; }
fin() { st "END rc=$RC"; echo "$(date +%T) OPT_CHECKA_DONE_20260930" >> $ST; }
trap fin EXIT
run() { local n=$1 to=$2; shift 2; st "START $n"; local t=$(date +%s)
  timeout $to "$@" >> $R/$n.log 2>&1; local r=$?; [ $r != 0 ] && RC=$r; st "END $n rc=$r $(( $(date +%s) - t ))s"; }
pgrep -f "opt_design.py" >/dev/null && { st "GPU busy - not started"; exit 1; }
source /root/autodl-tmp/OPL/S1/env_gpu.sh
D=$DEP/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$D PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/pylib_cpu PYPARDISO_MKL_RT=$D/libmkl_rt.so.3 MKL_NUM_THREADS=16
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets:$R/plate_packets OPL_GP_CACHE=0 OPL_CONV_FP32=1
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/OPL/src_v2
run checkA 7200 $PY -u opt_design.py $R/optA /root/autodl-tmp/OPL/S4/hlat222.json --model A3=/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt \
  --clamp y,min --load y,max --load-dir y --vfrac 0.8 --workers 8 --check 0,12,23 --t-cpu-fallback
