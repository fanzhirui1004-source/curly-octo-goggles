#!/bin/bash
# Section 6.11 step 1 pilot: opt_design.py on hlat222 (2x2x2, clamp y=min, consistent y load on y=max, load-face vertices
# fixed), 5 MMA iterations, NICE with fp64 correction (--park --resident 4). Waits for the E13 rerun marker (GPU free).
# Tag OPTP2; marker OPT_PILOT2_DONE_20260929 (EXIT trap). Rerun with --resident 2 after an OOM with 4 resident fp64-correction cells.
ST=/root/autodl-tmp/OPL/S1/V2/chain_p1.status; R=/root/autodl-tmp/OPL/S1/V2/R1/OPT; TAG=OPTP2; RC=0
st() { echo "$(date +%T) $TAG $*" >> $ST; }
fin() { st "END rc=$RC"; echo "$(date +%T) OPT_PILOT2_DONE_20260929" >> $ST; }
trap fin EXIT
st "QUEUED pid=$$ (waits for R1_E13B_DONE_20260929)"
true  # rerun 2026-09-29 16:25: E13 finished
st "START"
source /root/autodl-tmp/OPL/S1/env_gpu.sh
D=$DEP/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$D PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/pylib_cpu PYPARDISO_MKL_RT=$D/libmkl_rt.so.3 MKL_NUM_THREADS=16
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets OPL_GP_CACHE=0 OPL_CONV_FP32=1
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/OPL/src_v2
mkdir -p $R
t=$(date +%s)
timeout 10800 $PY -u opt_design.py $R/pilot222 /root/autodl-tmp/OPL/S4/hlat222.json --model A3=/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt \
  --clamp y,min --load y,max --load-dir y --vfrac 0.8 --stop-after 5 --park --resident 2 --workers 8 > $R/pilot222_r2.log 2>&1
RC=$?; st "pilot222 rc=$RC $(( $(date +%s)-t ))s"
