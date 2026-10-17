#!/bin/bash
# pipe_test.sh <tag> [ENV=VAL ...]: one design iteration of opt_design.py itself (bodies + analysis + MMA, --stop-after 1)
# on the 24-cell plate with fresh bodies and the ST20 arguments; with OPL_V3_PIPE_BODIES=1 OPL_V3_PIPE_FAULT=k it checks
# the fallback after a body failure in the overlapped front end. Root V3W/pipe/<tag>.
TAG=$1; shift
W=/root/autodl-tmp/OPL/V3W; P=/root/autodl-tmp/OPL; SC=$P/S1/V2/R1/SCALE
R=$W/pipe/$TAG; mkdir -p $R
source $P/S1/env_gpu.sh
D=/root/autodl-tmp/CUTFEM_DEPENDENCIES_20260924/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$D PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/pylib_cpu
export OPL_GP_CACHE=0 OPL_CONV_FP32=1 OPL_STREAM_PACK=1
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU OPL_T5_SKIP; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPL_PACKETS_EXTRA=$R/packets:$P/S4/packets:$P/S3/packets:$SC/plate_packets:$W/plate_packets
export OPL_BENCH_TMP=$W/bench_tmp
for kv in "$@"; do export "$kv"; done
export SRC=${SRC:-$P/src_v3}
cd $SRC
echo "$(date +%F_%T) START pipe $TAG env=[$*]" >> $W/status.txt
timeout 3600 $PY -u $SRC/opt_design.py $R $SC/layouts/plateS24.json --model A3=$P/S1/V2/A3_2grid/best.pt \
  --clamp x,min --load x,max --load-dir y --vfrac 0.8 --maxit 60 --stop-after 1 --fast --table5 --workers 14 --sens ad --ad-batch 2048 \
  --resident-gb 4 --warm --body-retry 6 > $R/run.log 2>&1; r=$?
echo "$(date +%F_%T) END pipe $TAG rc=$r" >> $W/status.txt
