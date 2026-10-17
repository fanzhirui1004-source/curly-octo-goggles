#!/bin/bash
# t5.sh <tag> <lattice> [ENV=VAL ...]: the Table 5 learned route (FINAL3 command and environment, as
# docs/paper_p1/review_r1/remeasure_20261017/remeasure.sh) on src_v3 (or SRC=...), whole-device memory sampled by
# nvidia-smi every 100 ms. Outputs: V3W/runs/<tag>/{learned_<LAT>.json,.log, sens_<LAT>.npz, gpu_<LAT>.csv}; status: V3W/status.txt.
# EXTRA_ARGS (env) is appended to the lat_scale command line.
TAG=$1; LAT=$2; shift 2
W=/root/autodl-tmp/OPL/V3W; O=$W/runs/$TAG; mkdir -p $O
P=/root/autodl-tmp/OPL
source $P/S1/env_gpu.sh
export OPL_GP_CACHE=0 OPL_CONV_FP32=1 OPL_PACKETS_EXTRA=$P/S4/packets:$P/S3/packets OPL_COARSE_PIVOT_FLOOR=1e-7
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU OPL_T5_SKIP OPL_STREAM_PACK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPL_BENCH_TMP=$W/bench_tmp
for kv in "$@"; do export "$kv"; done
SRC=${SRC:-$P/src_v3}
cd $SRC
ARGS="--fastidx --coarse-tpl --tet-triton --sparse-coarse --resident-gb 4 --tol 1e-6 --levels 1e-2,3e-3,1e-3,1e-4 --iters 1 --sens ad --sens-obj sum --ad-batch 2048"
nvidia-smi --query-gpu=timestamp,memory.used,clocks.sm,clocks.mem,power.draw,utilization.gpu,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader,nounits -lms 100 > $O/gpu_$LAT.csv 2>&1 & SMP=$!
echo "$(date +%F_%T) START $TAG $LAT src=$SRC env=[$*] extra=[$EXTRA_ARGS]" >> $W/status.txt
timeout ${TMO:-3600} $PY -u lat_scale.py $O/learned_$LAT.json $P/S4/$LAT.json $ARGS --save-sens $O/sens_$LAT.npz $EXTRA_ARGS > $O/learned_$LAT.log 2>&1; r=$?
sleep 2; kill $SMP
echo "$(date +%F_%T) END $TAG $LAT rc=$r gpu_max_MiB=$(cut -d, -f2 $O/gpu_$LAT.csv | sort -n | tail -1)" >> $W/status.txt
