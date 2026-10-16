#!/bin/bash
# shat.sh <tag> <body_dir>:<case> [args for v3_shat.py] : single-cell S_hat benchmark (env as the Table 5 route).
TAG=$1; SPEC=$2; shift 2
W=/root/autodl-tmp/OPL/V3W; O=$W/shat; mkdir -p $O
P=/root/autodl-tmp/OPL
source $P/S1/env_gpu.sh
export OPL_GP_CACHE=0 OPL_CONV_FP32=1 OPL_COARSE_PIVOT_FLOOR=1e-7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPL_PACKETS_EXTRA=$W/plate/base_S24/packets:$P/S4/packets:$P/S3/packets:$P/S1/V2/R1/SCALE/plate_packets
export OPL_BENCH_TMP=$W/bench_tmp
export SRC=${SRC:-$P/src_v3}
for kv in $SHAT_ENV; do export "$kv"; done
cd $SRC
echo "$(date +%F_%T) START shat $TAG $SPEC src=$SRC env=[$SHAT_ENV] [$*]" >> $W/status.txt
timeout 1800 $PY -u $W/scripts/v3_shat.py $O/$TAG.json $SPEC "$@" > $O/$TAG.log 2>&1; r=$?
echo "$(date +%F_%T) END shat $TAG rc=$r" >> $W/status.txt
