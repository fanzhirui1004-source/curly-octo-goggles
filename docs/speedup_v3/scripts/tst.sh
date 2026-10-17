#!/bin/bash
# tst.sh <tag> <script.py> [args] : run a test script with the Table 5 route environment (as shat.sh); log V3W/tests/<tag>.log
TAG=$1; SCR=$2; shift 2
W=/root/autodl-tmp/OPL/V3W; O=$W/tests; mkdir -p $O
P=/root/autodl-tmp/OPL
source $P/S1/env_gpu.sh
export OPL_GP_CACHE=0 OPL_CONV_FP32=1 OPL_COARSE_PIVOT_FLOOR=1e-7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPL_PACKETS_EXTRA=$W/plate/base_S24/packets:$P/S4/packets:$P/S3/packets:$P/S1/V2/R1/SCALE/plate_packets
export OPL_BENCH_TMP=$W/bench_tmp
export SRC=${SRC:-$P/src_v3}
for kv in $TST_ENV; do export "$kv"; done
cd $SRC
echo "$(date +%F_%T) START tst $TAG $SCR src=$SRC env=[$TST_ENV] [$*]" >> $W/status.txt
nvidia-smi --query-gpu=timestamp,memory.used,clocks.sm,clocks.mem,power.draw,utilization.gpu,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader,nounits -lms 200 > $O/$TAG.gpu.csv 2>&1 & SMP=$!
timeout ${TST_TIMEOUT:-1800} $PY -u $W/scripts/$SCR "$@" > $O/$TAG.log 2>&1; r=$?
kill $SMP 2>/dev/null
echo "$(date +%F_%T) END tst $TAG rc=$r" >> $W/status.txt
