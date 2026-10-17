#!/bin/bash
# lsrun.sh <tag> <layout> [lat_scale args]: lat_scale.py (plate cells of design iteration 0, bodies V3W/plate/body_S24)
# with the Table 5 route options, the 24-cell plate packets, resident budget 0, PCG tolerance 1e-6, reverse-mode
# sensitivities of the summed compliance; nvidia-smi sampling. Output V3W/ls/<tag>/out.json, run.log.
TAG=$1; LAY=$2; shift 2
W=/root/autodl-tmp/OPL/V3W; P=/root/autodl-tmp/OPL; SC=$P/S1/V2/R1/SCALE
R=$W/ls/$TAG; mkdir -p $R
source $P/S1/env_gpu.sh
export OPL_GP_CACHE=0 OPL_CONV_FP32=1 OPL_STREAM_PACK=1 OPL_COARSE_PIVOT_FLOOR=1e-7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPL_PACKETS_EXTRA=$W/plate/v3U_S24/packets:$P/S4/packets:$P/S3/packets:$SC/plate_packets:$W/plate_packets
export OPL_BENCH_TMP=$W/bench_tmp
export SRC=${SRC:-$P/src_v3}
cd $SRC
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 200 > $R/gpu.csv 2>&1 & SMP=$!
echo "$(date +%F_%T) START ls $TAG [$*]" >> $W/status.txt
timeout ${TMO:-7200} $PY -u $SRC/lat_scale.py $R/out.json $LAY --body $W/plate/body_S24 --model $P/S1/V2/A3_2grid/best.pt \
  --prec bnn:kpp:q1r --tol 1e-6 --sens ad --ad-batch ${ADB:-1024} --sens-obj sum --fastidx --coarse-tpl --tet-triton --sparse-coarse \
  --resident-gb 0 "$@" > $R/run.log 2>&1; r=$?
kill $SMP
echo "$(date +%F_%T) END ls $TAG rc=$r gpu_max_MiB=$(cut -d, -f2 $R/gpu.csv | sort -n | tail -1)" >> $W/status.txt
