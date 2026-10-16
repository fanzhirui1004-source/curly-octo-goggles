#!/bin/bash
# plate.sh <tag> <N> [ENV=VAL ...]: v3_plate.py on SCALE/layouts/plateS<N>.json with the arguments and environment of the
# ST20 scale runs (chain_final_route2.sh step 5: OPL_STREAM_PACK=1, --resident-gb 4, --workers 14, ...), design iteration 0
# only; nvidia-smi sampling every 100 ms. Root V3W/plate/<tag>_S<N> (bodies reused from V3W/plate/body_S<N> when present).
TAG=$1; N=$2; shift 2
W=/root/autodl-tmp/OPL/V3W; P=/root/autodl-tmp/OPL; SC=$P/S1/V2/R1/SCALE
R=$W/plate/${TAG}_S$N; mkdir -p $R
BODY=$W/plate/body_S$N; mkdir -p $BODY
[ -e $R/body ] || ln -s $BODY $R/body
source $P/S1/env_gpu.sh
D=/root/autodl-tmp/CUTFEM_DEPENDENCIES_20260924/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$D PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/pylib_cpu
export OPL_GP_CACHE=0 OPL_CONV_FP32=1 OPL_STREAM_PACK=1
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU OPL_T5_SKIP; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPL_PACKETS_EXTRA=$R/packets:$P/S4/packets:$P/S3/packets:$SC/plate_packets
export OPL_BENCH_TMP=$W/bench_tmp
for kv in "$@"; do export "$kv"; done
export SRC=${SRC:-$P/src_v3}
cd $SRC
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 100 > $R/gpu.csv 2>&1 & SMP=$!
echo "$(date +%F_%T) START plate $TAG S$N src=$SRC env=[$*]" >> $W/status.txt
timeout ${TMO:-14400} $PY -u $W/scripts/v3_plate.py $R/census.json $R $SC/layouts/plateS$N.json --model A3=$P/S1/V2/A3_2grid/best.pt \
  --clamp x,min --load x,max --load-dir y --vfrac 0.8 --maxit 60 --stop-after 1 --fast --table5 --workers 14 --sens ad --ad-batch 2048 \
  --resident-gb ${RGB:-4} --warm --body-retry 6 $EXTRA_ARGS > $R/run.log 2>&1 &
TPID=$!; sleep 2; PID=$(pgrep -P $TPID | head -1)
while kill -0 $TPID 2>/dev/null; do
  echo -e "$(date +%s)\t$(awk '/VmRSS/{print $2/1048576}' /proc/$PID/status 2>/dev/null)" >> $R/rss.tsv; sleep 1
done
wait $TPID; r=$?
sleep 2; kill $SMP
echo "$(date +%F_%T) END plate $TAG S$N rc=$r gpu_max_MiB=$(cut -d, -f2 $R/gpu.csv | sort -n | tail -1) rss_max_GiB=$(cut -f2 $R/rss.tsv | sort -g | tail -1)" >> $W/status.txt
