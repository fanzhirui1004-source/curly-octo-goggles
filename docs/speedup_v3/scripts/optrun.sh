#!/bin/bash
# optrun.sh <tag> <N> <K> [opt_design args]: opt_design.py on SCALE/layouts/plateS<N>.json for K design iterations with the
# arguments of the ST20 scale runs (as plate.sh) and resident budget 0; root V3W/opt/<tag>; the bodies of design iteration
# 0 linked from V3W/plate/body_S<N>, those of later iterations generated into the root's own body directory.
TAG=$1; N=$2; K=$3; shift 3
W=/root/autodl-tmp/OPL/V3W; P=/root/autodl-tmp/OPL; SC=$P/S1/V2/R1/SCALE
R=$W/opt/$TAG; mkdir -p $R/body
for d in $W/plate/body_S$N/*_o000 $W/plate/body_S$N/GP_TEMPLATES_n32.npz; do [ -e $R/body/$(basename $d) ] || ln -s $d $R/body/; done
source $P/S1/env_gpu.sh
D=/root/autodl-tmp/CUTFEM_DEPENDENCIES_20260924/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$D PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/pylib_cpu
export OPL_GP_CACHE=0 OPL_CONV_FP32=1 OPL_STREAM_PACK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPL_PACKETS_EXTRA=$R/packets:$P/S4/packets:$P/S3/packets:$SC/plate_packets:$W/plate_packets
export OPL_BENCH_TMP=$W/bench_tmp
export SRC=${SRC:-$P/src_v3}
cd $SRC
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 200 > $R/gpu.csv 2>&1 & SMP=$!
echo "$(date +%F_%T) START opt $TAG [$*]" >> $W/status.txt
timeout ${TMO:-14400} $PY -u $SRC/opt_design.py $R $SC/layouts/plateS$N.json --model A3=$P/S1/V2/A3_2grid/best.pt \
  --clamp x,min --load x,max --load-dir y --vfrac 0.8 --maxit 60 --stop-after $K --fast --table5 --workers 14 --sens ad \
  --ad-batch ${ADB:-1024} --resident-gb 0 --body-retry 6 "$@" > $R/run.log 2>&1; r=$?
kill $SMP
echo "$(date +%F_%T) END opt $TAG rc=$r gpu_max_MiB=$(cut -d, -f2 $R/gpu.csv | sort -n | tail -1)" >> $W/status.txt
