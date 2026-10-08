#!/bin/bash
# Section 6.11 step 6c: scale demonstration on one RTX 5090 with 90 GiB host memory.  Cut plates with the proportions of
# plate841 (cut from the bottom-right corner to the top edge at a quarter of the length; internal x = short side),
# scaled by s: 8x4 (24 cells), 12x6 (51), 16x8 (88), 18x9 (110), 20x10 (135); bottom clamp, top in-plane load (as B1),
# uniform tau0 0.40, vfrac 0.8; opt_design --fast --sens ad --warm --body-retry 6, 4 analyses (3 MMA updates), cells
# resident on the GPU up to 18 GB and streamed from host memory beyond, exact packed storage (OPL_STREAM_PACK=1), sparse
# coarse space (OPL_COARSE_SPARSE=1).  Rerun (after SCALE_DONE) of the sizes whose first run did not reach 4 iterations, with
# cells resident on the GPU only up to 4 GB (patched in place from 10 GB before it started: 18 GB left too little device
# memory for the front end at 135 cells and for the cuDSS K_PP factor at 51 cells).
# Order 88, 110, 135.  Tag SCALE2; marker SCALE2_DONE_20260930 (EXIT trap).
ST=/root/autodl-tmp/OPL/S1/V2/chain_p1.status; R=/root/autodl-tmp/OPL/S1/V2/R1/SCALE; TAG=SCALE2; RC=0
st() { echo "$(date +%T) $TAG $*" >> $ST; }
fin() { st "END rc=$RC"; echo "$(date +%T) SCALE2_DONE_20260930" >> $ST; }
trap fin EXIT
run() { local n=$1 to=$2; shift 2; st "START $n"; local t=$(date +%s)
  timeout $to "$@" >> $R/$n.log 2>&1; local r=$?; [ $r != 0 ] && RC=$r; st "END $n rc=$r $(( $(date +%s) - t ))s"; }
st "QUEUED pid=$$ (waits for SCALE_DONE_20260930)"
until grep -qE "^[0-9:]+ SCALE_DONE_20260930$" $ST; do sleep 60; done
sed 's/chain_scale.sh/chain_scale2.sh/' $R/rss_sampler.sh > $R/rss_sampler2.sh; nohup bash $R/rss_sampler2.sh > /dev/null 2>&1 &
source /root/autodl-tmp/OPL/S1/env_gpu.sh
D=$DEP/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$D PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/pylib_cpu PYPARDISO_MKL_RT=$D/libmkl_rt.so.3 MKL_NUM_THREADS=16
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets:$R/plate_packets OPL_GP_CACHE=0 OPL_CONV_FP32=1
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPL_STREAM_PACK=1 OPL_COARSE_SPARSE=1
cd /root/autodl-tmp/OPL/src_v2
mkdir -p $R/layouts $R/plate_packets
for P in "plateS88" "plateS110" "plateS135"; do set -- $P
  [ "$(cat $R/$1/history.jsonl 2>/dev/null | wc -l)" -ge 4 ] && continue
  run $1 36000 $PY -u opt_design.py $R/$1 $R/layouts/$1.json --model A3=/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt \
    --clamp x,min --load x,max --load-dir y --vfrac 0.8 --maxit 60 --stop-after 4 --fast --workers 14 --sens ad --ad-batch 2048 \
    --resident-gb 4  --warm --body-retry 6
done
