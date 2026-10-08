#!/bin/bash
# Section 6.11 step 6c: scale demonstration on one RTX 5090 with 90 GiB host memory.  Cut plates with the proportions of
# plate841 (cut from the bottom-right corner to the top edge at a quarter of the length; internal x = short side),
# scaled by s: 8x4 (24 cells), 12x6 (51), 16x8 (88), 18x9 (110), 20x10 (135); bottom clamp, top in-plane load (as B1),
# uniform tau0 0.40, vfrac 0.8; opt_design --fast --sens ad --warm --body-retry 6, 4 analyses (3 MMA updates), cells
# resident on the GPU up to 18 GB and streamed from host memory beyond, exact packed storage (OPL_STREAM_PACK=1), sparse
# coarse space (OPL_COARSE_SPARSE=1).  Uniform-configuration reruns (after SCALE2_DONE) into <name>r4: 24 and 51 cells, and 88 cells if its run
# in plateS88 did not use the 4 GB resident budget; cells resident on the GPU up to 4 GB.  Tag SCALE3; marker SCALE3_DONE_20260930.
ST=/root/autodl-tmp/OPL/S1/V2/chain_p1.status; R=/root/autodl-tmp/OPL/S1/V2/R1/SCALE; TAG=SCALE3; RC=0
st() { echo "$(date +%T) $TAG $*" >> $ST; }
fin() { st "END rc=$RC"; echo "$(date +%T) SCALE3_DONE_20260930" >> $ST; }
trap fin EXIT
run() { local n=$1 to=$2; shift 2; st "START $n"; local t=$(date +%s)
  timeout $to "$@" >> $R/$n.log 2>&1; local r=$?; [ $r != 0 ] && RC=$r; st "END $n rc=$r $(( $(date +%s) - t ))s"; }
st "QUEUED pid=$$ (waits for SCALE2_DONE_20260930)"
until grep -qE "^[0-9:]+ SCALE2_DONE_20260930$" $ST; do sleep 60; done
sed -e 's/chain_scale.sh/chain_scale3.sh/' -e 's/plateS\[0-9\]\*/plateS[0-9r]*/' $R/rss_sampler.sh > $R/rss_sampler3.sh; nohup bash $R/rss_sampler3.sh > /dev/null 2>&1 &
source /root/autodl-tmp/OPL/S1/env_gpu.sh
D=$DEP/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$D PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/pylib_cpu PYPARDISO_MKL_RT=$D/libmkl_rt.so.3 MKL_NUM_THREADS=16
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets:$R/plate_packets OPL_GP_CACHE=0 OPL_CONV_FP32=1
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPL_STREAM_PACK=1 OPL_COARSE_SPARSE=1
cd /root/autodl-tmp/OPL/src_v2
mkdir -p $R/layouts $R/plate_packets
for P in plateS24 plateS51 plateS88; do set -- $P
  if [ $1 = plateS88 ]; then grep '"event": "START"' $R/plateS88.log | tail -n 1 | grep -q '"resident_gb": 4.0' && continue; fi
  mkdir -p $R/${1}r4
  run ${1}r4 36000 $PY -u opt_design.py $R/${1}r4 $R/layouts/$1.json --model A3=/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt \
    --clamp x,min --load x,max --load-dir y --vfrac 0.8 --maxit 60 --stop-after 4 --fast --workers 14 --sens ad --ad-batch 2048 \
    --resident-gb 4  --warm --body-retry 6
done
