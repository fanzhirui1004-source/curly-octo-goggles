#!/bin/bash
# Section 6.11 cross-start test (after OPT_RESUME_DONE): NICE optimisation started from the homogenisation designs
# H_y / H_z under the absolute volume bound of B1/B2 (V* = 3.655259164261379), same constraints and MMA, 12 iterations at
# most: does NICE improve the homogenisation design (then B1/B2 stopped in a worse local optimum) or not?
# k=0 bodies from hevalH_y / hevalH_z.  Tag OPTXS; marker OPT_XSTART_DONE_20260930 (EXIT trap).
ST=/root/autodl-tmp/OPL/S1/V2/chain_p1.status; R=/root/autodl-tmp/OPL/S1/V2/R1/OPT; TAG=OPTXS; RC=0
st() { echo "$(date +%T) $TAG $*" >> $ST; }
fin() { st "END rc=$RC"; echo "$(date +%T) OPT_XSTART_DONE_20260930" >> $ST; }
trap fin EXIT
run() { local n=$1 to=$2; shift 2; st "START $n"; local t=$(date +%s)
  timeout $to "$@" >> $R/$n.log 2>&1; local r=$?; [ $r != 0 ] && RC=$r; st "END $n rc=$r $(( $(date +%s) - t ))s"; }
st "QUEUED pid=$$ (waits for OPT_RESUME_DONE_20260929)"
until grep -qE "^[0-9:]+ OPT_RESUME_DONE_20260929$" $ST; do sleep 60; done
source /root/autodl-tmp/OPL/S1/env_gpu.sh
D=$DEP/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$D PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/pylib_cpu PYPARDISO_MKL_RT=$D/libmkl_rt.so.3 MKL_NUM_THREADS=16
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets:$R/plate_packets OPL_GP_CACHE=0 OPL_CONV_FP32=1
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/OPL/src_v2
MODEL=A3=/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt
WARM=$(grep -q "warm-start test -> '--warm'" $ST && echo --warm)
for ld in y z; do X=$R/xstartH_$ld; mkdir -p $X/body $X/packets
  for bd in $R/hevalH_$ld/body/*_o000; do [ -d $bd ] && cp -rn $bd $X/body/; done
  cp -rn $R/hevalH_$ld/packets/*_o000 $X/packets/ 2>/dev/null; cp -n $R/hevalH_$ld/body/GP_TEMPLATES_n32.npz $X/body/ 2>/dev/null
  run xstartH_$ld 14400 $PY -u opt_design.py $X $R/homog/H_$ld.final_layout.json --model $MODEL \
    --clamp x,min --load x,max --load-dir $ld --vstar 3.655259164261379 --maxit 60 --stop-after 12 --fast --workers 12 \
    --sens ad --ad-batch 2048 --resident-gb 22 $WARM --body-retry 6
done
