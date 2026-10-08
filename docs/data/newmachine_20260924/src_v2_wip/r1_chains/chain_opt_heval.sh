#!/bin/bash
# Section 6.11: fine-scale NICE evaluation of the homogenisation-optimised plate designs (after OPT_PROD_DONE):
# opt_design --stop-after 1 on the final layout of H_y / H_z (analysis of the given design only; the MMA step it
# computes is not used).  Tag OPTH; marker OPT_HEVAL_DONE_20260929 (EXIT trap).
ST=/root/autodl-tmp/OPL/S1/V2/chain_p1.status; R=/root/autodl-tmp/OPL/S1/V2/R1/OPT; TAG=OPTH; RC=0
st() { echo "$(date +%T) $TAG $*" >> $ST; }
fin() { st "END rc=$RC"; echo "$(date +%T) OPT_HEVAL_DONE_20260929" >> $ST; }
trap fin EXIT
run() { local n=$1 to=$2; shift 2; st "START $n"; local t=$(date +%s)
  timeout $to "$@" > $R/$n.log 2>&1; local r=$?; [ $r != 0 ] && RC=$r; st "END $n rc=$r $(( $(date +%s) - t ))s"; }
st "QUEUED pid=$$ (waits for OPT_PROD_DONE_20260929)"
until grep -qE "^[0-9:]+ OPT_PROD_DONE_20260929$" $ST; do sleep 60; done
source /root/autodl-tmp/OPL/S1/env_gpu.sh
D=$DEP/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$D PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/pylib_cpu PYPARDISO_MKL_RT=$D/libmkl_rt.so.3 MKL_NUM_THREADS=16
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets:$R/plate_packets OPL_GP_CACHE=0 OPL_CONV_FP32=1
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/OPL/src_v2
for B in y z; do mkdir -p $R/hevalH_$B; cp -n $R/plateB1/body/GP_TEMPLATES_n32.npz $R/hevalH_$B/ 2>/dev/null
  run hevalH_$B 7200 $PY -u opt_design.py $R/hevalH_$B $R/homog/H_$B.final_layout.json --model A3=/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt \
    --clamp x,min --load x,max --load-dir $B --vfrac 0.8 --stop-after 1 --fast --workers 12 --sens ad --ad-batch 2048 --resident-gb 22
done
