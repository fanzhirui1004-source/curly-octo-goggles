#!/bin/bash
# C-plate validation (2026-10-02): (1) B1 rerun with --table5 (four Table-5 implementation options), 3 iterations, same
# args/env as plateB1, compared with plateB1/history.jsonl; (2) smoke run of the cut-band clamp on plate841: --clamp cut,
# load y,min (four full cells), direction x, 1 iteration.  Status lines in chain_cplate.status; marker CPLATE_VAL_DONE.
R=/root/autodl-tmp/OPL/S1/V2/R1/OPT; ST=$R/chain_cplate.status; TAG=CPVAL; RC=0
st() { echo "$(date +%F_%T) $TAG $*" >> $ST; }
fin() { st "END rc=$RC"; echo "$(date +%F_%T) CPLATE_VAL_DONE" >> $ST; }
trap fin EXIT
run() { local n=$1 to=$2; shift 2; st "START $n"; local t=$(date +%s)
  timeout $to "$@" > $R/$n.log 2>&1; local r=$?; [ $r != 0 ] && RC=$r; st "END $n rc=$r $(( $(date +%s) - t ))s"; }
source /root/autodl-tmp/OPL/S1/env_gpu.sh
D=$DEP/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$D PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/pylib_cpu PYPARDISO_MKL_RT=$D/libmkl_rt.so.3 MKL_NUM_THREADS=16
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets:$R/plate_packets OPL_GP_CACHE=0 OPL_CONV_FP32=1
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/OPL/src_v2
MODEL=A3=/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt
seed() { mkdir -p $2/body $2/packets
  for d in $1/body/*_o000; do [ -d $d ] && cp -rn $d $2/body/; done
  cp -rn $1/packets/*_o000 $2/packets/ 2>/dev/null; cp -n $1/body/GP_TEMPLATES_n32.npz $2/body/; }
st "QUEUED pid=$$"
seed $R/plateB1 $R/plateB1_t5val
run plateB1_t5val 7200 $PY -u opt_design.py $R/plateB1_t5val $R/layouts/plate841.json --model $MODEL \
  --clamp x,min --load x,max --load-dir y --vfrac 0.8 --maxit 60 --fast --workers 12 --sens ad --ad-batch 2048 --resident-gb 22 --warm \
  --table5 --stop-after 3
seed $R/plateB1 $R/cplate_smoke
run cplate_smoke 3600 $PY -u opt_design.py $R/cplate_smoke $R/layouts/plate841.json --model $MODEL \
  --clamp cut --load y,min --load-dir x --vfrac 0.8 --maxit 60 --fast --workers 12 --sens ad --ad-batch 2048 --resident-gb 22 --warm \
  --table5 --stop-after 1
