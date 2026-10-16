#!/bin/bash
# NICE (learned route, FINAL3 command) on hlat221a with the per-cell sensitivities saved: independent check of the exact
# CPU sensitivities (validation only).
P=/root/autodl-tmp/OPL
source $P/S1/env_gpu.sh
export OPL_GP_CACHE=0 OPL_CONV_FP32=1 OPL_PACKETS_EXTRA=$P/S4/packets:$P/S3/packets OPL_COARSE_PIVOT_FLOOR=1e-7
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU OPL_T5_SKIP OPL_STREAM_PACK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V=/root/autodl-tmp/cpu9754/val; mkdir -p $V; cd $P/src_v2
$PY -u lat_scale.py $V/nice_hlat221a.json $P/S4/hlat221a.json --fastidx --coarse-tpl --tet-triton --sparse-coarse --resident-gb 4 \
  --tol 1e-6 --levels 1e-2,3e-3,1e-3,1e-4 --iters 1 --sens ad --sens-obj sum --ad-batch 2048 --save-sens $V/nice_sens_hlat221a.npz > $V/nice.log 2>&1
echo "nice rc=$?" >> $V/status
