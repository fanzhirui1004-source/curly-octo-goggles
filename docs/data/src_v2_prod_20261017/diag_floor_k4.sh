#!/bin/bash
# Check of the coarse pivot floor (2026-10-03): the k = 4 design of case A analysed once on the final route (--table5, which now
# sets OPL_COARSE_PIVOT_FLOOR=1e-7), cold start.  Expect a recomputed residual at the level of the original route (1.2e-4)
# and a nonzero coarse shift on cells 100, 101, 111.  Marker DIAG_FLOOR_DONE in FINAL/diag.status.
R=/root/autodl-tmp/OPL/S1/V2/R1/FINAL; D=$R/diag; ST=$R/diag.status
source /root/autodl-tmp/OPL/S1/env_gpu.sh
export OPL_GP_CACHE=0 OPL_CONV_FP32=1 OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/OPL/src_v2
n=t5_floor; mkdir -p $D/$n/body; cp -n $R/optA/body/GP_TEMPLATES_n32.npz $D/$n/body/
echo "$(date +%F_%T) DIAG START $n" >> $ST
timeout 3600 $PY -u opt_design.py $D/$n $D/hlat222_k4.json --model A3=/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt --clamp y,min --load y,max \
  --load-dir y --vfrac 0.8 --fast --workers 8 --sens ad --ad-batch 4096 --resident-gb 28 --body-retry 6 --stop-after 1 --table5 > $D/$n.log 2>&1
echo "$(date +%F_%T) DIAG END $n rc=$? $(grep -o '"pcg": [0-9]*, "true_residual": [0-9.e+-]*' $D/$n.log | head -1)" >> $ST
echo "$(date +%F_%T) DIAG_FLOOR_DONE" >> $ST
