#!/bin/bash
# ELEV 2026-10-05: environment of FINAL2/chain_final_route2.sh (cplateN), 8 CPU threads, GPU capped by the script.
source /root/autodl-tmp/OPL/S1/env_gpu.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export OPL_GP_CACHE=0 OPL_CONV_FP32=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPL_TAILT_FUSED=1 OPL_COARSE_FP32=1 OPL_COARSE_ELEM=1 OPL_TET_TRITON=1 OPL_COARSE_SPARSE=1 OPL_COARSE_PIVOT_FLOOR=1e-7
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU OPL_T5_SKIP OPL_STREAM_PACK
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S1/V2/R1/FINAL2/cplateN/packets:/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets:/root/autodl-tmp/OPL/S1/V2/R1/OPT/plate_packets
R=/root/autodl-tmp/OPL/ELEV_20261005/render
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> $R/gpu_ncf.tsv; sleep 10; done ) &
SAMP=$!
rm -f $R/nice_q_k023.npz
taskset -c 0-7 $PY -u $R/nice_cell_field.py > $R/ncf.log 2>&1
rc=$?; echo "RC=$rc" >> $R/ncf.log
[ $rc = 0 ] && taskset -c 0-7 $PY -u $R/exact_cell_field.py > $R/ecf.log 2>&1; echo "RC=$?" >> $R/ecf.log
kill $SAMP
