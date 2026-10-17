#!/bin/bash
# ncu_symv.sh: Nsight Compute counters of the float32 element product alone vs inside S_hat (t_ncu_symv.py).
W=/root/autodl-tmp/OPL/V3W; O=$W/tests; P=/root/autodl-tmp/OPL
source $P/S1/env_gpu.sh
export OPL_GP_CACHE=0 OPL_CONV_FP32=1 OPL_COARSE_PIVOT_FLOOR=1e-7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPL_PACKETS_EXTRA=$W/plate/base_S24/packets:$P/S4/packets:$P/S3/packets:$P/S1/V2/R1/SCALE/plate_packets
export OPL_BENCH_TMP=$W/bench_tmp SRC=${SRC:-$P/src_v3}
cd $SRC
echo "$(date +%F_%T) START ncu_symv" >> $W/status.txt
timeout 2400 /usr/local/cuda/bin/ncu --nvtx --nvtx-include "alone/" --nvtx-include "inshat/" -k regex:_symv_up --launch-count 8 \
  --replay-mode application --cache-control none \
  --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__occupancy_limit_registers,smsp__cycles_active.avg,sm__cycles_elapsed.avg.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  $PY -u $W/scripts/t_ncu_symv.py $W/plate/body_S24:plateS24_040_o000 > $O/ncu_symv.log 2>&1; r=$?
echo "$(date +%F_%T) END ncu_symv rc=$r" >> $W/status.txt
