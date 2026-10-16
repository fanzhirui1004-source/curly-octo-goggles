#!/bin/bash
# Thread scaling of the CPU sensitivity pass on one cell (checkpoint off, batch 256): 16, 4 and 1 threads.
. /root/autodl-tmp/cpu9754/env_cpu9754.sh
cd /root/autodl-tmp/OPL/src_v2
for T in 16 4 1; do
  MKL_NUM_THREADS=$T OMP_NUM_THREADS=$T OPL_SENS_NOCKPT=1 timeout 1500 $PY -u /root/autodl-tmp/cpu9754/bench_sens.py hlat222_110 256 2>&1 | grep -v -i warn | grep '"case"' > /root/autodl-tmp/cpu9754/bench_threads_$T.json
done
echo DONE > /root/autodl-tmp/cpu9754/bench_threads.done
