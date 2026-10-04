#!/bin/bash
# Export hlat222 and hlat331 (sequentially) for the iterative-solver comparison.
source /root/autodl-tmp/OPL/S1/V2/R1/env_cpu.sh
export MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 OPL_GP_CACHE=0 PYTHONPATH=/root/autodl-tmp/OPL/src_v2:$PYTHONPATH
cd /root/autodl-tmp/OPL/src_v2
for L in hlat222 hlat331; do
  $PY -u /root/autodl-tmp/ITERCMP/lat_dump.py /root/autodl-tmp/OPL/S4/$L.json /root/autodl-tmp/ITERCMP/sys_$L > /root/autodl-tmp/ITERCMP/dump_$L.log 2>&1
  echo "$L rc=$? $(date +%T)" >> /root/autodl-tmp/ITERCMP/dump.status
done
