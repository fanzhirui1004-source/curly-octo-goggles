#!/bin/bash
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_17; O=$T/BENCH_02
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_COLS=64
cd $S
c=fresh_train_0013_d0_v1
for cfg in "1 16 DEFAULT" "1 16 AMD" "0 0 AMD" "1 16 NESTED_DISSECTION"; do
  set -- $cfg
  if [ $1 = 1 ]; then export CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0; else unset CUDSS_MT; fi
  CUDSS_THREADS=$2 CUDSS_REORDER=$3 timeout 600 /root/cutfem_neural_a_20260910/env/bin/python -u bench_cudss.py $c > $O/opt_${1}_${2}_$3.log 2>&1
  tail -1 $O/opt_${1}_${2}_$3.log >> $O/OPT.jsonl
done
echo done >> $O/status2
