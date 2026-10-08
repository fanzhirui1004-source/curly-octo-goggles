#!/bin/bash
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_17; O=$T/BENCH_02
export LD_LIBRARY_PATH=$T/pylib_cudss/nvidia/cu12/lib:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss
cd $S
for c in fresh_train_0003_d0_v0 fresh_train_0013_d0_v1 fresh_train_0003_full; do
  /root/cutfem_neural_a_20260910/env/bin/python -u bench_cudss.py $c > $O/cudss_$c.log 2>&1
  tail -1 $O/cudss_$c.log >> $O/CUDSS.jsonl
done
echo done >> $O/status
