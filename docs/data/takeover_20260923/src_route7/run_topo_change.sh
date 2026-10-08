#!/bin/bash
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; D=$T/xcase_src_36
cd $D
for e in 1/10000 -1/10000 1/1000 -1/1000 1/100 -1/100; do
  TAU_EPS=$e bash run_fast_prep3.sh 16 fresh_train_0013_cover01_r1 fresh_train_0021_cover01_r1 fresh_train_0013_full
done
/root/cutfem_neural_a_20260910/env/bin/python topo_change.py $T/R7_07 fresh_train_0013_cover01_r1 fresh_train_0021_cover01_r1 fresh_train_0013_full > $T/R7_07/topo_change.log 2>&1
echo "$(date -u +%FT%TZ) TOPO_DONE" >> $T/R7_07/status
