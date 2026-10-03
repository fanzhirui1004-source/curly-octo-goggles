#!/bin/bash
# Route 7: superset pattern (plan reuse across small topology changes): CPU superset stage, then GPU encoder checks.
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_36; O=$T/R7_12; B=$T/R7_07
cd $S
bash run_fast_superset.sh 16 1/50 fresh_train_0013_cover01_r1 fresh_train_0021_cover01_r1 fresh_train_0013_full
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/cutfem_neural_a_20260910/env/bin/python
mkdir -p $O
for c in fresh_train_0013_cover01_r1 fresh_train_0021_cover01_r1 fresh_train_0013_full; do
  OMP_NUM_THREADS=16 $PY -u superset_encoder.py $c $B $B/${c}_sup1_50 $O 0 1/10000 -1/10000 1/1000 -1/1000 1/100 -1/100 > $O/$c.log 2>&1
  echo "$(date -u +%FT%TZ) $c rc $?" >> $O/status
done
echo "$(date -u +%FT%TZ) SUPERSET_DONE" >> $O/status
