#!/bin/bash
# Route 7: design-loop re-encode cost (refactorization with a reused plan) and single-precision factorization.
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_36; O=$T/R7_09; B=$T/R7_07
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/cutfem_neural_a_20260910/env/bin/python
mkdir -p $O
cd $S
for c in fresh_train_0013_full; do
  OMP_NUM_THREADS=16 $PY -u refactor_test.py $c $B $O > $O/$c.log 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) $c rc $rc" >> $O/status
done
echo "$(date -u +%FT%TZ) REFACTOR_DONE" >> $O/status
