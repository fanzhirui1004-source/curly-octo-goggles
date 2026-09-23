#!/bin/bash
# Route 7: dense box operator from a partial cuDSS factorization (Schur mode), checked against the panel route.
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_36; O=$T/R7_14; B=$T/R7_07
cd $S
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/cutfem_neural_a_20260910/env/bin/python
mkdir -p $O
for c in fresh_train_0013_cover01_r1 fresh_train_0021_cover01_r1 fresh_train_0013_full; do
  OMP_NUM_THREADS=16 $PY -u schur_encoder.py $B $O 512 $c > $O/$c.log 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) $c rc $rc" >> $O/status
done
echo "$(date -u +%FT%TZ) SCHUR_DONE" >> $O/status
