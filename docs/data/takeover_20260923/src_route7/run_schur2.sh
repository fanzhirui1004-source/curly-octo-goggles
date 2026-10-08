#!/bin/bash
# Route 7: FULL cell in Schur mode (fp32 on device; fp64 with the Schur block copied to pinned host memory).
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_36; O=$T/R7_14; B=$T/R7_07
cd $S
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/cutfem_neural_a_20260910/env/bin/python
mkdir -p $O
c=fresh_train_0013_full
SCHUR_OUT_HOST=1 OMP_NUM_THREADS=16 $PY -u schur_encoder.py $B $O 512 $c > $O/${c}_2.log 2>&1
rc=$?
echo "$(date -u +%FT%TZ) $c host-out rc $rc" >> $O/status
echo "$(date -u +%FT%TZ) SCHUR2_DONE" >> $O/status
