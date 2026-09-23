#!/bin/bash
# Route 7: memory of one encoder, dense box operator cost, cuDSS reordering algorithms.
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_36; O=$T/R7_13; B=$T/R7_07
cd $S
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/cutfem_neural_a_20260910/env/bin/python
mkdir -p $O
OMP_NUM_THREADS=16 $PY -u budget_probe.py $B $O 512 0 fresh_train_0013_cover01_r1 fresh_train_0021_cover01_r1 > $O/cut.log 2>&1
rc=$?
echo "$(date -u +%FT%TZ) cut rc $rc" >> $O/status
OMP_NUM_THREADS=16 $PY -u budget_probe.py $B $O 512 1 fresh_train_0013_full > $O/full.log 2>&1
rc=$?
echo "$(date -u +%FT%TZ) full rc $rc" >> $O/status
echo "$(date -u +%FT%TZ) BUDGET_DONE" >> $O/status
