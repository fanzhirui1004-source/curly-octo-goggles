#!/bin/bash
# Route 7: fp32 Schur operator corrected on its soft subspace with k double-precision columns.
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_36; O=$T/R7_16; B=$T/R7_07
cd $S
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/cutfem_neural_a_20260910/env/bin/python
mkdir -p $O
cat /sys/fs/cgroup/memory.current > $O/mem_before
OMP_NUM_THREADS=16 $PY -u soft_correction.py $B $O fresh_train_0013_cover01_r1 fresh_train_0013_full 64,128,256,512 fresh_train_0021_cover01_r1 > $O/soft.log 2>&1
rc=$?
echo "$(date -u +%FT%TZ) soft rc $rc" >> $O/status
