#!/bin/bash
O=/root/autodl-tmp/OPL/V01
BASES="fresh_train_0013_cover01_r1 fresh_train_0021_cover01_r1 fresh_train_0013_full"
L=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/pylib_cudss:/root/autodl-tmp/OPL CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/OPL
OMP_NUM_THREADS=16 /root/autodl-tmp/gpuenv/bin/python -u v_rot.py $O $O/V_ROT.json $BASES > $O/v_rot.log 2>&1
rc=$?; echo "$(date -u +%FT%TZ) vrot2 rc $rc" >> $O/status
echo DONE2 >> $O/status
