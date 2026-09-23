#!/bin/bash
# New machine, step M: CPU geometry pipeline (frozen runtime, 16 workers) on 3 cases, then GPU reproduction check.
DEP=/root/autodl-tmp/CUTFEM_DEPENDENCIES_20260924
S=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/xcase_src_36
O=/root/autodl-tmp/OPL/M01; mkdir -p $O
export CUTFEM_EXECUTION_CONFIG=$DEP/EXECUTION_CONFIG_NEW16.json
cd $DEP/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/COVER_G/src
OMP_NUM_THREADS=1 $DEP/run_frozen_python.sh $S/fast_prep3.py 16 $O fresh_train_0013_cover01_r1 fresh_train_0021_cover01_r1 fresh_train_0013_full > $O/prep.log 2>&1
rc=$?; echo "$(date -u +%FT%TZ) prep rc $rc" >> $O/status
L=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/OPL
OMP_NUM_THREADS=16 /root/autodl-tmp/gpuenv/bin/python -u /root/autodl-tmp/OPL/repro_check.py $O > $O/repro.log 2>&1
rc=$?; echo "$(date -u +%FT%TZ) repro rc $rc" >> $O/status
echo DONE >> $O/status
