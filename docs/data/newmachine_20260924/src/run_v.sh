#!/bin/bash
# New machine, step V: fixed-coordinate check (metadata scan, cube-group equivariance of the pipeline).
DEP=/root/autodl-tmp/CUTFEM_DEPENDENCIES_20260924
O=/root/autodl-tmp/OPL/V01; mkdir -p $O
BASES="fresh_train_0013_cover01_r1 fresh_train_0021_cover01_r1 fresh_train_0013_full"
cd /root/autodl-tmp/OPL
/root/autodl-tmp/gpuenv/bin/python make_rot.py $BASES > $O/make_rot.log 2>&1
ROT=$(grep '^ROTATED' $O/make_rot.log | cut -d' ' -f2-)
export CUTFEM_EXECUTION_CONFIG=$DEP/EXECUTION_CONFIG_NEW16.json
cd $DEP/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/COVER_G/src
OMP_NUM_THREADS=1 $DEP/run_frozen_python.sh /root/autodl-tmp/OPL/src/fast_prep3.py 16 $O $BASES $ROT > $O/prep.log 2>&1
rc=$?; echo "$(date -u +%FT%TZ) prep rc $rc" >> $O/status
L=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/pylib_cudss:/root/autodl-tmp/OPL CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/OPL
OMP_NUM_THREADS=16 /root/autodl-tmp/gpuenv/bin/python -u v_rot.py $O $O/V_ROT.json $BASES > $O/v_rot.log 2>&1
rc=$?; echo "$(date -u +%FT%TZ) vrot rc $rc" >> $O/status
echo DONE >> $O/status
