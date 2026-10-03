#!/bin/bash
# Route 7: queries per cell for one lattice solve (uniform FULL cells tiled; thick 2/3 and thin 1/4).
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_36; O=$T/R7_17; B=$T/R7_07
mkdir -p $O
cd $S
for u in 2/3 1/4; do
  TAU_UNIFORM=$u bash run_fast_prep3.sh 16 fresh_train_0015_full
  echo "$(date -u +%FT%TZ) prep $u done" >> $O/status
done
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/cutfem_neural_a_20260910/env/bin/python
for u in 2/3 1/4; do
  sub=fresh_train_0015_full_uni${u/\//_}
  OMP_NUM_THREADS=16 $PY -u lattice_pcg.py $B $O fresh_train_0015_full $sub $u "2x1x1;2x2x2;3x3x3;10x2x2;5x5x4" > $O/$sub.log 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) $sub rc $rc" >> $O/status
done
echo "$(date -u +%FT%TZ) LATTICE_PCG_DONE" >> $O/status
