#!/bin/bash
# Route 1 E1b stage 1: teacher-free learned recurrence coefficients at fixed depth (pairs + Lanczos steps, quadratic coarse).
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_31; O=$T/R1_L1
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/cutfem_neural_a_20260910/env/bin/python
mkdir -p $O
cd $S
run() {  # name case depth extra-args...
  n=$1; c=$2; d=$3; shift 3
  rm -rf $O/$n
  $PY -u encode_r1.py --solver chebyshev --gpu-trace --gpu-pairs --case $c --cover-inputs $T/COVER_INPUTS/$c --elements polyref --s 4 --levels 1 \
      --degree1 2 --degree2 2 --omega-mode lanczos --safety 1.0 --learn-depth $d --output $O/$n "$@" > $O/$n.log 2>&1
  rc=$?
  ok=no; [ -f $O/$n/LEARN.json ] && ok=yes
  echo "$(date -u +%FT%TZ) $n rc $rc result $ok" >> $O/status
}
run 0013_L16 fresh_train_0013_cover01_r1 16 --learn-steps 300
run 0013_L24 fresh_train_0013_cover01_r1 24 --learn-steps 300
echo "$(date -u +%FT%TZ) L1_DONE" >> $O/status
