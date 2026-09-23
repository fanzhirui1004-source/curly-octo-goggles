#!/bin/bash
# Route 1: localization of the slowest modes of B A for the tuned hierarchy (Lanczos smoother steps, quadratic coarse).
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_29; O=$T/R1_D1
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/cutfem_neural_a_20260910/env/bin/python
mkdir -p $O
cd $S
run() {  # name case
  rm -rf $O/$1
  $PY -u encode_r1.py --solver chebyshev --gpu-trace --gpu-pairs --case $2 --cover-inputs $T/COVER_INPUTS/$2 --elements polyref --s 4 --levels 1 \
      --degree1 2 --degree2 2 --omega-mode lanczos --safety 1.0 --slow-diag 64 --output $O/$1 > $O/$1.log 2>&1
  rc=$?
  ok=no; [ -f $O/$1/SLOWDIAG.json ] && ok=yes
  echo "$(date -u +%FT%TZ) $1 rc $rc result $ok" >> $O/status
}
run 0013_d22_lz fresh_train_0013_cover01_r1
run 0020_d22_lz fresh_train_0020_cover01_r1
echo "$(date -u +%FT%TZ) D1_DONE" >> $O/status
