#!/bin/bash
# Route 1 oracle: level-1 space enriched with block restrictions of the slowest modes of the exact V-cycle.
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_26; O=$T/R1_E1
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/cutfem_neural_a_20260910/env/bin/python
mkdir -p $O; cd $S
run() {  # name case d1 d2 m k
  rm -rf $O/$1
  $PY -u encode_r1.py --solver chebyshev --gpu-trace --gpu-pairs --case $2 --cover-inputs $T/COVER_INPUTS/$2 --elements polyref --s 4 --levels 1 \
      --degree1 $3 --degree2 $4 --slow-m $5 --slow-k $6 --depths 8,16,32,64 --output $O/$1 > $O/$1.log 2>&1
  rc=$?
  ok=no; [ -f $O/$1/RESULT.json ] && ok=yes
  echo "$(date -u +%FT%TZ) $1 rc $rc result $ok" >> $O/status
}
run 0013_d22_m128_k12 fresh_train_0013_cover01_r1 2 2 128 12
run 0013_d22_m256_k24 fresh_train_0013_cover01_r1 2 2 256 24
run 0013_d11_m128_k12 fresh_train_0013_cover01_r1 1 1 128 12
echo "$(date -u +%FT%TZ) E1_DONE" >> $O/status
