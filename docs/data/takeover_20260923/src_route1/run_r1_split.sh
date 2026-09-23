#!/bin/bash
# Route 1: component-aware coarse blocks (strong-connection split), exact coefficients, fixed depths.
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_26; O=$T/R1_E2
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/cutfem_neural_a_20260910/env/bin/python
mkdir -p $O; cd $S
while pgrep -f "[r]un_r1_oracle.sh" > /dev/null; do sleep 20; done
run() {  # name case d1 d2 theta
  rm -rf $O/$1
  $PY -u encode_r1.py --solver chebyshev --gpu-trace --gpu-pairs --case $2 --cover-inputs $T/COVER_INPUTS/$2 --elements polyref --s 4 --levels 1 \
      --degree1 $3 --degree2 $4 --split-theta $5 --depths 8,16,32,64 --output $O/$1 > $O/$1.log 2>&1
  rc=$?
  ok=no; [ -f $O/$1/RESULT.json ] && ok=yes
  echo "$(date -u +%FT%TZ) $1 rc $rc result $ok" >> $O/status
}
run 0013_d11_s0.01 fresh_train_0013_cover01_r1 1 1 0.01
run 0013_d22_s0.01 fresh_train_0013_cover01_r1 2 2 0.01
run 0013_d11_s0.1 fresh_train_0013_cover01_r1 1 1 0.1
run 0020_d22_s0.01 fresh_train_0020_cover01_r1 2 2 0.01
echo "$(date -u +%FT%TZ) E2_DONE" >> $O/status
