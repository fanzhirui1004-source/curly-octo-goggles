#!/bin/bash
# Route 1: slow-mode localization with and without the weak-aggregate blocks (after the E4 batch).
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_31; O=$T/R1_D2
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/cutfem_neural_a_20260910/env/bin/python
mkdir -p $O
cd $S
while pgrep -f "[r]un_r1_weak.sh" > /dev/null; do sleep 15; done
run() {  # name case extra-args...
  n=$1; c=$2; shift 2
  rm -rf $O/$n
  $PY -u encode_r1.py --solver chebyshev --gpu-trace --gpu-pairs --case $c --cover-inputs $T/COVER_INPUTS/$c --elements polyref --s 4 --levels 1 \
      --degree1 2 --degree2 2 --omega-mode lanczos --safety 1.0 --slow-diag 64 --output $O/$n "$@" > $O/$n.log 2>&1
  rc=$?
  ok=no; [ -f $O/$n/SLOWDIAG.json ] && ok=yes
  echo "$(date -u +%FT%TZ) $n rc $rc result $ok" >> $O/status
}
run 0013_pairs fresh_train_0013_cover01_r1
run 0013_weak fresh_train_0013_cover01_r1 --weak-rel 0.01 --weak-theta 0.1 --weak-cap 32
echo "$(date -u +%FT%TZ) D2_DONE" >> $O/status
