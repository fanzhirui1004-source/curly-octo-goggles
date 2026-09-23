#!/bin/bash
# Route 1: smoother strength (Lanczos omegas) and smoothed aggregation, exact coefficients, fixed depths.
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_28; O=$T/R1_E3
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/cutfem_neural_a_20260910/env/bin/python
mkdir -p $O
cd $S
while pgrep -f "[r]un_r1_tune2.sh" > /dev/null; do sleep 20; done
run() {  # name case degree1 degree2 extra-args...
  n=$1; c=$2; d1=$3; d2=$4; shift 4
  rm -rf $O/$n
  $PY -u encode_r1.py --solver chebyshev --gpu-trace --gpu-pairs --case $c --cover-inputs $T/COVER_INPUTS/$c --elements polyref --s 4 --levels 1 \
      --degree1 $d1 --degree2 $d2 --depths 8,16,32,64 --output $O/$n "$@" > $O/$n.log 2>&1
  rc=$?
  ok=no; [ -f $O/$n/RESULT.json ] && ok=yes
  echo "$(date -u +%FT%TZ) $n rc $rc result $ok" >> $O/status
}
C13=fresh_train_0013_cover01_r1; C20=fresh_train_0020_cover01_r1
run 0013_d22_lz_s10 $C13 2 2 --omega-mode lanczos --safety 1.0
run 0020_d22_lz_s08 $C20 2 2 --omega-mode lanczos --safety 0.8
run 0013_d22_sa $C13 2 2 --omega-mode lanczos --smooth-p1 1.3333333333
run 0013_d22_sa_s08 $C13 2 2 --omega-mode lanczos --smooth-p1 1.3333333333 --safety 0.8
run 0020_d22_sa_s08 $C20 2 2 --omega-mode lanczos --smooth-p1 1.3333333333 --safety 0.8
echo "$(date -u +%FT%TZ) E3C_DONE" >> $O/status
