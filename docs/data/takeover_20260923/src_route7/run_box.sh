#!/bin/bash
# Route 7: direct box condensation (no trace coordinates): validation against the packet and timing.
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_34; O=$T/R7_06
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/cutfem_neural_a_20260910/env/bin/python
mkdir -p $O
cd $S
run() {  # name case elements extra...
  n=$1; c=$2; el=$3; shift 3
  ci=""; [ -d $T/COVER_INPUTS/$c ] && ci="--cover-inputs $T/COVER_INPUTS/$c"
  OMP_NUM_THREADS=16 $PY -u box_encode.py --case $c --elements $el $ci --output $O "$@" > $O/$n.log 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) $n rc $rc" >> $O/status
}
run 0013_body fresh_train_0013_cover01_r1 body --validate
run 0013_poly fresh_train_0013_cover01_r1 polyref --validate
run 0021_poly fresh_train_0021_cover01_r1 polyref
run 0015_poly fresh_train_0015_cover01_r1 polyref
run 0013full_poly fresh_train_0013_full polyref
echo "$(date -u +%FT%TZ) BOX_DONE" >> $O/status
