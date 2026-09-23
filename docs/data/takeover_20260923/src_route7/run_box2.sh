#!/bin/bash
# Route 7: GPU stage of the fast exact pipeline on the fast CPU stage's body-lite (no COVER_G, no P, ghost from faces).
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_36; O=$T/R7_08; B=$T/R7_07
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/cutfem_neural_a_20260910/env/bin/python
mkdir -p $O
cd $S
while pgrep -f "[r]un_r1_lattice2.sh" > /dev/null; do sleep 15; done
run() {  # name case extra...
  n=$1; c=$2; shift 2
  OMP_NUM_THREADS=16 $PY -u box_encode.py --case $c --elements polyref --body-dir $B --ghost faces --output $O "$@" > $O/$n.log 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) $n rc $rc" >> $O/status
}
run 0013 fresh_train_0013_cover01_r1 --validate --check-ghost --cover-inputs $T/COVER_INPUTS/fresh_train_0013_cover01_r1
run 0021 fresh_train_0021_cover01_r1 --check-ghost --cover-inputs $T/COVER_INPUTS/fresh_train_0021_cover01_r1
run 0015 fresh_train_0015_cover01_r1
run 0013full fresh_train_0013_full --validate
echo "$(date -u +%FT%TZ) BOX2_DONE" >> $O/status
