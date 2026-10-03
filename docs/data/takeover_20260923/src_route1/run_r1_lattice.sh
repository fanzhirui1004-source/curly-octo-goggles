#!/bin/bash
# Route 1: lattice acceptance gate (cut cell 0013 at the free end of a cantilever with its FULL parent as neighbour).
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_32; O=$T/R1_LAT
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/cutfem_neural_a_20260910/env/bin/python
mkdir -p $O
cd $S
run() {  # name case elements extra-args...
  n=$1; c=$2; el=$3; shift 3
  rm -rf $O/$n
  $PY -u encode_r1.py --solver chebyshev --gpu-trace --gpu-pairs --case $c --cover-inputs $T/COVER_INPUTS/$c --elements $el --s 4 --levels 1 \
      --degree1 2 --degree2 2 --omega-mode lanczos --safety 1.0 --output $O/$n "$@" > $O/$n.log 2>&1
  rc=$?
  ok=no; ls $O/$n/LATTICE_*.json > /dev/null 2>&1 && ok=yes
  echo "$(date -u +%FT%TZ) $n rc $rc result $ok" >> $O/status
}
C13=fresh_train_0013_cover01_r1
run 0013_body $C13 body --lattice x,y --lattice-variants c8,c16,c24,c32,c64
run 0013_polyref $C13 polyref --lattice x,y --lattice-variants c16,c32,c64
echo "$(date -u +%FT%TZ) LAT_DONE" >> $O/status
