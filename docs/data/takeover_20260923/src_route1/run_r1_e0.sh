#!/bin/bash
# Route 1 E0/E1a: exact-coefficient hierarchy at fixed Chebyshev depths, affine (degree 1) vs quadratic (degree 2)
# coarse fields, on second-version cases. Waits for the GPU to be free of other encode_gpu jobs first.
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_26; O=$T/R1_E0
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/cutfem_neural_a_20260910/env/bin/python
mkdir -p $O
while pgrep -f "encode_gpu.py --solver cudss --gpu-trace --full-spectrum" > /dev/null; do sleep 30; done
cd $S
for c in fresh_train_0013_cover01_r1 fresh_train_0020_cover01_r1; do
  for d1 in 1 2; do
    for d2 in 1 2; do
      name=${c}_d${d1}${d2}; rm -rf $O/$name
      $PY -u encode_r1.py --solver chebyshev --gpu-trace --gpu-pairs --case $c --cover-inputs $T/COVER_INPUTS/$c --elements polyref --s 4 --levels 1 \
          --degree1 $d1 --degree2 $d2 --depths 8,16,32,64 --output $O/$name > $O/$name.log 2>&1
      rc=$?
      ok=no; [ -f $O/$name/RESULT.json ] && ok=yes
      echo "$(date -u +%FT%TZ) $name rc $rc result $ok" >> $O/status
    done
  done
done
echo "$(date -u +%FT%TZ) E0_DONE" >> $O/status
