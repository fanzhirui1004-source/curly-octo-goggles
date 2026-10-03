#!/bin/bash
# Large second-batch cases with upper-only GPU P^T K P (CPU fallback on OOM) and transpose-aware audit backward.
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_19; O=$T/DIRECT_04
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $S
for c in fresh_train_0016_cover01_r1 fresh_train_0025_cover01_r1 fresh_train_0023_cover01_r1 fresh_train_0006_cover01_r1 fresh_train_0002_cover01_r1 fresh_train_0015_cover01_r1; do
  rm -rf $O/${c}_polyref
  /root/cutfem_neural_a_20260910/env/bin/python -u encode_gpu.py --solver cudss --gpu-trace --case $c --cover-inputs $T/COVER_INPUTS/$c --elements polyref --s 4 --levels 1 --output $O/${c}_polyref > $O/${c}_polyref.log 2>&1
  echo "$(date -u +%FT%TZ) ${c}_polyref exit $?" >> $O/status
done
echo "$(date -u +%FT%TZ) BATCH5_DONE" >> $O/status
