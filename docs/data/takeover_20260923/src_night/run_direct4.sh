#!/bin/bash
# After batch 3: the four worst second-batch cases at effective element resolution 16 (octree levels 2).
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_18; O=$T/DIRECT_03; mkdir -p $O
until grep -q BATCH3_DONE $T/DIRECT_02/status 2>/dev/null; do sleep 30; done
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $S
for c in fresh_train_0020_cover01_r1 fresh_train_0017_cover01_r1 fresh_train_0024_cover01_r1 fresh_train_0005_cover01_r1; do
  rm -rf $O/${c}_L2
  /root/cutfem_neural_a_20260910/env/bin/python -u encode_gpu.py --solver cudss --gpu-trace --case $c --cover-inputs $T/COVER_INPUTS/$c --elements polyref --s 4 --levels 2 --output $O/${c}_L2 > $O/${c}_L2.log 2>&1
  echo "$(date -u +%FT%TZ) ${c}_L2 exit $?" >> $O/status
done
echo "$(date -u +%FT%TZ) BATCH4_DONE" >> $O/status
