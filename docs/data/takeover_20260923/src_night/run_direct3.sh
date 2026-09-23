#!/bin/bash
# After the second batch: 0015 alone on the GPU, then re-time every direct case with the GPU P^T K P (DIRECT_02).
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_18; O=$T/DIRECT_02
until grep -q BATCH2_DONE $T/DIRECT_01/status; do sleep 30; done
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/cutfem_neural_a_20260910/env/bin/python
cd $S
run() { name=$1; shift; rm -rf $O/$name; $PY -u encode_gpu.py --solver cudss --gpu-trace --output $O/$name "$@" > $O/$name.log 2>&1; echo "$(date -u +%FT%TZ) $name exit $?" >> $O/status; }
run fresh_train_0015_cover01_r1_polyref --case fresh_train_0015_cover01_r1 --cover-inputs $T/COVER_INPUTS/fresh_train_0015_cover01_r1 --elements polyref --s 4 --levels 1
for c in fresh_train_0003_d0_v0 fresh_train_0003_full fresh_train_0013_d0_v1; do run ${c}_polyref --case $c --elements polyref --s 4 --levels 1; done
for c in fresh_train_0006_d1_v1 fresh_train_0006_full fresh_development_0000_d0_v0; do run ${c}_polyref --case $c --asset-root $T/ASSETS_DEV --elements polyref --s 4 --levels 1; done
for d in $T/COVER_INPUTS/*/; do
  c=$(basename $d); [ -f $d/RESULT.json ] || continue; [ $c = fresh_train_0015_cover01_r1 ] && continue
  run ${c}_polyref --case $c --cover-inputs $T/COVER_INPUTS/$c --elements polyref --s 4 --levels 1
done
echo "$(date -u +%FT%TZ) BATCH3_DONE" >> $O/status
