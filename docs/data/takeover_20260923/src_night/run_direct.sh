#!/bin/bash
# Direct (cuDSS) query path: geometry -> polyhedral elements (effective 8) -> P^T K P -> GPU sparse Cholesky, then the
# frozen finite witness against the teacher. Teacher-element runs time the solver alone. One job at a time.
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_17; O=$T/DIRECT_01
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/cutfem_neural_a_20260910/env/bin/python
cd $S
run() { name=$1; shift; rm -rf $O/$name; $PY -u encode_gpu.py --solver cudss --output $O/$name "$@" > $O/$name.log 2>&1; echo "$(date -u +%FT%TZ) $name exit $?" >> $O/status; }
for c in fresh_train_0013_d0_v1 fresh_train_0003_full; do run ${c}_teacher --case $c --elements teacher; done
for c in fresh_train_0011_cover01_r1 fresh_train_0008_cover01_r1 fresh_train_0013_cover01_r1 fresh_train_0026_cover01_r1 fresh_train_0007_cover01_r1 fresh_train_0021_cover01_r1; do
  run ${c}_polyref --case $c --cover-inputs $T/COVER_INPUTS/$c --elements polyref --s 4 --levels 1 --cpu-trace
done
for c in fresh_train_0003_d0_v0 fresh_train_0013_d0_v1 fresh_train_0006_d1_v1; do run ${c}_polyref --case $c --elements polyref --s 4 --levels 1; done
for c in fresh_train_0003_full fresh_train_0006_full; do run ${c}_polyref --case $c --elements polyref --s 4 --levels 1 --cpu-trace; done
echo "$(date -u +%FT%TZ) BATCH_DONE" >> $O/status
