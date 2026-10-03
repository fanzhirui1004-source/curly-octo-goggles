#!/bin/bash
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_16; O=$T/BENCH_01; mkdir -p $O
M=/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/diagnostics/MECHANICS_NETWORK_20260922_01
PY=/root/cutfem_neural_a_20260910/env/bin/python
( export CUTFEM_RUNTIME_ROOT=/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime CUTFEM_EXECUTION_CONFIG=$M/NATIVE_EXECUTION_CONFIG.json PYTHONPATH=$M/native_source_01
  for c in fresh_train_0003_d0_v0 fresh_train_0013_d0_v1 fresh_train_0003_full; do $PY $S/bench_pardiso.py $c 8 >> $O/PARDISO.jsonl 2>> $O/pardiso.err; done ) &
cd $S
for c in fresh_train_0003_d0_v0 fresh_train_0013_d0_v1 fresh_train_0003_full; do
  rm -rf $O/gpu_$c
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u encode_gpu.py --case $c --elements teacher --gpu-pairs --output $O/gpu_$c > $O/gpu_$c.log 2>&1
done
wait
echo done >> $O/status
