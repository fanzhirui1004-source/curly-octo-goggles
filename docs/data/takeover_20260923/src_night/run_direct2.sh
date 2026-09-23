#!/bin/bash
# Direct path, second round: train6 (dev assets), dev0000, the 0015 cover case that ran out of GPU memory on the
# Chebyshev path, then every remaining verified second-batch case that has a teacher target (teacher-free inputs built
# first on the CPU with the frozen P / GP code).
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_17; O=$T/DIRECT_01
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/cutfem_neural_a_20260910/env/bin/python
cd $S
run() { name=$1; shift; rm -rf $O/$name; $PY -u encode_gpu.py --solver cudss --output $O/$name "$@" > $O/$name.log 2>&1; echo "$(date -u +%FT%TZ) $name exit $?" >> $O/status; }
run fresh_train_0006_d1_v1_polyref --case fresh_train_0006_d1_v1 --asset-root $T/ASSETS_DEV --elements polyref --s 4 --levels 1
run fresh_train_0006_full_polyref --case fresh_train_0006_full --asset-root $T/ASSETS_DEV --elements polyref --s 4 --levels 1 --cpu-trace
run fresh_development_0000_d0_v0_polyref --case fresh_development_0000_d0_v0 --asset-root $T/ASSETS_DEV --elements polyref --s 4 --levels 1
run fresh_train_0015_cover01_r1_polyref --case fresh_train_0015_cover01_r1 --cover-inputs $T/COVER_INPUTS/fresh_train_0015_cover01_r1 --elements polyref --s 4 --levels 1 --cpu-trace
for v in $T/COVER_G/runs/*_G_VERIFY.json; do
  c=$(basename $v _G_VERIFY.json)
  grep -q '"MATCH"' $v || continue
  [ -f /root/autodl-tmp/CUTFEM_FRESH_GP_20260921/targets/${c}_v1/REFERENCE_RQ.npy ] || continue
  [ -f $O/${c}_polyref/RESULT.json ] && continue
  [ -f $T/COVER_INPUTS/$c/RESULT.json ] || bash $T/xcase_src_11/run_cover_blocks.sh $c > $T/COVER_INPUTS/$c.log 2>&1
  [ -f $T/COVER_INPUTS/$c/RESULT.json ] || { echo "$(date -u +%FT%TZ) ${c} blocks_failed" >> $O/status; continue; }
  run ${c}_polyref --case $c --cover-inputs $T/COVER_INPUTS/$c --elements polyref --s 4 --levels 1 --cpu-trace
done
echo "$(date -u +%FT%TZ) BATCH2_DONE" >> $O/status
