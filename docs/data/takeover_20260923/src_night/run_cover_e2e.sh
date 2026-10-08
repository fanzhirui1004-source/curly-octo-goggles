#!/bin/bash
# Teacher-free end-to-end on second-batch cases: frozen P/GP + exact-element probe check (CPU), then GPU encode with
# polyhedral elements (effective 8) and the finite witness against the teacher label. One case at a time.
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S11=$T/xcase_src_11; S16=$T/xcase_src_16
for c in "$@"; do
  [ -f $T/COVER_INPUTS/$c/RESULT.json ] || bash $S11/run_cover_blocks.sh $c > $T/COVER_INPUTS/$c.log 2>&1
  [ -f $T/COVER_INPUTS/$c/RESULT.json ] || { echo "$(date -u +%FT%TZ) $c blocks_failed" >> $T/COVER_INPUTS/e2e.status; continue; }
  ls /root/autodl-tmp/CUTFEM_FRESH_GP_20260921/targets/${c}_v1/REFERENCE_RQ.npy > /dev/null 2>&1 || { echo "$(date -u +%FT%TZ) $c no_target" >> $T/COVER_INPUTS/e2e.status; continue; }
  (cd $S16 && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /root/cutfem_neural_a_20260910/env/bin/python -u encode_gpu.py --case $c --cover-inputs $T/COVER_INPUTS/$c --elements polyref --s 4 --levels 1 --cpu-trace --output $T/ENCODE_GPU_01/${c}_polyref_L1 > $T/ENCODE_GPU_01/${c}_polyref_L1.log 2>&1)
  echo "$(date -u +%FT%TZ) $c encode exit $?" >> $T/COVER_INPUTS/e2e.status
done
echo "$(date -u +%FT%TZ) E2E_BATCH_FINISHED" >> $T/COVER_INPUTS/e2e.status
