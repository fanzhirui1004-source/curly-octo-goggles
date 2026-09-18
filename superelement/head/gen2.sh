#!/bin/bash
# Generalisation arm, resized so the presented labels stay in the container page cache.
PY=/root/cutfem_neural_a_20260910/env/bin/python
G=/root/autodl-tmp/CLAUDE_GEN_20260918
L=/root/autodl-tmp/CLAUDE_LABELS_20260917
D=/root/autodl-tmp/CLAUDE_SQRTHEAD_20260917

SEATS=$(grep '^SEATS=' "$G/plan_ram.txt" | cut -d= -f2-)
TRAIN=$(grep '^TRAIN_SEATS=' "$G/plan_ram.txt" | cut -d= -f2-)
EVAL=$(grep '^EVAL_SEATS=' "$G/plan_ram.txt" | cut -d= -f2-)
SHA=$(find "$D/src_v3/stage_cutfem_m4" -name '*.py' | sort | xargs sha256sum | sha256sum | cut -c1-16)
echo "GEN2 prepared $(echo $SEATS | wc -w) presented $(echo $TRAIN | wc -w) eval $(echo $EVAL | wc -w) sha $SHA"

cd "$D/src_v3" || exit 1
rm -rf "$G/GEN_CHOL_RAM"
$PY -u -m stage_cutfem_m4.run \
  --manifest "$L/V2_LABELS.json" \
  --seats $SEATS --train-seats $TRAIN --eval-seats $EVAL \
  --steps 60000 --warmup 1000 --pairs-per-bucket 8192 --calibrate-pairs 4096 \
  --head chol --diagonal-loss absolute --seed 20260917 --source-sha "$SHA" \
  --output "$G/GEN_CHOL_RAM"
echo "GEN2 rc=$? $(date -Is)"
