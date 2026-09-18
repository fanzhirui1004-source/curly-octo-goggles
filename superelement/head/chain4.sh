#!/bin/bash
# After the chol generalisation arm: build the A^{1/2} labels, then the sqrt arm.
PY=/root/cutfem_neural_a_20260910/env/bin/python
G=/root/autodl-tmp/CLAUDE_GEN_20260918
L=/root/autodl-tmp/CLAUDE_LABELS_20260917
D=/root/autodl-tmp/CLAUDE_SQRTHEAD_20260917

while pgrep -f "stage_cutfem_m4.run .*--head chol .*GEN_CHOL" > /dev/null; do sleep 60; done
echo "CHAIN4 chol arm exited $(date -Is)"
sleep 30

$PY -u "$G/expand_sqrt_labels.py" --manifest "$L/V2_LABELS.json" \
  --builder "$D/build_sqrt_label.py" --output "$G/SQRT_LABELS" --min-free-gib 40 \
  > "$G/sqrt_labels.log" 2>&1
echo "CHAIN4 sqrt labels rc=$? $(date -Is)"

$PY "$G/restrict_to_sqrt.py" --plan "$G/SPLIT_PLAN.json" \
  --manifest "$L/V2_LABELS.json" --output "$G" > "$G/restrict.log" 2>&1
echo "CHAIN4 restrict rc=$? $(date -Is)"
cat "$G/restrict.log"

SEATS=$(grep '^SEATS=' "$G/plan_sqrt.txt" | cut -d= -f2-)
TRAIN=$(grep '^TRAIN_SEATS=' "$G/plan_sqrt.txt" | cut -d= -f2-)
EVAL=$(grep '^EVAL_SEATS=' "$G/plan_sqrt.txt" | cut -d= -f2-)
SHA=$(find "$D/src_rebuild/stage_cutfem_m4" -name '*.py' | sort | xargs sha256sum | sha256sum | cut -c1-16)
echo "CHAIN4 sqrt arm seats $(echo $SEATS | wc -w) presented $(echo $TRAIN | wc -w) eval $(echo $EVAL | wc -w)"

cd "$D/src_rebuild" || exit 1
rm -rf "$G/GEN_SQRT"
$PY -u -m stage_cutfem_m4.run \
  --manifest "$L/V2_LABELS.json" \
  --seats $SEATS --train-seats $TRAIN --eval-seats $EVAL \
  --steps 60000 --warmup 1000 --pairs-per-bucket 8192 --calibrate-pairs 4096 \
  --head sqrt --seed 20260917 --source-sha "$SHA" \
  --output "$G/GEN_SQRT"
echo "CHAIN4 sqrt arm rc=$? $(date -Is)"
echo "CHAIN4 done $(date -Is)"
