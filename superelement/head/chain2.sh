#!/bin/bash
# Serial GPU queue: re-evaluate the sqrt arm, measure tau smoothness, expand labels.
PY=/root/cutfem_neural_a_20260910/env/bin/python
D=/root/autodl-tmp/CLAUDE_SQRTHEAD_20260917
TA=/root/autodl-tmp/CLAUDE_TAU_20260917
L=/root/autodl-tmp/CLAUDE_LABELS_20260917

echo "CHAIN2 start $(date -Is)"

rm -rf "$D/AB/S1_sqrt_EVAL"
"$PY" -u "$D/reeval.py" --source "$D/src_rebuild" --run "$D/AB/S1_sqrt" \
  --output "$D/AB/S1_sqrt_EVAL" > "$D/AB/reeval_sqrt.log" 2>&1
echo "CHAIN2 reeval_sqrt rc=$? $(date -Is)"

rm -rf "$TA/GATE"
"$PY" -u "$TA/tau_gate.py" --output "$TA/GATE" > "$TA/tau_gate.log" 2>&1
echo "CHAIN2 tau_gate rc=$? $(date -Is)"

"$PY" -u "$L/expand_labels.py" \
  --census /root/autodl-tmp/CLAUDE_TRACE_20260917/CENSUS.json \
  --existing /root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json \
  --output "$L/BATCH1" --max-q 20000 > "$L/expand.log" 2>&1
echo "CHAIN2 expand rc=$? $(date -Is)"

"$PY" "$L/merge_manifest.py" \
  --frozen /root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json \
  --new "$L/BATCH1/NEW_LABELS.json" --output "$L/V2_LABELS.json" >> "$L/expand.log" 2>&1
echo "CHAIN2 merge rc=$? $(date -Is)"
echo "CHAIN2 done $(date -Is)"
