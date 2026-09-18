#!/bin/bash
PY=/root/cutfem_neural_a_20260910/env/bin/python
D=/root/autodl-tmp/CLAUDE_SQRTHEAD_20260917
LP=/root/autodl-tmp/CLAUDE_LOGPIVOT_20260918
R5=/root/autodl-tmp/CLAUDE_ROUTE5_20260918
SM=/root/autodl-tmp/CLAUDE_SENSMODEL_20260918

echo "=== route 5 spectrum ==="
rm -rf "$R5/EVAL"
$PY -u "$R5/eval_inverse.py" --source "$D/src_v4" --run "$R5/S1_INVERSE" --output "$R5/EVAL"
echo "route5 eval rc=$?"

echo "=== sqrt+log assembled physics ==="
rm -rf "$SM/LOGPIVOT_SQRT"
$PY -u "$SM/sens_model.py" --seat 253 \
  --chol-factor /nonexistent.npy \
  --sqrt-factor "$LP/S1_LOGPIVOT_SQRT/EVAL_0253/M_PRED_UPPER.npy" \
  --output "$SM/LOGPIVOT_SQRT"
echo "sens sqrt+log rc=$?"
