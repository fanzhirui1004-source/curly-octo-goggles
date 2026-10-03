#!/bin/bash
# Route 3's symmetric head on route 5's target: M = A^{-1/2}, A_hat^-1 = M M.
# Equivariant, PSD by construction, soft-weighted. Same seat/steps/seed/loss as the
# other arms so the comparison is target-only.
PY=/root/cutfem_neural_a_20260910/env/bin/python
D=/root/autodl-tmp/CLAUDE_SQRTHEAD_20260917
O=/root/autodl-tmp/CLAUDE_ROUTE5_20260918
SHA=$(find "$D/src_v5/stage_cutfem_m4" -name '*.py' | sort | xargs sha256sum | sha256sum | cut -c1-16)
cd "$D/src_v5" || exit 1
rm -rf "$O/S1_INVSQRT"
$PY -u -m stage_cutfem_m4.run \
  --seats 253 --steps 20000 --warmup 16 --pairs-per-bucket 8192 \
  --head invsqrt --diagonal-loss log --eval-seats \
  --seed 20260917 --source-sha "$SHA" --output "$O/S1_INVSQRT"
echo "INVSQRT train rc=$? $(date -Is)"
rm -rf "$O/EVAL_INVSQRT"
$PY -u "$O/eval_inverse.py" --source "$D/src_v5" --run "$O/S1_INVSQRT" --output "$O/EVAL_INVSQRT" --symmetric
echo "INVSQRT eval rc=$? $(date -Is)"
