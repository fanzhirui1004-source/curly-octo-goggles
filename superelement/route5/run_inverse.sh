#!/bin/bash
# Route 5: train the existing upper-triangular head on the Cholesky factor of A^-1.
# --eval-seats with no values disables the built-in evaluation: the inverse head needs
# a different spectrum route, so it is evaluated by eval_inverse.py from the checkpoint.
PY=/root/cutfem_neural_a_20260910/env/bin/python
D=/root/autodl-tmp/CLAUDE_SQRTHEAD_20260917
O=/root/autodl-tmp/CLAUDE_ROUTE5_20260918
SHA=$(find "$D/src_v4/stage_cutfem_m4" -name '*.py' | sort | xargs sha256sum | sha256sum | cut -c1-16)
echo "ROUTE5 source sha $SHA"
cd "$D/src_v4" || exit 1
rm -rf "$O/S1_INVERSE"
$PY -u -m stage_cutfem_m4.run \
  --seats 253 --steps 20000 --warmup 16 --pairs-per-bucket 8192 \
  --head inverse --diagonal-loss log --eval-seats \
  --seed 20260917 --source-sha "$SHA" --output "$O/S1_INVERSE"
echo "ROUTE5 rc=$? $(date -Is)"
