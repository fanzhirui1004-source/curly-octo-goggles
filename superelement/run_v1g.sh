#!/bin/bash
# V1G: probe-free objective (exact log-det divergence per mode + extreme whitened modes), warm-started from V1E.
# Usage: run_v1g.sh <output_dir> <steps> [extra args]
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/cutfem_neural_a_20260910/superelement_v0
OUT=$1; STEPS=$2; shift 2
/root/cutfem_neural_a_20260910/env/bin/python v1_superelement.py --labels /root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json --output $OUT --steps $STEPS --chunk 50 --eval-every 3000 --eval-labels 3 --eval-max-q 21500 --train-max-q 23500 --rank 512 --width 64 --hidden 128 \
  --energy-weight 0 --action-weight 0 --dual-weight 0 --compliance-weight 0 --divergence-weight 1 --extreme-weight 0.1 --extreme-block 8 --extreme-iters 2 --trace-dtype float32 \
  --off-scale 10 --aug-prob 0.5 --volume-width 0 --lr 1e-3 --warmup 100 --lr-floor 0.1 "$@" > $(basename $OUT).log 2>&1
