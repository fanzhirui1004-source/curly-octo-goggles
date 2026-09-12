#!/bin/bash
# Back-end capacity test: free band coefficients + coarse-space compliance G = exp(H) (M = L Pq G^1/2), single label, no network.
# Usage: run_free_coarse.sh <seat> <output_dir> <steps> <init_checkpoint_or_none> [extra args]
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/cutfem_neural_a_20260910/superelement_v0
SEAT=$1; OUT=$2; STEPS=$3; INIT=$4; shift 4
INITARG=""; if [ "$INIT" != "none" ]; then INITARG="--init-checkpoint $INIT"; fi
/root/cutfem_neural_a_20260910/env/bin/python v1_superelement.py --labels /root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json --seats $SEAT --model free_coarse --aug-prob 0 --output $OUT --steps $STEPS --chunk 50 --eval-every 1000 --eval-labels 1 --eval-max-q 23000 \
  --rank 512 --width 64 --hidden 128 --volume-width 0 --off-scale 10 --energy-weight 0 --action-weight 0 --dual-weight 0 --compliance-weight 0 --divergence-weight 1 --extreme-weight 0.1 --extreme-block 8 --extreme-iters 2 --trace-dtype float32 \
  --coarse-stride 4 --coarse-prune 0.05 --coarse-init teacher --lr 3e-3 --warmup 50 --lr-floor 0.1 $INITARG "$@" > $(basename $OUT).log 2>&1
