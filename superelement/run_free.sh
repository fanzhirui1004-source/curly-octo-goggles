#!/bin/bash
# Free-coefficient capacity test under the network's bounded maps, initialized from a network checkpoint's own prediction.
# Usage: run_free.sh <seat> <output_dir> <steps> <init_checkpoint> [extra args]
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/cutfem_neural_a_20260910/superelement_v0
SEAT=$1; OUT=$2; STEPS=$3; INIT=$4; shift 4
/root/cutfem_neural_a_20260910/env/bin/python v1_superelement.py --labels /root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json --seats $SEAT --model free --aug-prob 0 --output $OUT --steps $STEPS --chunk 50 --eval-every 1000 --eval-labels 1 --eval-max-q 23000 \
  --rank 512 --width 64 --hidden 128 --volume-width 0 --off-scale 10 --energy-weight 0 --action-weight 0 --dual-weight 0 --compliance-weight 0 --divergence-weight 1 --extreme-weight 0.1 --extreme-block 8 --extreme-iters 2 --trace-dtype float32 \
  --lr 3e-3 --warmup 50 --lr-floor 0.1 --init-checkpoint $INIT "$@" > $(basename $OUT).log 2>&1
