#!/bin/bash
# V1 multi-label training. Usage: run_v1.sh <output_dir> <steps> [extra args]
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/cutfem_neural_a_20260910/superelement_v0
OUT=$1; STEPS=$2; shift 2
/root/cutfem_neural_a_20260910/env/bin/python v1_superelement.py --labels /root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json --output $OUT --steps $STEPS --chunk 50 --eval-every 3000 --eval-labels 3 --eval-max-q 23000 --rank 512 --width 64 --hidden 128 --probes 32 --white-probes 32 --action-probes 16 --action-weight 3 --dual-weight 1 --dual-probes 16 --lr 1e-3 --warmup 300 --lr-floor 0.1 "$@" > $(basename $OUT).log 2>&1
