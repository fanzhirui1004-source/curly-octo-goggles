#!/bin/bash
# tests18.sh: capacity with every change: S25 (cells spilled to disk around the K_PP planning peak, no budget reserve),
# the automatic resident budget (S23), and $XENV (e.g. OPL_V3_GRAPH=1 if S26 is adopted). Generated plates of about 256
# and 336 cells (gen_hlat S = 14, 16; bodies of the stopped v3i 256-cell run reused).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
export OPL_V3_SPILL=1 RGB=auto
for kv in $XENV; do export "$kv"; done
OPL_V3_PIPE_BODIES=0 bash $S/scale.sh ${TAG:-v3x} "${SIZES:-256 336}"
echo "$(date +%F_%T) TESTS18_DONE" >> $W/status.txt
