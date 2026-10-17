#!/bin/bash
# tests47.sh: the shared float64 weight store (kmf.wstore) and the parked ghost-face DOF arrays (GhostFaces.park):
# 51-cell plate with all current defaults, resident budget 4 and 0 (streamed bytes, time, memory, compliance).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
bash $S/plate.sh v3AB 51
RGB=0 bash $S/plate.sh v3AB_rg0 51
echo "$(date +%F_%T) TESTS47_DONE" >> $W/status.txt
