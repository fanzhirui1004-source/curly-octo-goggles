#!/bin/bash
# tests10b.sh: the 51-cell plate's solve phase was 5% slower with S14b-S18 + S15 (v3h) than v3f although the per-cell
# S_hat is not slower (t_cblk2: two-stream throughput 5% faster with S15): repeat v3h, then S15 off, then S17 off.
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
OPL_V3_PIPE_BODIES=0 bash $S/plate.sh v3h2 51 OPL_COARSE_BLK=1
OPL_V3_PIPE_BODIES=0 bash $S/plate.sh v3j 51 OPL_COARSE_BLK=0
OPL_V3_PIPE_BODIES=0 bash $S/plate.sh v3k 51 OPL_COARSE_BLK=1 OPL_V3_GC=full
echo "$(date +%F_%T) TESTS10B_DONE" >> $W/status.txt
