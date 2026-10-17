#!/bin/bash
# tests26.sh: P5, the next cell's setup in a worker thread (OPL_V3_FE_THREAD=1) on the 51-cell plate (v3P5; compare with
# tests25's v3P2: same code otherwise) and the 110-cell plate (v3P5 vs v3F).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
OPL_V3_PIPE_BODIES=0 TMO=1500 bash $S/plate.sh v3P5 51 OPL_V3_FE_THREAD=1
OPL_V3_PIPE_BODIES=0 TMO=2400 bash $S/plate.sh v3P5 110 OPL_V3_FE_THREAD=1
echo "$(date +%F_%T) TESTS26_DONE" >> $W/status.txt
