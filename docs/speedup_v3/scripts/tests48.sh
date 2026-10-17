#!/bin/bash
# tests48.sh: the weight store without the duplicated float64 view (stream_ops copies a view of other shape again):
# 51-cell plate with all current defaults, resident budget 4 and 0 (streamed bytes, time, memory, compliance).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
bash $S/plate.sh v3AC 51
RGB=0 bash $S/plate.sh v3AC_rg0 51
echo "$(date +%F_%T) TESTS48_DONE" >> $W/status.txt
