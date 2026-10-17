#!/bin/bash
# tests9.sh: plates of 162 and 190 cells with the host-memory budget (S14), then hlat554 (Table 5 route, shared operators).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
OPL_V3_PIPE_BODIES=0 bash $S/scale.sh v3g "162 190"
bash $S/t5.sh v3g hlat554
echo "$(date +%F_%T) TESTS9_DONE" >> $W/status.txt
