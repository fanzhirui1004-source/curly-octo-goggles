#!/bin/bash
# tests3.sh: the 51-cell plate with the new defaults (compare with v3a_S51), with two compute streams, and with fresh
# bodies generated before the analysis (as before) vs overlapped with the front end (OPL_V3_PIPE_BODIES=1).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
bash $S/plate.sh v3b 51
bash $S/plate.sh v3bs 51 OPL_V3_STREAMS=2
BODYDIR=$W/plate/body_S51_nopipe bash $S/plate.sh v3bn 51
BODYDIR=$W/plate/body_S51_pipe bash $S/plate.sh v3bp 51 OPL_V3_PIPE_BODIES=1
echo "$(date +%F_%T) TESTS3_DONE" >> $W/status.txt
