#!/bin/bash
# tests3.sh: the 51-cell plate with the new defaults (compare with v3a_S51), and with the body generation overlapped with
# the front end on fresh bodies (OPL_V3_PIPE_BODIES=1; bodies_s + analysis vs the overlapped wall time).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
bash $S/plate.sh v3b 51
BODYDIR=$W/plate/body_S51_pipe bash $S/plate.sh v3bp 51 OPL_V3_PIPE_BODIES=1
BODYDIR=$W/plate/body_S51_nopipe bash $S/plate.sh v3bn 51
echo "$(date +%F_%T) TESTS3_DONE" >> $W/status.txt
