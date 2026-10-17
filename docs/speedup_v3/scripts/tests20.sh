#!/bin/bash
# tests20.sh: S26 with stage buffers (streamed cells' states copied ahead on the copy stream into 4 fixed device
# buffers; the graphs decode and compute): the 51-cell plate on two (v3z) and four (v3z4) streams, with S27 if
# XENV says so; hlat222 (v3z).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
OPL_V3_PIPE_BODIES=0 TMO=1500 bash $S/plate.sh v3z 51 OPL_V3_GRAPH=1 $XENV
OPL_V3_PIPE_BODIES=0 TMO=1500 bash $S/plate.sh v3z4 51 OPL_V3_GRAPH=1 OPL_V3_STREAMS=4 $XENV
TMO=900 bash $S/t5.sh v3z hlat222 OPL_V3_GRAPH=1 $XENV
echo "$(date +%F_%T) TESTS20_DONE" >> $W/status.txt
