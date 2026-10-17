#!/bin/bash
# tests17.sh: CUDA graphs (S26). t_graph.py again (one capture stream per replay stream; watchdog), then the 51-cell plate
# with OPL_V3_GRAPH=1 on two (v3u) and three (v3v) compute streams, and hlat222 with graphs (v3u).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
bash $S/tst.sh graph3 t_graph.py $W/plate/body_S24:plateS24_040_o000,plateS24_060_o000,plateS24_150_o000,plateS24_160_o000
OPL_V3_PIPE_BODIES=0 bash $S/plate.sh v3u 51 OPL_V3_GRAPH=1
OPL_V3_PIPE_BODIES=0 bash $S/plate.sh v3v 51 OPL_V3_GRAPH=1 OPL_V3_STREAMS=3
bash $S/t5.sh v3u hlat222 OPL_V3_GRAPH=1
echo "$(date +%F_%T) TESTS17_DONE" >> $W/status.txt
