#!/bin/bash
# tests17.sh: CUDA graphs (S26). t_graph.py again (one capture stream per replay stream; watchdog), then, only if its
# multi-stream replays completed, the 51-cell plate with OPL_V3_GRAPH=1 on two (v3u) and three (v3v) compute streams and
# hlat222 with graphs (v3u); the runs time out after 25 min (a 51-cell plate takes ~4.5 min).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
bash $S/tst.sh graph3 t_graph.py $W/plate/body_S24:plateS24_040_o000,plateS24_060_o000,plateS24_150_o000,plateS24_160_o000
if grep -q '"streams_2"' $W/tests/graph3.log && ! grep -q '"hang"' $W/tests/graph3.log; then
  OPL_V3_PIPE_BODIES=0 TMO=1500 bash $S/plate.sh v3u 51 OPL_V3_GRAPH=1
  OPL_V3_PIPE_BODIES=0 TMO=1500 bash $S/plate.sh v3v 51 OPL_V3_GRAPH=1 OPL_V3_STREAMS=3
  TMO=900 bash $S/t5.sh v3u hlat222 OPL_V3_GRAPH=1
else
  echo "$(date +%F_%T) SKIP graph plates (graph3 did not complete its multi-stream replays)" >> $W/status.txt
fi
echo "$(date +%F_%T) TESTS17_DONE" >> $W/status.txt
