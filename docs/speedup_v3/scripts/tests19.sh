#!/bin/bash
# tests19.sh: S26 again after the eligibility fix (tests17's v3u / v3v captured no graph: PCG runs with autograd
# enabled): the 51-cell plate with graphs on two (v3u2) and four (v3w) streams; S27 (transposed convolutions as forward
# ones) checked per convolution and on S_hat (t_convt.py); graphs + S27 on the plate (v3y) and on hlat222 (v3y).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
OPL_V3_PIPE_BODIES=0 TMO=1500 bash $S/plate.sh v3u2 51 OPL_V3_GRAPH=1
OPL_V3_PIPE_BODIES=0 TMO=1500 bash $S/plate.sh v3w 51 OPL_V3_GRAPH=1 OPL_V3_STREAMS=4
bash $S/tst.sh convt t_convt.py $W/plate/body_S24:plateS24_040_o000
OPL_V3_PIPE_BODIES=0 TMO=1500 bash $S/plate.sh v3y 51 OPL_V3_GRAPH=1 OPL_V3_CONVT=1
TMO=900 bash $S/t5.sh v3y hlat222 OPL_V3_GRAPH=1 OPL_V3_CONVT=1
echo "$(date +%F_%T) TESTS19_DONE" >> $W/status.txt
