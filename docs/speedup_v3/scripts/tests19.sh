#!/bin/bash
# tests19.sh: S27 (transposed convolutions as forward ones) checked per convolution and on S_hat (t_convt.py); then the
# 51-cell plate with graphs on four streams (v3w) and with graphs + S27 (v3y), and hlat222 with graphs + S27 (v3y).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
bash $S/tst.sh convt t_convt.py $W/plate/body_S24:plateS24_040_o000
OPL_V3_PIPE_BODIES=0 TMO=1500 bash $S/plate.sh v3w 51 OPL_V3_GRAPH=1 OPL_V3_STREAMS=4
OPL_V3_PIPE_BODIES=0 TMO=1500 bash $S/plate.sh v3y 51 OPL_V3_GRAPH=1 OPL_V3_CONVT=1
TMO=900 bash $S/t5.sh v3y hlat222 OPL_V3_GRAPH=1 OPL_V3_CONVT=1
echo "$(date +%F_%T) TESTS19_DONE" >> $W/status.txt
