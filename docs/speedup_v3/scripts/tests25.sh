#!/bin/bash
# tests25.sh: P2 on S_hat (t_symv2.py), then the 51-cell plate and hlat222 with the retuned element product (v3P2).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
bash $S/tst.sh symv2 t_symv2.py $W/plate/body_S24:plateS24_040_o000
OPL_V3_PIPE_BODIES=0 TMO=1500 bash $S/plate.sh v3P2 51
TMO=900 bash $S/t5.sh v3P2 hlat222
echo "$(date +%F_%T) TESTS25_DONE" >> $W/status.txt
