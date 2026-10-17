#!/bin/bash
# tests7.sh: concatenated ghost-face product, cuDNN benchmark mode.
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; SPEC=$W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh gfcat t_gfcat.py $SPEC
bash $S/tst.sh cudnn t_cudnn.py $SPEC
echo "$(date +%F_%T) TESTS7_DONE" >> $W/status.txt
