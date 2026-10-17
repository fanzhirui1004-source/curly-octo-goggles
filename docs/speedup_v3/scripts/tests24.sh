#!/bin/bash
# tests24.sh: P2 micro-benchmark of the float32 element product variants (t_symv.py).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
bash $S/tst.sh symv t_symv.py $W/plate/body_S24:plateS24_040_o000
echo "$(date +%F_%T) TESTS24_DONE" >> $W/status.txt
