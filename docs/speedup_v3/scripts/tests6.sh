#!/bin/bash
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
bash $S/tst.sh sens2 t_sens2.py $W/plate/body_S24:plateS24_040_o000
echo "$(date +%F_%T) TESTS6_DONE" >> $W/status.txt
