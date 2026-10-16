#!/bin/bash
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; B24=$W/plate/body_S24
bash $S/tst.sh sync t_sync.py $B24:plateS24_040_o000
echo "$(date +%F_%T) TESTS1B_DONE" >> $W/status.txt
