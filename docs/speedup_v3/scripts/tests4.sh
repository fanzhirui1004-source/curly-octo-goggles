#!/bin/bash
# tests4.sh: sensitivity-phase profile and the front-end profile after the pinned-buffer / constructor changes.
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; B24=$W/plate/body_S24
bash $S/tst.sh sens t_sens.py $B24:plateS24_040_o000
bash $S/tst.sh prep2 t_prep.py $B24 plateS24_040_o000,plateS24_030_o000,plateS24_050_o000,plateS24_060_o000
echo "$(date +%F_%T) TESTS4_DONE" >> $W/status.txt
