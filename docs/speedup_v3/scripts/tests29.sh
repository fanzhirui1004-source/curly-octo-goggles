#!/bin/bash
# tests29.sh: S_hat kernel time attributed to call sites (t_kprof2.py, B = 1, 16) on a full plate cell.
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
bash $S/tst.sh kprof2 t_kprof2.py $W/plate/body_S24:plateS24_040_o000
echo "$(date +%F_%T) TESTS29_DONE" >> $W/status.txt
