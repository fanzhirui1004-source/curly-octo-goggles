#!/bin/bash
# tests45.sh: P15b (gkw_kernel: Kronecker ghost faces with lanes over the columns, from 8 columns on; OPL_V3_KMF_GKW):
# t_gk.py (B = 1..32, nb 0: gkw from 8 columns, nb 1: the 15-thread gk_kernel), S_hat with gkw off (--sw kmf.GKW=0:
# the "on" run is the old kernel).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; SP=$W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh gk2 t_gk.py $SP
bash $S/tst.sh sw_gkw t_switch.py $SP --sw kmf.GKW=0
echo "$(date +%F_%T) TESTS45_DONE" >> $W/status.txt
