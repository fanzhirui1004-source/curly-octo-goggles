#!/bin/bash
# tests38.sh: P16 (OPL_V3_HCUDA): fastnet's hyperedge gather / scatter as CUDA kernels: t_hyperk.py (bitwise /
# rounding against the Triton kernels, time per layer type, B = 1..4), S_hat with it on; then the 51-cell plate with
# the plain defaults (v3V: P13, P14, P15 on).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; SP=$W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh hyperk t_hyperk.py $SP
bash $S/tst.sh sw_hk t_switch.py $SP --sw hyperk.ON
bash $S/plate.sh v3V 51
echo "$(date +%F_%T) TESTS38_DONE" >> $W/status.txt
