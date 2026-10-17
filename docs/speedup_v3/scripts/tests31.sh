#!/bin/bash
# tests31.sh: P9b two-phase ghost-face product (no atomics): the product alone (t_switch.py --gf) and S_hat with it.
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; SP=$W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh sw_gf3 t_switch.py $SP --sw gffused.MODE=2 --gf
echo "$(date +%F_%T) TESTS31_DONE" >> $W/status.txt
