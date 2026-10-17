#!/bin/bash
# tests30.sh: S_hat's matrix-product call sites (t_gemmsites.py); P11 (where-free clamped layers): the port-row invariant
# at every layer and the forward map off / on (t_clampfree.py), S_hat off / on (t_switch.py).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; SP=$W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh gemmsites t_gemmsites.py $SP
bash $S/tst.sh clampfree t_clampfree.py $SP
bash $S/tst.sh sw_clampfree t_switch.py $SP --sw fastnet.CLAMPFREE
echo "$(date +%F_%T) TESTS30_DONE" >> $W/status.txt
