#!/bin/bash
# tests30.sh: S_hat's matrix-product call sites (t_gemmsites.py); the element product in situ vs alone, back to back
# vs spaced (clocks; t_symv6.py); P11 (where-free clamped layers): the port-row invariant
# at every layer and the forward map off / on (t_clampfree.py), S_hat off / on (t_switch.py); P10b sparse fused
# convolution (t_conv2.py per convolution, t_switch.py on S_hat).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; SP=$W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh gemmsites t_gemmsites.py $SP
bash $S/tst.sh fesites t_fesites.py $W/plate/body_S24:plateS24_060_o000,plateS24_040_o000
bash $S/tst.sh symv6 t_symv6.py $SP
bash $S/tst.sh color t_color.py $SP
bash $S/tst.sh clampfree t_clampfree.py $SP
bash $S/tst.sh sw_clampfree t_switch.py $SP --sw fastnet.CLAMPFREE
bash $S/tst.sh hfuse2 t_hfuse2.py $SP
bash $S/tst.sh sw_gf2 t_switch.py $SP --sw gffused.ON --gf
bash $S/tst.sh conv2s t_conv2.py $SP
bash $S/tst.sh sw_conv3s t_switch.py $SP --sw fastnet.CONV3
echo "$(date +%F_%T) TESTS30_DONE" >> $W/status.txt
