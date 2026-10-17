#!/bin/bash
# tests44.sh: P18 (the hyperedge layer in one CUDA kernel, hyperfused.layer -> hyperk.layer, used by fastnet with
# OPL_V3_HFUSE=1): t_hl.py (against P16 gather + cuBLAS mix + P16 scatter, per layer type, B = 1, 2, 4, 16), S_hat with
# it on (B = 1, 3, 16).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; SP=$W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh hl t_hl.py $SP
bash $S/tst.sh sw_hl t_switch.py $SP --sw fastnet.HFUSE
echo "$(date +%F_%T) TESTS44_DONE" >> $W/status.txt
