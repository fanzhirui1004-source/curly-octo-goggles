#!/bin/bash
# tests29.sh: the float32 element product's kernel choice timed inside S_hat graph replays (t_symv5.py); S_hat kernel time attributed to call sites (t_kprof2.py, B = 1, 16) on a full plate cell; the U-Net
# convolutions' layout / cuDNN algorithm choices (t_conv2.py); P8 hyperedge layer in one kernel (t_hfuse2.py); P9 ghost-face product in one
# kernel per template (t_switch.py --gf), P8 + P9 together; then the
# 51-cell plate with device-level profiles of PCG iteration 40 (the census instruments matvecs 20-22 eagerly on one
# stream) and of front-end cells 20..22. (P2b dropped: t_kdedupe found no two full elements with equal float32 rows.)
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
bash $S/tst.sh symv5 t_symv5.py $W/plate/body_S24:plateS24_040_o000,plateS24_060_o000,plateS24_150_o000
bash $S/tst.sh kprof2 t_kprof2.py $W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh conv2 t_conv2.py $W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh hfuse2 t_hfuse2.py $W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh sw_gf t_switch.py $W/plate/body_S24:plateS24_040_o000 --sw gffused.ON --gf
bash $S/tst.sh sw_both t_switch.py $W/plate/body_S24:plateS24_040_o000 --sw fastnet.HFUSE,gffused.ON
mkdir -p $W/plate/v3Q2_S51
OPL_V3_PIPE_BODIES=0 TMO=1500 bash $S/plate.sh v3Q2 51 OPL_V3_PROF_ITER=40 OPL_V3_PROF_OUT=$W/plate/v3Q2_S51/prof40.json OPL_V3_PROF_TRACE=1 OPL_V3_PROF_FE=20 OPL_V3_PROF_FE_OUT=$W/plate/v3Q2_S51/prof_fe20.json
echo "$(date +%F_%T) TESTS29_DONE" >> $W/status.txt
