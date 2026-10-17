#!/bin/bash
# tests29.sh: (PCG iteration 40 profiled: the census instruments matvecs 20-22 eagerly on one stream) S_hat kernel time attributed to call sites (t_kprof2.py, B = 1, 16) on a full plate cell; the U-Net
# convolutions' layout / cuDNN algorithm choices (t_conv2.py); P8 hyperedge layer in one kernel (t_hfuse2.py); then P2b
# (OPL_V3_KMAP=1: full elements with a bitwise-equal float32 matrix read one row) on hlat222 and the 51-cell plate, the
# plate also with a device-level profile of front-end cells 20..22 (OPL_V3_PROF_FE=20).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
bash $S/tst.sh kprof2 t_kprof2.py $W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh conv2 t_conv2.py $W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh hfuse2 t_hfuse2.py $W/plate/body_S24:plateS24_040_o000
mkdir -p $W/plate/v3Q2_S51
OPL_V3_PIPE_BODIES=0 TMO=1500 bash $S/plate.sh v3Q2 51 OPL_V3_PROF_ITER=40 OPL_V3_PROF_OUT=$W/plate/v3Q2_S51/prof40.json OPL_V3_PROF_TRACE=1
TMO=900 bash $S/t5.sh v3K hlat222 OPL_V3_KMAP=1
mkdir -p $W/plate/v3K_S51
OPL_V3_PIPE_BODIES=0 TMO=1500 bash $S/plate.sh v3K 51 OPL_V3_KMAP=1 OPL_V3_PROF_FE=20 OPL_V3_PROF_FE_OUT=$W/plate/v3K_S51/prof_fe20.json OPL_V3_PROF_ITER=40 OPL_V3_PROF_OUT=$W/plate/v3K_S51/prof40.json
echo "$(date +%F_%T) TESTS29_DONE" >> $W/status.txt
