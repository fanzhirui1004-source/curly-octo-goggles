#!/bin/bash
# tests14.sh: checks of S21 (K_PP CSR released), S22 (coarse columns without V), S24 (fused network paths), the wide
# S_hat op profile; then hlat222 and the 51-cell plate with all of them (v3p), the category-B float32 K_PP factor (v3q,
# reported separately) and the automatic resident budget (v3r).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; SPEC=$W/plate/body_S24:plateS24_040_o000
GF=${GF:-OPL_V3_GFCAT=1}
bash $S/tst.sh kppfree t_kppfree.py
bash $S/tst.sh coarsecols t_coarsecols.py $W/plate/body_S24:plateS24_040_o000,plateS24_060_o000,plateS24_150_o000,plateS24_160_o000
bash $S/tst.sh netfuse t_netfuse.py $SPEC
TST_ENV="OPL_V3_NETFUSE=1 $GF" bash $S/tst.sh prof16 t_prof16.py $SPEC
ALL="$GF OPL_V3_NETFUSE=1 OPL_V3_KPP_FREEA=1 OPL_V3_COARSE_COLS=1"
bash $S/t5.sh v3p hlat222 $ALL
OPL_V3_PIPE_BODIES=0 bash $S/plate.sh v3p 51 $ALL
bash $S/t5.sh v3q hlat222 $ALL OPL_V3_KPP_FP32=1
OPL_V3_PIPE_BODIES=0 bash $S/plate.sh v3q 51 $ALL OPL_V3_KPP_FP32=1
OPL_V3_PIPE_BODIES=0 RGB=auto bash $S/plate.sh v3r 51 $ALL
echo "$(date +%F_%T) TESTS14_DONE" >> $W/status.txt
