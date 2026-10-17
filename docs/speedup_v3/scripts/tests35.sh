#!/bin/bash
# tests35.sh: P13 / P14 with equispaced points (exact float32 tables) and the Lame constants folded into per-element
# weights (OPL_V3_KMF_PTS=equi, default): t_kmf.py (accuracy against float64, time); S_hat with the matrix-free products
# also for 16 columns (OPL_V3_KMF_MAXB=16 against 4, P13 + P14 on), the DOF-parallel face kernel (OPL_V3_KMF_GF2); full runs with P13 + P14: hlat222, 51-cell plate.
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; SP=$W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh kmf_equi t_kmf.py $SP
TST_ENV="OPL_V3_KMF=1 OPL_V3_KMF64=1" bash $S/tst.sh sw_maxb16 t_switch.py $SP --sw kmf.MAXB=16
TST_ENV="OPL_V3_KMF=1 OPL_V3_KMF64=1" bash $S/tst.sh sw_gf2k t_switch.py $SP --sw kmf.GF2
bash $S/t5.sh v3S_equi_t5 hlat222 OPL_V3_KMF=1 OPL_V3_KMF64=1
bash $S/plate.sh v3S_equi 51 OPL_V3_KMF=1 OPL_V3_KMF64=1
echo "$(date +%F_%T) TESTS35_DONE" >> $W/status.txt
