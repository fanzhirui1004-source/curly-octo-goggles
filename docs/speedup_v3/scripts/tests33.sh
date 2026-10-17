#!/bin/bash
# tests33.sh: P13 with several columns per block (B = 1..4; OPL_V3_KMF_MAXB=4): t_kmf.py, S_hat in graph replay
# (t_switch.py); then full runs with the new defaults (P10b sparse convolutions, P11, P12) and with P13 (OPL_V3_KMF=1):
# Table 5 route on hlat222 (three load cases: B = 3), 51-cell plate (B = 1).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; SP=$W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh kmf2 t_kmf.py $SP
bash $S/tst.sh sw_kmf4 t_switch.py $SP --sw kmf.ON
bash $S/t5.sh v3S_t5 hlat222
bash $S/t5.sh v3S_kmf_t5 hlat222 OPL_V3_KMF=1
bash $S/plate.sh v3S 51
bash $S/plate.sh v3S_kmf 51 OPL_V3_KMF=1
echo "$(date +%F_%T) TESTS33_DONE" >> $W/status.txt
