#!/bin/bash
# tests37.sh: P15 (OPL_V3_KMF_GK): ghost faces in the Kronecker form (Gram = gamma A (x) m (x) m): t_gk.py (accuracy,
# time against the current kernels, float32 and float64), S_hat with it on (P13 + P14 now default on); 24-cell plate
# with the plain defaults (v3U: P13 now on by default, no switches set) and with GK.
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; SP=$W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh gk t_gk.py $SP
bash $S/tst.sh sw_gk t_switch.py $SP --sw kmf.GK
bash $S/plate.sh v3U 24
bash $S/plate.sh v3U_gk 24 OPL_V3_KMF_GK=1
echo "$(date +%F_%T) TESTS37_DONE" >> $W/status.txt
