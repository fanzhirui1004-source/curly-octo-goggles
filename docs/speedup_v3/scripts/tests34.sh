#!/bin/bash
# tests34.sh: P13 / P14 again with the library building (tests33's runs with OPL_V3_KMF=1 fell back to the Triton form:
# the multi-column version did not compile). P13 float32 kernels (t_kmf.py, S_hat sw_kmf5), P14 float64 product
# (t_kmf64.py: against the stored-matrix product; S_hat sw_kmf64 on top of P13), full runs: Table 5 route on hlat222
# and the 51-cell plate with P13, and with P13 + P14.
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; SP=$W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh kmf3 t_kmf.py $SP
bash $S/tst.sh kmf64 t_kmf64.py $SP
bash $S/tst.sh sw_kmf5 t_switch.py $SP --sw kmf.ON
TST_ENV="OPL_V3_KMF=1" bash $S/tst.sh sw_kmf64 t_switch.py $SP --sw kmf.ON64
bash $S/t5.sh v3S_kmf_t5b hlat222 OPL_V3_KMF=1
bash $S/t5.sh v3S_kmf64_t5 hlat222 OPL_V3_KMF=1 OPL_V3_KMF64=1
bash $S/plate.sh v3S_kmfb 51 OPL_V3_KMF=1
bash $S/plate.sh v3S_kmf64 51 OPL_V3_KMF=1 OPL_V3_KMF64=1
echo "$(date +%F_%T) TESTS34_DONE" >> $W/status.txt
