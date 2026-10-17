#!/bin/bash
# tests40.sh: P16 (OPL_V3_HCUDA) with the gather only up to B F = 128 (Triton's beyond) and a fourth scatter head-sum
# order (butterfly); S_hat with it on; the 32-column products of a B = 16 S_hat matrix-free too (kmf.MAXB=32, P16 on);
# 51-cell plate with P16 on, resident budget 4 (as before), 12 and auto (S23): PCG is near the host-to-device limit.
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; SP=$W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh hyperk3 t_hyperk.py $SP --B 1,2,4,16
bash $S/tst.sh sw_hk3 t_switch.py $SP --sw hyperk.ON
TST_ENV="OPL_V3_HCUDA=1" bash $S/tst.sh sw_maxb32 t_switch.py $SP --sw kmf.MAXB=32
bash $S/plate.sh v3W_hk 51 OPL_V3_HCUDA=1
RGB=12 bash $S/plate.sh v3W_rgb12 51 OPL_V3_HCUDA=1
RGB=auto bash $S/plate.sh v3W_rgba 51 OPL_V3_HCUDA=1
echo "$(date +%F_%T) TESTS40_DONE" >> $W/status.txt
