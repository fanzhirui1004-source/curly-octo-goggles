#!/bin/bash
# tests51.sh: item 2c micro test (coarse Z^T / T^T without copies, 24-cell plate); the 51-cell plate (resident budget 0)
# with the new defaults of items 1a-1c, 2a and the graph change (v3AD); the same with the float32 K_PP factor (item 2b,
# category B).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; P=/root/autodl-tmp/OPL; SC=$P/S1/V2/R1/SCALE
PK=$P/S4/packets:$P/S3/packets:$SC/plate_packets:$W/plate_packets
TST_ENV="OPL_PACKETS_EXTRA=$W/plate/v3U_S24/packets:$PK" bash $S/tst.sh t51_coarseT t_coarseT.py $W/plate/body_S24 $SC/layouts/plateS24.json
RGB=0 bash $S/plate.sh v3AD_rg0 51
RGB=0 bash $S/plate.sh v3AD_kf32_rg0 51 OPL_V3_KPP_FP32=1
echo "$(date +%F_%T) TESTS51_DONE" >> $W/status.txt
