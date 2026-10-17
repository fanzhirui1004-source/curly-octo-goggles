#!/bin/bash
# tests55.sh: item 4 (category B): opt_design on the 24-cell plate, 4 iterations with --warm and the preconditioner kept
# for 2 / 4 design iterations (--reuse-prec: applied on the matched DOFs, Jacobi on the others); cuDSS reordering
# algorithms on the 51-cell plate's K_PP (host peak of the plan, factor size, times; t_reorder.py, one process each);
# the 51-cell plate with the A Z blocks in float32 (category B), alone and with the float32 K_PP factor.
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; P=/root/autodl-tmp/OPL; SC=$P/S1/V2/R1/SCALE
PK=$P/S4/packets:$P/S3/packets:$SC/plate_packets:$W/plate_packets
for RP in 4 2; do bash $S/optrun.sh opt_S24_rp$RP 24 4 --warm --reuse-prec $RP; done
mkdir -p $W/tests/reorder
for ALG in DEFAULT NESTED_DISSECTION AMD; do
  TST_ENV="OPL_PACKETS_EXTRA=$W/plate/v3AC_S51/packets:$PK OPL_V3_CUDSS_REORDER=$ALG" bash $S/tst.sh t55_reorder_$ALG t_reorder.py $W/plate/body_S51 $SC/layouts/plateS51.json $W/tests/reorder/$ALG.npy
done
RGB=0 bash $S/plate.sh v3AF_azb32_rg0 51 OPL_V3_AZB_FP32=1
RGB=0 bash $S/plate.sh v3AF_gpu32_rg0 51 OPL_V3_AZB_FP32=1 OPL_V3_KPP_FP32=1
echo "$(date +%F_%T) TESTS55_DONE" >> $W/status.txt
