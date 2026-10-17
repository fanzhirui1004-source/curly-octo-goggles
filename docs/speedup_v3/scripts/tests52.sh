#!/bin/bash
# tests52.sh: item 3 (t_sensb: the sensitivity pass of a full plate cell: row batches, forward / backward split, two cells
# on two streams), item 2c (51-cell plate with T^T as a view of T, OPL_V3_COARSE_TT=0), item 4 (lat_scale on the 24-cell
# plate, 4 design steps of dtau = 0.0255 at fixed topology: preconditioner rebuilt every step vs built once (--reuse-prec);
# opt_design on the 24-cell plate, 4 MMA iterations with and without --warm).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; P=/root/autodl-tmp/OPL; SC=$P/S1/V2/R1/SCALE
PK=$P/S4/packets:$P/S3/packets:$SC/plate_packets:$W/plate_packets
TST_ENV="OPL_PACKETS_EXTRA=$W/plate/v3U_S24/packets:$PK" bash $S/tst.sh t52_sensb t_sensb.py $W/plate/body_S24:plateS24_000_o000
RGB=0 bash $S/plate.sh v3AE_tt_rg0 51 OPL_V3_COARSE_TT=0
mkdir -p $W/layouts
python3 -c "
import json; L=json.load(open('$SC/layouts/plateS24.json'))
for c in L['cells']: c['case'] = c['case'] + '_o000'
L['name'] = L['name'] + '_o000'; json.dump(L, open('$W/layouts/plateS24_o000.json', 'w'))"
for RP in 0 4; do bash $S/lsrun.sh ls_S24_rp$RP $W/layouts/plateS24_o000.json --iters 4 --dtau 0.0255 --reuse-prec $RP; done
for WARM in warm cold; do bash $S/optrun.sh opt_S24_$WARM 24 4 $([ $WARM = warm ] && echo --warm); done
echo "$(date +%F_%T) TESTS52_DONE" >> $W/status.txt
