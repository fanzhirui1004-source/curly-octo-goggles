#!/bin/bash
# tests54.sh: the 51-cell plate (resident budget 0) with the new defaults of items 2c / 3 (T^T a view, the allocator cache
# returned after PCG step 2, sensitivity pass without uncut tetrahedra, row batch 1024): v3AF; item 4: opt_design on the
# 24-cell plate, 4 iterations with --warm and Krylov subspace recycling (OPL_V3_PCG_RECYCLE=m,k,kmax, category B);
# which per-cell device tensors are still read after the setup (t_live.py), a full and a cut cell.
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; P=/root/autodl-tmp/OPL; SC=$P/S1/V2/R1/SCALE
PK=$P/S4/packets:$P/S3/packets:$SC/plate_packets:$W/plate_packets
RGB=0 bash $S/plate.sh v3AF_rg0 51
for RC in 40,8,16 60,12,24; do OPL_V3_PCG_RECYCLE=$RC bash $S/optrun.sh opt_S24_rc${RC//,/_} 24 4 --warm; done
TST_ENV="OPL_PACKETS_EXTRA=$W/plate/v3U_S24/packets:$PK" bash $S/tst.sh t54_live_full t_live.py $W/plate/body_S24:plateS24_000_o000
TST_ENV="OPL_PACKETS_EXTRA=$W/plate/v3U_S24/packets:$PK" bash $S/tst.sh t54_live_cut t_live.py $W/plate/body_S24:plateS24_070_o000
echo "$(date +%F_%T) TESTS54_DONE" >> $W/status.txt
