#!/bin/bash
# tests56.sh: sensitivity row-batch study. t_sensd (full and cut cells of the 24-cell plate: row batches 512-8192, the
# tetrahedron VJP as before and row-indexed (OPL_V3_SENS_ROWS): time in interleaved rounds, device peak, kernel launches,
# host synchronisations, device time, forward / backward split, bitwise check); the 51-cell plate (rg0, as v3AF_rg0 with
# batch 1024) with the VJP as before at batch 2048 and row-indexed at batches 2048 and 4096.
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; P=/root/autodl-tmp/OPL; SC=$P/S1/V2/R1/SCALE
PK=$P/S4/packets:$P/S3/packets:$SC/plate_packets:$W/plate_packets
TST_ENV="OPL_PACKETS_EXTRA=$W/plate/v3U_S24/packets:$PK" bash $S/tst.sh t56_sensd t_sensd.py $W/plate/body_S24 plateS24_000_o000 plateS24_060_o000 plateS24_070_o000
ADB=2048 RGB=0 bash $S/plate.sh v3AG_o2048_rg0 51
ADB=2048 RGB=0 bash $S/plate.sh v3AG_r2048_rg0 51 OPL_V3_SENS_ROWS=1
ADB=4096 RGB=0 bash $S/plate.sh v3AG_r4096_rg0 51 OPL_V3_SENS_ROWS=1
echo "$(date +%F_%T) TESTS56_DONE" >> $W/status.txt
