#!/bin/bash
# tests50.sh: items 1b / 1c (moments and ghost-face DOF arrays dropped after the setup; int32 hyperedge node lists, one
# copy of the fringe subsets, shared transfer indices): t_drop.py on a full cell of the 24-cell plate.
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; P=/root/autodl-tmp/OPL; SC=$P/S1/V2/R1/SCALE
PK=$P/S4/packets:$P/S3/packets:$SC/plate_packets:$W/plate_packets
TST_ENV="OPL_PACKETS_EXTRA=$W/plate/v3U_S24/packets:$PK" bash $S/tst.sh t50_drop t_drop.py $W/plate/body_S24:plateS24_000_o000
echo "$(date +%F_%T) TESTS50_DONE" >> $W/status.txt
