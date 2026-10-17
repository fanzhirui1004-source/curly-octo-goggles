#!/bin/bash
# tests49.sh: items 1a / 2a (per-cell K_PP triplets in kppstore files, K_PP assembled in row blocks straight to CSR):
# t_kppcsr.py on the 24- and 51-cell plates (bodies of design iteration 0).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; P=/root/autodl-tmp/OPL; SC=$P/S1/V2/R1/SCALE
PK=$P/S4/packets:$P/S3/packets:$SC/plate_packets:$W/plate_packets
TST_ENV="OPL_PACKETS_EXTRA=$W/plate/v3U_S24/packets:$PK" bash $S/tst.sh t49_kppcsr_S24 t_kppcsr.py $W/plate/body_S24 $SC/layouts/plateS24.json
TST_ENV="OPL_PACKETS_EXTRA=$W/plate/v3AC_S51/packets:$PK" bash $S/tst.sh t49_kppcsr_S51 t_kppcsr.py $W/plate/body_S51 $SC/layouts/plateS51.json
echo "$(date +%F_%T) TESTS49_DONE" >> $W/status.txt
