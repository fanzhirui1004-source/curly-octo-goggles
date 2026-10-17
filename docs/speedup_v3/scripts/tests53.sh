#!/bin/bash
# tests53.sh: item 3 (t_sensc: the sensitivity pass without the tetrahedra no tau-dependent clip cut, full and cut cells
# of the 24-cell plate); item 4 (lat_scale on the 24-cell plate, x-direction load as the plate runs, 4 design steps of
# dtau = 0.0255 at fixed topology: preconditioner rebuilt every step vs built once).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; P=/root/autodl-tmp/OPL; SC=$P/S1/V2/R1/SCALE
PK=$P/S4/packets:$P/S3/packets:$SC/plate_packets:$W/plate_packets
TST_ENV="OPL_PACKETS_EXTRA=$W/plate/v3U_S24/packets:$PK" bash $S/tst.sh t53_sensc t_sensc.py $W/plate/body_S24 plateS24_000_o000 plateS24_060_o000 plateS24_070_o000
for RP in 0 4; do bash $S/lsrun.sh ls_S24x_rp$RP $W/layouts/plateS24_o000.json --iters 4 --dtau 0.0255 --reuse-prec $RP --clamp x,min --load x,max; done
echo "$(date +%F_%T) TESTS53_DONE" >> $W/status.txt
