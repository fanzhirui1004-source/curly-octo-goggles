#!/bin/bash
# tests55.sh: item 4 (category B): opt_design on the 24-cell plate, 4 iterations with --warm and the preconditioner kept
# for 2 / 4 design iterations (--reuse-prec: applied on the matched DOFs, Jacobi on the others).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
for RP in 4 2; do bash $S/optrun.sh opt_S24_rp$RP 24 4 --warm --reuse-prec $RP; done
echo "$(date +%F_%T) TESTS55_DONE" >> $W/status.txt
