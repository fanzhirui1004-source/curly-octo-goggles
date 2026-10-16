#!/bin/bash
# sw.sh: S_hat timings of the three src_v2 switches (one at a time and all) on a plate cell and hlat222_000, B = 1,3,16,32.
W=/root/autodl-tmp/OPL/V3W
for CELL in /root/autodl-tmp/OPL/V3W/plate/body_S24:plateS24_040_o000 /root/autodl-tmp/OPL/S4/body:hlat222_000; do
  N=$(basename ${CELL#*:})
  for V in "off:" "symv:OPL_KE_SYMV=1" "symv64:OPL_KE_SYMV64=1" "cinv:OPL_COARSE_INV=1" "all:OPL_KE_SYMV=1 OPL_KE_SYMV64=1 OPL_COARSE_INV=1"; do
    T=${V%%:*}; E=${V#*:}
    SRC=/root/autodl-tmp/OPL/src_v2 SHAT_ENV="$E" bash $W/scripts/shat.sh sw_${N}_$T $CELL --B 1,3,16,32 --rep 5 --save $W/shat/sw_${N}_$T.npz --ref $W/shat/sw_${N}_off.npz
  done
done
echo "$(date +%F_%T) SW_DONE" >> $W/status.txt
