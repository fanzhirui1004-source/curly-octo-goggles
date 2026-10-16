#!/bin/bash
W=/root/autodl-tmp/OPL/V3W; C=/root/autodl-tmp/OPL/S4/body:hlat222_000; R=$W/shat/iso_v2.npz
SRC=/root/autodl-tmp/OPL/src_v2 bash $W/scripts/shat.sh iso_v2 $C --B 1,3 --rep 3 --save $R
SRC=/root/autodl-tmp/OPL/src_v2 SHAT_ENV="OPL_KE_SYMV=1 OPL_KE_SYMV64=1 OPL_COARSE_INV=1" bash $W/scripts/shat.sh iso_v2all $C --B 1,3 --rep 3 --ref $R
for V in "fk1:" "fk0:OPL_FULL_KE=0" "fk0sym0:OPL_FULL_KE=0 OPL_KE_SYMV=0 OPL_KE_SYMV64=0" "fk1sym0:OPL_KE_SYMV=0 OPL_KE_SYMV64=0" "slim0:OPL_V3_SLIM=0 OPL_FULL_KE=0 OPL_KE_SYMV=0 OPL_KE_SYMV64=0 OPL_COARSE_PACK=0 OPL_FI_TRITON=0"; do
  T=${V%%:*}; E=${V#*:}
  SHAT_ENV="$E" bash $W/scripts/shat.sh iso_$T $C --B 1,3 --rep 3 --ref $R
done
echo "$(date +%F_%T) ISO2_DONE" >> $W/status.txt
