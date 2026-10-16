#!/bin/bash
W=/root/autodl-tmp/OPL/V3W
for V in "nb:OPL_COARSE_BLOCKS=0" "nfk:OPL_FULL_KE=0" "nsym:OPL_KE_SYMV=0 OPL_KE_SYMV64=0"; do
  T=${V%%:*}; E=${V#*:}
  bash $W/scripts/t5.sh il_$T hlat222 $E
done
echo "$(date +%F_%T) ISOLAT_DONE" >> $W/status.txt
