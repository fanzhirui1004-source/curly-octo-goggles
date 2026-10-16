#!/bin/bash
# d5.sh: the switch diagnosis of chain_diag5.sh (2026-09-28, never run) on hlat222 with src_v2 and the FINAL3 environment.
W=/root/autodl-tmp/OPL/V3W
for V in "symv:OPL_KE_SYMV=1" "symv64:OPL_KE_SYMV64=1" "cinv:OPL_COARSE_INV=1" "all:OPL_KE_SYMV=1 OPL_KE_SYMV64=1 OPL_COARSE_INV=1"; do
  T=${V%%:*}; E=${V#*:}
  SRC=/root/autodl-tmp/OPL/src_v2 bash $W/scripts/t5.sh d5_$T hlat222 $E
done
echo "$(date +%F_%T) D5V3_DONE" >> $W/status.txt
