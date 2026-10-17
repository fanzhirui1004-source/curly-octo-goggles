#!/bin/bash
# tests22.sh: the noise band of hlat331's compliance (its third load case differed from the src_v2 baseline by 1.6e-7 in
# v3F): src_v3 twice with every default (v3G, v3H), once without S27 (v3I: OPL_V3_CONVT=0), src_v2 once more (v2R).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
bash $S/t5.sh v3G hlat331
bash $S/t5.sh v3H hlat331
bash $S/t5.sh v3I hlat331 OPL_V3_CONVT=0
SRC=/root/autodl-tmp/OPL/src_v2 bash $S/t5.sh v2R hlat331
echo "$(date +%F_%T) TESTS22_DONE" >> $W/status.txt
