#!/bin/bash
# tests11.sh: capacity on the 5090 / 90 GiB container with S14b-S18 and the block coarse inverse (S15, OPL_COARSE_BLK=1):
# the 190-cell plate again (vs v3g), then generated plates of about 256, 336 and 424 cells (gen_hlat S = 14, 16, 18).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
export OPL_COARSE_BLK=1
OPL_V3_PIPE_BODIES=0 bash $S/scale.sh v3i "190 256 336 424"
echo "$(date +%F_%T) TESTS11_DONE" >> $W/status.txt
