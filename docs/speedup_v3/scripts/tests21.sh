#!/bin/bash
# tests21.sh: the final re-measure with every default of src_v3 (S1-S27; graphs and S27 on): the Table 5 lattices, the
# 110-cell plate and the 192-cell plate (S190 layout) with the ST20 arguments (--resident-gb 4), then the capacity runs:
# generated plates of about 256, 336 and 424 cells with S25 (spill around the K_PP planning) and the automatic resident
# budget (S23).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
for L in hlat221a hlat221b hlat222 hlat331; do bash $S/t5.sh v3F $L; done
OPL_V3_PIPE_BODIES=0 bash $S/plate.sh v3F 110
LAYOUT=$W/layouts/plateS190.json OPL_V3_PIPE_BODIES=0 bash $S/plate.sh v3F 190
export OPL_V3_SPILL=1 RGB=auto
OPL_V3_PIPE_BODIES=0 bash $S/scale.sh v3X "256 336 424"
echo "$(date +%F_%T) TESTS21_DONE" >> $W/status.txt
