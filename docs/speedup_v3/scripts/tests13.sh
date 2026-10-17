#!/bin/bash
# tests13.sh: fewer kernel launches per S_hat (the two-stream PCG matvec is bound by the CPU's launches):
# S15b (block coarse solve in nine kernels) and the concatenated ghost-face product (OPL_V3_GFCAT=1, 8 instead of 12
# launches per K product). Cell checks, then hlat222 and the 51-cell plate for each.
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
bash $S/tst.sh cblk4 t_cblk.py $W/plate/body_S24:plateS24_040_o000,plateS24_060_o000,plateS24_150_o000,plateS24_160_o000
bash $S/tst.sh cblk2b t_cblk2.py $W/plate/body_S24:plateS24_040_o000,plateS24_050_o000
bash $S/t5.sh v3m hlat222
OPL_V3_PIPE_BODIES=0 bash $S/plate.sh v3m 51
bash $S/t5.sh v3n hlat222 OPL_V3_GFCAT=1
OPL_V3_PIPE_BODIES=0 bash $S/plate.sh v3n 51 OPL_V3_GFCAT=1
echo "$(date +%F_%T) TESTS13_DONE" >> $W/status.txt
