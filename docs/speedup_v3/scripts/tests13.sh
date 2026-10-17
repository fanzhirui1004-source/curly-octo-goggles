#!/bin/bash
# tests13.sh: fewer kernel launches per S_hat (the two-stream PCG matvec is bound by the CPU's launches):
# S15b (block coarse solve in nine kernels), the concatenated ghost-face product (OPL_V3_GFCAT=1, 8 instead of 12
# launches per K product) and the padded batched one (S20, OPL_V3_GFBMM=1, 4 launches). Cell checks, single-cell S_hat
# against a saved reference, then hlat222 and the 51-cell plate for each.
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; SPEC=$W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh cblk4 t_cblk.py $W/plate/body_S24:plateS24_040_o000,plateS24_060_o000,plateS24_150_o000,plateS24_160_o000
bash $S/tst.sh cblk2b t_cblk2.py $W/plate/body_S24:plateS24_040_o000,plateS24_050_o000
bash $S/shat.sh gf_ref $SPEC --B 1,3 --rep 10 --save $W/shat/gf_ref.npz
SHAT_ENV="OPL_V3_GFCAT=1" bash $S/shat.sh gf_cat $SPEC --B 1,3 --rep 10 --ref $W/shat/gf_ref.npz
SHAT_ENV="OPL_V3_GFBMM=1" bash $S/shat.sh gf_bmm $SPEC --B 1,3 --rep 10 --ref $W/shat/gf_ref.npz
bash $S/t5.sh v3m hlat222
OPL_V3_PIPE_BODIES=0 bash $S/plate.sh v3m 51
bash $S/t5.sh v3n hlat222 OPL_V3_GFCAT=1
OPL_V3_PIPE_BODIES=0 bash $S/plate.sh v3n 51 OPL_V3_GFCAT=1
bash $S/t5.sh v3o hlat222 OPL_V3_GFBMM=1
OPL_V3_PIPE_BODIES=0 bash $S/plate.sh v3o 51 OPL_V3_GFBMM=1
echo "$(date +%F_%T) TESTS13_DONE" >> $W/status.txt
