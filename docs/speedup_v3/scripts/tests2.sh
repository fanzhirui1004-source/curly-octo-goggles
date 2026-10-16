#!/bin/bash
# tests2.sh: single-cell S_hat with the new defaults (tiled hyperedge kernels on / off), then the hlat222 Table 5 route
# (census.sh): new defaults (S7 lean K_PP, S8 chebfused, S9 stored float64 partial-element matrices, tiled hyperedge
# kernels), a rerun (noise), two compute streams, tiled kernels off.
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
SPEC=$W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh graph2 t_graph2.py $W/plate/body_S24 plateS24_040_o000,plateS24_030_o000,plateS24_050_o000
bash $S/shat.sh v3b_ht $SPEC --B 1,3,16 --rep 10 --segments --ref $W/shat/blk_512.npz
SHAT_ENV="OPL_V3_HTILE=0" bash $S/shat.sh v3b_noht $SPEC --B 1,3,16 --rep 10 --segments --ref $W/shat/blk_512.npz
bash $S/tst.sh sync t_sync.py $SPEC
bash $S/census.sh v3b hlat222
bash $S/census.sh v3b_r hlat222
bash $S/census.sh v3b_st2 hlat222 OPL_V3_STREAMS=2
bash $S/census.sh v3b_noht hlat222 OPL_V3_HTILE=0
echo "$(date +%F_%T) TESTS2_DONE" >> $W/status.txt
