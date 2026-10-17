#!/bin/bash
# tests10.sh: S15 block-structured coarse inverse (OPL_COARSE_BLK=1) and S16 K_PP host-triplet release: single-cell check
# (full, cut and small cells), hlat222 (Table 5 route, lat_scale) and the 51-cell plate (opt_design) against v3f.
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
bash $S/tst.sh cblk t_cblk.py $W/plate/body_S24:plateS24_040_o000,plateS24_060_o000,plateS24_150_o000,plateS24_160_o000
bash $S/t5.sh v3h hlat222 OPL_COARSE_BLK=1
OPL_V3_PIPE_BODIES=0 bash $S/plate.sh v3h 51 OPL_COARSE_BLK=1
# front-end steps per cell (S15 block build, S17 young gc) against the full collections of src_v2
SPEC=$W/plate/body_S24 CASES=plateS24_040_o000,plateS24_030_o000,plateS24_050_o000,plateS24_060_o000,plateS24_100_o000
TST_ENV="OPL_V3_GC=full" bash $S/tst.sh prep_gcfull t_prep.py $SPEC $CASES --top 25
bash $S/tst.sh prep_gcyoung t_prep.py $SPEC $CASES --top 25
TST_ENV="OPL_COARSE_BLK=1" bash $S/tst.sh prep_blk t_prep.py $SPEC $CASES --top 25
# S_hat kernel profiles of the current code (B = 1 and B = 16, the coarse A Z width), float64 matrices kept as in the plates
bash $S/shat.sh prof1_v3h $W/plate/body_S24:plateS24_040_o000 --B 1,3 --rep 10 --profile --segments --store64
bash $S/shat.sh prof16_v3h $W/plate/body_S24:plateS24_040_o000 --B 16 --rep 5 --profile --segments --store64
SHAT_ENV="OPL_V3_SLOTS=recompute" bash $S/shat.sh prof1_rc $W/plate/body_S24:plateS24_040_o000 --B 1,3 --rep 10 --profile --segments --store64
echo "$(date +%F_%T) TESTS10_DONE" >> $W/status.txt
