#!/bin/bash
# tests36.sh: 51-cell plate with P13 + P14 (OPL_V3_KMF=1 OPL_V3_KMF64=1): device-level profile of the front end (cells
# 20-22) and of PCG iteration 40 (gpuprof; where the time goes now); the coarse-space operator products on two streams
# (OPL_V3_STREAM_MAXCOLS=16: the B = 16 applies of A Z like the PCG matvec's). First P13b (OPL_V3_KMF_EF, elements and
# ghost faces in one kernel): t_kef.py (accuracy, time against the two kernels), S_hat with it on (P13 + P14 on).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; SP=$W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh kef t_kef.py $SP
TST_ENV="OPL_V3_KMF=1 OPL_V3_KMF64=1" bash $S/tst.sh sw_kef t_switch.py $SP --sw kmf.EF
OPL_V3_PIPE_BODIES=0 TMO=1500 bash $S/plate.sh v3T 51 OPL_V3_KMF=1 OPL_V3_KMF64=1 OPL_V3_PROF_ITER=40 OPL_V3_PROF_OUT=$W/plate/v3T_S51/prof40.json OPL_V3_PROF_FE=20 OPL_V3_PROF_FE_OUT=$W/plate/v3T_S51/prof_fe20.json
bash $S/plate.sh v3T_st16 51 OPL_V3_KMF=1 OPL_V3_KMF64=1 OPL_V3_STREAM_MAXCOLS=16
echo "$(date +%F_%T) TESTS36_DONE" >> $W/status.txt
