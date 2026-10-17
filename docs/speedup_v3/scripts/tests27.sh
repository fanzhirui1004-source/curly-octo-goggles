#!/bin/bash
# tests27.sh: P2 under CUDA-graph replay (t_symv3.py: one cell at B = 1, 3; three cells on two streams), then the 51-cell
# plate with a device-level profile of PCG iteration 20 (gpuprof: OPL_V3_PROF_ITER=20, chrome trace kept).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; B=$W/plate/body_S24
bash $S/tst.sh symv3 t_symv3.py $B:plateS24_040_o000,plateS24_060_o000,plateS24_150_o000
mkdir -p $W/plate/v3Q_S51
OPL_V3_PIPE_BODIES=0 TMO=1500 bash $S/plate.sh v3Q 51 OPL_V3_PROF_ITER=20 OPL_V3_PROF_OUT=$W/plate/v3Q_S51/prof20.json OPL_V3_PROF_TRACE=1
echo "$(date +%F_%T) TESTS27_DONE" >> $W/status.txt
