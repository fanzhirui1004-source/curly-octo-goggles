#!/bin/bash
# tests32.sh: P13 matrix-free float32 element product with the CUDA C++ sum-factorised kernel (kmf_kernel.cu, built
# with nvcc on first use): t_kmf.py (accuracy against float64, device time alone, elements per block; the Triton form
# and the formed-matrix product alongside), then S_hat with it (t_switch.py, graph replay, B = 1 uses it, B = 3 / 16 not).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; SP=$W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh kmf t_kmf.py $SP
bash $S/tst.sh sw_kmf2 t_switch.py $SP --sw kmf.ON
echo "$(date +%F_%T) TESTS32_DONE" >> $W/status.txt
