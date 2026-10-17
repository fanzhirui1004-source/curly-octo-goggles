#!/bin/bash
# tests32.sh: P13 matrix-free float32 correction product with CUDA C++ kernels (kmf_kernel.cu, built with nvcc on first
# use): t_kmf.py (element kernel and ghost-face kernel alone: accuracy against float64, device time, block shapes; the
# Triton form and the current paths alongside), then S_hat with them (t_switch.py, graph replay; B = 1 uses them,
# B = 3 / 16 not): elements and faces (sw_kmf2), elements only (sw_kmf3).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; SP=$W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh kmf t_kmf.py $SP
bash $S/tst.sh sw_kmf2 t_switch.py $SP --sw kmf.ON
bash $S/tst.sh sw_kmf3 t_switch.py $SP --sw kmf.ON,kmf.GF=0
echo "$(date +%F_%T) TESTS32_DONE" >> $W/status.txt
