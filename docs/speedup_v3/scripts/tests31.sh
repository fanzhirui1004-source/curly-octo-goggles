#!/bin/bash
# tests31.sh: P13 matrix-free float32 element product (t_kmf.py: accuracy against float64, device time, tiles; S_hat
# with it in graph replay); P12 in-place adjoint mask split (t_msplit.py: bitwise, then S_hat in graph replay with t_switch.py);
# P9b two-phase ghost-face product (no atomics): the product alone (t_switch.py --gf) and S_hat with it;
# Nsight Compute counters of the element product alone vs inside S_hat (ncu_symv.sh).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; SP=$W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh kmf t_kmf.py $SP
bash $S/tst.sh sw_kmf t_switch.py $SP --sw kmf.ON
bash $S/tst.sh msplit t_msplit.py $SP
bash $S/tst.sh sw_msplit t_switch.py $SP --sw fastnet.MSPLIT_IP
bash $S/tst.sh sw_gf3 t_switch.py $SP --sw gffused.MODE=2 --gf
bash $S/ncu_symv.sh
echo "$(date +%F_%T) TESTS31_DONE" >> $W/status.txt
