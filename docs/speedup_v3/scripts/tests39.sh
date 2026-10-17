#!/bin/bash
# tests39.sh: P16 (OPL_V3_HCUDA) after the fixes: the Triton kernels' division instruction (div.full.f32: the adjoint
# gather bitwise too), column slices of 128 (any B F, B = 16 included), the scatter's head-sum order (HS 0 / 1 / 2 on a
# collision-free subset); t_hyperk.py B = 1, 2, 3, 4, 8, 16; S_hat with it on (B = 1, 3, 16).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; SP=$W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh hyperk2 t_hyperk.py $SP
bash $S/tst.sh sw_hk2 t_switch.py $SP --sw hyperk.ON
echo "$(date +%F_%T) TESTS39_DONE" >> $W/status.txt
