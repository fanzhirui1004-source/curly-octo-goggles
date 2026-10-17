#!/bin/bash
# tests15.sh: repeat checks with their own references (S21: the cuDSS solve against itself, and after the freed blocks were
# overwritten with NaN; S24: each convolution against its own rerun and with deterministic cuDNN); S19b (fused slot
# assignment) per layer bitwise; CUDA-graph prototype of one cell's S_hat (t_graph.py).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; SPEC=$W/plate/body_S24:plateS24_040_o000
GF=${GF:-OPL_V3_GFCAT=1}
bash $S/tst.sh kppfree2 t_kppfree.py
bash $S/tst.sh netfuse2 t_netfuse.py $SPEC
bash $S/tst.sh slotfuse2 t_slotfuse.py $SPEC
TST_ENV="OPL_V3_NETFUSE=1 $GF" bash $S/tst.sh graph t_graph.py $W/plate/body_S24:plateS24_040_o000,plateS24_060_o000
echo "$(date +%F_%T) TESTS15_DONE" >> $W/status.txt
