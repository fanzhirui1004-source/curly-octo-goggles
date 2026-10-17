#!/bin/bash
# tests12.sh: block coarse inverse with COO couplings (no cuSPARSE) and default on; fused slot-head tail check (S19);
# the 51-cell plate twice (solve-phase jitter seen with the CSR couplings in v3h / v3h2).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
bash $S/tst.sh cblk3 t_cblk.py $W/plate/body_S24:plateS24_040_o000,plateS24_060_o000,plateS24_150_o000,plateS24_160_o000
bash $S/tst.sh slotfuse t_slotfuse.py $W/plate/body_S24:plateS24_040_o000
OPL_V3_PIPE_BODIES=0 bash $S/plate.sh v3l 51
OPL_V3_PIPE_BODIES=0 bash $S/plate.sh v3l2 51
echo "$(date +%F_%T) TESTS12_DONE" >> $W/status.txt
