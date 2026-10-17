#!/bin/bash
# tests16.sh: CUDA-graph prototype on four plate cells, 1-4 streams (t_graph.py; the slot recompute is capturable now);
# S25 spill / unspill check (t_spill.py); the 51-cell plate with S21/S22/S24 on by default (v3s) and with a forced
# spill of 6 GiB around the K_PP factorisation (v3t: results must equal v3s up to the run-to-run noise).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
GF=${GF:-OPL_V3_GFCAT=1}
bash $S/tst.sh graph2 t_graph.py $W/plate/body_S24:plateS24_040_o000,plateS24_060_o000,plateS24_150_o000,plateS24_160_o000
bash $S/tst.sh spill t_spill.py --gb 2
OPL_V3_PIPE_BODIES=0 bash $S/plate.sh v3s 51 $GF
OPL_V3_PIPE_BODIES=0 bash $S/plate.sh v3t 51 $GF OPL_V3_SPILL=1 OPL_V3_SPILL_FORCE_GB=6
echo "$(date +%F_%T) TESTS16_DONE" >> $W/status.txt
