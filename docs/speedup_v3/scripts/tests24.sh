#!/bin/bash
# tests24.sh: P2 micro-benchmark of the float32 element product variants (t_symv.py); P1: the Ritz values of the
# preconditioned operator (bnn:kpp:q1r) on hlat222 and the 51-cell plate.
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
bash $S/tst.sh symv t_symv.py $W/plate/body_S24:plateS24_040_o000
TMO=900 bash $S/t5.sh lz hlat222 OPL_V3_PCG_LANCZOS=1                 # Ritz values of the preconditioned operator
OPL_V3_PIPE_BODIES=0 TMO=1500 bash $S/plate.sh lz 51 OPL_V3_PCG_LANCZOS=1
echo "$(date +%F_%T) TESTS24_DONE" >> $W/status.txt
