#!/bin/bash
# tests28.sh: P2b check (t_kdedupe.py: bitwise-equal full-element matrices, product with the element-to-row map), the
# float32 product's kernels timed with a cold L2 (t_symv4.py), then P1 solvers on hlat222 (three right-hand sides): block PCG (OPL_V3_PCG_BLOCK=1) and the Ritz-deflated restart
# (OPL_V3_PCG_RITZ=m,k) against the default (v3P2_t5: 129 iterations, solve 26.1 s).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
bash $S/tst.sh kdedupe t_kdedupe.py $W/plate/body_S24:plateS24_040_o000,plateS24_060_o000,plateS24_150_o000,plateS24_070_o000
bash $S/tst.sh symv4 t_symv4.py $W/plate/body_S24:plateS24_040_o000,plateS24_070_o000
TMO=900 bash $S/t5.sh v3B hlat222 OPL_V3_PCG_BLOCK=1
TMO=900 bash $S/t5.sh v3R hlat222 OPL_V3_PCG_RITZ=40,10
TMO=900 bash $S/t5.sh v3R2 hlat222 OPL_V3_PCG_RITZ=60,20
echo "$(date +%F_%T) TESTS28_DONE" >> $W/status.txt
