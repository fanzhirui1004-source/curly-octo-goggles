#!/bin/bash
# tests28.sh: P1 solvers on hlat222 (three right-hand sides): block PCG (OPL_V3_PCG_BLOCK=1) and the Ritz-deflated restart
# (OPL_V3_PCG_RITZ=m,k) against the default (v3P2_t5: 129 iterations, solve 26.1 s).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
TMO=900 bash $S/t5.sh v3B hlat222 OPL_V3_PCG_BLOCK=1
TMO=900 bash $S/t5.sh v3R hlat222 OPL_V3_PCG_RITZ=40,10
TMO=900 bash $S/t5.sh v3R2 hlat222 OPL_V3_PCG_RITZ=60,20
echo "$(date +%F_%T) TESTS28_DONE" >> $W/status.txt
