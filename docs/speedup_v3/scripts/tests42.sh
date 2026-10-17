#!/bin/bash
# tests42.sh: device memory with all current defaults (P13-P17): 51-cell plate with the allocator's per-pool breakdown at
# the census points (OPL_V3_CENSUS_POOLS=1: CUDA-graph pools vs the default pool), the same without CUDA graphs
# (OPL_V3_GRAPH=0: memory and time the graphs cost / save); hlat222 with all current defaults.
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
bash $S/plate.sh v3Y 51 OPL_V3_CENSUS_POOLS=1
bash $S/plate.sh v3Y_ng 51 OPL_V3_CENSUS_POOLS=1 OPL_V3_GRAPH=0
bash $S/t5.sh v3Y_t5 hlat222
echo "$(date +%F_%T) TESTS42_DONE" >> $W/status.txt
