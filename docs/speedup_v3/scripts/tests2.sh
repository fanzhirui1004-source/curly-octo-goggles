#!/bin/bash
# tests2.sh: hlat222 Table 5 route (census.sh) with the new defaults (chebfused, stored float64 partial-element matrices,
# lean K_PP assembly), then with the opt-in fused hyperedge layer and two compute streams.
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
bash $S/census.sh v3b hlat222
bash $S/census.sh v3b_hf hlat222 OPL_V3_HFUSE=1
bash $S/census.sh v3b_st2 hlat222 OPL_V3_STREAMS=2
bash $S/census.sh v3b_hfst2 hlat222 OPL_V3_HFUSE=1 OPL_V3_STREAMS=2
echo "$(date +%F_%T) TESTS2_DONE" >> $W/status.txt
