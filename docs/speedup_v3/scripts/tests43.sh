#!/bin/bash
# tests43.sh: with P17 a streamed cell copies ~178 MiB per PCG step, so fewer resident cells may cost little time:
# 51-cell plate with resident budget 0 and 2 GiB (device memory against time; the ST20 runs use 4).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
RGB=0 bash $S/plate.sh v3Z_rg0 51 OPL_V3_CENSUS_POOLS=1
RGB=2 bash $S/plate.sh v3Z_rg2 51 OPL_V3_CENSUS_POOLS=1
echo "$(date +%F_%T) TESTS43_DONE" >> $W/status.txt
