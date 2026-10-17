#!/bin/bash
# tests8.sh: host-memory budget (hostmem.Budget; OPL_V3_KE64=auto, OPL_V3_SLOTS=auto): hlat222 check, then the plates
# that exceed the container's 90 GiB with every item stored (135, 162), and 190.
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
bash $S/census.sh v3g hlat222
OPL_V3_PIPE_BODIES=0 bash $S/plate.sh v3g 135
OPL_V3_PIPE_BODIES=0 bash $S/plate.sh v3g 162
echo "$(date +%F_%T) TESTS8_DONE" >> $W/status.txt
