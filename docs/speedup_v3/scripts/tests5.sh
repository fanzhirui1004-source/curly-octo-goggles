#!/bin/bash
# tests5.sh: opt_design iteration on the 24-cell plate: sequential bodies, overlapped bodies, overlapped with an injected
# body failure at the 10th cell (fallback to the retry loop).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
bash $S/pipe_test.sh seq
bash $S/pipe_test.sh pipe OPL_V3_PIPE_BODIES=1
bash $S/pipe_test.sh fault OPL_V3_PIPE_BODIES=1 OPL_V3_PIPE_FAULT=10
echo "$(date +%F_%T) TESTS5_DONE" >> $W/status.txt
