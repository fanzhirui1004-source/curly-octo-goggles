#!/bin/bash
# tests41.sh: P17 (OPL_V3_SLOTDEC: slot weights recomputed by the fused heads, not stored / streamed) on the 51-cell
# plate against the current defaults (P13-P16, MAXB 32), resident budget 4 GiB as the ST20 runs: time, host / device
# peaks, streamed bytes, compliance.
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
bash $S/plate.sh v3X 51
bash $S/plate.sh v3X_sd 51 OPL_V3_SLOTDEC=1
echo "$(date +%F_%T) TESTS41_DONE" >> $W/status.txt
