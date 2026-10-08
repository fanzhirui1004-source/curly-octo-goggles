#!/bin/bash
# Recompute all teacher sensitivity blocks (three bases through the same code path, twelve perturbations), two at a time.
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_24; X=$T/SENS_01
L=""
for b in fresh_train_0008_cover01_r1 fresh_train_0020_cover01_r1 fresh_train_0013_cover01_r1; do
  L="$L $b ${b}_tau0"
  for t in m10000 p10000 m1000 p1000; do L="$L $b ${b}_tau$t"; done
done
echo $L | xargs -n 2 -P 2 bash $S/blocks24_one.sh
echo "$(date -u +%FT%TZ) BLOCKS24_DONE" >> $X/blocks24.status
