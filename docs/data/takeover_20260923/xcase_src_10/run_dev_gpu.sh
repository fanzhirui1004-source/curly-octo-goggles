#!/bin/bash
# Overnight GPU queue (one GPU job at a time). Exact FE coefficients only; no training.
# 1) pair-smoother (theta 0.9) finite witness on six unseen development cases
# 2) the same without the pair smoother (control: is the pair smoother needed there?)
# 3) complete spectrum at the nominated depth for every development case that nominates one, smallest first
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923
S=$T/xcase_src_10
AR=$T/ASSETS_DEV
PY=/root/cutfem_neural_a_20260910/env/bin/python
P=$T/DEV_PAIR0p9_01
B=$T/DEV_BASE_01
mkdir -p $P $B
CASES="fresh_development_0003_d0_v0 fresh_development_0006_d0_v0 fresh_development_0000_d0_v0 fresh_development_0002_full fresh_development_0002_d0_v2 fresh_development_0004_d1_v2"
until [ $(cat $T/CROSS_CASE_04_PAIR0p9/fullspec.status 2>/dev/null | wc -l) -ge 3 ]; do sleep 30; done
echo "$(date -u +%FT%TZ) QUEUE_START" >> $P/run.status
for c in $CASES; do
  $PY -u $S/cross_case_chebyshev.py --case $c --asset-root $AR --pair-theta 0.9 --output $P/$c > $P/$c.log 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) finite_pair $c exit $?" >> $P/run.status
done
for c in $CASES; do
  $PY -u $S/cross_case_chebyshev.py --case $c --asset-root $AR --output $B/$c > $B/$c.log 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) finite_base $c exit $?" >> $P/run.status
done
for c in $CASES; do
  d=$($PY -c "import json;print(json.load(open('$P/$c/RESULT.json'))['selected_full_spectrum_depth'])" 2>/dev/null)
  if [ "$d" = "None" ] || [ -z "$d" ]; then echo "$(date -u +%FT%TZ) fullspec $c no_nomination" >> $P/run.status; continue; fi
  $PY -u $S/cross_case_full_spectrum.py --case $c --asset-root $AR --finite $P/$c --depth $d --pair-theta 0.9 --output $P/FULLSPEC_${c}_$(printf %03d $d) > $P/FULLSPEC_$c.log 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) fullspec $c depth $d exit $?" >> $P/run.status
done
echo "$(date -u +%FT%TZ) QUEUE_FINISHED" >> $P/run.status
