#!/bin/bash
# All second-batch (cover) cases present as local packets: recompute G (6 workers), then verify against the
# producer's published CONTEXT. Sequential; skips cases already verified.
S=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/xcase_src_11
W=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/COVER_G
R=/root/autodl-tmp/CUTFEM_FRESH_GP_20260921
PY=/root/cutfem_neural_a_20260910/env/bin/python
for c in $(ls $R/packets | grep cover); do
  [ -f $W/runs/${c}_G_VERIFY.json ] && grep -q '"MATCH"' $W/runs/${c}_G_VERIFY.json && continue
  bash $S/run_cover_g.sh 6 $c
  $PY $S/verify_cover_g.py $c >> $W/verify.log 2>&1
  echo "$(date -u +%FT%TZ) verified $c $(grep -o '"status": "[A-Z]*"' $W/runs/${c}_G_VERIFY.json 2>/dev/null)" >> $W/run.status
done
echo "$(date -u +%FT%TZ) ALL_COVER_G_FINISHED" >> $W/run.status
