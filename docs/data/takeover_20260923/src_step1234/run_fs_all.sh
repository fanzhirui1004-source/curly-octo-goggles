#!/bin/bash
# Two-sided full spectrum (direct exact action, polyhedral elements at effective resolution 8) for every second-batch
# case with teacher-free inputs, one at a time.
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_20; O=$T/FULLSPEC_DIRECT_01
for d in $T/COVER_INPUTS/*/; do
  c=$(basename $d); [ -f $d/RESULT.json ] || continue
  [ -f $O/${c}_res8/FULL_SPECTRUM.json ] && continue
  bash $S/run_fs_one.sh ${c}_res8 --case $c --cover-inputs $T/COVER_INPUTS/$c --elements polyref --s 4 --levels 1
done
echo "$(date -u +%FT%TZ) FS_ALL_DONE" >> $O/status
