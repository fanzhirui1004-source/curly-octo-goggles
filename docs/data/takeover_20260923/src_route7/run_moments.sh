#!/bin/bash
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_36; O=$T/R7_10
PY=/root/cutfem_neural_a_20260910/env/bin/python
mkdir -p $O; cd $S
$PY -u moments_test.py $T/R7_07 fresh_train_0013_cover01_r1 fresh_train_0021_cover01_r1 fresh_train_0013_full > $O/moments.log 2>&1
echo "$(date -u +%FT%TZ) moments rc $?" >> $O/status
