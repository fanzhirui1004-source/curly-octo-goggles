#!/bin/bash
# Minimal exact-check set: (1) start designs B1:0+B2:0 (shared T), compliance only; (2) final designs B1:22, B2:29 with
# gradients; (3) bending homogenisation design and its NICE cross-start, compliance only.
source /root/autodl-tmp/xc/xc_env.sh
cd $W/plates_exact/src_v2
run() { echo "=== STEP $1 START $(date '+%F %T')"; shift; python3 -u exact_check_cpu.py "$@" $COMMON; rc=$?; echo "=== STEP rc=$rc $(date '+%F %T')"; return $rc; }
run 1 $R/plateB1:0 $R/plateB2:0 --no-sens || exit 1
run 2 $R/plateB1:last $R/plateB2:last || exit 2
run 3 $R/hevalH_z:0 $R/xstartH_z:last --no-sens || exit 3
echo "=== CHAIN DONE $(date '+%F %T')"
