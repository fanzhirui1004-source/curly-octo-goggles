#!/bin/bash
# E2 hard-direction attribution needs dense background-node frames (tens of GB): run after the spectrum batch.
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923
until grep -q FS_ALL_DONE $T/FULLSPEC_DIRECT_01/status 2>/dev/null; do sleep 60; done
cd $T/xcase_src_23 && /root/cutfem_neural_a_20260910/env/bin/python -u attr_e12.py E2 $T/ATTR_01/E2.json > $T/ATTR_01/E2.log 2>&1
echo "$(date -u +%FT%TZ) E2 exit $?" >> $T/ATTR_01/status
