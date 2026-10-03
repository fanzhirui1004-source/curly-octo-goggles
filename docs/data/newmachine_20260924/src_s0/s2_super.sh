#!/bin/bash
# Supervisor for the step-2 run: resume from out/ckpt.pt after an unexpected exit (no DONE event), at most 5 restarts.
source /root/autodl-tmp/OPL/S1/env_gpu.sh
cd /root/autodl-tmp/OPL/src
S1=/root/autodl-tmp/OPL/S1
CFG=$S1/s2_full.json
for try in 0 1 2 3 4 5; do
  echo "$(date +%F_%T) launch try=$try" >> $S1/s2_full.super
  SENS_REASSOC=1 $PY -u train2.py $CFG >> $S1/s2_full.log 2>&1
  rc=$?
  echo "$(date +%F_%T) exit rc=$rc" >> $S1/s2_full.super
  grep -q '"event": "DONE"' $S1/s2_full/train.log && { echo "$(date +%F_%T) DONE" >> $S1/s2_full.super; exit 0; }
  python3 - <<'P'
import json; p='/root/autodl-tmp/OPL/S1/s2_full.json'; d=json.load(open(p)); d['resume']=True; json.dump(d, open(p,'w'), indent=1)
P
  sleep 30
done
echo "$(date +%F_%T) GAVE_UP" >> $S1/s2_full.super
