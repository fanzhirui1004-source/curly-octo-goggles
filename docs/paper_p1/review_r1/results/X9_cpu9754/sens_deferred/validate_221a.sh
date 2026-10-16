#!/bin/bash
# Validation of the sensitivity routes on the GPU machine's CPU (not a timing run): hlat221a, route (a) then (b).
. /root/autodl-tmp/cpu9754/env_cpu9754.sh
export MKL_NUM_THREADS=16 OMP_NUM_THREADS=16
V=/root/autodl-tmp/cpu9754/val; mkdir -p $V; cd /root/autodl-tmp/OPL/src_v2
IPT=/root/autodl-tmp/exact_tools/iparm_tuned.json
$PY -u /root/autodl-tmp/cpu9754/lat_direct_sens.py $V/a_hlat221a.json /root/autodl-tmp/OPL/S4/hlat221a.json --iparm-file $IPT > $V/a.log 2>&1; echo "a rc=$?" >> $V/status
$PY -u /root/autodl-tmp/cpu9754/lat_cond_sens.py $V/b_hlat221a.json /root/autodl-tmp/OPL/S4/hlat221a.json --iparm-file $IPT --ref $V/a_hlat221a.json > $V/b.log 2>&1; echo "b rc=$?" >> $V/status
echo DONE >> $V/status
