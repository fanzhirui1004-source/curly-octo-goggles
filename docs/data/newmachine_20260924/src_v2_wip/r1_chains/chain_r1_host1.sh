#!/bin/bash
# Revision round 1, single-core column of Table 5 (agent tag R1ONE). Starts only after the line
# "HH:MM:SS R1_HOSTCOND_DONE_20260928" (anchored) and a quiet container (as E1; at most 3 h, then NOT_QUIET and proceed).
# All runs with MKL_NUM_THREADS = OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = 1 (front end included).
#  (1) whole-lattice PARDISO Cholesky, lat_direct_cpu2.py, E1's tuned iparm with the parallel-only controls switched off
#      (iparm(24) = 0 classic factorisation, iparm(25) = 0 sequential solve); ordering iparm(2) as tuned, and if that run
#      fails, iparm(2) = 2 (METIS) is used and recorded: hlat221a, hlat221b once each; hlat222 once (E1's memory guard,
#      margin 4 GiB); hlat331 --analysis-only (predicted memory only).
#  (2) SciPy SuperLU (lat_scipy_cpu.py) on hlat221a, variants colamd (splu / spsolve defaults) and mmd (MMD_AT_PLUS_A,
#      diag_pivot_thresh 0, SymmetricMode), each in a child with RLIMIT_AS 70 GiB, RSS watchdog 70 GiB and a 2 h wall limit
#      enforced by the child itself; if neither finishes on hlat221a, both variants again on the 2-cell smoke lattice
#      (650k DOFs) with the same limits.
# Writes the line "HH:MM:SS R1_HOST1_DONE_20260928" at the end, also on failure (EXIT trap).
# SMOKE=1: no waiting, 1-cell layout, SuperLU limits 6 GiB / 300 s, output in R1/smoke/one, no marker.
O=/root/autodl-tmp/OPL/S1/V2; ST=$O/chain_p1.status; R=$O/R1; SRC=/root/autodl-tmp/OPL/src_v2; S4=/root/autodl-tmp/OPL/S4
if [ "$SMOKE" = 1 ]; then
  H=$R/smoke/one; TAG="R1ONE SMOKE"; IPT=$R/smoke/host/iparm_tuned.json; RH=$R/smoke/host
  PLATS="$R/smoke/hlatsmoke1.json"; P8=""; PAN=""; SLAT=$R/smoke/hlatsmoke1.json; SFB=""; ASG=6; WAL=300
else
  H=$R/host1; TAG="R1ONE"; IPT=$R/host/iparm_tuned.json; RH=$R/host
  PLATS="$S4/hlat221a.json $S4/hlat221b.json"; P8="$S4/hlat222.json"; PAN="$S4/hlat331.json"; SLAT=$S4/hlat221a.json
  SFB=$R/smoke/hlatsmoke2.json; ASG=70; WAL=7200
fi
mkdir -p $H
st() { echo "$(date +%T) $TAG $*" >> $ST; }
RC=0
fin() { if [ "$SMOKE" = 1 ]; then st "SMOKE_END rc=$RC"; else st "END rc=$RC"; echo "$(date +%T) R1_HOST1_DONE_20260928" >> $ST; fi; }
trap fin EXIT
if [ "$SMOKE" != 1 ]; then
  st "QUEUED pid=$$ (after the host condensation chain and a quiet container)"
  until grep -qE "^[0-9]{2}:[0-9]{2}:[0-9]{2} R1_HOSTCOND_DONE_20260928$" $ST; do sleep 300; done
  t0=$(date +%s); Q=0
  while [ $(( $(date +%s) - t0 )) -lt 10800 ]; do
    u0=$(awk '/^usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat); sleep 60; u1=$(awk '/^usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat)
    cores=$(python3 -c "print(round(($u1-$u0)/60e6,2))")
    big=$(ps -eo pid,rss,comm --no-headers | awk '$3 ~ /python/ && $2 > 2097152 {print $1}' | wc -l)
    gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    if python3 -c "import sys; sys.exit(0 if $cores < 1.5 else 1)" && [ $big = 0 ] && [ $gpu = 0 ]; then Q=1; break; fi
    sleep 240
  done
  if [ $Q = 1 ]; then st "QUIET cores=$cores loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"; else st "NOT_QUIET after 3 h: cores=$cores big_python=$big gpu_procs=$gpu; proceeding"; fi
fi
st "SKIPPED (author decision 2026-09-29: CPU baselines only on the dedicated AMD machine; the EXIT trap writes the completion line)"; exit 0
st "START pid=$$"
source $R/env_cpu.sh
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPL_GP_CACHE=0
cd $SRC
# single-thread iparm: tuned values, parallel-only controls off; fallback ordering METIS (2)
$PY - $IPT $H <<'PY'
import json, sys
t = json.load(open(sys.argv[1])); ip = dict(t['iparm']); ip['24'] = 0; ip['25'] = 0
json.dump(dict(iparm=ip, derived_from=sys.argv[1], note='single thread: iparm(24)=0, iparm(25)=0; ordering as tuned'), open(sys.argv[2] + '/iparm_1thread.json', 'w'), indent=1)
ip2 = dict(ip); ip2['2'] = 2
json.dump(dict(iparm=ip2, derived_from=sys.argv[1], note='single thread, fallback ordering iparm(2)=2 (METIS)'), open(sys.argv[2] + '/iparm_1thread_metis.json', 'w'), indent=1)
PY
pdir() {  # pdir <layout.json> [extra args]: one run, fallback to iparm(2)=2 if the tuned ordering fails
  local lay=$1; shift; local L=$(basename $lay .json); local t0=$(date +%s)
  $PY -u lat_direct_cpu2.py $H/lat1_${L}_chol.json $lay --mtypes 2 --iparm-file $H/iparm_1thread.json --margin-gib 4 "$@" > $H/lat1_${L}_chol.log 2>&1
  local rc=$?
  if ! grep -q '"factor_s"\|"skipped"' $H/lat1_${L}_chol.json 2>/dev/null; then
    st "pardiso1 $L tuned ordering failed rc=$rc; retry with iparm(2)=2"
    $PY -u lat_direct_cpu2.py $H/lat1_${L}_chol_metis.json $lay --mtypes 2 --iparm-file $H/iparm_1thread_metis.json --margin-gib 4 "$@" > $H/lat1_${L}_chol_metis.log 2>&1
    rc=$?
  fi
  [ $rc = 0 ] || RC=$rc; st "pardiso1 $L rc=$rc $(( $(date +%s)-t0 ))s"; }
for lay in $PLATS $P8; do pdir $lay; done
[ -n "$PAN" ] && pdir $PAN --analysis-only
# SciPy SuperLU
slu() {  # slu <layout.json>
  local lay=$1; local L=$(basename $lay .json); local t0=$(date +%s)
  local REF=$RH/lat2_${L}_chol_rep1.json,$H/lat1_${L}_chol.json,$(ls $O/lat_direct_${L}*.json 2>/dev/null | tr '\n' ',')
  $PY -u lat_scipy_cpu.py $H/scipy_${L}.json $lay --as-gib $ASG --watch-gib $ASG --wall-s $WAL --variants colamd,mmd --ref "$REF" --keep-npz-dir $H/npz > $H/scipy_${L}.log 2>&1
  local rc=$?; [ $rc = 0 ] || RC=$rc
  st "superlu $L rc=$rc $(( $(date +%s)-t0 ))s $($PY -c "import json; d=json.load(open('$H/scipy_${L}.json')); print(' '.join(v+'='+str(c.get('status')) for v, c in d.get('solves', {}).items()))" 2>/dev/null)"; }
slu $SLAT
if [ -n "$SFB" ] && ! grep -q '"status": "ok"' $H/scipy_$(basename $SLAT .json).json 2>/dev/null; then
  st "superlu: no variant finished on $(basename $SLAT .json); running both on the 2-cell lattice"
  slu $SFB
fi
$PY lat_scipy_cpu.py --selftest $H > $H/scipy_selftest.json 2>&1; st "superlu selftest rc=$?"
