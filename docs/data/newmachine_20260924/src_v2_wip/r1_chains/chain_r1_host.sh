#!/bin/bash
# Revision round 1, E1 (agent tag R1HOST): host baseline done properly (Table 5 per-cell condensation, Table 6 whole lattice).
# Waits until the accuracy/derivative queue and the E7 refinement have finished (their completion markers), then until the
# container is quiet (container CPU use < 1.5 cores over 60 s, no other python process > 2 GiB RSS, no GPU compute process;
# at most 3 h, then it proceeds and records NOT_QUIET). Then, with MKL_NUM_THREADS = OMP_NUM_THREADS = 16 (= the container's
# CPU quota, cpu.max 1600000/100000; os.cpu_count() reports the host's 128):
#   0 environment capture (lscpu, cgroup, MKL, library versions)
#   1 tuning on G3 (iparm(2) x iparm(24), then iparm(25)) -> host/iparm_tuned.json
#   2 Table 5, PARDISO Cholesky, direct phase-33, Schur option + blocked route, G1-G4, 3 repetitions (separate processes)
#   3 Table 5 LU reference (old configuration via pypardiso, plus direct phase 33 on the same factor; wrapper overhead), once
#   4 Table 6: hlat221a, hlat221b Cholesky x 3 and LU (MKL defaults) x 1; hlat222 Cholesky x 3 (memory guard, margin 4 GiB);
#     hlat331 Cholesky once (guard; repeated twice more only if it was factorised)
#   5 summary host/host_summary.json (median / min over repetitions)
# Writes the line "HH:MM:SS R1_HOST_DONE_20260928" (also on failure, EXIT trap). Afterwards (not part of the timing): retry of the E7 solves the
# E7 memory guard refused (ref_valid2.py --retry-errors, headroom 4 GiB), then the line "HH:MM:SS R1_REF64_DONE_20260928".
# SMOKE=1: no waiting, G2 only, 1 repetition, s-budget 20, a 2-cell layout, output in R1/smoke/host, no markers.
O=/root/autodl-tmp/OPL/S1/V2; ST=$O/chain_p1.status; R=$O/R1; SRC=/root/autodl-tmp/OPL/src_v2; S4=/root/autodl-tmp/OPL/S4
G="fresh_train_0020_cover01_r2,fresh_train_0020_cover01_r1,fresh_train_0020_full,fresh_train_0007_full"   # G1-G4
if [ "$SMOKE" = 1 ]; then
  H=$R/smoke/host; TAG="R1HOST SMOKE"; G=fresh_train_0020_cover01_r1; NREP=1; SB=20; LATS4="hlatsmoke2"; LAT8=""; TUNEC=fresh_train_0020_cover01_r1
else
  H=$R/host; TAG="R1HOST"; NREP=3; SB=300; LATS4="hlat221a hlat221b"; LAT8="hlat222"; TUNEC=fresh_train_0020_full
fi
mkdir -p $H
st() { echo "$(date +%T) $TAG $*" >> $ST; }
RC=0; MARKED=""
mark() { [ -n "$MARKED" ] && return; MARKED=1; if [ "$SMOKE" = 1 ]; then st "SMOKE_END rc=$RC"; else st "END rc=$RC"; echo "$(date +%T) R1_HOST_DONE_20260928" >> $ST; fi; }
trap mark EXIT
# completion line = "HH:MM:SS <MARKER>", "HH:MM:SS <TAG> <MARKER> ..." or "HH:MM:SS ... <MARKER>" (last token); never a line mentioning "wait"
has() { grep -E "^[0-9]{2}:[0-9]{2}:[0-9]{2} (([A-Za-z0-9_]+ )?$1( |$)|.* $1[[:space:]]*$)" $ST | grep -viq "wait"; }
if [ "$SMOKE" != 1 ]; then
  st "QUEUED pid=$$ (waits for the derivative-chain and refinement completion markers)"
  until has R1_DERIV_DONE_20260928 && has R1_REF_DONE_20260928; do sleep 300; done
  # quiet check
  t0=$(date +%s); Q=0
  while [ $(( $(date +%s) - t0 )) -lt 10800 ]; do
    u0=$(awk '/^usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat); sleep 60; u1=$(awk '/^usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat)
    cores=$(python3 -c "print(round(($u1-$u0)/60e6,2))")
    big=$(ps -eo pid,rss,comm --no-headers | awk -v me=$$ '$3 ~ /python/ && $2 > 2097152 {print $1}' | wc -l)
    gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    if python3 -c "import sys; sys.exit(0 if $cores < 1.5 else 1)" && [ $big = 0 ] && [ $gpu = 0 ]; then Q=1; break; fi
    sleep 240
  done
  if [ $Q = 1 ]; then st "QUIET cores=$cores loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"; else st "NOT_QUIET after 3 h: cores=$cores big_python=$big gpu_procs=$gpu; proceeding"; fi
fi
st "SKIPPED (author decision 2026-09-29: CPU baselines only on the dedicated AMD machine; the EXIT trap writes the completion line)"; echo "$(date +%T) R1_REF64_DONE_20260928" >> $ST; exit 0
st "START pid=$$"
source $R/env_cpu.sh
export MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 OPL_GP_CACHE=0
cd $SRC
# 0. environment
{ date; uname -a; lscpu; nproc; cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.high; free -g;
  nvidia-smi -L; cat /proc/loadavg; $PY -c "import numpy, scipy, torch; print(numpy.__version__, scipy.__version__, torch.__version__)";
  ls /root/autodl-tmp/pylib_cpu | grep dist-info; } > $H/env_host.txt 2>&1
$PY -c "import json, pardiso_direct as PD; print(json.dumps(PD.env_record(), indent=1))" > $H/env_host.json 2>> $H/env_host.txt
run() {  # run <label> <cmd...>
  local l=$1; shift; local t0=$(date +%s); "$@" >> $H/$l.log 2>&1; local rc=$?; [ $rc = 0 ] || RC=$rc
  st "$l rc=$rc $(( $(date +%s)-t0 ))s load=$(cut -d' ' -f1 /proc/loadavg)"; }
# 1. tuning
run tune $PY -u bench_cpu2.py $H/bench_cpu2_tune.json $TUNEC --mode tune --reps 3 --iparm-file $H/iparm_tuned.json
# 2. Table 5 Cholesky, 3 repetitions
for k in $(seq 1 $NREP); do
  run t5_chol_rep$k $PY -u bench_cpu2.py $H/bench_cpu2_chol_rep$k.json $G --mode main --reps 3 --s-budget $SB --iparm-file $H/iparm_tuned.json
done
# 3. Table 5 LU reference
run t5_luref $PY -u bench_cpu2.py $H/bench_cpu2_luref.json $G --mode luref --reps 3
# 4. Table 6
for L in $LATS4; do
  LAY=$S4/$L.json; [ "$SMOKE" = 1 ] && LAY=$R/smoke/$L.json
  for k in $(seq 1 $NREP); do
    run t6_${L}_chol_rep$k $PY -u lat_direct_cpu2.py $H/lat2_${L}_chol_rep$k.json $LAY --mtypes 2 --iparm-file $H/iparm_tuned.json --margin-gib 4
  done
  run t6_${L}_lu $PY -u lat_direct_cpu2.py $H/lat2_${L}_lu.json $LAY --mtypes 11 --margin-gib 4
done
for L in $LAT8; do
  for k in $(seq 1 $NREP); do
    run t6_${L}_chol_rep$k $PY -u lat_direct_cpu2.py $H/lat2_${L}_chol_rep$k.json $S4/$L.json --mtypes 2 --iparm-file $H/iparm_tuned.json --margin-gib 4
  done
done
if [ "$SMOKE" != 1 ]; then
  run t6_hlat331_chol_rep1 $PY -u lat_direct_cpu2.py $H/lat2_hlat331_chol_rep1.json $S4/hlat331.json --mtypes 2 --iparm-file $H/iparm_tuned.json --margin-gib 4
  if grep -q '"factor_s"' $H/lat2_hlat331_chol_rep1.json 2>/dev/null; then
    for k in 2 3; do
      run t6_hlat331_chol_rep$k $PY -u lat_direct_cpu2.py $H/lat2_hlat331_chol_rep$k.json $S4/hlat331.json --mtypes 2 --iparm-file $H/iparm_tuned.json --margin-gib 4
    done
  fi
fi
# 5. summary
run summary $PY r1_host_summary.py $H $H/host_summary.json
mark
# 6. E7 retry of guard-refused solves (after the timing work; quiet host, headroom 4 GiB)
if [ "$SMOKE" != 1 ] && [ -f $R/ref_valid_r1.json ]; then
  export MKL_NUM_THREADS=16 OMP_NUM_THREADS=16
  RC2=0
  for stage in sweep studies; do for c in fresh_val_2010_d0_v0 fresh_val_2005_d1_v0 fresh_val_2074_d0_v0 fresh_val_2051_d1_v1; do
    [ $stage = studies ] && [ $c = fresh_val_2005_d1_v0 ] && continue
    t0=$(date +%s)
    $PY -u ref_valid2.py $R/ref_valid_r1.json $c --ns 24,32,40,48,56,64 --stage $stage --headroom-gib 4 --retry-errors >> $R/ref_logs/ref_${c}_retry_${stage}.log 2>&1
    rc=$?; [ $rc = 0 ] || RC2=$rc; st "REF_RETRY $stage $c rc=$rc $(( $(date +%s)-t0 ))s"
  done; done
  st "REF_RETRY END rc=$RC2"; echo "$(date +%T) R1_REF64_DONE_20260928" >> $ST
fi
