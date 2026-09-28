#!/bin/bash
# Revision round 1, lattice-level route (b) (agent tag R1COND): conventional exact condensation of the Table 6 lattices on
# the host (lat_cond_cpu.py). Starts only after the line "HH:MM:SS R1_HOST_DONE_20260928" (anchored), after the E1 chain's
# own follow-up (E7 retry) has ended, and when the container is quiet (as E1: < 1.5 cores over 60 s, no other python
# > 2 GiB, no GPU process; at most 3 h, then NOT_QUIET is recorded and it proceeds).
# Runs (MKL_NUM_THREADS = OMP_NUM_THREADS = 16, tuned iparm of E1), solver 'block' (dense-kernel block Cholesky in the
# substructuring order): hlat221a, hlat221b x 3 repetitions; hlat222, hlat331 x 1; plus one 'pardiso' run (assembled CSR,
# sparse PARDISO Cholesky) on hlat221b for comparison (all memory-guarded, margin 4 GiB). Summary R1/hostcond/hostcond_summary.json.
# Writes the line "HH:MM:SS R1_HOSTCOND_DONE_20260928" at the end, also on failure (EXIT trap).
# SMOKE=1: no waiting, 2-cell layout, 8 threads, block + pardiso (pardiso_64 forced, tests the ILP64 path), output R1/smoke/cond.
O=/root/autodl-tmp/OPL/S1/V2; ST=$O/chain_p1.status; R=$O/R1; SRC=/root/autodl-tmp/OPL/src_v2; S4=/root/autodl-tmp/OPL/S4
if [ "$SMOKE" = 1 ]; then
  H=$R/smoke/cond; TAG="R1COND SMOKE"; RUNS="hlatsmoke2:1:block hlatsmoke2:1:pardiso"; TH=8; X="--ilp64"; IPF=$R/smoke/host/iparm_tuned.json; RH=$R/smoke/host
else
  H=$R/hostcond; TAG="R1COND"; RUNS="hlat221a:3:block hlat221b:3:block hlat222:1:block hlat331:1:block hlat221b:1:pardiso"; TH=16; X=""; IPF=$R/host/iparm_tuned.json; RH=$R/host
fi
mkdir -p $H
st() { echo "$(date +%T) $TAG $*" >> $ST; }
RC=0
fin() { if [ "$SMOKE" = 1 ]; then st "SMOKE_END rc=$RC"; else st "END rc=$RC"; echo "$(date +%T) R1_HOSTCOND_DONE_20260928" >> $ST; fi; }
trap fin EXIT
if [ "$SMOKE" != 1 ]; then
  st "QUEUED pid=$$ (after the host-timing chain and a quiet container)"
  until grep -qE "^[0-9]{2}:[0-9]{2}:[0-9]{2} R1_HOST_DONE_20260928$" $ST; do sleep 300; done
  until grep -qE "^[0-9]{2}:[0-9]{2}:[0-9]{2} R1_REF64_DONE_20260928$" $ST || ! pgrep -f chain_r1_host.sh > /dev/null; do sleep 300; done
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
st "START pid=$$"
source $R/env_cpu.sh
export MKL_NUM_THREADS=$TH OMP_NUM_THREADS=$TH OPL_GP_CACHE=0
cd $SRC
$PY lat_cond_cpu.py --selftest > $H/selftest.json 2>&1; rc=$?; [ $rc = 0 ] || RC=$rc; st "selftest rc=$rc $(tail -c 200 $H/selftest.json)"
for item in $RUNS; do
  L=$(echo $item | cut -d: -f1); N=$(echo $item | cut -d: -f2); SV=$(echo $item | cut -d: -f3); SUF=""; [ $SV = pardiso ] && SUF=_pardiso
  LAY=$S4/$L.json; [ "$SMOKE" = 1 ] && LAY=$R/smoke/$L.json
  REF=$RH/lat2_${L}_chol_rep1.json,$RH/lat2_${L}_chol_rep2.json,$(ls $O/lat_direct_${L}*.json 2>/dev/null | tr '\n' ',')
  for k in $(seq 1 $N); do
    t0=$(date +%s)
    XX=""; [ $SV = pardiso ] && XX=$X
    $PY -u lat_cond_cpu.py $H/latcond${SUF}_${L}_rep$k.json $LAY --solver $SV --iparm-file $IPF --margin-gib 4 --ref "$REF" $XX > $H/latcond${SUF}_${L}_rep$k.log 2>&1
    rc=$?; [ $rc = 0 ] || RC=$rc; st "latcond $SV $L rep$k rc=$rc $(( $(date +%s)-t0 ))s load=$(cut -d' ' -f1 /proc/loadavg)"
  done
done
$PY - $H <<'PY' > $H/summary.log 2>&1
import json, glob, sys, numpy as np
from pathlib import Path
H = sys.argv[1]; out = {}
agg = lambda v: None if not [x for x in v if x is not None] else dict(median=float(np.median([x for x in v if x is not None])), min=float(min(x for x in v if x is not None)), n=len([x for x in v if x is not None]))
import re
keys = sorted({(m.group(1) or '_block', m.group(2)) for f in glob.glob(f'{H}/latcond*_rep*.json')
               for m in [re.match(r'latcond(_pardiso)?_(.+)_rep\d+\.json$', Path(f).name)] if m})
for sv, L in keys:
    pre = 'latcond_pardiso' if sv == '_pardiso' else 'latcond'
    rs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(f'{H}/{pre}_{L}_rep*.json'))]
    ok = [r for r in rs if 'compliance' in r]
    o = dict(reps=len(rs), completed=len(ok), skipped=[r.get('skipped') for r in rs if 'skipped' in r],
             free_retained=rs[0].get('free_retained'), nnz_upper=rs[0].get('nnz_upper'), csr_GiB=rs[0].get('csr_GiB'),
             dense_S_total_GiB=rs[0].get('dense_S_total_GiB', rs[0].get('dense_S_total_GiB_predicted')),
             predicted_GiB=rs[0].get('predicted_GiB'), interface=rs[0].get('interface'), solver=rs[0].get('solver'),
             n_shared=rs[0].get('n_shared'), interface_dense_GiB=rs[0].get('interface_dense_GiB'), held_factor_GiB=rs[0].get('held_factor_GiB'))
    for k in ('front_end_s', 'schur_s', 'lattice_s', 'structure_s', 'indices_s', 'values_s', 'analysis_s', 'factor_s', 'solve_s',
              'condensed_s', 'total_s', 'process_peak_rss_GiB', 'assembly_peak_rss_GiB', 'factor_solve_peak_rss_GiB', 'throttled_s',
              'private_elimination_s', 'interface_factor_solve_s', 'back_substitution_s', 'block_peak_rss_GiB'):
        o[k] = agg([r.get(k) for r in ok])
    if ok:
        if 'pardiso' in ok[0]:
            o['pardiso_GiB'] = ok[0]['pardiso']['peak_mem_GiB']; o['nnz_factor'] = ok[0]['pardiso']['nnz_factor']
        o['schur_pardiso_GiB_max'] = max(c['schur']['pardiso']['peak_mem_GiB']['total'] for c in ok[0]['cells'].values())
        o['rel_residual_max'] = max((max(r['rel_residual']) for r in ok if r.get('rel_residual')), default=None)
        o['compliance'] = ok[0]['compliance']; o['reference'] = ok[0].get('reference')
    out[f'{sv.strip("_")}:{L}'] = o
Path(f'{H}/hostcond_summary.json').write_text(json.dumps(out, indent=1, default=float)); print('ok')
PY
st "summary rc=$?"
