#!/bin/bash
# Revision round 1, E7 (agent tag R1REF): CutFEM reference refinement to n = 56 / 64, CPU only.
# Cells: H1 fresh_val_2005_d1_v0 (existing n = 24..48 record seeded from ref_valid_h1.json; adds 56, 64),
#        H2 fresh_val_2010_d0_v0, A3's worst geometry fresh_val_2051_d1_v1, thinnest-wall validation cell fresh_val_2074_d0_v0
#        (all corners tau 0.1768-0.1777; n = 24..64 each).
# Steps: packets <case>_n<n> (context with n replaced, as run_refvalid.sh) -> bodies by fast_prep4 (frozen env) ->
#        ref_valid2.py --stage sweep per case (memory guard, see below) -> --stage studies (n = 32).
# Writes the line "HH:MM:SS R1_REF_DONE_20260928" to chain_p1.status at the end, also on failure (EXIT trap).
# Memory (coordinator: X2's LAT_CPU gates need 20-40 GB from ~21:30 to ~03:00): before each process the chain waits until
# memory.high - container anon >= 60 GiB; inside, a factorisation runs only if predicted PARDISO memory + 45 GiB fits in
# memory.high - container anon. E7's own footprint therefore stays below ~43 GiB and >= 45 GiB remain for the pilot / X2.
# Solves refused by the guard are recorded as errors and retried in the quiet window of the E1 chain (headroom 4 GiB).
# SMOKE=1: H2 only, n = 24, 32, output in R1/smoke, no marker.
O=/root/autodl-tmp/OPL/S1/V2; ST=$O/chain_p1.status; R=$O/R1; S4=/root/autodl-tmp/OPL/S4; SRC=/root/autodl-tmp/OPL/src_v2
DEP=/root/autodl-tmp/CUTFEM_DEPENDENCIES_20260924
if [ "$SMOKE" = 1 ]; then
  OUT=$R/smoke/ref_valid_r1_smoke.json; LOGD=$R/smoke; TAG="R1REF SMOKE"; NS=24,32; NB="24"
  CASES="fresh_val_2010_d0_v0"; NEWC="fresh_val_2010_d0_v0"
else
  OUT=$R/ref_valid_r1.json; LOGD=$R/ref_logs; TAG="R1REF"; NS=24,32,40,48,56,64; NB="24 40 48 56 64"
  CASES="fresh_val_2010_d0_v0 fresh_val_2005_d1_v0 fresh_val_2074_d0_v0 fresh_val_2051_d1_v1"
  NEWC="fresh_val_2010_d0_v0 fresh_val_2074_d0_v0 fresh_val_2051_d1_v1"
fi
mkdir -p $LOGD
st() { echo "$(date +%T) $TAG $*" >> $ST; }
RC=0
fin() { if [ "$SMOKE" = 1 ]; then st "SMOKE_END rc=$RC"; else st "END rc=$RC"; echo "$(date +%T) R1_REF_DONE_20260928" >> $ST; fi; }
trap fin EXIT
st "START pid=$$ cases=$(echo $CASES | tr ' ' ',') ns=$NS"
avail() { python3 -c "
a=int([l.split()[1] for l in open('/sys/fs/cgroup/memory.stat') if l.startswith('anon ')][0]); h=int(open('/sys/fs/cgroup/memory.high').read())
print(int((h-a)/2**30))"; }
waitmem() {  # waitmem <GiB needed incl. 25 GiB headroom>
  local n=0; while [ $(avail) -lt $1 ]; do [ $n = 0 ] && st "WAIT_MEM need $1 GiB avail $(avail) GiB"; n=1; sleep 120; done; }
# 1. packets
/root/autodl-tmp/gpuenv/bin/python - "$NB" $NEWC <<'PY'
import json, shutil, sys
from pathlib import Path
S3 = Path('/root/autodl-tmp/OPL/S3/packets'); S4 = Path('/root/autodl-tmp/OPL/S4/packets')
ns = [int(x) for x in sys.argv[1].split()]
plan = {c: ns for c in sys.argv[2:]}
if len(ns) > 2: plan['fresh_val_2005_d1_v0'] = [56, 64]
for c, nl in plan.items():
    for n in nl:
        d = S4 / f'{c}_n{n}'
        if (d / 'FRESH_CONTEXT.json').exists(): continue
        d.mkdir(parents=True, exist_ok=True)
        ctx = json.loads((S3 / c / 'FRESH_CONTEXT.json').read_text()); ctx['n'] = n
        ctx['provenance']['refinement_of'] = c
        (d / 'FRESH_CONTEXT.json').write_text(json.dumps(ctx, indent=1)); shutil.copy(S3 / c / 'SAMPLE.json', d / 'SAMPLE.json')
        print('packet', d)
PY
# 2. bodies (sequential; existing ones kept)
export OPL_PACKETS_EXTRA=$S4/packets:/root/autodl-tmp/OPL/S3/packets
for c in $CASES; do for n in $NB; do
  [ -d $S4/packets/${c}_n$n ] || continue
  [ -f $S4/body/${c}_n$n/NODES.npy ] && continue
  t0=$(date +%s)
  (cd $DEP/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/COVER_G/src && OMP_NUM_THREADS=1 timeout 7200 $DEP/run_frozen_python.sh $SRC/fast_prep4.py 1 $S4/body ${c}_n$n) > $S4/body_${c}_n$n.log 2>&1
  rc=$?; [ $rc = 0 ] || RC=$rc; st "BODY ${c}_n$n rc=$rc $(( $(date +%s)-t0 ))s"
done; done
# 3. sweeps, then 4. studies (one process per case and stage)
source $R/env_cpu.sh
export MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 OPL_GP_CACHE=0
cd $SRC
for stage in sweep studies; do for c in $CASES; do
  [ $stage = studies ] && [ $c = fresh_val_2005_d1_v0 ] && continue
  SEED=""; [ $c = fresh_val_2005_d1_v0 ] && SEED="--seed $O/ref_valid_h1.json"
  waitmem 60
  t0=$(date +%s)
  $PY -u ref_valid2.py $OUT $c --ns $NS --stage $stage --headroom-gib 45 $SEED >> $LOGD/ref_${c}_${stage}.log 2>&1
  rc=$?; [ $rc = 0 ] || RC=$rc; st "$stage $c rc=$rc $(( $(date +%s)-t0 ))s"
done; done
