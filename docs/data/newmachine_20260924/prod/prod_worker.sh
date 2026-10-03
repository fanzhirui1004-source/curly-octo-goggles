#!/bin/bash
# prod_worker.sh: claim cells from the shared queue (atomic mkdir), build bodies (CPU, frozen env), prep_geo + prep_geo2
# (GPU; glued: safe mode, explicit continuous neighbours), drop the regenerable ghost-penalty caches, ship one tar per cell
# to the transit dir (tar -> md5 -> status; the .md5 appears last). Local copies are removed once the main machine ACKs.
# STOP file in the queue dir: finish the current cell and exit.
Q=/autodl-fs/data/OPL_QUEUE; T=/autodl-fs/data/OPL_TRANSIT; S=/root/autodl-tmp/OPL/S3
DEP=/root/autodl-tmp/CUTFEM_DEPENDENCIES_20260924
source /root/autodl-tmp/OPL/S1/env_gpu.sh
export OPL_PACKETS_EXTRA=$S/packets CUTFEM_EXECUTION_CONFIG=$DEP/EXECUTION_CONFIG_NEW16.json
export BANK_SPLITS=512,64,64 SENS_REASSOC=1 GLUED_SAFE=1 GLUED_NEIGHBOURS=explicit GLUED_ALL_FACES=1
H=$(hostname | sed 's/autodl-container-//')
L=$S/logs; mkdir -p $L $T/ACK $Q/claims $Q/done
cd /root/autodl-tmp/OPL/src_v2
log(){ echo "$(date +%F_%T) $H $*" >> $L/worker.log; echo "$(date +%F_%T) $H $*" >> $Q/events.log; }
cleanup_acked(){ for a in $(ls $T/ACK 2>/dev/null); do
  [ -e $S/data/$a ] || [ -e $S/data_v2/$a ] || [ -e $S/body/$a ] || continue
  rm -rf $S/data/$a $S/data_v2/$a $S/body/$a $S/body/${a}_nb* $L/$a.*.log; done; }
log WORKER_START
while read c; do
  [ -f $Q/STOP ] && { log STOP; break; }
  mkdir $Q/claims/$c 2>/dev/null || continue
  echo $H > $Q/claims/$c/host
  cleanup_acked
  while [ $(df --output=used -BG /autodl-fs/data | tail -1 | tr -dc 0-9) -gt 170 ]; do sleep 60; done
  t0=$(date +%s); log START $c
  build(){ (cd $DEP/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/COVER_G/src && OMP_NUM_THREADS=1 timeout 600 $DEP/run_frozen_python.sh /root/autodl-tmp/OPL/src_v2/fast_prep4.py 8 $S/body $1) >> $L/$c.body.log 2>&1; }
  build $c; fixed=""
  if [ -f $S/body/$c/PREP.json ]; then           # glue face from the body's material (planned face if it has material)
    tag=$(cd /tmp && OMP_NUM_THREADS=1 $DEP/run_frozen_python.sh /root/autodl-tmp/OPL/src_v2/gen_new.py $S --seed 2026092602 --make-neighbour $c --body $S/body 2>> $L/$c.body.log | tail -1)
    echo "{\"glue_tag\": \"$tag\"}" >> $L/$c.body.log
  fi
  nbs=$(ls $S/packets | grep "^${c}_nb" | tr '\n' ' ')
  for nb in $nbs; do [ -f $S/body/$nb/PREP.json ] || build $nb                   # one body per call: a failing neighbour cannot abort the others
    if [ ! -f $S/body/$nb/PREP.json ]; then     # e.g. topology not certified: far corners := shared face, rebuild
      (cd /tmp && OMP_NUM_THREADS=1 $DEP/run_frozen_python.sh /root/autodl-tmp/OPL/src_v2/gen_new.py $S --seed 2026092602 --fix-neighbour $nb) >> $L/$c.body.log 2>&1
      build $nb; fixed="$fixed packets/$nb"; log NEIGHBOUR_FIXED $nb $([ -f $S/body/$nb/PREP.json ] && echo ok || echo still_failing)
    fi
  done
  t1=$(date +%s); t2=$t1; st=OK
  if [ ! -f $S/body/$c/PREP.json ]; then st=BODY_FAIL; else
    timeout 3600 $PY -u prep_geo.py $S/body $S/data $c >> $L/$c.prep_geo.log 2>&1
    t2=$(date +%s)
    if [ -f $S/data/$c/DONE.json ]; then
      timeout 5400 $PY -u prep_geo2.py $S/body $S/data --out $S/data_v2 --classes force_c,face_c,support_k,glued --F-old $c >> $L/$c.prep_geo2.log 2>&1
      [ -f $S/data_v2/$c/DONE2.json ] || st=PREP2_INCOMPLETE
    else st=PREP_FAIL; fi
  fi
  t3=$(date +%s)
  rm -f $S/body/$c/GP_UPPER.npz $S/body/${c}_nb*/GP_UPPER.npz
  python3 - "$c" "$H" "$st" $t0 $t1 $t2 $t3 > $L/$c.status.json <<'PY'
import json, sys, os
c, h, st, t0, t1, t2, t3 = sys.argv[1], sys.argv[2], sys.argv[3], *map(int, sys.argv[4:8])
S = '/root/autodl-tmp/OPL/S3'
rec = dict(case=c, host=h, status=st, body_s=t1 - t0, prep_geo_s=t2 - t1, prep_geo2_s=t3 - t2, total_s=t3 - t0)
for k, p in (('done', f'{S}/data/{c}/DONE.json'), ('done2', f'{S}/data_v2/{c}/DONE2.json'), ('failed', f'{S}/data/{c}/FAILED.json')):
    if os.path.exists(p):
        try:
            rec[k] = json.load(open(p))
        except Exception as e:
            rec[k] = repr(e)
ev = {}
lp = f'{S}/logs/{c}.prep_geo2.log'
if os.path.exists(lp):
    for l in open(lp):
        if l.startswith('{'):
            try:
                x = json.loads(l)
            except Exception:
                continue
            e = x.get('event', '')
            if e in ('CLASS_FAILED2', 'PARTIAL2', 'FAILED2', 'GLUED_OFFSET_SKIPPED', 'GLUED_SAFE_RETRY', 'GLUED_SAFE_FP32', 'GLUED_OFFSET_REJECTED', 'CLASS_SKIPPED2'):
                ev.setdefault(e, []).append({k: x.get(k) for k in ('cls', 'offset', 'failed', 'error', 'skipped', 'stage') if k in x})
rec['events'] = ev
print(json.dumps(rec, default=str))
PY
  items="$( [ -d $S/body/$c ] && echo body/$c ) $(cd $S && ls -d body/${c}_nb* 2>/dev/null | tr '\n' ' ') logs/$c.body.log logs/$c.status.json $(cd $S && ls -d packets/${c}_nb* 2>/dev/null | tr '\n' ' ')"
  [ -d $S/data/$c ] && items="$items data/$c"; [ -d $S/data_v2/$c ] && items="$items data_v2/$c"
  for f in prep_geo prep_geo2; do [ -f $L/$c.$f.log ] && items="$items logs/$c.$f.log"; done
  if (cd $S && tar cf $T/$c.tar.part $items) && m=$(md5sum < $T/$c.tar.part | cut -c1-32) && mv $T/$c.tar.part $T/$c.tar && echo $m > $T/$c.md5; then
    touch $Q/done/$c; log END $c $st $((t3 - t0))s
  else
    log SHIP_FAIL $c; rm -f $T/$c.tar.part
  fi
done < $Q/ORDER.txt
log WORKER_EXIT
