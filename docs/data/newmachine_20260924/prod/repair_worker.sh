#!/bin/bash
# repair_worker.sh: redo the glued class of shipped cells listed in the repair queue (cells made before the glue-face /
# refinement / neighbour fixes), then continue with the production worker. Case data come from the main machine as
# /autodl-fs/data/OPL_REPAIR/<case>.tar; the repaired cell ships like a produced one (the puller overwrites it).
Q=/autodl-fs/data/OPL_QUEUE; T=/autodl-fs/data/OPL_TRANSIT; R=/autodl-fs/data/OPL_REPAIR; S=/root/autodl-tmp/OPL/S3
DEP=/root/autodl-tmp/CUTFEM_DEPENDENCIES_20260924
source /root/autodl-tmp/OPL/S1/env_gpu.sh
export OPL_PACKETS_EXTRA=$S/packets CUTFEM_EXECUTION_CONFIG=$DEP/EXECUTION_CONFIG_NEW16.json
export BANK_SPLITS=512,64,64 SENS_REASSOC=1 GLUED_SAFE=1 GLUED_NEIGHBOURS=explicit GLUED_ALL_FACES=1
H=$(hostname | sed 's/autodl-container-//'); L=$S/logs; mkdir -p $L $Q/repair_claims
cd /root/autodl-tmp/OPL/src_v2
log(){ echo "$(date +%F_%T) $H $*" >> $Q/events.log; }
build(){ (cd $DEP/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/COVER_G/src && OMP_NUM_THREADS=1 timeout 600 $DEP/run_frozen_python.sh /root/autodl-tmp/OPL/src_v2/fast_prep4.py 8 $S/body $1) >> $L/$c.repair.log 2>&1; }
[ -f $Q/REPAIR.txt ] && while read c; do
  [ -f $Q/STOP ] && break
  [ -f $R/$c.tar ] || continue
  mkdir $Q/repair_claims/$c 2>/dev/null || continue
  t0=$(date +%s); log REPAIR_START $c
  cp $R/$c.tar /root/autodl-tmp/OPL/_rep.tar && (cd $S && tar xf /root/autodl-tmp/OPL/_rep.tar --overwrite) && rm -f /root/autodl-tmp/OPL/_rep.tar
  tag=$(cd /tmp && OMP_NUM_THREADS=1 $DEP/run_frozen_python.sh /root/autodl-tmp/OPL/src_v2/gen_new.py $S --seed 2026092602 --make-neighbour $c --body $S/body 2>> $L/$c.repair.log | tail -1)
  for nb in $(ls $S/packets | grep "^${c}_nb"); do
    [ -f $S/body/$nb/PREP.json ] || build $nb
    if [ ! -f $S/body/$nb/PREP.json ]; then
      (cd /tmp && OMP_NUM_THREADS=1 $DEP/run_frozen_python.sh /root/autodl-tmp/OPL/src_v2/gen_new.py $S --seed 2026092602 --fix-neighbour $nb) >> $L/$c.repair.log 2>&1; build $nb
    fi
  done
  timeout 5400 $PY -u prep_geo2.py $S/body $S/data --out $S/data_v2 --classes glued --force $c >> $L/$c.repair.log 2>&1
  rm -f $S/body/$c/GP_UPPER.npz $S/body/${c}_nb*/GP_UPPER.npz
  ok=$([ -f $S/data_v2/$c/train_glued.npy ] && echo glued_ok || echo glued_missing)
  items="body/$c $(cd $S && ls -d body/${c}_nb* packets/${c}_nb* 2>/dev/null | tr '\n' ' ') data/$c data_v2/$c logs/$c.repair.log"
  if (cd $S && tar cf $T/$c.tar.part $items) && m=$(md5sum < $T/$c.tar.part | cut -c1-32) && mv $T/$c.tar.part $T/$c.tar && echo $m > $T/$c.md5; then
    log REPAIR_END $c $ok tag=$tag $(( $(date +%s) - t0 ))s; rm -f $R/$c.tar
  else log REPAIR_SHIP_FAIL $c; rm -f $T/$c.tar.part; fi
  rm -rf $S/data/$c $S/data_v2/$c $S/body/$c $S/body/${c}_nb*
done < $Q/REPAIR.txt
exec bash /root/autodl-tmp/OPL/prod_worker.sh
