source /root/autodl-tmp/OPL/queue/env.sh
D=/root/autodl-tmp/OPL/S2/data
export BANK_SPLITS=512,64,64 PREP_LOCK=$Q PREP_WAIT_READY=1
cd /root/autodl-tmp/OPL/src
# pass 1: one process per geometry (a failed cuDSS allocation cannot leak into the next geometry)
echo "PASS1 start $(date +%T)" >> $S1/step2_gpu.log
for c in $(cat $S1/step2_cases.txt); do
  { [ -f $D/$c/DONE.json ] || [ -f $D/$c/FAILED.json ]; } && continue
  PREP_MIN_FREE_GB=9 PREP_WAIT_MAX=3600 $PY -u prep_geo.py $S $D $c >> $S1/step2_gpu.log 2>&1
done
# pass 2: memory failures again, when the GPU has room (after the running training)
echo "PASS2 start $(date +%T)" >> $S1/step2_gpu.log
for f in $D/*/FAILED.json; do [ -f "$f" ] && grep -q "ALLOC_FAILED\|out of memory\|OutOfMemory" $f && rm -f $f; done
for c in $(cat $S1/step2_cases.txt); do
  { [ -f $D/$c/DONE.json ] || [ -f $D/$c/FAILED.json ]; } && continue
  rm -f $D/$c/*.npy $D/$c/NETDATA.npz $D/$c/PORTS.json
  PREP_MIN_FREE_GB=22 PREP_WAIT_MAX=36000 $PY -u prep_geo.py $S $D $c >> $S1/step2_gpu.log 2>&1
done
echo "STEP2_ALL_DONE $(date +%T)" >> $S1/step2_gpu.log; touch $S1/STEP2_ALL_DONE
