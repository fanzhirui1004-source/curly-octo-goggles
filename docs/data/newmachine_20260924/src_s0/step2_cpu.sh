DEP=/root/autodl-tmp/CUTFEM_DEPENDENCIES_20260924
O=/root/autodl-tmp/OPL/S0; S1=/root/autodl-tmp/OPL/S1; PY=/root/autodl-tmp/gpuenv/bin/python
export CUTFEM_EXECUTION_CONFIG=$DEP/EXECUTION_CONFIG_NEW16.json
for c in $(cat $S1/step2_cases.txt); do
  if [ -f $O/$c/READY ] || [ -f $O/$c/FAILED_BODY ]; then continue; fi
  if [ -f $O/$c/PREP.json ] && [ -f $O/$c/GP_UPPER.npz ]; then touch $O/$c/READY; continue; fi
  if [ ! -f $O/$c/PREP.json ]; then
    (cd $DEP/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/COVER_G/src && OMP_NUM_THREADS=1 timeout 900 $DEP/run_frozen_python.sh /root/autodl-tmp/OPL/src/fast_prep4.py 8 $O $c >> $S1/step2_body.log 2>&1)
  fi
  if [ ! -f $O/$c/PREP.json ]; then mkdir -p $O/$c; touch $O/$c/FAILED_BODY; echo "BODY_FAIL $c" >> $S1/step2_body.log; continue; fi
  while [ $(pgrep -fc "[g]p_cpu.py") -ge 4 ]; do sleep 2; done
  (cd /root/autodl-tmp/OPL/src && OMP_NUM_THREADS=2 $PY -u gp_cpu.py $O $c >> $S1/step2_gp.log 2>&1 && touch $O/$c/READY || { touch $O/$c/FAILED_BODY; echo "GP_FAIL $c" >> $S1/step2_body.log; }) &
done
wait
echo STEP2_CPU_DONE >> $S1/step2_body.log
