source /root/autodl-tmp/OPL/queue/env.sh
export BANK_SPLITS=512,64,64 PREP_LOCK=/root/autodl-tmp/OPL/queue PREP_WAIT_READY=1
cd /root/autodl-tmp/OPL/src
$PY -u prep_geo.py $S /root/autodl-tmp/OPL/S2/data $(cat $S1/step2_cases.txt) >> $S1/step2_gpu.log 2>&1
echo STEP2_GPU_DONE >> $S1/step2_gpu.log
