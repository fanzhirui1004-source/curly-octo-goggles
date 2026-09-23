#!/bin/bash
# Route 2 D1 in the frozen runtime environment (PARDISO libraries). Usage: run_d1.sh <case> <H>
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; D=$T/xcase_src_27; W=$T/COVER_G; V=/root/autodl-tmp/CUTFEM_INGEST_R38/environment/.venv/bin/python
export CUTFEM_EXECUTION_CONFIG=$W/EXECUTION_CONFIG.json CUTFEM_RUNTIME_ROOT=/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime
eval "$(cd $W/src && PYTHONPATH=$W/src $V -B -c "import json,shlex;from stage_cutfem_runtime.environment import runtime_environment;c=json.load(open('$W/EXECUTION_CONFIG.json'));e,b=runtime_environment('$W/src',c['runtime_names']);print('\n'.join('export %s=%s'%(k,shlex.quote(v)) for k,v in e.items()))")"
mkdir -p $T/R2_D1
cd $W/src && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 $V -B $D/dual_d1.py $1 $2 $T/R2_D1/${1}_H$2_halo$3.json $3 > $T/R2_D1/${1}_H$2_halo$3.log 2>&1
rc=$?
echo "$(date -u +%FT%TZ) D1 $1 H$2 halo$3 rc $rc" >> $T/R2_D1/status
