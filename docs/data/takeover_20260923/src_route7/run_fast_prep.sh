#!/bin/bash
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; D=$T/xcase_src_25; W=$T/COVER_G; V=/root/autodl-tmp/CUTFEM_INGEST_R38/environment/.venv/bin/python
export CUTFEM_EXECUTION_CONFIG=$W/EXECUTION_CONFIG.json CUTFEM_RUNTIME_ROOT=/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime
eval "$(cd $W/src && PYTHONPATH=$W/src $V -B -c "import json,shlex;from stage_cutfem_runtime.environment import runtime_environment;c=json.load(open('$W/EXECUTION_CONFIG.json'));e,b=runtime_environment('$W/src',c['runtime_names']);print('\n'.join('export %s=%s'%(k,shlex.quote(v)) for k,v in e.items()))")"
mkdir -p $T/R7_04
cd $W/src
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 $V -B $D/fast_prep.py "$@" >> $T/R7_04/run.log 2>&1
rc=$?
echo "$(date -u +%FT%TZ) fast_prep rc $rc" >> $T/R7_04/status
