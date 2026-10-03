#!/bin/bash
# Geometry stage of one registered second-batch case into a given directory (same private source copy, config and
# runtime as COVER_G; 8 workers = the host's assembly allocation). Usage: run_e2e_g.sh <case> <out dir>
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; W=$T/COVER_G
PANEL=/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/CUT_COVER80_20260922_CONFIG_V3/panel
V=/root/autodl-tmp/CUTFEM_INGEST_R38/environment/.venv/bin/python
export CUTFEM_EXECUTION_CONFIG=$W/EXECUTION_CONFIG.json CUTFEM_RUNTIME_ROOT=/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime
eval "$(cd $W/src && PYTHONPATH=$W/src $V -B -c "import json,shlex;from stage_cutfem_runtime.environment import runtime_environment;c=json.load(open('$W/EXECUTION_CONFIG.json'));e,b=runtime_environment('$W/src',c['runtime_names']);print('\n'.join('export %s=%s'%(k,shlex.quote(v)) for k,v in e.items()))")"
rm -rf $2; mkdir -p $2
cd $W/src && CUTFEM_MEMORY_POLICY=container_headroom_1g CUTFEM_ASSEMBLY_WORKERS=8 CUTFEM_NUMERIC_THREADS=8 CUTFEM_STAGE_OUTPUT=$2 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  $V -B -m stage_cutfem_multiconstraint.trial geometry --panel-run $PANEL --case-id $1 --n 32 > $2.log 2>&1
