#!/bin/bash
# Geometry (G) stage for the private tau-perturbed sensitivity panel, with the same private copy of the frozen source,
# execution config and runtime as COVER_G. Usage: run_sens_g.sh <workers> <case_id> [...]
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923
W=$T/COVER_G; X=$T/SENS_01
PANEL=$X/panel
V=/root/autodl-tmp/CUTFEM_INGEST_R38/environment/.venv/bin/python
NW=$1; shift
mkdir -p $X/runs
export CUTFEM_EXECUTION_CONFIG=$W/EXECUTION_CONFIG.json CUTFEM_RUNTIME_ROOT=/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime
eval "$(cd $W/src && PYTHONPATH=$W/src $V -B -c "import json,shlex;from stage_cutfem_runtime.environment import runtime_environment;c=json.load(open('$W/EXECUTION_CONFIG.json'));e,b=runtime_environment('$W/src',c['runtime_names']);print('\n'.join('export %s=%s'%(k,shlex.quote(v)) for k,v in e.items()))")"
[ -n "$LD_LIBRARY_PATH" ] || { echo NO_RUNTIME_ENV >> $X/run.status; exit 1; }
for c in "$@"; do
  O=$X/runs/${c}_G
  [ -f $O/RESULT.json ] && continue
  rm -rf $O; mkdir -p $O
  t0=$(date +%s)
  ( cd $W/src && CUTFEM_MEMORY_POLICY=container_headroom_1g CUTFEM_ASSEMBLY_WORKERS=$NW CUTFEM_NUMERIC_THREADS=$NW CUTFEM_STAGE_OUTPUT=$O \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
    $V -B -m stage_cutfem_multiconstraint.trial geometry --panel-run $PANEL --case-id $c --n 32 ) > $O.log 2>&1
  echo "$(date -u +%FT%TZ) G $c exit $? seconds $(( $(date +%s) - t0 ))" >> $X/run.status
done
