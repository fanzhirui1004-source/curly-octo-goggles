#!/bin/bash
# Recompute the geometry (G) stage of second-batch cases on this host from a private copy of the frozen
# source tree (the frozen tree itself is never written). Usage: run_cover_g.sh <workers> <case> [<case> ...]
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923
W=$T/COVER_G
F=/root/autodl-tmp/CUTFEM_INGEST_R38/source_independent_6624dc8_20260910
R=/root/autodl-tmp/CUTFEM_FRESH_GP_20260921
PANEL=$R/CUT_COVER80_20260922_CONFIG_V3/panel
V=/root/autodl-tmp/CUTFEM_INGEST_R38/environment/.venv/bin/python
NW=$1; shift
mkdir -p $W/runs
if [ ! -d $W/src ]; then
  mkdir -p $W/src.pending && tar -C $F --exclude=./artifacts --exclude=./.git -cf - . | tar -C $W/src.pending -xf - && mv $W/src.pending $W/src
  (cd $F && git rev-parse HEAD > $W/SOURCE_HEAD.txt 2>&1)
fi
if [ ! -f $W/EXECUTION_CONFIG.json ]; then
  $V -c "import json,platform;json.dump(dict(schema='CUTFEM_EXECUTION_CONFIG_V1',target='gpu',targets=dict(gpu=dict(hostname=platform.node(),project_root='$W',scratch_root='$W/scratch',cold_archive_root='$W/cold',allocated_cores=16,allocated_memory_gib=90)),parallelism=dict(assembly_workers=8,numeric_threads=8,worker_numeric_threads=1),runtime_names=['r9_reference_v1','r10_algoim_v3','r13_pardiso_v1','r14_fitted_v2','r17_algoim_v2','r22_gmpy2_v1'],authorization=dict(bulk_data_production=False,neural_training=False)),open('$W/EXECUTION_CONFIG.json','w'),indent=2)"
  mkdir -p $W/scratch $W/cold
fi
export CUTFEM_EXECUTION_CONFIG=$W/EXECUTION_CONFIG.json CUTFEM_RUNTIME_ROOT=/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime
# Same import/dynamic-library conventions as the frozen runner (stage_cutfem_runtime.environment).
eval "$(cd $W/src && PYTHONPATH=$W/src $V -B -c "import json,shlex;from stage_cutfem_runtime.environment import runtime_environment;c=json.load(open('$W/EXECUTION_CONFIG.json'));e,b=runtime_environment('$W/src',c['runtime_names']);print('\n'.join('export %s=%s'%(k,shlex.quote(v)) for k,v in e.items()))")"
[ -n "$LD_LIBRARY_PATH" ] || { echo NO_RUNTIME_ENV >> $W/run.status; exit 1; }
for c in "$@"; do
  O=$W/runs/${c}_G
  [ -f $O/RESULT.json ] && { echo "$(date -u +%FT%TZ) $c already" >> $W/run.status; continue; }
  rm -rf $O; mkdir -p $O
  echo "$(date -u +%FT%TZ) START $c" >> $W/run.status
  ( cd $W/src && CUTFEM_MEMORY_POLICY=container_headroom_1g CUTFEM_ASSEMBLY_WORKERS=$NW CUTFEM_NUMERIC_THREADS=$NW CUTFEM_STAGE_OUTPUT=$O \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
    $V -B -m stage_cutfem_multiconstraint.trial geometry --panel-run $PANEL --case-id $c --n 32 ) > $O.log 2>&1
  echo "$(date -u +%FT%TZ) END $c exit $?" >> $W/run.status
done
