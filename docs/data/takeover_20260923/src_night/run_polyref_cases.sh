#!/bin/bash
# Exact 125-moment truth (J4a), polyhedral reference moments (s=4,8), then the assembled-spectrum replacement
# test with PARDISO, for several first-batch cases. CPU only, one case at a time.
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_15
M=/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/diagnostics/MECHANICS_NETWORK_20260922_01
PY=/root/cutfem_neural_a_20260910/env/bin/python
export CUTFEM_RUNTIME_ROOT=/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime CUTFEM_EXECUTION_CONFIG=$M/NATIVE_EXECUTION_CONFIG.json PYTHONPATH=$M/native_source_01
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $S
for c in "$@"; do
  [ -f $T/ELEMENT_MOMENTS_01/$c/MOMENTS125.npz ] || $PY -u element_moments.py --case $c --orders --output $T/ELEMENT_MOMENTS_01/$c > $T/ELEMENT_MOMENTS_01/$c.log 2>&1
  [ -f $T/POLYREF_01/$c/POLYREF_S8.npz ] || $PY -u element_polyref.py --case $c --moments $T/ELEMENT_MOMENTS_01/$c --subgrids 4 8 --output $T/POLYREF_01/$c > $T/POLYREF_01_$c.log 2>&1
  for s in 4 8; do
    $PY -u element_tolerance_v3.py --case $c --solver pardiso --threads 4 --families replace --eps 0 --replace-moments $T/POLYREF_01/$c/POLYREF_S$s.npz --work $T/ELEMENT_WORK --output $T/ELEMENT_TOL_03/${c}_S${s}_all > $T/ELEMENT_TOL_03/${c}_S${s}_all.log 2>&1
    echo "$(date -u +%FT%TZ) $c S$s exit $?" >> $T/ELEMENT_TOL_03/cases.status
  done
done
