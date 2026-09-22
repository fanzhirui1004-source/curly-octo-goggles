#!/bin/bash
# J1: CPU-only asset preparation for six development cases, one case at a time (Codex's frozen scripts,
# 8 numeric threads each, their own 30 GiB / 900 s guards). A failed case is recorded and skipped.
R=/root/autodl-tmp/CUTFEM_FRESH_GP_20260921
M=$R/diagnostics/MECHANICS_NETWORK_20260922_01
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923
O=$T/ASSETS_DEV
S=$T/xcase_src_09
PY=/root/cutfem_neural_a_20260910/env/bin/python
RT=$(for d in /root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime /root/autodl-tmp/CUTFEM_GP_TASK_V2_20260915_R1/RUNTIME_RECOVERY/environment/runtime; do sha256sum $d/r13_pardiso_v1/RUNTIME_MANIFEST.json | grep -q ^118030de7cce1acd1659e0ef3bb644941f4066bac7f58adadc4eda9525d891ae && { echo $d; break; }; done)
[ -n "$RT" ] || { echo NO_MATCHING_RUNTIME > /dev/stderr; exit 1; }
mkdir -p $O
sha256sum $M/source_assets_01/prepare_assets_v3.py $M/source_assets_01/validate_saved_native.py $M/source_v1/run_mechanics.py $S/dev_reference.py > $O/SOURCE_SHA256.txt
for c in fresh_development_0003_d0_v0 fresh_development_0006_d0_v0 fresh_development_0000_d0_v0 fresh_development_0004_d1_v2 fresh_development_0002_d0_v2 fresh_development_0002_full; do
  echo "$(date -u +%FT%TZ) START $c" >> $O/run.status
  if ! $PY $M/source_assets_01/prepare_assets_v3.py --root $R --case $c --output $O/$c --compile-only > $O/$c.compile.log 2>&1; then
    echo "$(date -u +%FT%TZ) FAIL_COMPILE $c" >> $O/run.status; continue; fi
  if ! CUTFEM_RUNTIME_ROOT=$RT CUTFEM_EXECUTION_CONFIG=$M/NATIVE_EXECUTION_CONFIG.json PYTHONPATH=$M/source_assets_01:$M/native_source_01 $PY $M/source_assets_01/validate_saved_native.py --assets $O/$c --packet $R/packets/$c --output $O/${c}_NATIVE > $O/$c.native.log 2>&1; then
    echo "$(date -u +%FT%TZ) FAIL_NATIVE $c" >> $O/run.status; continue; fi
  if ! OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 $PY $S/dev_reference.py --case $c --asset $O/$c --qualification $O/${c}_NATIVE --output $O/${c}_REFERENCE > $O/$c.reference.log 2>&1; then
    echo "$(date -u +%FT%TZ) FAIL_REFERENCE $c" >> $O/run.status; continue; fi
  echo "$(date -u +%FT%TZ) DONE $c" >> $O/run.status
done
echo "$(date -u +%FT%TZ) ALL_FINISHED" >> $O/run.status
