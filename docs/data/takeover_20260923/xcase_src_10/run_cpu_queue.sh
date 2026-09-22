#!/bin/bash
# Overnight CPU queue: element-level tolerance (P2) on the cases the default sparse LU cannot hold,
# using Codex's native PARDISO (same runtime manifest 118030de...), 6 threads, one case at a time.
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923
S=$T/xcase_src_10
M=/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/diagnostics/MECHANICS_NETWORK_20260922_01
PY=/root/cutfem_neural_a_20260910/env/bin/python
O=$T/ELEMENT_TOL_01
RT=$(for d in /root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime /root/autodl-tmp/CUTFEM_GP_TASK_V2_20260915_R1/RUNTIME_RECOVERY/environment/runtime; do sha256sum $d/r13_pardiso_v1/RUNTIME_MANIFEST.json | grep -q ^118030de7cce1acd1659e0ef3bb644941f4066bac7f58adadc4eda9525d891ae && { echo $d; break; }; done)
[ -n "$RT" ] || { echo NO_MATCHING_RUNTIME >> $O/cpu_queue.status; exit 1; }
export CUTFEM_RUNTIME_ROOT=$RT CUTFEM_EXECUTION_CONFIG=$M/NATIVE_EXECUTION_CONFIG.json PYTHONPATH=$M/native_source_01
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
until [ -f $O/fresh_train_0003_d0_v0/RESULT.json ] && [ $(cat $O/j2b.status 2>/dev/null | wc -l) -ge 3 ]; do sleep 30; done
for c in fresh_train_0013_d0_v1 fresh_development_0002_d0_v2 fresh_development_0004_d1_v2; do
  $PY -u $S/element_tolerance_v2.py --case $c --solver pardiso --threads 6 --work $T/ELEMENT_WORK --output $O/$c > $O/$c.log 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) $c exit $?" >> $O/cpu_queue.status
done
echo "$(date -u +%FT%TZ) CPU_QUEUE_FINISHED" >> $O/cpu_queue.status
