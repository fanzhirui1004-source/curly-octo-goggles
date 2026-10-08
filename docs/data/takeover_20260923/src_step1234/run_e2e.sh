#!/bin/bash
# Measured end-to-end preparation: geometry parameters -> query-ready operator, one case at a time on an idle host.
# Stages: G (frozen geometry stage) -> P and GP (frozen trace / face selection) -> GPU encode (elements, P^T K P, cuDSS).
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_22
PY=/root/cutfem_neural_a_20260910/env/bin/python
until grep -q SENS_DONE $T/SENS_01/pipeline.status 2>/dev/null; do sleep 60; done
L=$T/pylib_cudss/nvidia/cu12/lib
for c in "$@"; do
  O=$T/E2E_01/$c; rm -rf $O; mkdir -p $O
  $PY $S/measure.py $O/STAGE_G.json -- bash $S/run_e2e_g.sh $c $O/G_run
  $PY $S/measure.py $O/STAGE_INPUTS.json -- bash $S/run_e2e_inputs.sh $O/G_run $O/inputs > $O/inputs.log 2>&1
  ( cd $S && LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True G_BODY_DIR=$O/G_run/body \
    $PY $S/measure.py $O/STAGE_ENCODE.json -- $PY -u encode_gpu.py --solver cudss --gpu-trace --encode-only --case $c \
      --cover-inputs $O/inputs --elements polyref --s 4 --levels 1 --output $O/encode > $O/encode.log 2>&1 )
  $PY $S/e2e_cost.py $O > $O/cost.log 2>&1
  echo "$(date -u +%FT%TZ) e2e $c done" >> $T/E2E_01/status
done
echo "$(date -u +%FT%TZ) E2E_DONE" >> $T/E2E_01/status
