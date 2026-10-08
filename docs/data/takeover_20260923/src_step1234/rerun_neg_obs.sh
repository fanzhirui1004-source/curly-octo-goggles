#!/bin/bash
# Re-run the geometry-side observables at negative eps: v1 passed "--tau-eps -1/10000", which argparse reads as an
# option. Pass it as --tau-eps=<value>. Exit codes are captured before any other command runs.
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_21; X=$T/SENS_01
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/cutfem_neural_a_20260910/env/bin/python
cd $S
for b in fresh_train_0008_cover01_r1 fresh_train_0020_cover01_r1 fresh_train_0013_cover01_r1; do
  for t in m10000 m1000; do
    e=-1/10000; [ $t = m1000 ] && e=-1/1000
    o=$X/obs/${b}_geometry_$t; rm -rf $o
    $PY -u encode_gpu.py --solver cudss --gpu-trace --observables --direct-width 64 --spectrum-batch 512 --case $b --cover-inputs $T/COVER_INPUTS/$b \
        --elements polyref --s 4 --levels 1 --tau-eps=$e --output $o > $o.log 2>&1
    rc=$?
    ok=no; [ -f $o/OBSERVABLES.json ] && ok=yes
    echo "$(date -u +%FT%TZ) obs geometry $b $t rc $rc observables $ok (rerun, --tau-eps=$e)" >> $X/pipeline.status
  done
done
$PY $S/sens_slopes.py fresh_train_0008_cover01_r1 fresh_train_0020_cover01_r1 fresh_train_0013_cover01_r1 > $X/SLOPES.txt 2>&1
rc=$?
echo "$(date -u +%FT%TZ) SENS_DONE_V2 slopes rc $rc" >> $X/pipeline.status
