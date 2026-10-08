#!/bin/bash
# GPU part of the design-sensitivity pipeline (replaces the waiting tail of run_sens_pipeline.sh, whose blocks used the
# v1 admission): after BLOCKS24_DONE and the full-spectrum batch, observables for geometry and teacher sides, then slopes.
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_21; X=$T/SENS_01
BASES="fresh_train_0008_cover01_r1 fresh_train_0020_cover01_r1 fresh_train_0013_cover01_r1"
TAGS="m1000 m10000 p10000 p1000"
declare -A EPS=([m1000]=-1/1000 [m10000]=-1/10000 [p10000]=1/10000 [p1000]=1/1000)
until grep -q BLOCKS24_DONE $X/blocks24.status 2>/dev/null; do sleep 30; done
until grep -q FS_ALL_DONE $T/FULLSPEC_DIRECT_01/status 2>/dev/null; do sleep 60; done
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/cutfem_neural_a_20260910/env/bin/python
cd $S; mkdir -p $X/obs
for b in $BASES; do
  for t in 0 $TAGS; do
    e=${EPS[$t]:-0}
    o=$X/obs/${b}_geometry_$t; rm -rf $o
    $PY -u encode_gpu.py --solver cudss --gpu-trace --observables --direct-width 64 --spectrum-batch 512 --case $b --cover-inputs $T/COVER_INPUTS/$b \
        --elements polyref --s 4 --levels 1 --tau-eps $e --output $o > $o.log 2>&1
    echo "$(date -u +%FT%TZ) obs geometry $b $t exit $?" >> $X/pipeline.status
    p=${b}_tau$t
    if grep -q '"admitted": true' $X/blocks/$p/SENS_BLOCKS.json 2>/dev/null; then
      o=$X/obs/${b}_teacher_$t; rm -rf $o
      $PY -u encode_gpu.py --solver cudss --observables --direct-width 64 --spectrum-batch 512 --case $p --asset-root $X/blocks \
          --elements teacher --output $o > $o.log 2>&1
      echo "$(date -u +%FT%TZ) obs teacher $b $t exit $?" >> $X/pipeline.status
    else
      echo "$(date -u +%FT%TZ) obs teacher $b $t skipped (not admitted)" >> $X/pipeline.status
    fi
  done
done
$PY $S/sens_slopes.py $BASES > $X/SLOPES.txt 2>&1
echo "$(date -u +%FT%TZ) SENS_DONE" >> $X/pipeline.status
