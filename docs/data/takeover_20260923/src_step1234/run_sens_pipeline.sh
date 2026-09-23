#!/bin/bash
# Design-sensitivity pipeline. CPU part (after the perturbed G stages): teacher blocks with admission checks.
# GPU part (after the full-spectrum batch, to avoid sharing the GPU): observables for teacher and geometry sides.
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_21; X=$T/SENS_01
BASES="fresh_train_0008_cover01_r1 fresh_train_0020_cover01_r1 fresh_train_0013_cover01_r1"
TAGS="m1000 m10000 p10000 p1000"
declare -A EPS=([m1000]=-1/1000 [m10000]=-1/10000 [p10000]=1/10000 [p1000]=1/1000)
until [ "$(grep -c ' G ' $X/run.status 2>/dev/null)" -ge 12 ]; do sleep 30; done
for b in $BASES; do
  mkdir -p $X/blocks/${b}_tau0/compiled
  for f in A.npz C.npz D.npz Q_RIGID.npy INTERIOR_RIGID.npy INTERIOR_POINTS.npy QUALIFICATION_PROBES.npz; do cp $T/COVER_INPUTS/$b/$f $X/blocks/${b}_tau0/compiled/; done
  for t in $TAGS; do
    p=${b}_tau$t
    [ -f $X/blocks/$p/SENS_BLOCKS.json ] || bash $S/run_sens_blocks.sh $b $p >> $X/sens_blocks.log 2>&1
    echo "$(date -u +%FT%TZ) blocks $p exit $?" >> $X/pipeline.status
  done
done
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
    p=${b}_tau$t; [ $t = 0 ] && p=${b}_tau0
    if [ $t = 0 ] || grep -q '"admitted": true' $X/blocks/$p/SENS_BLOCKS.json 2>/dev/null; then
      o=$X/obs/${b}_teacher_$t; rm -rf $o
      $PY -u encode_gpu.py --solver cudss --observables --direct-width 64 --spectrum-batch 512 --case $p --asset-root $X/blocks \
          --elements teacher --output $o > $o.log 2>&1
      echo "$(date -u +%FT%TZ) obs teacher $b $t exit $?" >> $X/pipeline.status
    fi
  done
done
$PY $S/sens_slopes.py $BASES > $X/SLOPES.txt 2>&1
echo "$(date -u +%FT%TZ) SENS_DONE" >> $X/pipeline.status
