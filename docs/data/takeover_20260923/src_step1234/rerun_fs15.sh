#!/bin/bash
# Re-run the one full spectrum that ran out of GPU memory at panel width 64, with width 16. Exit code captured first.
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_20; O=$T/FULLSPEC_DIRECT_01
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $S; name=fresh_train_0015_cover01_r1_res8
mv $O/$name $O/${name}_OOM_width64 2>/dev/null; mv $O/$name.log $O/${name}_OOM_width64.log 2>/dev/null
/root/cutfem_neural_a_20260910/env/bin/python -u encode_gpu.py --solver cudss --gpu-trace --full-spectrum --direct-width 16 --spectrum-batch 128 \
    --output $O/$name --case fresh_train_0015_cover01_r1 --cover-inputs $T/COVER_INPUTS/fresh_train_0015_cover01_r1 --elements polyref --s 4 --levels 1 > $O/$name.log 2>&1
rc=$?
ok=no; [ -f $O/$name/FULL_SPECTRUM.json ] && ok=yes
echo "$(date -u +%FT%TZ) $name rerun width16 rc $rc result $ok" >> $O/status
