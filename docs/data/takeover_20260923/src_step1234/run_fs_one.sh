#!/bin/bash
# One two-sided full spectrum with the direct (cuDSS) exact action. Usage: run_fs_one.sh <name> <encode_gpu args...>
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_20; O=$T/FULLSPEC_DIRECT_01
L=$T/pylib_cudss/nvidia/cu12/lib
export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $S; name=$1; shift; rm -rf $O/$name
/root/cutfem_neural_a_20260910/env/bin/python -u encode_gpu.py --solver cudss --gpu-trace --full-spectrum --direct-width 64 --output $O/$name "$@" > $O/$name.log 2>&1
echo "$(date -u +%FT%TZ) $name exit $?" >> $O/status
