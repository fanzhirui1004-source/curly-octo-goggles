#!/bin/bash
# Route 2 D2: certify route 1 skeleton fields. prep (CPU, frozen env) -> fields (GPU) -> certify (CPU). Usage: d2_batch.sh <case>
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; D=$T/xcase_src_27; S=$T/xcase_src_29; O=$T/R2_D2; W=$T/COVER_G
V=/root/autodl-tmp/CUTFEM_INGEST_R38/environment/.venv/bin/python; PY=/root/cutfem_neural_a_20260910/env/bin/python
c=$1; mkdir -p $O
st() { echo "$(date -u +%FT%TZ) $*" >> $O/status; }
cpu() {  # frozen runtime environment (PARDISO), in a subshell so the GPU steps keep their own environment
  ( export CUTFEM_EXECUTION_CONFIG=$W/EXECUTION_CONFIG.json CUTFEM_RUNTIME_ROOT=/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime
    eval "$(cd $W/src && PYTHONPATH=$W/src $V -B -c "import json,shlex;from stage_cutfem_runtime.environment import runtime_environment;c=json.load(open('$W/EXECUTION_CONFIG.json'));e,b=runtime_environment('$W/src',c['runtime_names']);print('\n'.join('export %s=%s'%(k,shlex.quote(v)) for k,v in e.items()))")"
    cd $W/src && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 $V -B $D/dual_d2.py "$@" )
}
gpu() {  # name elements
  ( L=$T/pylib_cudss/nvidia/cu12/lib
    export LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH PYTHONPATH=$T/pylib_cudss CUDSS_MT=$L/libcudss_mtlayer_gomp.so.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd $S && rm -rf $O/$1 && $PY -u encode_r1.py --solver chebyshev --gpu-trace --gpu-pairs --case $c --cover-inputs $T/COVER_INPUTS/$c --elements $2 --s 4 --levels 1 \
      --degree1 2 --degree2 2 --omega-mode lanczos --safety 0.8 --fields $O/${c}_prep.npz --field-depths 8,16,32,64 --output $O/$1 )
}
cpu prep $c $O/${c}_prep.npz > $O/${c}_prep.log 2>&1; rc=$?; st prep $c rc $rc
while pgrep -f "[r]un_r1_tune3.sh" > /dev/null; do sleep 20; done
for el in body polyref; do
  gpu ${c}_${el} $el > $O/${c}_${el}_fields.log 2>&1; rc=$?; ok=no; [ -f $O/${c}_${el}/FIELDS.npz ] && ok=yes; st fields $c $el rc $rc result $ok
done
for el in body polyref; do
  [ -f $O/${c}_${el}/FIELDS.npz ] || continue
  cpu certify $c 2 1 $O/${c}_${el}/FIELDS.npz $O/${c}_${el}_certify.json > $O/${c}_${el}_certify.log 2>&1; rc=$?; st certify $c $el rc $rc
done
st D2_DONE $c
