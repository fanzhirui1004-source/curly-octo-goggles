#!/bin/bash
# Section 6.11: case A exact twin (after OPT_HEVAL_DONE): the same optimisation as optA with exact
# condensation (--exact: dense condensed cell matrices, PCG to 1e-10, exact sensitivities), same start, constraints, MMA.
ST=/root/autodl-tmp/OPL/S1/V2/chain_p1.status; R=/root/autodl-tmp/OPL/S1/V2/R1/OPT; TAG=OPTX; RC=0
st() { echo "$(date +%T) $TAG $*" >> $ST; }
fin() { st "END rc=$RC"; echo "$(date +%T) OPT_EXACT_DONE_20260929" >> $ST; }
trap fin EXIT
run() { local n=$1 to=$2; shift 2; st "START $n"; local t=$(date +%s)
  timeout $to "$@" > $R/$n.log 2>&1; local r=$?; [ $r != 0 ] && RC=$r; st "END $n rc=$r $(( $(date +%s) - t ))s"; }
st "QUEUED pid=$$ (waits for OPT_HEVAL_DONE_20260929)"
until grep -qE "^[0-9:]+ OPT_HEVAL_DONE_20260929$" $ST; do sleep 60; done
source /root/autodl-tmp/OPL/S1/env_gpu.sh
D=$DEP/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$D PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/pylib_cpu PYPARDISO_MKL_RT=$D/libmkl_rt.so.3 MKL_NUM_THREADS=16
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets:$R/plate_packets OPL_GP_CACHE=0 OPL_CONV_FP32=1
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/OPL/src_v2
mkdir -p $R/optAx/body $R/optAx/packets
for d in $R/pilot222f/body/*_o000; do [ -d $d ] && cp -rn $d $R/optAx/body/; done
cp -rn $R/pilot222f/packets/*_o000 $R/optAx/packets/ 2>/dev/null; cp -n $R/pilot222f/body/GP_TEMPLATES_n32.npz $R/optAx/body/
run optAx 43200 $PY -u opt_design.py $R/optAx /root/autodl-tmp/OPL/S4/hlat222.json \
  --clamp y,min --load y,max --load-dir y --vfrac 0.8 --maxit 60 --workers 8 --exact
