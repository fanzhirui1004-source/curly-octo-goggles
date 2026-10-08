#!/bin/bash
# Section 6.11: after the fp64 pilot (OPT_PILOT2_DONE): (1) the --fast deployment route on hlat222 for 3 iterations (its
# iteration-0 design equals the pilot's: bodies copied), (2) periodic homogenisation of the uniform-thickness cell (12 tau).
# Tag OPTF; marker OPT_FAST_DONE_20260929 (EXIT trap).
ST=/root/autodl-tmp/OPL/S1/V2/chain_p1.status; R=/root/autodl-tmp/OPL/S1/V2/R1/OPT; TAG=OPTF; RC=0
st() { echo "$(date +%T) $TAG $*" >> $ST; }
fin() { st "END rc=$RC"; echo "$(date +%T) OPT_FAST_DONE_20260929" >> $ST; }
trap fin EXIT
run() { local n=$1 to=$2; shift 2; st "START $n"; local t=$(date +%s)
  timeout $to "$@" > $R/$n.log 2>&1; local r=$?; [ $r != 0 ] && RC=$r; st "END $n rc=$r $(( $(date +%s) - t ))s"; }
st "QUEUED pid=$$ (waits for OPT_PILOT2_DONE_20260929)"
until grep -qE "^[0-9:]+ OPT_PILOT2_DONE_20260929$" $ST; do sleep 60; done
source /root/autodl-tmp/OPL/S1/env_gpu.sh
D=$DEP/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$D PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/pylib_cpu PYPARDISO_MKL_RT=$D/libmkl_rt.so.3 MKL_NUM_THREADS=16
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets:$R/plate_packets OPL_GP_CACHE=0 OPL_CONV_FP32=1
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/OPL/src_v2
mkdir -p $R/pilot222f/body $R/pilot222f/packets
for d in $R/pilot222/body/*_o000; do [ -d $d ] && cp -rn $d $R/pilot222f/body/; done
cp -rn $R/pilot222/packets/*_o000 $R/pilot222f/packets/ 2>/dev/null; cp -n $R/pilot222/body/GP_TEMPLATES_n32.npz $R/pilot222f/body/
run pilot222f 7200 $PY -u opt_design.py $R/pilot222f /root/autodl-tmp/OPL/S4/hlat222.json --model A3=/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt \
  --clamp y,min --load y,max --load-dir y --vfrac 0.8 --stop-after 3 --fast --workers 8
run homog_cell 10800 $PY -u homog_cell.py $R/homog --template $R/plate_packets/plate841_000 --workers 8
