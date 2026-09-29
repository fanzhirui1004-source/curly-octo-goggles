#!/bin/bash
# Section 6.11: (1) --fast --sens ad on hlat222 k=0 (bodies copied from pilot222f): reverse-mode sensitivities vs the
# central-difference route of pilot222f k=0, and timing with a larger reverse batch; (2) homog_cell rerun with tau 0.18
# instead of 0.17 (below the generator range).  Tag OPTA; marker OPT_AD_DONE_20260929 (EXIT trap).
ST=/root/autodl-tmp/OPL/S1/V2/chain_p1.status; R=/root/autodl-tmp/OPL/S1/V2/R1/OPT; TAG=OPTA; RC=0
st() { echo "$(date +%T) $TAG $*" >> $ST; }
fin() { st "END rc=$RC"; echo "$(date +%T) OPT_AD_DONE_20260929" >> $ST; }
trap fin EXIT
run() { local n=$1 to=$2; shift 2; st "START $n"; local t=$(date +%s)
  timeout $to "$@" > $R/$n.log 2>&1; local r=$?; [ $r != 0 ] && RC=$r; st "END $n rc=$r $(( $(date +%s) - t ))s"; }
source /root/autodl-tmp/OPL/S1/env_gpu.sh
D=$DEP/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$D PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/pylib_cpu PYPARDISO_MKL_RT=$D/libmkl_rt.so.3 MKL_NUM_THREADS=16
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets:$R/plate_packets OPL_GP_CACHE=0 OPL_CONV_FP32=1
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/OPL/src_v2
mkdir -p $R/pilot222a/body $R/pilot222a/packets
for d in $R/pilot222f/body/*_o000; do [ -d $d ] && cp -rn $d $R/pilot222a/body/; done
cp -rn $R/pilot222f/packets/*_o000 $R/pilot222a/packets/ 2>/dev/null; cp -n $R/pilot222f/body/GP_TEMPLATES_n32.npz $R/pilot222a/body/
st "QUEUED/START pid=$$"
run pilot222a 3600 $PY -u opt_design.py $R/pilot222a /root/autodl-tmp/OPL/S4/hlat222.json --model A3=/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt \
  --clamp y,min --load y,max --load-dir y --vfrac 0.8 --stop-after 1 --fast --workers 8 --sens ad --ad-batch 4096 --resident-gb 28
run homog_cell2 10800 $PY -u homog_cell.py $R/homog --template $R/plate_packets/plate841_000 --workers 8 \
  --taus 0.18,0.22,0.27,0.32,0.37,0.42,0.47,0.52,0.57,0.62,0.67,0.70
