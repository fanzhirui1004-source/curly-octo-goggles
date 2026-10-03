#!/bin/bash
# Section 6.11 production (after OPT_AD_DONE_20260929): (0) warm-start test on hlat222 (3 iterations, --fast --sens ad
# --warm) against pilot222f (cold, same designs up to MMA): warm start is used below only if |dC|/C < 1e-5 at every
# iteration and the PCG count drops; (1) case A: hlat222 NICE optimisation, clamp y,min, load y,max, direction y;
# (2) case B1: cut plate plate841, clamp x,min (8-cell long edge), load x,max (top face), direction y (in plane);
# (3) case B2: same plate, direction z (bending of the single layer).  All --fast --sens ad, vfrac 0.8, maxit 60.
# Tag OPTR; marker OPT_PROD_DONE_20260929 (EXIT trap).
ST=/root/autodl-tmp/OPL/S1/V2/chain_p1.status; R=/root/autodl-tmp/OPL/S1/V2/R1/OPT; TAG=OPTR; RC=0
st() { echo "$(date +%T) $TAG $*" >> $ST; }
fin() { st "END rc=$RC"; echo "$(date +%T) OPT_PROD_DONE_20260929" >> $ST; }
trap fin EXIT
run() { local n=$1 to=$2; shift 2; st "START $n"; local t=$(date +%s)
  timeout $to "$@" > $R/$n.log 2>&1; local r=$?; [ $r != 0 ] && RC=$r; st "END $n rc=$r $(( $(date +%s) - t ))s"; }
st "QUEUED pid=$$ (waits for OPT_AD_DONE_20260929)"
until grep -qE "^[0-9:]+ OPT_AD_DONE_20260929$" $ST; do sleep 60; done
source /root/autodl-tmp/OPL/S1/env_gpu.sh
D=$DEP/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$D PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/pylib_cpu PYPARDISO_MKL_RT=$D/libmkl_rt.so.3 MKL_NUM_THREADS=16
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets:$R/plate_packets OPL_GP_CACHE=0 OPL_CONV_FP32=1
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/OPL/src_v2
MODEL=A3=/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt
seed() { mkdir -p $2/body $2/packets
  for d in $1/body/*_o000; do [ -d $d ] && cp -rn $d $2/body/; done
  cp -rn $1/packets/*_o000 $2/packets/ 2>/dev/null; cp -n $1/body/GP_TEMPLATES_n32.npz $2/body/; }
# (0) warm-start test
seed $R/pilot222f $R/pilot222w
run pilot222w 3600 $PY -u opt_design.py $R/pilot222w /root/autodl-tmp/OPL/S4/hlat222.json --model $MODEL \
  --clamp y,min --load y,max --load-dir y --vfrac 0.8 --stop-after 3 --fast --workers 8 --sens ad --ad-batch 4096 --resident-gb 28 --warm
WARM=$($PY - <<'P'
import json
R='/root/autodl-tmp/OPL/S1/V2/R1/OPT'
try:
    a=[json.loads(l) for l in open(R+'/pilot222f/history.jsonl')]; b=[json.loads(l) for l in open(R+'/pilot222w/history.jsonl')]
    ok = len(b) >= 3 and all(abs(y['C']-x['C'])/x['C'] < 1e-5 for x, y in zip(a, b)) and sum(y['pcg'] for y in b[1:]) < sum(x['pcg'] for x in a[1:len(b)])
except Exception:
    ok = False
print('--warm' if ok else '')
P
)
st "warm-start test -> '${WARM}'"
# (1) case A
seed $R/pilot222f $R/optA
run optA 21600 $PY -u opt_design.py $R/optA /root/autodl-tmp/OPL/S4/hlat222.json --model $MODEL \
  --clamp y,min --load y,max --load-dir y --vfrac 0.8 --maxit 60 --fast --workers 8 --sens ad --ad-batch 4096 --resident-gb 28 $WARM
# (2) case B1, (3) case B2 (k=0 bodies from plateB1)
for B in B1:y B2:z; do n=${B%%:*}; d=${B##*:}
  [ $n = B1 ] || seed $R/plateB1 $R/plate$n
  run plate$n 43200 $PY -u opt_design.py $R/plate$n $R/layouts/plate841.json --model $MODEL \
    --clamp x,min --load x,max --load-dir $d --vfrac 0.8 --maxit 60 --fast --workers 12 --sens ad --ad-batch 2048 --resident-gb 22 $WARM
done
