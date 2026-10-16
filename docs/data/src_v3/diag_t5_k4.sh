#!/bin/bash
# Diagnosis (2026-10-03): the final-route case A rerun (R1/FINAL/optA, --table5) did not converge at k = 4 (PCG 3000,
# true residual 661) on a design identical to that of the original run (OPT/optA, converged in 117).  Analyse the k = 4
# design once (--stop-after 1, cold start) with: the original route; --table5; --table5 with one option skipped
# (OPL_T5_SKIP).  New directories under R1/FINAL/diag only.  Status in FINAL/diag.status; marker DIAG_T5_DONE.
R=/root/autodl-tmp/OPL/S1/V2/R1/FINAL; D=$R/diag; mkdir -p $D; ST=$R/diag.status
st() { echo "$(date +%F_%T) DIAG $*" >> $ST; }
trap 'echo "$(date +%F_%T) DIAG_T5_DONE" >> $ST' EXIT
source /root/autodl-tmp/OPL/S1/env_gpu.sh
D2=$DEP/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$D2 PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/pylib_cpu PYPARDISO_MKL_RT=$D2/libmkl_rt.so.3 MKL_NUM_THREADS=16
export OPL_GP_CACHE=0 OPL_CONV_FP32=1 OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/OPL/src_v2
$PY - <<PYEOF
import json
L = json.load(open('/root/autodl-tmp/OPL/S4/hlat222.json')); m = json.load(open('$R/optA/meta.json'))
H = [json.loads(l) for l in open('$R/optA/history.jsonl')]; tv = H[4]['tv']
for c, v in zip(L['cells'], m['vid']):
    c['tau_corners'] = [tv[i] for i in v]
json.dump(L, open('$D/hlat222_k4.json', 'w'))
PYEOF
COMMON="--model A3=/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt --clamp y,min --load y,max --load-dir y --vfrac 0.8 --fast --workers 8 --sens ad --ad-batch 4096 --resident-gb 28 --body-retry 6 --stop-after 1"
one() { local n=$1 skip=$2; shift 2; st "START $n skip=$skip"; mkdir -p $D/$n/body; cp -n $R/optA/body/GP_TEMPLATES_n32.npz $D/$n/body/ 2>/dev/null
  OPL_T5_SKIP=$skip timeout 3600 $PY -u opt_design.py $D/$n $D/hlat222_k4.json $COMMON "$@" > $D/$n.log 2>&1
  st "END $n rc=$? $(grep -o '"pcg": [0-9]*, "true_residual": [0-9.e+-]*' $D/$n.log | head -1)"; }
one orig ''
one t5 ''           --table5
one t5_noELEM   COARSE_ELEM   --table5
one t5_noTET    TET_TRITON    --table5
one t5_noSPARSE COARSE_SPARSE --table5
one t5_noFI     FI            --table5
