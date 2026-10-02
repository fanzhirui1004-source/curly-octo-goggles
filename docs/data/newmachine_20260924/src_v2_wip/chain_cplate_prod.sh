#!/bin/bash
# C-plate production (2026-10-02, pre-registered): plate841, cut-band clamp (--clamp cut), consistent unit traction on the
# y = min face (four full cells) in direction x, vfrac 0.8, move 0.05, maxit 60, same start / bounds / constraints for both
# routes.  (1) homogenised optimisation (homog_macro.py --clamp cut, CPU); (2) NICE optimisation (--fast --table5 --warm);
# (3) NICE analysis of the homogenised final design (--stop-after 1); (4) exact-condensation checks of the start design
# (cplateN k=0), the NICE final design (last analysed iteration) and the homogenised final design (hevalHC k=0).
# Status in chain_cplate.status; marker CPLATE_PROD_DONE.
R=/root/autodl-tmp/OPL/S1/V2/R1/OPT; ST=$R/chain_cplate.status; TAG=CPPROD; RC=0
st() { echo "$(date +%F_%T) $TAG $*" >> $ST; }
fin() { st "END rc=$RC"; echo "$(date +%F_%T) CPLATE_PROD_DONE" >> $ST; }
trap fin EXIT
run() { local n=$1 to=$2; shift 2; st "START $n"; local t=$(date +%s)
  timeout $to "$@" > $R/$n.log 2>&1; local r=$?; [ $r != 0 ] && RC=$r; st "END $n rc=$r $(( $(date +%s) - t ))s"; return $r; }
source /root/autodl-tmp/OPL/S1/env_gpu.sh
D=$DEP/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$D PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/pylib_cpu PYPARDISO_MKL_RT=$D/libmkl_rt.so.3 MKL_NUM_THREADS=16
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets:$R/plate_packets OPL_GP_CACHE=0 OPL_CONV_FP32=1
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/OPL/src_v2
MODEL=A3=/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt
LAY=$R/layouts/plate841.json
BC="--clamp cut --load y,min --load-dir x"
COMMON="$BC --vfrac 0.8 --move 0.05 --maxit 60 --tmin 0.18 --tmax 0.69 --span 0.45 --grad 0.45"
seed() { mkdir -p $2/body $2/packets
  for d in $1/body/*_o000; do [ -d $d ] && cp -rn $d $2/body/; done
  cp -rn $1/packets/*_o000 $2/packets/ 2>/dev/null; cp -n $1/body/GP_TEMPLATES_n32.npz $2/body/; }
st "QUEUED pid=$$"
# (1) homogenised optimisation
run homogHC_x 3600 /root/miniconda3/bin/python -u homog_macro.py $R/homog/HC_x $LAY $R/homog/homog_cells.json --m 6 $COMMON --pen 1e6
# (2) NICE optimisation
seed $R/plateB1 $R/cplateN
run cplateN 43200 $PY -u opt_design.py $R/cplateN $LAY --model $MODEL $COMMON \
  --fast --table5 --warm --workers 12 --sens ad --ad-batch 2048 --resident-gb 22 --body-retry 6
# (3) NICE analysis of the homogenised final design
mkdir -p $R/hevalHC/body; cp -n $R/plateB1/body/GP_TEMPLATES_n32.npz $R/hevalHC/body/
run hevalHC 7200 $PY -u opt_design.py $R/hevalHC $R/homog/HC_x/final_layout.json --model $MODEL $COMMON \
  --fast --table5 --workers 12 --sens ad --ad-batch 2048 --resident-gb 22 --body-retry 6 --stop-after 1
# (4) exact checks
KF=$($PY -c "import json;print(max(json.loads(l)['k'] for l in open('$R/cplateN/history.jsonl')))")
st "NICE final analysed iteration k=$KF"
run checkN 43200 $PY -u opt_design.py $R/cplateN $LAY --model $MODEL $COMMON --check 0,$KF --t-cpu-fallback
run checkHC 21600 $PY -u opt_design.py $R/hevalHC $R/homog/HC_x/final_layout.json --model $MODEL $COMMON --check 0 --t-cpu-fallback
