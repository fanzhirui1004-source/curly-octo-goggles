#!/bin/bash
# Section 6.11 follow-up (after OPT_PROD_DONE; replaces the queued OPTH / OPTX chains): (1) every production run that did
# not finish (no CONVERGED / MAXIT event in its log) is resumed from its state.npz with --body-retry 3 (a cell whose body
# generation fails is regenerated with its corners scaled by 1 +- 1e-9, ...); (2) fine-scale NICE evaluation of the
# homogenisation designs H_y / H_z; (3) case A exact twin.  Tag OPTF2; marker OPT_FOLLOW_DONE_20260929 (EXIT trap).
ST=/root/autodl-tmp/OPL/S1/V2/chain_p1.status; R=/root/autodl-tmp/OPL/S1/V2/R1/OPT; TAG=OPTF2; RC=0
st() { echo "$(date +%T) $TAG $*" >> $ST; }
fin() { st "END rc=$RC"; echo "$(date +%T) OPT_FOLLOW_DONE_20260929" >> $ST; }
trap fin EXIT
run() { local n=$1 to=$2; shift 2; st "START $n"; local t=$(date +%s)
  timeout $to "$@" >> $R/$n.log 2>&1; local r=$?; [ $r != 0 ] && RC=$r; st "END $n rc=$r $(( $(date +%s) - t ))s"; }
st "QUEUED pid=$$ (waits for OPT_PROD_DONE_20260929)"
until grep -qE "^[0-9:]+ OPT_PROD_DONE_20260929$" $ST; do sleep 60; done
source /root/autodl-tmp/OPL/S1/env_gpu.sh
D=$DEP/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$D PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/pylib_cpu PYPARDISO_MKL_RT=$D/libmkl_rt.so.3 MKL_NUM_THREADS=16
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets:$R/plate_packets OPL_GP_CACHE=0 OPL_CONV_FP32=1
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/OPL/src_v2
MODEL=A3=/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt
WARM=$(grep -q "warm-start test -> '--warm'" $ST && echo --warm)
done_() { grep -qE '"event": "(CONVERGED|MAXIT)"' $R/$1.log 2>/dev/null; }
# (1) resumes
done_ optA || run optA 21600 $PY -u opt_design.py $R/optA /root/autodl-tmp/OPL/S4/hlat222.json --model $MODEL \
  --clamp y,min --load y,max --load-dir y --vfrac 0.8 --maxit 60 --fast --workers 8 --sens ad --ad-batch 4096 --resident-gb 28 $WARM --body-retry 3
for B in B1:y B2:z; do n=${B%%:*}; d=${B##*:}
  done_ plate$n || run plate$n 43200 $PY -u opt_design.py $R/plate$n $R/layouts/plate841.json --model $MODEL \
    --clamp x,min --load x,max --load-dir $d --vfrac 0.8 --maxit 60 --fast --workers 12 --sens ad --ad-batch 2048 --resident-gb 22 $WARM --body-retry 3
done
# (2) fine-scale evaluation of the homogenisation designs
for B in y z; do
  run hevalH_$B 7200 $PY -u opt_design.py $R/hevalH_$B $R/homog/H_$B.final_layout.json --model $MODEL \
    --clamp x,min --load x,max --load-dir $B --vfrac 0.8 --stop-after 1 --fast --workers 12 --sens ad --ad-batch 2048 --resident-gb 22 --body-retry 3
done
# (3) case A exact twin
mkdir -p $R/optAx/body $R/optAx/packets
for d in $R/pilot222f/body/*_o000; do [ -d $d ] && cp -rn $d $R/optAx/body/; done
cp -rn $R/pilot222f/packets/*_o000 $R/optAx/packets/ 2>/dev/null; cp -n $R/pilot222f/body/GP_TEMPLATES_n32.npz $R/optAx/body/
run optAx 43200 $PY -u opt_design.py $R/optAx /root/autodl-tmp/OPL/S4/hlat222.json \
  --clamp y,min --load y,max --load-dir y --vfrac 0.8 --maxit 60 --workers 8 --exact --body-retry 3
