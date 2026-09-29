#!/bin/bash
# Section 6.11 resume chain (replaces OPTF2 after B2): waits for the running plateB2 process (pid in $1) to exit, then
# (1) resumes every unfinished production run (optA, plateB1, plateB2) with --body-retry 6 (design-vertex perturbation
# on body-generation failure), (2) fine-scale evaluation of H_y / H_z, (3) case A exact twin.
# Tag OPTR2; marker OPT_RESUME_DONE_20260929 (EXIT trap).
ST=/root/autodl-tmp/OPL/S1/V2/chain_p1.status; R=/root/autodl-tmp/OPL/S1/V2/R1/OPT; TAG=OPTR2; RC=0
st() { echo "$(date +%T) $TAG $*" >> $ST; }
fin() { st "END rc=$RC"; echo "$(date +%T) OPT_RESUME_DONE_20260929" >> $ST; }
trap fin EXIT
run() { local n=$1 to=$2; shift 2; st "START $n"; local t=$(date +%s)
  timeout $to "$@" >> $R/$n.log 2>&1; local r=$?; [ $r != 0 ] && RC=$r; st "END $n rc=$r $(( $(date +%s) - t ))s"; }
st "QUEUED pid=$$ (waits for pid $1 = plateB2)"
while kill -0 $1 2>/dev/null; do sleep 60; done
source /root/autodl-tmp/OPL/S1/env_gpu.sh
D=$DEP/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$D PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/pylib_cpu PYPARDISO_MKL_RT=$D/libmkl_rt.so.3 MKL_NUM_THREADS=16
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets:$R/plate_packets OPL_GP_CACHE=0 OPL_CONV_FP32=1
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/OPL/src_v2
MODEL=A3=/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt
WARM=$(grep -q "warm-start test -> '--warm'" $ST && echo --warm)
done_() { grep -qE '"event": "(CONVERGED|MAXIT)"' $R/$1.log 2>/dev/null; }
done_ optA || run optA 21600 $PY -u opt_design.py $R/optA /root/autodl-tmp/OPL/S4/hlat222.json --model $MODEL \
  --clamp y,min --load y,max --load-dir y --vfrac 0.8 --maxit 60 --fast --workers 8 --sens ad --ad-batch 4096 --resident-gb 28 $WARM --body-retry 6
for B in B1:y B2:z; do n=${B%%:*}; ld=${B##*:}
  done_ plate$n || run plate$n 43200 $PY -u opt_design.py $R/plate$n $R/layouts/plate841.json --model $MODEL \
    --clamp x,min --load x,max --load-dir $ld --vfrac 0.8 --maxit 60 --fast --workers 12 --sens ad --ad-batch 2048 --resident-gb 22 $WARM --body-retry 6
done
for ld in y z; do
  run hevalH_$ld 7200 $PY -u opt_design.py $R/hevalH_$ld $R/homog/H_$ld.final_layout.json --model $MODEL \
    --clamp x,min --load x,max --load-dir $ld --vfrac 0.8 --stop-after 1 --fast --workers 12 --sens ad --ad-batch 2048 --resident-gb 22 --body-retry 6
done
mkdir -p $R/optAx/body $R/optAx/packets
for bd in $R/pilot222f/body/*_o000; do [ -d $bd ] && cp -rn $bd $R/optAx/body/; done
cp -rn $R/pilot222f/packets/*_o000 $R/optAx/packets/ 2>/dev/null; cp -n $R/pilot222f/body/GP_TEMPLATES_n32.npz $R/optAx/body/
run optAx 43200 $PY -u opt_design.py $R/optAx /root/autodl-tmp/OPL/S4/hlat222.json --model $MODEL \
  --clamp y,min --load y,max --load-dir y --vfrac 0.8 --maxit 60 --workers 8 --exact --body-retry 6
