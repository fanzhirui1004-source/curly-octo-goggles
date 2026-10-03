#!/bin/bash
# Table 5 learned route re-timed on the final route (2026-10-03): the four lattices of Table 5 with the arguments of the
# original timed runs (S1/V2/learned_hlat221a.json, learned_hlat221b.json, d5_off.json, learned_hlat331.json) and the coarse
# pivot rule (OPL_COARSE_PIVOT_FLOOR=1e-7, Appendix F.2).  Waits for FINAL2_DONE (GPU free).  Outputs in R1/FINAL3;
# status FINAL3/chain.status (tag T5RT); marker T5RT_DONE.
R=/root/autodl-tmp/OPL/S1/V2/R1/FINAL3; mkdir -p $R; ST=$R/chain.status; TAG=T5RT; RC=0
st() { echo "$(date +%F_%T) $TAG $*" >> $ST; }
fin() { st "END rc=$RC"; echo "$(date +%F_%T) T5RT_DONE" >> $ST; }
trap fin EXIT
run() { local n=$1 to=$2; shift 2; st "START $n"; local t=$(date +%s)
  timeout $to "$@" > $R/$n.log 2>&1; local r=$?; [ $r != 0 ] && RC=$r; st "END $n rc=$r $(( $(date +%s) - t ))s"; }
st "QUEUED pid=$$ (waits for FINAL2_DONE)"
until grep -qE "^[0-9_:-]+ FINAL2_DONE$" /root/autodl-tmp/OPL/S1/V2/R1/FINAL2/chain.status; do sleep 60; done
source /root/autodl-tmp/OPL/S1/env_gpu.sh
export OPL_GP_CACHE=0 OPL_CONV_FP32=1 OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets
export OPL_COARSE_PIVOT_FLOOR=1e-7
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU OPL_T5_SKIP OPL_STREAM_PACK; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/OPL/src_v2
ARGS="--fastidx --coarse-tpl --tet-triton --sparse-coarse --resident-gb 4 --tol 1e-6 --levels 1e-2,3e-3,1e-3,1e-4 --iters 1 --sens ad --sens-obj sum --ad-batch 2048"
for L in hlat221a hlat221b hlat222 hlat331; do
  run learned_$L 7200 $PY -u lat_scale.py $R/learned_$L.json /root/autodl-tmp/OPL/S4/$L.json $ARGS
done
