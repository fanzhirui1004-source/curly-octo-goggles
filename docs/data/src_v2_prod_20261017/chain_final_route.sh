#!/bin/bash
# Final-route reruns (2026-10-03): every reported optimisation timing on the route of Table 5 (opt_design --table5).
# (1) case A (hlat222) NICE optimisation, settings of optA plus --table5 (GPU; runs now, beside the CPU exact checks);
# (2) after CPLATE_EXACT_DONE: exact checks of the new case-A designs at k = 0, 12 and the last iteration
#     (exact_check_cpu.py --mmap-T);
# (3) then the scale demonstration plates (24, 51, 88, 110, 135 cells; layouts of R1/SCALE), clamp x,min, load x,max in y,
#     4 analyses, 4 GB resident budget, packed streamed state, --table5, with the 30-s RSS / GPU-memory sampler.
# New directories only (R1/FINAL); nothing in OPT or SCALE is modified.  Status in FINAL/chain_final.status (tag FINAL);
# marker FINAL_ROUTE_DONE.
R=/root/autodl-tmp/OPL/S1/V2/R1/FINAL; OPT=/root/autodl-tmp/OPL/S1/V2/R1/OPT; SC=/root/autodl-tmp/OPL/S1/V2/R1/SCALE
mkdir -p $R; ST=$R/chain_final.status; TAG=FINAL; RC=0
st() { echo "$(date +%F_%T) $TAG $*" >> $ST; }
fin() { st "END rc=$RC"; echo "$(date +%F_%T) FINAL_ROUTE_DONE" >> $ST; }
trap fin EXIT
run() { local n=$1 to=$2; shift 2; st "START $n"; local t=$(date +%s)
  timeout $to "$@" > $R/$n.log 2>&1; local r=$?; [ $r != 0 ] && RC=$r; st "END $n rc=$r $(( $(date +%s) - t ))s"; return $r; }
seed() { mkdir -p $2/body $2/packets
  for d in $1/body/*_o000; do [ -d $d ] && cp -rn $d $2/body/; done
  cp -rn $1/packets/*_o000 $2/packets/ 2>/dev/null; cp -n $1/body/GP_TEMPLATES_n32.npz $2/body/ 2>/dev/null; }
source /root/autodl-tmp/OPL/S1/env_gpu.sh
D=$DEP/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$D PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/pylib_cpu PYPARDISO_MKL_RT=$D/libmkl_rt.so.3 MKL_NUM_THREADS=16
export OPL_GP_CACHE=0 OPL_CONV_FP32=1
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/OPL/src_v2
MODEL=A3=/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt
st "QUEUED pid=$$"
# (1) case A on the final route
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets:$OPT/plate_packets
seed $OPT/optA $R/optA
run optA 21600 $PY -u opt_design.py $R/optA /root/autodl-tmp/OPL/S4/hlat222.json --model $MODEL \
  --clamp y,min --load y,max --load-dir y --vfrac 0.8 --move 0.05 --maxit 60 --tmin 0.18 --tmax 0.69 --span 0.45 --grad 0.45 \
  --fast --table5 --warm --workers 8 --sens ad --ad-batch 4096 --resident-gb 28 --body-retry 6
# (2) exact checks of case A, after the C-plate exact checks have released the CPU memory
st "WAIT CPLATE_EXACT_DONE"
until grep -qE "^[0-9_:-]+ CPLATE_EXACT_DONE$" $OPT/chain_cplate.status; do sleep 60; done
run exact_optA 43200 $PY -u exact_check_cpu.py $R/optA:0,12,last --work $R/exact_optA --mmap-T --t-route schur --threads 32
# (3) scale demonstration on the final route
export OPL_STREAM_PACK=1
cat > $R/rss_sampler.sh <<EOS
#!/bin/bash
while pgrep -f chain_final_route.sh >/dev/null; do
  for p in \$(pgrep -f "opt_design.py $R/plateS"); do
    n=\$(ps -o args= -p \$p | grep -o "plateS[0-9]*" | head -1); r=\$(ps -o rss= -p \$p 2>/dev/null)
    [ -n "\$r" ] && echo -e "\$(date +%T)\t\$(awk "BEGIN{print \$r/1048576}")\t\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> $R/rss_\$n.tsv
  done; sleep 30; done
EOS
nohup bash $R/rss_sampler.sh > /dev/null 2>&1 &
for P in plateS24 plateS51 plateS88 plateS110 plateS135; do
  export OPL_PACKETS_EXTRA=$R/$P/packets:/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets:$SC/plate_packets
  for src in $SC/${P}r4 $SC/$P; do [ -d $src/body ] && { seed $src $R/$P; break; }; done
  run $P 36000 $PY -u opt_design.py $R/$P $SC/layouts/$P.json --model $MODEL \
    --clamp x,min --load x,max --load-dir y --vfrac 0.8 --maxit 60 --stop-after 4 --fast --table5 --workers 14 --sens ad \
    --ad-batch 2048 --resident-gb 4 --warm --body-retry 6
done
