#!/bin/bash
# Re-measurement on the new AutoDL machine (2026-10-17).  Waits until the data transfer has finished (code, lattice
# geometry and environments present, data-disk usage unchanged over two readings 5 min apart), then
#  (1) re-runs the Table 5 learned route with the command and environment of chain_t5_retime.sh (FINAL3), sampling the
#      whole-device GPU memory with nvidia-smi every 100 ms;
#  (2) measures the deployed per-cell state of the ST13 cells G1-G4 (st13_deploy_mem.py).
# Everything is written to /root/remeasure_20261017 (system disk); nothing on the data disk is modified except the
# netdata scratch files that lat_scale.py and st13_deploy_mem.py write to OPL/S1/V2/bench_tmp, as the original runs did.
# Progress: /root/remeasure_20261017/status.txt (last line ALL_DONE or GIVE_UP).
O=/root/remeasure_20261017; mkdir -p $O; ST=$O/status.txt
st() { echo "$(date +%F_%T) $*" >> $ST; }
D=/root/autodl-tmp; P=$D/OPL
ready() {
  [ -f $P/src_v2/lat_scale.py ] && [ -f $P/src_v2/teacher.py ] && [ -f $P/S1/env_gpu.sh ] || return 1
  for L in hlat221a hlat221b hlat222 hlat331; do [ -f $P/S4/$L.json ] || return 1; done
  [ "$(ls $P/S4/body 2>/dev/null | wc -l)" -ge 149 ] || return 1
  for d in gpuenv exact_tools pylib_cpu; do [ -n "$(ls -A $D/$d 2>/dev/null)" ] || return 1; done
  return 0
}
st "WAIT start"
t0=$(date +%s); prev=-1; stable=0
while true; do
  used=$(df -B1M --output=used $D | tail -1 | tr -d ' ')
  if ready && [ "$used" = "$prev" ]; then stable=$((stable + 1)); else stable=0; fi
  st "used=${used}M ready=$(ready && echo 1 || echo 0) stable=$stable"
  [ $stable -ge 2 ] && break
  [ $(( $(date +%s) - t0 )) -gt 43200 ] && { st "GIVE_UP after 12 h"; exit 1; }
  prev=$used; sleep 300
done
st "TRANSFER_DONE used=${used}M"

source $P/S1/env_gpu.sh
export OPL_GP_CACHE=0 OPL_CONV_FP32=1 OPL_PACKETS_EXTRA=$P/S4/packets:$P/S3/packets
export OPL_COARSE_PIVOT_FLOOR=1e-7
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU OPL_T5_SKIP OPL_STREAM_PACK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi > $O/nvidia_smi_before.txt 2>&1
$PY -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))" > $O/env_check.txt 2>&1
st "ENV $(tr '\n' ' ' < $O/env_check.txt)"

sample() { nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 100 > $1 2>&1 & echo $!; }

cd $P/src_v2
ARGS="--fastidx --coarse-tpl --tet-triton --sparse-coarse --resident-gb 4 --tol 1e-6 --levels 1e-2,3e-3,1e-3,1e-4 --iters 1 --sens ad --sens-obj sum --ad-batch 2048"
for L in hlat221a hlat221b hlat222 hlat331; do
  st "START t5 $L"
  SP=$(sample $O/gpu_$L.csv)
  timeout 3600 $PY -u lat_scale.py $O/learned_$L.json $P/S4/$L.json $ARGS > $O/learned_$L.log 2>&1; r=$?
  sleep 2; kill $SP 2>/dev/null
  st "END t5 $L rc=$r gpu_max_MiB=$(cut -d, -f2 $O/gpu_$L.csv | sort -n | tail -1)"
done

st "START st13"
SP=$(sample $O/gpu_st13.csv)
timeout 3600 $PY -u $O/st13_deploy_mem.py $O/st13_deploy_mem.json > $O/st13.log 2>&1; r=$?
sleep 2; kill $SP 2>/dev/null
st "END st13 rc=$r"
nvidia-smi > $O/nvidia_smi_after.txt 2>&1
st "ALL_DONE"
