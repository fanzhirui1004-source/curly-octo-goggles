#!/bin/bash
# Revision round 1, accuracy chain (tag R1ACC; 2026-09-28). Waits for DF_PILOT_DONE_20260928, then
#   E2a + E5  one pass over the 80 validation geometries (view 0), per-direction errors, four models:
#             A3 (NICE), BW (B+W = v2L1 + W at evaluation), C (A0_ctrl), CW (C+W = A0_ctrl + W at evaluation)
#   E2b       C+W on every archived pair configuration (lat_full --nb-mode explicit, as gate_B2grid_* in chain_p1.sh)
#   E6        held-out pairs: 9 pre-registered sel=false cells x {x, y}; dense exact T of cell + neighbours first
#             (make_T_gpu.py, host fallback make_T_cpu.py), then arms A3 and BW
#   summary   r1_acc_summary.py -> R1/ACC/R1_ACC_SUMMARY.{json,txt} (after E2 and again at the end)
# and ALWAYS writes R1_ACC_DONE_20260928 to chain_p1.status at the end (also on failure: EXIT trap).
# Pre-registration (written before any result): R1/ACC/PREREG_R1ACC.json. No TF32 / fp16 / bf16 (OPL_CONV_FP32=1).
O=/root/autodl-tmp/OPL/S1/V2; R=$O/R1/ACC; ST=$O/chain_p1.status; S=/root/autodl-tmp/OPL/src_v2; B=/root/autodl-tmp/OPL/S0
D=/root/autodl-tmp/CUTFEM_DEPENDENCIES_20260924/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
W='{"smooth_k": 8, "smooth_alpha": 30.0, "coarse_space": "Q1_17"}'
st() { echo "$(date +%T) R1ACC $*" >> $ST; }
finish() { grep -q "R1_ACC_DONE_20260928" $ST || echo "$(date +%T) R1ACC R1_ACC_DONE_20260928 $1" >> $ST; }
trap 'finish "(exit trap)"' EXIT
mkdir -p $R
st "CHAIN_WAIT DF_PILOT_DONE_20260928 pid=$$"
until grep -q DF_PILOT_DONE_20260928 $ST; do sleep 300; done
for i in $(seq 1 30); do pgrep -f chain_dfp.sh > /dev/null || break; sleep 60; done     # pilot chain gone (<= 30 min)
sleep 60
st "CHAIN_START pid=$$"
source /root/autodl-tmp/OPL/S1/env_gpu.sh
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S3/packets OPL_GP_CACHE=0 OPL_CONV_FP32=1
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER LAT_CPU SENS_REASSOC
cd $S
{ date; nvidia-smi; free -g; md5sum eval_views_pd.py r1_gate.py r1_acc_summary.py lat_full.py eval_views.py models.py trainlib.py \
  $O/A3_2grid/best.pt $O/v2L1/best.pt $O/A0_ctrl/best.pt $R/PREREG_R1ACC.json; env | grep -E '^(OPL_|LAT_|FUSED|SENS|PY|OMP|MKL|PYTORCH|CUDA)'; } > $R/env_chain_start.txt 2>&1
VAL=$(python3 -c "import json;print(','.join(json.load(open('/root/autodl-tmp/OPL/S2/SPLIT_ARMS.json'))['val_s3']))")

# ---------------------------------------------------------------- E2a + E5 (one pass, four models)
ev() { timeout 18000 $PY -u eval_views_pd.py $R --prefix newval3_ --model A3=$O/A3_2grid/best.pt --model "BW=$O/v2L1/best.pt;$W" \
         --model C=$O/A0_ctrl/best.pt --model "CW=$O/A0_ctrl/best.pt;$W" --views 0 --cases $VAL \
         --body /root/autodl-tmp/OPL/S0 --data /root/autodl-tmp/OPL/S2/data_v2 "$@"; }
st "START E5 newval3 (A3,BW,C,CW x 80 geometries, view 0)"
ev > $R/newval3.log 2>&1; rc=$?
st "END E5 newval3 rc=$rc"
if [ $rc -ne 0 ]; then
  st "RETRY E5 newval3 --resume"; ev --resume >> $R/newval3.log 2>&1; rc=$?; st "END E5 newval3 retry rc=$rc"
fi

# ---------------------------------------------------------------- pair gates (r1_gate.py = lat_full.py + env / failure record)
gate() {   # name ckpt case conf [override]
  local n=$1 ck=$2 c=$3 conf=$4 ov=$5
  if [ -f $R/$n.json ] && grep -q '"r1_status": "ok"' $R/$n.json; then st "SKIP $n (done)"; return; fi
  if [ -n "$ov" ]; then export OPL_MODEL_ARGS_OVERRIDE="$ov"; else unset OPL_MODEL_ARGS_OVERRIDE; fi
  st "START $n"
  LAT_CPU=1 FUSED_HYPER=1 timeout 3600 $PY -u r1_gate.py $R/$n.json --pairs '' --custom "$ck:$c:$ck:" --nb-mode explicit --sets test --configs $conf > $R/$n.log 2>&1
  local rc=$?; unset OPL_MODEL_ARGS_OVERRIDE
  st "END $n rc=$rc"
}
# E2b: C+W on the archived pair configurations (7 development cells x {x, y}, plus H2 = 2010 x / y, ill-posed / empty for every arm)
for c in fresh_val_2000_full fresh_val_2001_full fresh_val_2005_d1_v0 fresh_val_2006_d0_v1 fresh_val_2003_d1_v1 fresh_val_2004_d0_v2 fresh_val_2002_d0_v0 fresh_val_2010_d0_v0; do
  for conf in x y; do gate gate_CW_${c}_$conf $O/A0_ctrl/best.pt $c $conf "$W"; done
done
st "START summary (after E2)"; $PY -u r1_acc_summary.py $R $O > $R/summary_after_E2.log 2>&1; st "END summary (after E2) rc=$?"

# ---------------------------------------------------------------- E6: held-out pairs (pre-registered cells, PREREG_R1ACC.json)
HO="fresh_val_2051_d1_v1 fresh_val_2045_d1_v0 fresh_val_2074_d0_v0 fresh_val_2021_d1_v0 fresh_val_2063_d1_v2 fresh_val_2032_full fresh_val_2047_d1_v2 fresh_val_2078_d0_v1 fresh_val_2026_d0_v0"
st "START E6 dense T (cells + nbmx / nbmy)"
for c in $HO; do for cc in $c ${c}_nbmx ${c}_nbmy; do
  [ -f $B/${cc}_portview/T64.npy ] && continue
  timeout 1800 $PY -u make_T_gpu.py $B $cc >> $R/makeT.log 2>&1
  [ -f $B/${cc}_portview/T64.npy ] && r=ok || r=FAIL; st "makeT_gpu $cc $r"
done; done
M=""; for c in $HO; do for cc in $c ${c}_nbmx ${c}_nbmy; do [ -f $B/${cc}_portview/T64.npy ] || M="$M,$cc"; done; done
if [ -n "$M" ]; then
  ( export CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH=$D PYPARDISO_MKL_RT=$D/libmkl_rt.so.3 PYTHONPATH=/root/autodl-tmp/pylib_cpu MKL_NUM_THREADS=16 \
           OPL_DEV=cpu OMP_NUM_THREADS=16 OPL_GP_CACHE=0 OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S3/packets
    timeout 21600 /root/autodl-tmp/gpuenv/bin/python -u make_T_cpu.py $B ${M#,} >> $R/makeT.log 2>&1 )
  st "makeT_cpu ${M#,} rc=$?"
fi
for arm in A3 BW; do
  case $arm in A3) ck=$O/A3_2grid/best.pt; ov=;; BW) ck=$O/v2L1/best.pt; ov=$W;; esac
  for c in $HO; do for conf in x y; do gate ho_gate_${arm}_${c}_$conf $ck $c $conf "$ov"; done; done
  st "E6 arm $arm done"
done

# ---------------------------------------------------------------- summary + marker
st "START summary (final)"; $PY -u r1_acc_summary.py $R $O > $R/summary_final.log 2>&1; st "END summary (final) rc=$?"
finish "(chain end)"
