#!/bin/bash
# Revision round 1, extra chain (tag R1EXT; 2026-09-29): E6 repairs, E11 (NICE in cube view 17), E13 (one-corner tau sweeps).
# Waits for the derivative chain's completion line "HH:MM:SS R1_DERIV_DONE_20260928 ..." (anchored; lines containing "wait"
# ignored). A heartbeat line "HH:MM:SS R1EXT HEARTBEAT" is written every 10 min while this chain lives (the host-timing
# chain waits for this chain's completion line, or for 30 min without an R1EXT line). Then:
#   E6 (a)  the four held-out NICE gates SIGKILLed in the accuracy chain (2047 x/y, 2078 x/y): the accuracy chain's exact
#           command (r1_gate.py, LAT_CPU=1 FUSED_HYPER=1, OPL_PACKETS_EXTRA=S3 packets, --custom ckpt:case:ckpt: --nb-mode
#           explicit --sets test), outputs R1/ACC/ho_gate_A3_<case>_<conf>.json; >= 45 GiB host memory free before each
#   E6 (b)  heavy-stratum replacement for fresh_val_2026_d0_v0: r1x3_e6_replace.py (continuation of the pre-registered
#           default_rng(20260928) draw; record R1/ACC/E6_REPLACEMENT_R1X3.json, PREREG unchanged); dense T of the cell and
#           its _nbmx / _nbmy (make_T_gpu.py, host fallback make_T_cpu.py), gates x and y; if both fail, the next draw
#           (at most 3 attempts); then r1_acc_summary.py R1/ACC $O
#   plan    r1_sweep.py plan (derived packets, point lists; corners from the E3 records)
#   bodies  fast_prep4 (frozen CPU environment, as the E7 chain), 8 in parallel, in visiting order, background
#   E11     eval_views_pd.py NICE view 17 on the 80 val_s3 geometries (body S0, data S2/data_v2, OPL_CONV_FP32=1) + summary
#   E13     r1_sweep.py run: m1x (M1/x max corner, 201 + 101), u1y (U1/y max corner, 201 + 101), l221 (hlat221a cut cell,
#           101 + 51), m1xm (M1/x median corner, 101 + 51), u1ym (U1/y median corner, 101 + 51), each with a deadline
#           inside the global budget (points are visited coarse-to-fine, so a stopped sweep still spans the range);
#           then r1_sweep.py summary -> R1/X3/E13/E13_SUMMARY.json
# GLOBAL BUDGET for the GPU work after START_WORK: 16200 s by default (E6 + E11 + E13); override by writing a number of
# seconds into R1/X3/R1EXT_BUDGET_S before the work starts.
# Always ends with the line "HH:MM:SS R1_EXTRA_DONE_20260928" (EXIT trap). Status lines never contain other markers.
O=/root/autodl-tmp/OPL/S1/V2; R=$O/R1/X3; E=$R/E13; ACC=$O/R1/ACC; ST=$O/chain_p1.status; S=/root/autodl-tmp/OPL/src_v2
B=/root/autodl-tmp/OPL/S0; DEP=/root/autodl-tmp/CUTFEM_DEPENDENCIES_20260924; CK=$O/A3_2grid/best.pt
st() { echo "$(date +%T) R1EXT $*" >> $ST; }
RC=0; MARKED=""
mark() { [ -n "$MARKED" ] && return; MARKED=1; st "END rc=$RC"; echo "$(date +%T) R1_EXTRA_DONE_20260928" >> $ST; }
trap mark EXIT
( while kill -0 $$ 2>/dev/null; do sleep 600; kill -0 $$ 2>/dev/null && st "HEARTBEAT"; done ) &
avail() { python3 -c "
m=int(open('/sys/fs/cgroup/memory.max').read().strip() or 0)
s=dict(l.split() for l in open('/sys/fs/cgroup/memory.stat'))
print(int((m-int(s['anon'])-int(s['shmem']))/2**30))" 2>/dev/null || echo 999; }
memwait() { local n=0 t0=$(date +%s); while [ $(avail) -lt $1 ] && [ $(( $(date +%s) - t0 )) -lt 7200 ]; do
  [ $n = 0 ] && st "WAIT_MEM need $1 GiB avail $(avail) GiB"; n=1; sleep 120; done; }
run() { local n=$1 to=$2; shift 2; st "START $n"; local t=$(date +%s)
  timeout $to "$@" > $R/$n.log 2>&1; local r=$?; [ $r != 0 ] && RC=$r; st "END $n rc=$r $(( $(date +%s) - t ))s"; }
mkdir -p $E/body $E/tmp $R/E11
st "QUEUED pid=$$ (waits for the derivative-chain completion line)"
has_deriv() { grep -E "^[0-9]{2}:[0-9]{2}:[0-9]{2} R1_DERIV_DONE_20260928( |$)" $ST | grep -viq "wait"; }
until has_deriv; do sleep 120; done
T0=$(date +%s); BUDGET=16200; [ -f $R/R1EXT_BUDGET_S ] && BUDGET=$(cat $R/R1EXT_BUDGET_S); GEND=$(( T0 + BUDGET ))
st "START_WORK pid=$$ budget_s=$BUDGET"
source /root/autodl-tmp/OPL/S1/env_gpu.sh
D=$DEP/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$D PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/pylib_cpu PYPARDISO_MKL_RT=$D/libmkl_rt.so.3 MKL_NUM_THREADS=16
export OPL_PACKETS_EXTRA=$E/packets:/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets OPL_GP_CACHE=0 OPL_CONV_FP32=1
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER LAT_CPU SENS_REASSOC
cd $S
{ date; nvidia-smi; free -g; md5sum r1_sweep.py r1x3_common.py r1x3_e11_summary.py r1x3_e6_replace.py r1_gate.py r1_acc_summary.py \
  eval_views_pd.py fast_prep4.py make_T_gpu.py make_T_cpu.py lattice3.py lat_multi.py teacher.py trainlib.py fastnet.py models.py $CK \
  $ACC/PREREG_R1ACC.json; env | grep -E "^OPL_|^PYTORCH|^OMP|^MKL" | sort; } > $R/chain_ext_env.txt 2>&1
# ---------------------------------------------------------------- plan + bodies (CPU, background)
SPECS="m1x:pair:fresh_val_2003_d1_v1:x:max u1y:pair:fresh_val_2000_full:y:max l221:lattice:hlat221a:/root/autodl-tmp/OPL/S4/hlat221a.json:lat:101:51 m1xm:pair:fresh_val_2003_d1_v1:x:med:101:51 u1ym:pair:fresh_val_2000_full:y:med:101:51"
$PY -u r1_sweep.py plan $E $SPECS > $E/bodies.txt 2> $R/e13_plan.log; r=$?; st "PLAN rc=$r $(wc -l < $E/bodies.txt) points"; [ $r != 0 ] && RC=$r
cp -n $B/GP_TEMPLATES_n32.npz $E/body/ 2>/dev/null
cat > $E/body_one.sh <<'EOB'
#!/bin/bash
# body_one.sh <body_root> <case>: fast_prep4 in the frozen CPU environment; <case>.failed on failure
[ -f $1/$2/PREP.json ] && exit 0
cd /root/autodl-tmp/CUTFEM_DEPENDENCIES_20260924/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/COVER_G/src && \
  OMP_NUM_THREADS=1 timeout 1800 /root/autodl-tmp/CUTFEM_DEPENDENCIES_20260924/run_frozen_python.sh /root/autodl-tmp/OPL/src_v2/fast_prep4.py 1 $1 $2 > $1/$2.log 2>&1 \
  || touch $1/$2.failed
EOB
chmod +x $E/body_one.sh
( t=$(date +%s); xargs -a $E/bodies.txt -P 8 -I{} $E/body_one.sh $E/body {}; st "BODIES_END $(ls $E/body/*/PREP.json 2>/dev/null | wc -l) ok $(ls $E/body/*.failed 2>/dev/null | wc -l) failed $(( $(date +%s) - t ))s" ) &
BPID=$!
# ---------------------------------------------------------------- E6 (a) reruns and (b) replacement
gate() {   # name case conf  (the accuracy chain's gate(), NICE)
  local n=$1 c=$2 conf=$3
  if [ -f $ACC/$n.json ] && grep -q '"r1_status": "ok"' $ACC/$n.json; then st "SKIP $n (done)"; return; fi
  memwait 45
  st "START $n"
  ( export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S3/packets; unset OPL_MODEL_ARGS_OVERRIDE
    LAT_CPU=1 FUSED_HYPER=1 timeout 3600 $PY -u r1_gate.py $ACC/$n.json --pairs '' --custom "$CK:$c:$CK:" --nb-mode explicit --sets test --configs $conf > $ACC/$n.log 2>&1 )
  local rc=$?; [ $rc != 0 ] && RC=$rc; st "END $n rc=$rc"
}
for c in fresh_val_2047_d1_v2 fresh_val_2078_d0_v1; do for conf in x y; do gate ho_gate_A3_${c}_$conf $c $conf; done; done
FAILED=""
for attempt in 1 2 3; do
  REP=$($PY r1x3_e6_replace.py $ACC $ACC/valmeta.json "" "$FAILED" 2>> $R/e6_replace.log | tail -1)
  st "E6 replacement draw $attempt: $REP"
  [ -z "$REP" ] || [ "$REP" = NONE ] && break
  for cc in $REP ${REP}_nbmx ${REP}_nbmy; do
    [ -f $B/${cc}_portview/T64.npy ] && continue
    ( export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S3/packets; timeout 1800 $PY -u make_T_gpu.py $B $cc >> $R/e6_makeT.log 2>&1 )
    [ -f $B/${cc}_portview/T64.npy ] && r=ok || r=FAIL; st "makeT_gpu $cc $r"
  done
  M=""; for cc in $REP ${REP}_nbmx ${REP}_nbmy; do [ -f $B/${cc}_portview/T64.npy ] || M="$M,$cc"; done
  if [ -n "$M" ]; then
    ( export CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH=$D PYPARDISO_MKL_RT=$D/libmkl_rt.so.3 PYTHONPATH=/root/autodl-tmp/pylib_cpu MKL_NUM_THREADS=16 \
             OPL_DEV=cpu OMP_NUM_THREADS=16 OPL_GP_CACHE=0 OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S3/packets
      timeout 21600 /root/autodl-tmp/gpuenv/bin/python -u make_T_cpu.py $B ${M#,} >> $R/e6_makeT.log 2>&1 )
    st "makeT_cpu ${M#,} rc=$?"
  fi
  for conf in x y; do gate ho_gate_A3_${REP}_$conf $REP $conf; done
  nok=0; for conf in x y; do grep -q '"r1_status": "ok"' $ACC/ho_gate_A3_${REP}_$conf.json 2>/dev/null && nok=$((nok+1)); done
  st "E6 replacement $REP gates ok=$nok/2"
  [ $nok -gt 0 ] && break
  FAILED="$FAILED,$REP"
done
$PY r1x3_e6_replace.py $ACC $ACC/valmeta.json "" "$FAILED" > /dev/null 2>> $R/e6_replace.log      # final record
( export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S3/packets; timeout 1800 $PY -u r1_acc_summary.py $ACC $O > $ACC/summary_r1ext.log 2>&1 ); st "E6 summary rc=$?"
# ---------------------------------------------------------------- E11 (GPU, while the bodies are built)
VAL=$(python3 -c "import json;print(','.join(json.load(open('/root/autodl-tmp/OPL/S2/SPLIT_ARMS.json'))['val_s3']))")
run e11_view17 5400 $PY -u eval_views_pd.py $R/E11 --prefix view17_ --model A3=$CK --views 17 --cases $VAL \
  --body /root/autodl-tmp/OPL/S0 --data /root/autodl-tmp/OPL/S2/data_v2
run e11_summary 600 $PY -u r1x3_e11_summary.py $R/E11/view17_A3.json $R/E11/E11_SUMMARY.json
# ---------------------------------------------------------------- E13 sweeps (budget shares of the remaining time)
sweep() { local tag=$1 frac=$2; local now=$(date +%s); local left=$(( GEND - now )); [ $left -lt 120 ] && { st "SKIP $tag (budget)"; return; }
  local dl=$(( now + left * frac / 100 )); [ $dl -gt $GEND ] && dl=$GEND
  run e13_$tag $(( dl - now + 1200 )) $PY -u r1_sweep.py run $E $tag --deadline $dl; }
sweep m1x 25
sweep u1y 33
sweep l221 40
sweep m1xm 50
sweep u1ym 100
run e13_summary 900 $PY -u r1_sweep.py summary $E
wait $BPID 2>/dev/null
st "ALL_STEPS_DONE rc=$RC"
