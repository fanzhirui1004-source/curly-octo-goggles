#!/bin/bash
# Revision round 1, extra chain (tag R1EXT; 2026-09-28/29): E11 (NICE in cube view 17) and E13 (one-corner tau sweeps).
# Waits for the derivative chain's completion line "HH:MM:SS R1_DERIV_DONE_20260928 ..." (anchored; lines containing
# "wait" ignored), then:
#   plan    r1_sweep.py plan: derived packets + point lists (corners from the E3 records: max / median |s_c|; lattice:
#           largest-|s| corner of a cut cell at an unshared vertex)
#   bodies  fast_prep4 (frozen CPU environment, as the E7 chain) for every sweep point, 8 in parallel, in visiting order,
#           in the background (a failed body leaves <case>.failed; the sweep skips it)
#   E11     eval_views_pd.py, NICE (A3_2grid, model args of the checkpoint = the paper's W), view 17, the 80 val_s3
#           geometries, body S0, data S2/data_v2, OPL_CONV_FP32=1; then r1x3_e11_summary.py (view 17 vs view 0)
#   E13     r1_sweep.py run, sweeps in priority order, each with its own deadline inside a global budget:
#           m1x (M1/x, max corner, 201 + 101), u1y (U1/y, max corner, 201 + 101), l221 (hlat221a cut cell, 101 + 51),
#           m1xm (M1/x, median corner, 101 + 51), u1ym (U1/y, median corner, 101 + 51); then r1_sweep.py summary
# GLOBAL BUDGET: E1 (host timing) starts its quiet-container check at the same derivative marker and proceeds noisy after
# 3 h, so by default this chain ends its GPU work 2 h 45 min after it starts. Override: write a number of seconds into
# R1/X3/R1EXT_BUDGET_S before the chain starts its work (read at start).
# Always ends with the line "HH:MM:SS R1_EXTRA_DONE_20260928 rc=<rc>" (EXIT trap). Status lines never contain markers.
O=/root/autodl-tmp/OPL/S1/V2; R=$O/R1/X3; E=$R/E13; ST=$O/chain_p1.status; S=/root/autodl-tmp/OPL/src_v2
DEP=/root/autodl-tmp/CUTFEM_DEPENDENCIES_20260924
TAG=R1EXT
st() { echo "$(date +%T) $TAG $*" >> $ST; }
RC=0; MARKED=""
mark() { [ -n "$MARKED" ] && return; MARKED=1; st "END rc=$RC"; echo "$(date +%T) R1_EXTRA_DONE_20260928 rc=$RC" >> $ST; }
trap mark EXIT
run() { local n=$1 to=$2; shift 2; st "START $n"; local t=$(date +%s)
  timeout $to "$@" > $R/$n.log 2>&1; local r=$?; [ $r != 0 ] && RC=$r; st "END $n rc=$r $(( $(date +%s) - t ))s"; }
mkdir -p $E/body $E/tmp $R/E11
st "QUEUED pid=$$ (waits for the derivative-chain completion line)"
has_deriv() { grep -E "^[0-9]{2}:[0-9]{2}:[0-9]{2} R1_DERIV_DONE_20260928( |$)" $ST | grep -viq "wait"; }
until has_deriv; do sleep 120; done
T0=$(date +%s); BUDGET=9900; [ -f $R/R1EXT_BUDGET_S ] && BUDGET=$(cat $R/R1EXT_BUDGET_S); GEND=$(( T0 + BUDGET ))
st "START_WORK pid=$$ budget_s=$BUDGET"
source /root/autodl-tmp/OPL/S1/env_gpu.sh
D=$DEP/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$D PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/pylib_cpu PYPARDISO_MKL_RT=$D/libmkl_rt.so.3 MKL_NUM_THREADS=16
export OPL_PACKETS_EXTRA=$E/packets:/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets OPL_GP_CACHE=0 OPL_CONV_FP32=1
cd $S
{ date; nvidia-smi; free -g; md5sum r1_sweep.py r1x3_common.py r1x3_e11_summary.py eval_views_pd.py fast_prep4.py lattice3.py lat_multi.py \
  teacher.py trainlib.py fastnet.py models.py $O/A3_2grid/best.pt; env | grep -E "^OPL_|^PYTORCH|^OMP|^MKL" | sort; } > $R/chain_ext_env.txt 2>&1
# ---------------------------------------------------------------- plan + bodies (CPU, background)
SPECS="m1x:pair:fresh_val_2003_d1_v1:x:max u1y:pair:fresh_val_2000_full:y:max l221:lattice:hlat221a:/root/autodl-tmp/OPL/S4/hlat221a.json:lat:101:51 m1xm:pair:fresh_val_2003_d1_v1:x:med:101:51 u1ym:pair:fresh_val_2000_full:y:med:101:51"
$PY -u r1_sweep.py plan $E $SPECS > $E/bodies.txt 2> $R/e13_plan.log; r=$?; st "PLAN rc=$r $(wc -l < $E/bodies.txt) points"; [ $r != 0 ] && RC=$r
cp -n /root/autodl-tmp/OPL/S0/GP_TEMPLATES_n32.npz $E/body/ 2>/dev/null
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
# ---------------------------------------------------------------- E11 (GPU, while the bodies are built)
VAL=$(python3 -c "import json;print(','.join(json.load(open('/root/autodl-tmp/OPL/S2/SPLIT_ARMS.json'))['val_s3']))")
run e11_view17 5400 $PY -u eval_views_pd.py $R/E11 --prefix view17_ --model A3=$O/A3_2grid/best.pt --views 17 --cases $VAL \
  --body /root/autodl-tmp/OPL/S0 --data /root/autodl-tmp/OPL/S2/data_v2
run e11_summary 600 $PY -u r1x3_e11_summary.py $R/E11/view17_A3.json $R/E11/E11_SUMMARY.json
# ---------------------------------------------------------------- E13 sweeps (budget shares of the remaining time)
sweep() { local tag=$1 frac=$2; local now=$(date +%s); local left=$(( GEND - now )); [ $left -lt 120 ] && { st "SKIP $tag (budget)"; return; }
  local dl=$(( now + left * frac / 100 )); [ $dl -gt $GEND ] && dl=$GEND
  run e13_$tag $(( dl - now + 900 )) $PY -u r1_sweep.py run $E $tag --deadline $dl; }
sweep m1x 30
sweep u1y 43
sweep l221 50
sweep m1xm 50
sweep u1ym 100
run e13_summary 900 $PY -u r1_sweep.py summary $E
wait $BPID 2>/dev/null
st "ALL_STEPS_DONE rc=$RC"
