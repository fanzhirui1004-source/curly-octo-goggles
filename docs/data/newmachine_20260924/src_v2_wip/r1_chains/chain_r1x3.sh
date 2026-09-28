#!/bin/bash
# Revision round 1, derivative chain (tag R1X3; 2026-09-28). Queued behind the accuracy chain: waits for its completion
# line "HH:MM:SS R1ACC <accuracy marker> ..." (anchored match), then runs on the GPU
#   E14  r1_psd.py      element PSD of K_e and K_e,c (FD and forward-mode AD moments) on the 8 pair test cells
#   E3+E4 pairs         r1_pairs.py, one process per test cell (x and y; fresh_val_2010 excluded: ill-posed), LAT_CPU=1,
#                       FUSED_HYPER=1 as the archived gates; steps 1e-2..1e-4, frozen-interval variant at 1e-3,
#                       full re-solve check on M1/x and U1/y (largest and median |s_c| corners, steps 1e-2 and 3e-3)
#   E3+E4 lattices fp64 r1_lat.py --deriv: hlat222 with its layers hlat221a/b (--park --resident 2, as lat_hetero222_A3),
#                       then hlat331 (--lean --park --resident 5, as lat_hetero331_A3); steps 3e-3, 1e-3, 3e-4
#   E4 lattices fp32    r1_lat.py --deploy (fp32 correction and coarse solve, the timed route): hlat222, hlat331
#   summary             r1x3_summary.py -> R1/X3/SUMMARY_X3.json (pre-registered gradient decision rule)
# Every step records its rc; the chain ALWAYS ends with the line "HH:MM:SS R1_DERIV_DONE_20260928 rc=<rc>" in
# chain_p1.status (EXIT trap), so the host-timing chain can never deadlock. Status lines never contain other markers.
# Host memory: before each heavy step the chain waits (at most 2 h per step) until the cgroup has the stated GiB free
# of anonymous memory (the E7 refinement may run concurrently on the host).
O=/root/autodl-tmp/OPL/S1/V2; R=$O/R1/X3; ST=$O/chain_p1.status; S=/root/autodl-tmp/OPL/src_v2
S4=/root/autodl-tmp/OPL/S4; CK=$O/A3_2grid/best.pt
TAG=R1X3
st() { echo "$(date +%T) $TAG $*" >> $ST; }
RC=0; MARKED=""
mark() { [ -n "$MARKED" ] && return; MARKED=1; st "END rc=$RC"; echo "$(date +%T) R1_DERIV_DONE_20260928 rc=$RC" >> $ST; }
trap mark EXIT
avail() { python3 -c "
m=int(open('/sys/fs/cgroup/memory.max').read().strip() or 0)
s=dict(l.split() for l in open('/sys/fs/cgroup/memory.stat'))
print(int((m-int(s['anon'])-int(s['shmem']))/2**30))" 2>/dev/null || echo 999; }
memwait() { local n=0 t0=$(date +%s); while [ $(avail) -lt $1 ] && [ $(( $(date +%s) - t0 )) -lt 7200 ]; do
  [ $n = 0 ] && st "WAIT_MEM need $1 GiB avail $(avail) GiB"; n=1; sleep 120; done; }
run() { local n=$1 to=$2; shift 2; st "START $n"; local t=$(date +%s)
  timeout $to "$@" > $R/$n.log 2>&1; local r=$?; [ $r != 0 ] && RC=$r; st "END $n rc=$r $(( $(date +%s) - t ))s"; }
mkdir -p $R/pairs $R/tmp
st "QUEUED pid=$$ (waits for the accuracy-chain completion line)"
has_acc() { grep -E "^[0-9]{2}:[0-9]{2}:[0-9]{2} (R1ACC )?R1_ACC_DONE_20260928( |$)" $ST | grep -viq "wait"; }
until has_acc; do sleep 300; done
st "START_WORK pid=$$"
source /root/autodl-tmp/OPL/S1/env_gpu.sh
D=/root/autodl-tmp/CUTFEM_DEPENDENCIES_20260924/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$D PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/pylib_cpu PYPARDISO_MKL_RT=$D/libmkl_rt.so.3 MKL_NUM_THREADS=16
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets OPL_GP_CACHE=0 OPL_CONV_FP32=1
cd $S
{ date; nvidia-smi; free -g; md5sum r1x3_common.py r1_pairs.py r1_lat.py r1_psd.py r1x3_summary.py lat_hetero.py lat_full.py lattice3.py \
  teacher.py trainlib.py fastnet.py models.py moments_ad.py $CK; env | grep -E "^OPL_|^PYTORCH|^OMP|^MKL" | sort; } > $R/chain_env.txt 2>&1
# ---------------------------------------------------------------- E14
run psd 3600 $PY -u r1_psd.py $R/psd.json fresh_val_2001_full,fresh_val_2003_d1_v1,fresh_val_2006_d0_v1,fresh_val_2005_d1_v0,fresh_val_2010_d0_v0,fresh_val_2000_full,fresh_val_2002_d0_v0,fresh_val_2004_d0_v2 --body /root/autodl-tmp/OPL/S0
# ---------------------------------------------------------------- E3 + E4 pairs
RES=fresh_val_2003_d1_v1:x:auto2,fresh_val_2000_full:y:auto2
for c in fresh_val_2003_d1_v1 fresh_val_2000_full fresh_val_2001_full fresh_val_2005_d1_v0 fresh_val_2006_d0_v1 fresh_val_2002_d0_v0 fresh_val_2004_d0_v2; do
  need=40; [ $c = fresh_val_2004_d0_v2 ] && need=60
  memwait $need
  run pair_$c 7200 env LAT_CPU=1 FUSED_HYPER=1 $PY -u r1_pairs.py $R/pairs $c --ckpt $CK --resolve $RES
done
# ---------------------------------------------------------------- E3 + E4 lattices, fp64 correction (Section 6.9 route)
memwait 45
run lat64_222 21600 $PY -u r1_lat.py $R/lat64_222 $S4/hlat222.json,$S4/hlat221a.json,$S4/hlat221b.json --model A3=$CK --max-cols 16 --park --resident 2 --deriv
memwait 50
run lat64_331 21600 $PY -u r1_lat.py $R/lat64_331 $S4/hlat331.json --model A3=$CK --max-cols 16 --lean --park --resident 5 --deriv
# ---------------------------------------------------------------- E4 lattices, fp32 correction (timed route of Table 6)
memwait 45
run lat32_222 14400 $PY -u r1_lat.py $R/lat32_222 $S4/hlat222.json --model A3=$CK --max-cols 16 --deploy --park --resident 2
memwait 50
run lat32_331 14400 $PY -u r1_lat.py $R/lat32_331 $S4/hlat331.json --model A3=$CK --max-cols 16 --deploy --park --resident 5
# ---------------------------------------------------------------- summary
run summary 1800 $PY -u r1x3_summary.py $R
st "ALL_STEPS_DONE rc=$RC"
