#!/bin/bash
# Revision round 1, dedicated CPU host (AMD EPYC 9654, cgroup v1: 120 GiB, 32-CPU quota): ALL remaining CPU work, sequential.
# Tag R1CPU; status file /root/autodl-tmp/OPL/S1/V2/chain_cpu.status (this machine). Stage markers (plain lines):
#   R1_CPU_E1_DONE_20260928, R1_CPU_REF_DONE_20260928, R1_CPU_COND_DONE_20260928, R1_CPU_ONE_DONE_20260928,
#   and R1_CPU_ALL_DONE_20260928 at the end (EXIT trap, also on failure).
# Stage A  E1 (16 threads): env capture, tuning (G3), ST17 per-cell Cholesky x 3 (bench_cpu2 main), LU reference + wrapper
#          overhead (luref), Table 6 lattice Cholesky x 3 for hlat221a, hlat221b, hlat222, hlat331 (memory guard, margin
#          4 GiB, ceiling limit - 2 GiB), LU x 1 for the 4-cell lattices; supplementary: one 32-thread Cholesky per lattice
#          (lat2t32_*.json); summary host/host_summary.json.
# Stage B  E7 retries (16 threads): ref_valid2.py --retry-errors --headroom-gib 4, sweeps then studies, 4 cells.
# Stage C  route (b) conventional exact condensation (16 threads): lat_cond_cpu.py block solver x 3 for all four lattices;
#          one --solver pardiso run on hlat221b; summary hostcond/hostcond_summary.json.
# Stage D  single core (1 thread): lat_direct_cpu2 Cholesky (iparm(24)=iparm(25)=0; ordering as tuned, METIS fallback) on
#          hlat221a, hlat221b, hlat222, hlat331 (guarded); SciPy SuperLU colamd/mmd on hlat221a (70 GiB / 2 h self-limits),
#          fallback both on the 2-cell lattice.
# Every run records loadavg and container CPU throttling (cpu.stat) itself. SMOKE=1: tiny versions, output R1/smoke_cpu.
O=/root/autodl-tmp/OPL/S1/V2; ST=$O/chain_cpu.status; R=$O/R1; SRC=/root/autodl-tmp/OPL/src_v2; S4=/root/autodl-tmp/OPL/S4
G="fresh_train_0020_cover01_r2,fresh_train_0020_cover01_r1,fresh_train_0020_full,fresh_train_0007_full"
LATS="hlat221a hlat221b hlat222 hlat331"; NREP=3; SB=300; TUNEC=fresh_train_0020_full; B=$R/cpu; TAG=R1CPU
ASG=70; WAL=7200; SLAT=$S4/hlat221a.json; SFB=$R/smoke/hlatsmoke2.json
lay() { echo $S4/$1.json; }
if [ "$SMOKE" = 1 ]; then
  G=fresh_train_0020_cover01_r1; LATS="hlatsmoke1"; NREP=1; SB=20; TUNEC=fresh_train_0020_cover01_r1; B=$R/smoke_cpu; TAG="R1CPU SMOKE"
  ASG=6; WAL=300; SLAT=$R/smoke/hlatsmoke1.json; SFB=""
  lay() { echo $R/smoke/$1.json; }
fi
H=$B/host; HC=$B/hostcond; H1=$B/host1; mkdir -p $H $HC $H1 $R/ref_logs
st() { echo "$(date +%T) $TAG $*" >> $ST; }
mk() { [ "$SMOKE" = 1 ] || echo "$(date +%T) $1" >> $ST; }
RC=0
fin() { st "END rc=$RC"; mk R1_CPU_ALL_DONE_20260928; }
trap fin EXIT
run() {  # run <logdir> <label> <cmd...>
  local d=$1 l=$2; shift 2; local t0=$(date +%s); "$@" >> $d/$l.log 2>&1; local rc=$?; [ $rc = 0 ] || RC=$rc
  st "$l rc=$rc $(( $(date +%s)-t0 ))s load=$(cut -d' ' -f1 /proc/loadavg) thr_ns=$(awk '/throttled_time/{print $2}' /sys/fs/cgroup/cpu/cpu.stat)"; }
st "START pid=$$"
source $R/env_cpu.sh
cd $SRC
# ---------------- A: E1
export MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 OPL_GP_CACHE=0
{ date; uname -a; lscpu; nproc; cat /sys/fs/cgroup/memory/memory.limit_in_bytes /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us;
  cat /sys/fs/cgroup/cpu/cpu.stat; free -g; cat /proc/loadavg; $PY -c "import numpy, scipy, torch; print(numpy.__version__, scipy.__version__, torch.__version__)";
  ls /root/autodl-tmp/pylib_cpu /root/autodl-tmp/mklenv/lib/python3.12/site-packages 2>/dev/null | grep dist-info; } > $H/env_host.txt 2>&1
$PY -c "import json, pardiso_direct as PD; print(json.dumps(PD.env_record(), indent=1))" > $H/env_host.json 2>> $H/env_host.txt
MKL_VERBOSE=1 $PY -c "
import numpy as np, scipy.sparse as sp, pardiso_direct as PD
A = sp.random(300, 300, density=0.05, random_state=0); A = (A @ A.T + 300 * sp.identity(300)).tocsr(); U = sp.triu(A).tocsr()
P = PD.Pardiso(U, 2, PD.tuned_iparm()); P.phase(11); P.phase(22); P.solve(np.ones(300)); P.release()
import ctypes; L = PD.lib(); a = np.ones((64, 64)); c = np.zeros((64, 64))
L.cblas_dgemm.argtypes = [ctypes.c_int] * 6 + [ctypes.c_double, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_void_p, ctypes.c_int]
L.cblas_dgemm(101, 111, 111, 64, 64, 64, 1.0, a.ctypes.data, 64, a.ctypes.data, 64, 0.0, c.ctypes.data, 64)
" > $H/mkl_verbose.txt 2>&1
run $H tune $PY -u bench_cpu2.py $H/bench_cpu2_tune.json $TUNEC --mode tune --reps 3 --iparm-file $H/iparm_tuned.json
for k in $(seq 1 $NREP); do
  run $H t5_chol_rep$k $PY -u bench_cpu2.py $H/bench_cpu2_chol_rep$k.json $G --mode main --reps 3 --s-budget $SB --iparm-file $H/iparm_tuned.json
done
run $H t5_luref $PY -u bench_cpu2.py $H/bench_cpu2_luref.json $G --mode luref --reps 3
for L in $LATS; do
  for k in $(seq 1 $NREP); do
    run $H t6_${L}_chol_rep$k $PY -u lat_direct_cpu2.py $H/lat2_${L}_chol_rep$k.json $(lay $L) --mtypes 2 --iparm-file $H/iparm_tuned.json --margin-gib 4
  done
  case $L in hlat221a|hlat221b|hlatsmoke1)
    run $H t6_${L}_lu $PY -u lat_direct_cpu2.py $H/lat2_${L}_lu.json $(lay $L) --mtypes 11 --margin-gib 4 ;; esac
done
for L in $LATS; do   # supplementary: one 32-thread repetition (container quota 32 CPUs)
  MKL_NUM_THREADS=32 OMP_NUM_THREADS=32 run $H t6_${L}_chol_t32 $PY -u lat_direct_cpu2.py $H/lat2t32_${L}_chol.json $(lay $L) --mtypes 2 --iparm-file $H/iparm_tuned.json --margin-gib 4
done
run $H summary $PY r1_host_summary.py $H $H/host_summary.json
mk R1_CPU_E1_DONE_20260928
# ---------------- B: E7 retries -- MOVED to the third CPU machine (chain_r1_cpu3.sh; in-place edit 2026-09-29)
st "stage B (E7 retries) moved to the third CPU machine (see chain_cpu3.status there)"
mk R1_CPU_REF_DONE_20260928
# ---------------- C: route (b)
for L in $LATS; do
  REF=$H/lat2_${L}_chol_rep1.json,$H/lat2_${L}_chol_rep2.json,$(ls $O/lat_direct_${L}*.json 2>/dev/null | tr '\n' ',')
  for k in $(seq 1 $NREP); do
    run $HC latcond_${L}_rep$k $PY -u lat_cond_cpu.py $HC/latcond_${L}_rep$k.json $(lay $L) --solver block --iparm-file $H/iparm_tuned.json --margin-gib 4 --ref "$REF"
  done
done
PL=hlat221b; [ "$SMOKE" = 1 ] && PL=hlatsmoke1
run $HC latcond_pardiso_${PL}_rep1 $PY -u lat_cond_cpu.py $HC/latcond_pardiso_${PL}_rep1.json $(lay $PL) --solver pardiso --iparm-file $H/iparm_tuned.json --margin-gib 4 --ref $H/lat2_${PL}_chol_rep1.json
run $HC selftest $PY lat_cond_cpu.py --selftest
run $HC summary $PY r1_cond_summary.py $HC
mk R1_CPU_COND_DONE_20260928
# ---------------- D: single core
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$PY - $H/iparm_tuned.json $H1 <<'PY'
import json, sys
t = json.load(open(sys.argv[1])); ip = dict(t['iparm']); ip['24'] = 0; ip['25'] = 0
json.dump(dict(iparm=ip, derived_from=sys.argv[1], note='single thread: iparm(24)=0, iparm(25)=0; ordering as tuned'), open(sys.argv[2] + '/iparm_1thread.json', 'w'), indent=1)
ip2 = dict(ip); ip2['2'] = 2
json.dump(dict(iparm=ip2, derived_from=sys.argv[1], note='single thread, fallback ordering iparm(2)=2 (METIS)'), open(sys.argv[2] + '/iparm_1thread_metis.json', 'w'), indent=1)
PY
for L in $LATS; do
  run $H1 lat1_${L}_chol $PY -u lat_direct_cpu2.py $H1/lat1_${L}_chol.json $(lay $L) --mtypes 2 --iparm-file $H1/iparm_1thread.json --margin-gib 4
  if ! grep -q '"factor_s"\|"skipped"' $H1/lat1_${L}_chol.json 2>/dev/null; then
    st "lat1 $L: tuned ordering failed single-threaded; retry with iparm(2)=2"
    run $H1 lat1_${L}_chol_metis $PY -u lat_direct_cpu2.py $H1/lat1_${L}_chol_metis.json $(lay $L) --mtypes 2 --iparm-file $H1/iparm_1thread_metis.json --margin-gib 4
  fi
done
slu() {
  local lf=$1; local L=$(basename $lf .json)
  local REF=$H/lat2_${L}_chol_rep1.json,$H1/lat1_${L}_chol.json,$(ls $O/lat_direct_${L}*.json 2>/dev/null | tr '\n' ',')
  run $H1 scipy_${L} $PY -u lat_scipy_cpu.py $H1/scipy_${L}.json $lf --as-gib $ASG --watch-gib $ASG --wall-s $WAL --variants colamd,mmd --ref "$REF" --keep-npz-dir $H1/npz
  st "superlu $L: $($PY -c "import json; d=json.load(open('$H1/scipy_${L}.json')); print(' '.join(v+'='+str(c.get('status')) for v, c in d.get('solves', {}).items()))" 2>/dev/null)"; }
slu $SLAT
st "SuperLU 2-cell fallback moved to the third CPU machine (in-place edit 2026-09-29)"
run $H1 scipy_selftest $PY lat_scipy_cpu.py --selftest $H1
mk R1_CPU_ONE_DONE_20260928
