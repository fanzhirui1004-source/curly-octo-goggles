#!/bin/bash
# Revision round 1, second CPU machine of the same model (AMD EPYC 9654, cgroup v2: memory.max 60 GiB, memory.high
# 58 GiB, 32-CPU quota). Work split with the 120 GiB machine (chain_r1_cpu.sh), same environment and versions, same iparm
# (iparm_tuned.json copied from the 120 GiB machine), same thread settings; machine name in every JSON (field 'machine').
#  B  E7 retries (accuracy only): ref_valid2.py --retry-errors --headroom-gib 4, sweeps then studies, 16 threads
#  C  route (b) block solver x 3 for hlat221a and hlat221b; PARDISO-path route (b) on hlat221b once (memory guard), 16 threads
#  D  single core: 1-thread PARDISO Cholesky (iparm(24)=iparm(25)=0, ordering as tuned, METIS fallback) on hlat221a, hlat221b
#  S  SciPy SuperLU colamd + mmd on the 2-cell lattice hlatsmoke2, caps RLIMIT_AS 50 GiB, watchdog 50 GiB, wall 2 h
# Environment capture: lscpu, cgroup, cpu.stat, MKL_VERBOSE (cpu3/host/). Every run records loadavg and cpu.stat throttling.
# Status /root/autodl-tmp/OPL/S1/V2/chain_cpu3.status (this machine); markers R1_CPU3_REF_DONE_20260929,
# R1_CPU3_COND_DONE_20260929, R1_CPU3_ONE_DONE_20260929, R1_CPU3_SLU_DONE_20260929, R1_CPU3_ALL_DONE_20260929 (EXIT trap).
O=/root/autodl-tmp/OPL/S1/V2; ST=$O/chain_cpu3.status; R=$O/R1; SRC=/root/autodl-tmp/OPL/src_v2; S4=/root/autodl-tmp/OPL/S4
B=$R/cpu3; H=$B/host; HC=$B/hostcond; H1=$B/host1; mkdir -p $H $HC $H1 $R/ref_logs
IPT=$H/iparm_tuned.json                                  # copied from the 120 GiB machine (identical settings)
st() { echo "$(date +%T) R1CPU3 $*" >> $ST; }
mk() { echo "$(date +%T) $1" >> $ST; }
RC=0
fin() { st "END rc=$RC"; mk R1_CPU3_ALL_DONE_20260929; }
trap fin EXIT
run() { local d=$1 l=$2; shift 2; local t0=$(date +%s); "$@" >> $d/$l.log 2>&1; local rc=$?; [ $rc = 0 ] || RC=$rc
  st "$l rc=$rc $(( $(date +%s)-t0 ))s load=$(cut -d' ' -f1 /proc/loadavg) thr_us=$(awk '/throttled_usec/{print $2}' /sys/fs/cgroup/cpu.stat)"; }
st "START pid=$$ machine=$(hostname)"
source $R/env_cpu.sh
cd $SRC
export MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 OPL_GP_CACHE=0
{ date; hostname; uname -a; lscpu; nproc; cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.high /sys/fs/cgroup/cpu.max /sys/fs/cgroup/cpu.stat;
  free -g; cat /proc/loadavg; $PY -c "import numpy, scipy, torch; print(numpy.__version__, scipy.__version__, torch.__version__)";
  ls /root/autodl-tmp/pylib_cpu /root/autodl-tmp/mklenv/lib/python3.12/site-packages 2>/dev/null | grep dist-info; } > $H/env_host.txt 2>&1
$PY -c "import json, platform, pardiso_direct as PD; r = PD.env_record(); r['machine'] = platform.node(); print(json.dumps(r, indent=1))" > $H/env_host.json 2>> $H/env_host.txt
MKL_VERBOSE=1 $PY -c "
import ctypes, numpy as np, pardiso_direct as PD
L = PD.lib(); a = np.ones((64, 64)); c = np.zeros((64, 64))
L.cblas_dgemm.argtypes = [ctypes.c_int] * 6 + [ctypes.c_double, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_void_p, ctypes.c_int]
L.cblas_dgemm(101, 111, 111, 64, 64, 64, 1.0, a.ctypes.data, 64, a.ctypes.data, 64, 0.0, c.ctypes.data, 64)
" > $H/mkl_verbose.txt 2>&1
# ---------------- B: E7 retries
for stage in sweep studies; do for c in fresh_val_2010_d0_v0 fresh_val_2005_d1_v0 fresh_val_2074_d0_v0 fresh_val_2051_d1_v1; do
  [ $stage = studies ] && [ $c = fresh_val_2005_d1_v0 ] && continue
  run $R/ref_logs ref_${c}_cpu3_${stage} $PY -u ref_valid2.py $R/ref_valid_r1.json $c --ns 24,32,40,48,56,64 --stage $stage --headroom-gib 4 --retry-errors
done; done
mk R1_CPU3_REF_DONE_20260929
# ---------------- C: route (b), 4-cell lattices
for L in hlat221a hlat221b; do
  REF=$(ls $O/lat_direct_${L}*.json 2>/dev/null | tr '\n' ',')
  for k in 1 2 3; do
    run $HC latcond_${L}_rep$k $PY -u lat_cond_cpu.py $HC/latcond_${L}_rep$k.json $S4/$L.json --solver block --iparm-file $IPT --margin-gib 4 --ref "$REF"
  done
done
run $HC latcond_pardiso_hlat221b_rep1 $PY -u lat_cond_cpu.py $HC/latcond_pardiso_hlat221b_rep1.json $S4/hlat221b.json --solver pardiso --iparm-file $IPT --margin-gib 4 --ref "$(ls $O/lat_direct_hlat221b*.json | tr '\n' ',')"
run $HC selftest $PY lat_cond_cpu.py --selftest
run $HC summary $PY r1_cond_summary.py $HC
mk R1_CPU3_COND_DONE_20260929
# ---------------- D: single core, 4-cell lattices
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$PY - $IPT $H1 <<'PY'
import json, sys
t = json.load(open(sys.argv[1])); ip = dict(t['iparm']); ip['24'] = 0; ip['25'] = 0
json.dump(dict(iparm=ip, derived_from=sys.argv[1], note='single thread: iparm(24)=0, iparm(25)=0; ordering as tuned'), open(sys.argv[2] + '/iparm_1thread.json', 'w'), indent=1)
ip2 = dict(ip); ip2['2'] = 2
json.dump(dict(iparm=ip2, derived_from=sys.argv[1], note='single thread, fallback ordering iparm(2)=2 (METIS)'), open(sys.argv[2] + '/iparm_1thread_metis.json', 'w'), indent=1)
PY
for L in hlat221a hlat221b; do
  run $H1 lat1_${L}_chol $PY -u lat_direct_cpu2.py $H1/lat1_${L}_chol.json $S4/$L.json --mtypes 2 --iparm-file $H1/iparm_1thread.json --margin-gib 4
  if ! grep -q '"factor_s"\|"skipped"' $H1/lat1_${L}_chol.json 2>/dev/null; then
    st "lat1 $L: tuned ordering failed single-threaded; retry with iparm(2)=2"
    run $H1 lat1_${L}_chol_metis $PY -u lat_direct_cpu2.py $H1/lat1_${L}_chol_metis.json $S4/$L.json --mtypes 2 --iparm-file $H1/iparm_1thread_metis.json --margin-gib 4
  fi
done
mk R1_CPU3_ONE_DONE_20260929
# ---------------- S: SuperLU on the 2-cell lattice
run $H1 scipy_hlatsmoke2 $PY -u lat_scipy_cpu.py $H1/scipy_hlatsmoke2.json $R/smoke/hlatsmoke2.json --as-gib 50 --watch-gib 50 --wall-s 7200 --variants colamd,mmd --ref $B/ref_lat2_hlatsmoke2_chol_rep1.json --keep-npz-dir $H1/npz
st "superlu hlatsmoke2: $($PY -c "import json; d=json.load(open('$H1/scipy_hlatsmoke2.json')); print(' '.join(v+'='+str(c.get('status'))+' factor_s='+str(c.get('factor_s')) for v, c in d.get('solves', {}).items()))" 2>/dev/null)"
run $H1 summary_one $PY r1_host_summary.py $H $H/host_summary_cpu3.json --one $H1
mk R1_CPU3_SLU_DONE_20260929
