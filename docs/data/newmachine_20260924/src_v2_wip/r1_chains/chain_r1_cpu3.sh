#!/bin/bash
# Revision round 1, third CPU machine (AMD EPYC 9654, cgroup v2: memory.max 60 GiB, memory.high 58 GiB, 32-CPU quota).
# Accuracy-only / non-table work moved off the 120 GiB machine (no timing from here enters a table column):
#  B  E7 retries: ref_valid2.py --retry-errors --headroom-gib 4 (sweeps, then studies) for H2, H1, 2074, 2051
#     (2051 n = 64 and the 2074 studies are the open items; H2 n = 56 stays missing), 16 threads
#  S  SciPy SuperLU on the 2-cell lattice hlatsmoke2 (650k DOFs), both variants colamd and mmd, factor + 6 solves,
#     unconditionally; self-enforced caps adapted to this machine: RLIMIT_AS 50 GiB, RSS watchdog 50 GiB, wall 2 h
#     (recorded in the JSON); single thread
# Status /root/autodl-tmp/OPL/S1/V2/chain_cpu3.status (this machine); markers R1_CPU3_REF_DONE_20260929,
# R1_CPU3_SLU_DONE_20260929, R1_CPU3_ALL_DONE_20260929 (EXIT trap).
O=/root/autodl-tmp/OPL/S1/V2; ST=$O/chain_cpu3.status; R=$O/R1; SRC=/root/autodl-tmp/OPL/src_v2
mkdir -p $R/ref_logs $R/cpu3
st() { echo "$(date +%T) R1CPU3 $*" >> $ST; }
mk() { echo "$(date +%T) $1" >> $ST; }
RC=0
fin() { st "END rc=$RC"; mk R1_CPU3_ALL_DONE_20260929; }
trap fin EXIT
run() { local d=$1 l=$2; shift 2; local t0=$(date +%s); "$@" >> $d/$l.log 2>&1; local rc=$?; [ $rc = 0 ] || RC=$rc
  st "$l rc=$rc $(( $(date +%s)-t0 ))s load=$(cut -d' ' -f1 /proc/loadavg)"; }
st "START pid=$$"
source $R/env_cpu.sh
cd $SRC
export MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 OPL_GP_CACHE=0
for stage in sweep studies; do for c in fresh_val_2010_d0_v0 fresh_val_2005_d1_v0 fresh_val_2074_d0_v0 fresh_val_2051_d1_v1; do
  [ $stage = studies ] && [ $c = fresh_val_2005_d1_v0 ] && continue
  run $R/ref_logs ref_${c}_cpu3_${stage} $PY -u ref_valid2.py $R/ref_valid_r1.json $c --ns 24,32,40,48,56,64 --stage $stage --headroom-gib 4 --retry-errors
done; done
mk R1_CPU3_REF_DONE_20260929
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run $R/cpu3 scipy_hlatsmoke2 $PY -u lat_scipy_cpu.py $R/cpu3/scipy_hlatsmoke2.json $R/smoke/hlatsmoke2.json --as-gib 50 --watch-gib 50 --wall-s 7200 --variants colamd,mmd --ref $R/cpu3/ref_lat2_hlatsmoke2_chol_rep1.json --keep-npz-dir $R/cpu3/npz
st "superlu hlatsmoke2: $($PY -c "import json; d=json.load(open('$R/cpu3/scipy_hlatsmoke2.json')); print(' '.join(v+'='+str(c.get('status'))+' factor_s='+str(c.get('factor_s')) for v, c in d.get('solves', {}).items()))" 2>/dev/null)"
mk R1_CPU3_SLU_DONE_20260929
