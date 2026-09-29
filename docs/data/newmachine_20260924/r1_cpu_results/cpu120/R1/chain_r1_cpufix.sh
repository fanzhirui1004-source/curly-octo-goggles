#!/bin/bash
# Revision round 1, 120 GiB AMD machine: re-run the 4-cell route (b) and single-core runs here, so that every CPU number of
# Table 5 comes from one machine (the 60 GiB machine's single-thread front end was ~2x slower on identical cells).
# Tag R1CPUFIX; status file chain_cpu.status; marker R1_CPU_FIX_DONE_20260929 (EXIT trap).
O=/root/autodl-tmp/OPL/S1/V2; ST=$O/chain_cpu.status; R=$O/R1; SRC=/root/autodl-tmp/OPL/src_v2; S4=/root/autodl-tmp/OPL/S4
B=$R/cpu; H=$B/host; HC=$B/hostcond; H1=$B/host1; TAG=R1CPUFIX
st() { echo "$(date +%T) $TAG $*" >> $ST; }
RC=0
fin() { st "END rc=$RC"; echo "$(date +%T) R1_CPU_FIX_DONE_20260929" >> $ST; }
trap fin EXIT
run() { local d=$1 l=$2; shift 2; local t0=$(date +%s); "$@" >> $d/$l.log 2>&1; local rc=$?; [ $rc = 0 ] || RC=$rc
  st "$l rc=$rc $(( $(date +%s)-t0 ))s load=$(cut -d' ' -f1 /proc/loadavg) thr_ns=$(awk '/throttled_time/{print $2}' /sys/fs/cgroup/cpu/cpu.stat)"; }
st "START pid=$$"
source $R/env_cpu.sh
cd $SRC
export MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 OPL_GP_CACHE=0
for L in hlat221a hlat221b; do
  REF=$H/lat2_${L}_chol_rep1.json,$(ls $O/lat_direct_${L}*.json 2>/dev/null | tr '\n' ',')
  run $HC latcond_${L}_rep1 $PY -u lat_cond_cpu.py $HC/latcond_${L}_rep1.json $S4/$L.json --solver block --iparm-file $H/iparm_tuned.json --margin-gib 4 --ref "$REF"
done
run $HC summary2 $PY r1_cond_summary.py $HC
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for L in hlat221a hlat221b; do
  run $H1 lat1_${L}_chol $PY -u lat_direct_cpu2.py $H1/lat1_${L}_chol.json $S4/$L.json --mtypes 2 --iparm-file $H1/iparm_1thread.json --margin-gib 4
done
