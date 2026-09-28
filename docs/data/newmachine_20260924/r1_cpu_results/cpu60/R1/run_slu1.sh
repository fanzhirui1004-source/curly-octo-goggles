#!/bin/bash
R=/root/autodl-tmp/OPL/S1/V2/R1; H1=$R/cpu3/host1; ST=/root/autodl-tmp/OPL/S1/V2/chain_cpu3.status
source $R/env_cpu.sh; cd /root/autodl-tmp/OPL/src_v2
echo "$(date +%T) R1CPU3 START scipy_hlatsmoke1 (coordinator, 1-cell SuperLU)" >> $ST
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -u lat_scipy_cpu.py $H1/scipy_hlatsmoke1.json $R/smoke/hlatsmoke1.json --as-gib 50 --watch-gib 50 --wall-s 7200 --variants colamd,mmd --keep-npz-dir $H1/npz > $H1/scipy_hlatsmoke1.log 2>&1
echo "$(date +%T) R1CPU3 END scipy_hlatsmoke1 rc=$? $(python3 -c "import json; d=json.load(open('$H1/scipy_hlatsmoke1.json')); print(' '.join(v+'='+str(c.get('status'))+' factor_s='+str(c.get('factor_s')) for v, c in d.get('solves', {}).items()))" 2>/dev/null)" >> $ST
echo "$(date +%T) R1_CPU3_SLU1_DONE_20260929" >> $ST
