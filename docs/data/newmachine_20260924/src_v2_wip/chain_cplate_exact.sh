#!/bin/bash
# C-plate exact checks (2026-10-03), after chain_cplate_prod.sh (whose in-process checks exceed the 96.6 GB container
# limit when loading the dense T of a 24-cell design): exact_check_cpu.py --mmap-T (T memory-mapped) on the start design
# (cplateN k=0), the NICE final design (cplateN last k) and the homogenised final design (hevalHC k=0).
# Status in chain_cplate.status (tag CPEXACT); marker CPLATE_EXACT_DONE.
R=/root/autodl-tmp/OPL/S1/V2/R1/OPT; ST=$R/chain_cplate.status; TAG=CPEXACT; RC=0
st() { echo "$(date +%F_%T) $TAG $*" >> $ST; }
fin() { st "END rc=$RC"; echo "$(date +%F_%T) CPLATE_EXACT_DONE" >> $ST; }
trap fin EXIT
st "QUEUED pid=$$ (waits for CPLATE_PROD_DONE)"
until grep -qE "^[0-9_:-]+ CPLATE_PROD_DONE$" $ST; do sleep 60; done
source /root/autodl-tmp/OPL/S1/env_gpu.sh
D=$DEP/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$D PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/pylib_cpu PYPARDISO_MKL_RT=$D/libmkl_rt.so.3
cd /root/autodl-tmp/OPL/src_v2
for spec in cplateN:0,last hevalHC:0; do n=${spec%%:*}
  st "START exact_$n"; t=$(date +%s)
  timeout 43200 $PY -u exact_check_cpu.py $R/$spec --work $R/exact_cplate --mmap-T --t-route schur --threads 32 \
    > $R/exact_$n.log 2>&1; r=$?; [ $r != 0 ] && RC=$r; st "END exact_$n rc=$r $(( $(date +%s) - t ))s"
done
