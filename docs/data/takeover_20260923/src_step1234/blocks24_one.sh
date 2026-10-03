#!/bin/bash
# One teacher sensitivity block with the corrected admission (sens_blocks.py v2). Usage: blocks24_one.sh <base> <id>
T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_24; X=$T/SENS_01
W=$T/COVER_G; V=/root/autodl-tmp/CUTFEM_INGEST_R38/environment/.venv/bin/python
export CUTFEM_EXECUTION_CONFIG=$W/EXECUTION_CONFIG.json CUTFEM_RUNTIME_ROOT=/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime
eval "$(cd $W/src && PYTHONPATH=$W/src $V -B -c "import json,shlex;from stage_cutfem_runtime.environment import runtime_environment;c=json.load(open('$W/EXECUTION_CONFIG.json'));e,b=runtime_environment('$W/src',c['runtime_names']);print('\n'.join('export %s=%s'%(k,shlex.quote(v)) for k,v in e.items()))")"
b=$1; p=$2; d=$X/blocks/$p
mkdir -p $d/compiled
[ -f $d/SENS_BLOCKS.json ] && mv $d/SENS_BLOCKS.json $d/SENS_BLOCKS_v1.json
rm -f $d/compiled/A.npz $d/compiled/C.npz $d/compiled/D.npz
cd $W/src && OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 $V -B $S/sens_blocks.py $b $p > $X/blocks24_$p.log 2>&1
echo "$(date -u +%FT%TZ) blocks24 $p exit $?" >> $X/blocks24.status
