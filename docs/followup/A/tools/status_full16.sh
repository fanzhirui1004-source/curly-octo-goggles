# Remote status of an a0_eval run directory (run with c3n.sh). Successful records have no "error" key; the last record of
# a (cell, map) pair counts, because resumed runs append.
cd /root/autodl-tmp/OPL/A0
python - <<'PY'
import json, collections
R=[json.loads(l) for l in open('runs/full16/RESULTS.jsonl')]
last={}
for r in R: last[(r['case'],r['map'])]=r
ok=collections.Counter(c for (c,m),r in last.items() if 'error' not in r); err=collections.Counter(c for (c,m),r in last.items() if 'error' in r)
print('ok', sum(ok.values()), 'err', sum(err.values()), 'pairs', len(last))
PY
pgrep -af "a0_eval.py|run_rest.sh" | cut -c1-120 || echo "no a0 process"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
