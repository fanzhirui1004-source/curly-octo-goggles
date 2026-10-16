import json, glob, os
from fractions import Fraction
R='/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/packets'
rows=[]
for d in sorted(glob.glob(R+'/*')):
    c=os.path.basename(d)
    try: ctx=json.load(open(d+'/FRESH_CONTEXT.json'))
    except Exception as e: continue
    cs=ctx.get('case',{})
    keys=list(cs.keys())
    rows.append((c, cs.get('normal') is not None, {k:cs[k] for k in keys if k not in ('tau_corners',)}))
print(len(rows))
print(json.dumps(rows[:3], default=str)[:3000])
import collections
print(collections.Counter(tuple(sorted(r[2].keys())) for r in rows))
