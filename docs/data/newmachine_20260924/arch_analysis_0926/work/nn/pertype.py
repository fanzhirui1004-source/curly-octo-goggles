import json, numpy as np, re
def typ(n):
    if n.endswith('full'): return 'FULL'
    m=re.search(r'_v(\d)$',n)
    if m: return {'0':'heavy','1':'medium','2':'light'}[m.group(1)]
    return 'other'
def load(f): return json.load(open(f))['per_geo']
res={}
for f in ['newval_c_oh.json','newval_v2s.json','newval_v2L1.json','trainfit_v2L1.json']:
    pg=load(f)
    print('==',f)
    for cls in ['force','support','face','force_c','macro','grf']:
        row=[]
        for t in ['FULL','light','medium','heavy']:
            v=[pg[g]['0'].get(cls,np.nan) for g in pg if typ(g)==t]
            v=np.array(v,float); v=v[np.isfinite(v)]
            if f.startswith('trainfit'): v=v[v<10]
            if len(v)==0: row.append('   -   '); continue
            row.append(f'{100*v.mean():5.1f}/{100*np.median(v):5.1f}/{100*v.max():5.1f}(n{len(v)})')
        print(f'{cls:8s}',' | '.join(row))
