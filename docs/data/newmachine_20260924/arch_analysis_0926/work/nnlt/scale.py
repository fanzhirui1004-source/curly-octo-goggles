import json, numpy as np, re
def typ(n):
    if n.endswith('full'): return 'FULL'
    m=re.search(r'_v(\d)$',n); return {'0':'heavy','1':'medium','2':'light'}[m.group(1)]
L=lambda f: json.load(open(f))['per_geo']
a,b,c=L('newval_c_oh.json'),L('newval_v2s.json'),L('newval_v2L1.json')
for cls in ['force','support','face','macro','grf']:
  print('==',cls)
  for t in ['FULL','light','medium','heavy']:
    g=[k for k in c if typ(k)==t]
    A=np.array([a[k]['0'][cls] for k in g]);B=np.array([b[k]['0'][cls] for k in g]);C=np.array([c[k]['0'][cls] for k in g])
    r=np.log(C/B); r2=np.log(B/A)
    # alpha per cell for 148->305
    al=-r/np.log(305/148)
    print(f'{t:6s} mean {100*B.mean():5.2f}->{100*C.mean():5.2f} geomean {100*np.exp(np.log(B).mean()):5.2f}->{100*np.exp(np.log(C).mean()):5.2f}  per-cell ratio median {np.exp(np.median(r)):.2f} IQR {np.exp(np.percentile(r,25)):.2f}-{np.exp(np.percentile(r,75)):.2f}  frac improved {np.mean(r<0):.2f}  alpha(median) {np.median(al):.2f}  alpha(mean-based) {np.log(B.mean()/C.mean())/np.log(305/148):.2f} | c_oh->v2s ratio med {np.exp(np.median(r2)):.2f}')
  # correlation of improvement with difficulty
  g=list(c); B=np.array([b[k]['0'][cls] for k in g]);C=np.array([c[k]['0'][cls] for k in g])
  print(' spearman(log err v2s, log ratio)', np.corrcoef(np.argsort(np.argsort(np.log(B))),np.argsort(np.argsort(np.log(C/B))))[0,1])
