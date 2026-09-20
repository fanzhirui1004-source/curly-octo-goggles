import json, os, numpy as np, glob
L=json.load(open('/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json'))
out=[]
for e in L:
    seat=int(e['seat']); rec={}
    ip=os.path.join(e['reference'],'INVERSE_RESULT.json')
    mp=os.path.join(e['reference'],'MQ_RESULT.json')
    if os.path.exists(ip):
        d=json.load(open(ip)); rec['kappa']=d.get('condition_of_A'); rec['d']=d.get('d')
        rec['pivmin']=d.get('pivot_min'); rec['pivmax']=d.get('pivot_max')
    if os.path.exists(mp):
        d=json.load(open(mp)); rec['lmin_Ainv']=d.get('lambda_min_of_A_inverse'); rec['lmax_Ainv']=d.get('lambda_max_of_A_inverse')
    tc=e.get('trace_cache')
    if tc and os.path.exists(tc):
        z=np.load(tc, allow_pickle=True)
        k=z['kind']; rec['n_kind0']=int((k==0).sum()); rec['n_kind1']=int((k==1).sum())
        tau=z['tau_corners']; rec['tau_mean']=float(tau.mean()); rec['tau_min']=float(tau.min()); rec['tau_max']=float(tau.max())
        cp=z['cut_plane']; rec['cut_plane']=[float(x) for x in cp]
    rec['seat']=seat; rec['q']=e.get('q'); rec['split']=e.get('split'); rec['thickness']=e.get('thickness')
    out.append(rec)
json.dump(out, open('/root/_seat_desc.json','w'))
cut=[r for r in out if r.get('n_kind1',0)>0]; box=[r for r in out if r.get('n_kind1',0)==0]
print('labelled seats', len(out), 'cut', len(cut), 'box', len(box))
def q(v,p): 
    v=sorted(v); return v[int(p*(len(v)-1))]
for name,g in [('CUT',cut),('BOX',box)]:
    ks=[r['kappa'] for r in g if r.get('kappa')]
    print(f"{name}: n={len(g)} kappa min={min(ks):.3g} p25={q(ks,.25):.3g} med={q(ks,.5):.3g} p75={q(ks,.75):.3g} max={max(ks):.3g}")
print()
print("seat, kappa(A*), d, q, kind1, tau_mean, cut_plane, split")
for r in sorted(out, key=lambda r: -(r.get('kappa') or 0))[:12]:
    print(f"{r['seat']:>7} {r.get('kappa',0):>12.4g} {str(r.get('d')):>7} {str(r.get('q')):>7} {str(r.get('n_kind1')):>6} {r.get('tau_mean',0):>7.3f} {str(r.get('cut_plane'))[:46]:>46} {r.get('split')}")
