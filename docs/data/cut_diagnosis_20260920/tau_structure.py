"""What does the tau axis actually look like, for cut cells and for the box cells that DID generalise?"""
import json, math, itertools, statistics as st, numpy as np
inv={r['seat']: r for r in json.load(open('/root/_inventory.json'))}
L={int(e['seat']): e for e in json.load(open('/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json'))}
desc={r['seat']: r for r in json.load(open('/root/_seat_desc.json'))}
# full 8-vector tau per seat
tau={}
for s,e in L.items():
    try:
        z=np.load(e['trace_cache'], allow_pickle=False); tau[s]=np.asarray(z['tau_corners'], float)
    except Exception: pass
cut=[s for s,r in desc.items() if (r.get('n_kind1') or 0)>0 and inv.get(s,{}).get('mq') and s in tau]
box=[s for s,r in desc.items() if (r.get('n_kind1') or 0)==0 and inv.get(s,{}).get('mq') and s in tau]
dom=[s for s in cut if (desc[s].get('tau_mean') or 9)<=0.45]
print(f'trainable cut {len(cut)} (in domain {len(dom)}) | trainable box {len(box)}')
def report(name, ss):
    T=np.array([tau[s] for s in ss])
    print(f'{name}: n={len(ss)}')
    print(f'   per-corner min {T.min(axis=0).round(3)}')
    print(f'   per-corner max {T.max(axis=0).round(3)}')
    print(f'   mean tau over corners: {T.mean(axis=1).min():.3f}-{T.mean(axis=1).max():.3f}')
    grad=np.abs(T.max(axis=1)-T.min(axis=1))            # within-cell tau spread = the grading
    print(f'   within-cell grading |max-min| : min {grad.min():.4f} med {np.median(grad):.4f} max {grad.max():.4f}')
    # nearest-neighbour distance in the 8-vector, relative to the mean tau
    nn=[]
    for i,s in enumerate(ss):
        d=[np.linalg.norm(tau[s]-tau[t])/np.linalg.norm(tau[s]) for t in ss if t!=s]
        nn.append(min(d))
    print(f'   nearest neighbour in tau (relative): med {st.median(nn):.3f} min {min(nn):.3f} max {max(nn):.3f}')
report('BOX (these DID generalise: 0.3-3.3 % on 12 seats)', box)
report('CUT in design domain', dom)
print()
band=[s for s in dom if desc[s]['cut_plane'][1] < 0.10]
print(f'=== the b<0.10 slope band: {len(band)} trainable in-domain seats ===')
T=np.array([tau[s] for s in band])
print(f'   tau_mean span {T.mean(axis=1).min():.3f}-{T.mean(axis=1).max():.3f}')
print(f'   per-corner min {T.min(axis=0).round(3)}')
print(f'   per-corner max {T.max(axis=0).round(3)}')
print()
print('=== near-duplicate cut planes: can we vary tau at a nearly FIXED cut? ===')
pairs=[]
for a,b in itertools.combinations(dom,2):
    dp=math.hypot(desc[a]['cut_plane'][1]-desc[b]['cut_plane'][1], desc[a]['cut_plane'][3]-desc[b]['cut_plane'][3])
    dt=np.linalg.norm(tau[a]-tau[b])/np.linalg.norm(tau[a])
    pairs.append((dp,dt,a,b))
pairs.sort()
print(f"{'|d(b,d)|':>9} {'rel d(tau)':>10}  seats")
for dp,dt,a,b in pairs[:12]: print(f'{dp:>9.4f} {dt:>10.3f}  {a} vs {b}')
tight=[p for p in pairs if p[0]<=0.03]
print(f'pairs with |d(b,d)| <= 0.03: {len(tight)}; their relative tau distance: '
      f'med {st.median([p[1] for p in tight]):.3f} max {max(p[1] for p in tight):.3f}')
