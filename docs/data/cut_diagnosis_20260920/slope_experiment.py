"""Arm S vs arm W, fixed: both arms must COVER the held-out slope; only the concentration differs.

A part's facet has one slope b and its boundary cells differ only in offset d, so the application
needs "generalise over d and tau at fixed b".  Arm S spends all 15 training seats at that slope;
arm W spends the same 15 across the whole family, including 1-2 at that slope.  Same n, steps,
capacity, g=0, no sharding, and the SAME held-out seats.
"""
import json
inv={r['seat']: r for r in json.load(open('/root/_inventory.json'))}
desc={r['seat']: r for r in json.load(open('/root/_seat_desc.json'))}
ok=[s for s,r in desc.items()
    if (r.get('n_kind1') or 0)>0 and (r.get('tau_mean') or 9)<=0.45 and inv.get(s,{}).get('mq')]
band=sorted([s for s in ok if desc[s]['cut_plane'][1] < 0.10], key=lambda s: desc[s]['cut_plane'][3])
held=[band[i] for i in (2, 5, len(band)//2, len(band)-3)]
train_S=[s for s in band if s not in held]
pool=sorted([s for s in ok if s not in held], key=lambda s: desc[s]['cut_plane'][1])
k=len(train_S)
# even quantiles of b over the WHOLE pool, so W covers the held-out slope too
idx=[round(i*(len(pool)-1)/(k-1)) for i in range(k)]
train_W=[]
for i in idx:
    j=i
    while pool[j] in train_W: j=(j+1) % len(pool)
    train_W.append(pool[j])
def span(ss,key):
    v=[desc[s]['cut_plane'][key] for s in ss]; return f'{min(v):.3f}-{max(v):.3f}'
print(f'HELD  ({len(held)}): {held}')
print(f'   b {span(held,1)}   d {span(held,3)}   tau {min(desc[s]["tau_mean"] for s in held):.3f}-{max(desc[s]["tau_mean"] for s in held):.3f}')
for nm, ts in (('ARM_S', train_S), ('ARM_W', train_W)):
    nearband=sum(1 for s in ts if desc[s]['cut_plane'][1] < 0.10)
    print(f'{nm} ({len(ts)}): b {span(ts,1)}  d {span(ts,3)}  seats with b<0.10: {nearband}')
    print(f'   {" ".join(map(str,ts))}')
W='/root/autodl-tmp/CLAUDE_EQUI_20260918'
for nm, ts in (('S', train_S), ('W', train_W)):
    open(f'{W}/split_slope_{nm}.txt','w').write(
        f"ALL={' '.join(map(str,ts+held))}\nPRESENTED={' '.join(map(str,ts))}\nEVAL={' '.join(map(str,held))}\n")
json.dump(dict(scope='arm S (all training seats at one facet slope) against arm W (the same number spread '
                     'over the whole cut-plane family, still covering that slope)',
               declared='2026-09-20',
               matched=['n=15','steps','capacity','g=0','no sharding','identical held-out seats'],
               family='every cut plane in the dataset has normal (1, b, 0); b in 0.003-0.987, offset d in 0.034-1.658',
               per_part_geometry='one flat facet gives all its boundary cells the SAME b and different d; '
                                 'the number of distinct local offsets is 1 for a lattice-aligned facet, '
                                 '2 for slope 1/2, 4 for slope 1/4, and ~N for a generic slope (38 at N=32)',
               held_out=held, arm_S=train_S, arm_W=train_W,
               reading='S >> W on the same held-out seats means the requirement the application actually has '
                       'is much easier than the family we have been training for, and a per-part specialist '
                       'at n<=4 (aligned or simple-rational facets) or n~N (generic facets) is the route'),
          open(W+'/SLOPE_EXPERIMENT.json','w'), indent=1)
print('\nwritten split_slope_S.txt / split_slope_W.txt / SLOPE_EXPERIMENT.json')
