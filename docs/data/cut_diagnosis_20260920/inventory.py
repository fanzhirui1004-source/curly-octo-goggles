import json, os
L=json.load(open('/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json'))
rows=[]
for e in L:
    ref=e['reference']; pk=e['packet']
    rows.append(dict(seat=int(e['seat']), q=e['q'], split=e['split'], thickness=e['thickness'],
                     mq=os.path.exists(os.path.join(ref,'MQ_UPPER.npy')),
                     rup=os.path.exists(os.path.join(ref,'R_UPPER.npy')),
                     inv=os.path.exists(os.path.join(ref,'INVERSE_RESULT.json')),
                     sup=os.path.exists(os.path.join(pk,'S_UPPER.npy')),
                     tc=os.path.exists(e['trace_cache'])))
import numpy as np
desc={r['seat']:r for r in json.load(open('/root/_seat_desc.json'))}
def kind(s):
    d=desc.get(s,{}); k=d.get('n_kind1')
    return 'cut' if (k or 0)>0 else ('box' if k==0 else '?')
for label in ['cut','box','?']:
    g=[r for r in rows if kind(r['seat'])==label]
    if not g: continue
    print(f"{label:>4}: n={len(g):>4}  MQ={sum(r['mq'] for r in g):>4}  R_UPPER={sum(r['rup'] for r in g):>4}"
          f"  INVERSE={sum(r['inv'] for r in g):>4}  S_UPPER={sum(r['sup'] for r in g):>4}  cache={sum(r['tc'] for r in g):>4}")
cut=[r for r in rows if kind(r['seat'])=='cut']
ind=[r for r in cut if (desc[r['seat']].get('tau_mean') or 9)<=0.45]
print()
print('cut seats in design domain tau<=0.45:', len(ind),
      '| with MQ label:', sum(r['mq'] for r in ind), '| with S_UPPER (teacher, needed for condensation):', sum(r['sup'] for r in ind))
print('cut seats with S_UPPER but no MQ label (condensable, not yet trainable):',
      sum(1 for r in cut if r['sup'] and not r['mq']))
print('q range of cut seats with S_UPPER:', min(r['q'] for r in cut if r['sup']), max(r['q'] for r in cut if r['sup']))
json.dump(rows, open('/root/_inventory.json','w'))
