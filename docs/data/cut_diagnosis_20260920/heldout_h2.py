import json, glob
desc={r['seat']:r for r in json.load(open('/root/_seat_desc.json'))}
W='/root/autodl-tmp/CLAUDE_EQUI_20260918'
pres={}
for f in glob.glob(W+'/split_*.txt'):
    for line in open(f):
        if line.startswith('PRESENTED='):
            pres[f.split('/')[-1]] = {int(x) for x in line.split('=',1)[1].split()}
# arms whose presented set is not in a split file
pres['ROT_CUT'] = set(json.loads([l for l in open(W+'/rot_cut.log') if '"training_start"' in l][0])['seats'])
allpres=set().union(*pres.values())
cut={s for s,r in desc.items() if (r.get('n_kind1') or 0)>0 and r.get('kappa') is not None}
inscope={s for s in cut if (desc[s].get('tau_mean') or 9)<=0.45}
print('labelled cut seats', len(cut), '| in design domain tau<=0.45', len(inscope))
print('presented somewhere:', len(allpres & cut), 'of the cut seats')
never = sorted(cut - allpres)
print('NEVER presented anywhere (any tau):', len(never))
print('  of those, in the design domain:', len([s for s in never if s in inscope]))
for s in never[:20]:
    r=desc[s]; print(f"   {s:>8} q={r['q']:>6} tau={r.get('tau_mean',0):.3f} kappa={r.get('kappa',0):.3g}")
N8={100032,100002,100024,224,100046,100000,100021,100079}
P28={int(x) for x in open(W+'/split_cutonly.txt').read().split('PRESENTED=')[1].split('\n')[0].split()}
h2 = sorted(s for s in inscope if s not in N8 and s not in P28)
print()
print('held out by BOTH H2 arms (n=8 set and CUTONLY_NOAUG), in domain:', len(h2))
h2s=sorted(h2, key=lambda s: desc[s]['q'])
step=max(1,len(h2s)//12); pick=[h2s[i] for i in range(0,len(h2s),step)][:12]
print(f"{'seat':>8} {'q':>7} {'kappa':>10} {'tau':>6} {'kind1':>6}")
for s in pick:
    r=desc[s]; print(f"{s:>8} {r['q']:>7} {r['kappa']:>10.3g} {r['tau_mean']:>6.3f} {r['n_kind1']:>6}")
print('H2_HELDOUT=' + ' '.join(map(str,pick)))
json.dump(dict(scope='pre-registered held-out set for the H2 comparison only',
               declared='2026-09-20',
               criterion='labelled cut seat, tau_mean<=0.45, presented by NEITHER H2 arm',
               caveat='these seats WERE presented in other arms (ROT_CUT / WIDE_CUT / NCUT_59), which does '
                      'not contaminate H2 because neither H2 arm saw them; there is no seat in the design '
                      'domain that is unpresented everywhere',
               seats=pick, n8=sorted(N8), cutonly_presented=sorted(P28)),
          open(W+'/HELDOUT_H2.json','w'), indent=1)
