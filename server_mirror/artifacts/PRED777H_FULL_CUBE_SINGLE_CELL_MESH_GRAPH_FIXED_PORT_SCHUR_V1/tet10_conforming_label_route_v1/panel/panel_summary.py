import json, sys, os
D='/root/autodl-tmp/_claude_diag/step2s'
rows=[]
for c in ['G0','G1','G2','G3','G4','G5']:
    for sub,ref in (('gates','k2_del'),('gates_k2','k3_del')):
        p=f'{D}/{c}/{sub}/TET10_LABEL_GATES.json'
        if not os.path.exists(p): continue
        r=json.load(open(p)); g=r['gates'].get('TET10',{})
        pw=g.get('pairwise',{}); cv=g.get('convergence_vs_reference',{})
        allmax=max([v['lowfreq_frobenius'] for v in pw.values()] or [float('nan')])
        ng=[v['lowfreq_frobenius'] for k,v in pw.items() if 'nong' not in k]
        ngmax=max(ng) if ng else float('nan')
        cvn={k:v['lowfreq_frobenius'] for k,v in cv.items()}
        cv_ng=max([v for k,v in cvn.items() if 'nong' not in k] or [float('nan')]); cv_all=max(cvn.values()) if cvn else float('nan')
        rows.append((c,sub,r['q'],r['lowfreq_mode_count'],len(pw),allmax,ngmax,cv_all,cv_ng,ref))
print(f"{'case':5s} {'set':8s} {'q':>5s} {'modes':>5s} {'pairs':>5s} {'repro_all':>9s} {'repro_netgen':>12s} {'conv_all':>8s} {'conv_netgen':>11s} ref")
for c,sub,q,m,n,a,b,ca,cn,ref in rows:
    print(f"{c:5s} {sub:8s} {q:5d} {m:5d} {n:5d} {a:9.4f} {b:12.4f} {ca:8.4f} {cn:11.4f} {ref}")
