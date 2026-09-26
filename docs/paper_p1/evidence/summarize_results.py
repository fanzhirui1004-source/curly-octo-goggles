import json, glob, numpy as np
meta = json.load(open('valmeta.json'))
def strat(c):
    m = meta[c]
    if m['kind'] == 'FULL': return 'FULL'
    v = m['vol']; return 'light' if v > 2/3 else ('moderate' if v > 1/3 else 'heavy')
out = []
P = lambda x: f'{100*x:.3g}'
# 1. population
models = [('B (v2L1)', 'newval_v2L1.json'), ('C (A0_ctrl)', 'newval_A0_ctrl.json'), ('S8 (A2_tail8)', 'newval_A2_tail8.json'), ('A3', 'newval_A3_2grid.json')]
for subset, keep in (('all 80', lambda c: True), ('60 non-selection', lambda c: not meta[c]['sel'])):
    out.append(f'\n### Population, directional energy excess (%), mean / max over geometries — {subset}\n')
    for cls in ('force_c', 'glued', 'support_k', 'face_c', 'force', 'support'):
        out.append(f'\n**{cls}**\n\n| model | FULL | light | moderate | heavy | all |\n|---|---|---|---|---|---|')
        for name, f in models:
            d = json.load(open(f))['per_geo']; row = [name]
            for st in ('FULL', 'light', 'moderate', 'heavy', None):
                v = [g['0'][cls] for c, g in d.items() if c in meta and keep(c) and (st is None or strat(c) == st) and cls in g['0'] and np.isfinite(g['0'][cls])]
                row.append(f'{P(np.mean(v))} / {P(np.max(v))} (n={len(v)})' if v else '—')
            out.append('| ' + ' | '.join(row) + ' |')
# 2. gates
lab = {'2000_full': 'U1', '2001_full': 'U2', '2003_d1_v1': 'M1', '2005_d1_v0': 'H1', '2006_d0_v1': 'M2', '2010_d0_v0': 'H2', '2002_d0_v0': '2002', '2004_d0_v2': '2004'}
out.append('\n### Two-cell continuous-neighbour gates: max compliance / max sensitivity error (%) over the six face loads\n\n| cell/config | C | S8 | A3 |\n|---|---|---|---|')
def g(f):
    try:
        x = json.load(open(f))['results'][0]['test']
        return f"{P(x['gate_compliance_max'])} / {P(x['gate_sens_max'])} {'PASS' if x['gate_pass'] else 'fail'}"
    except Exception: return '—'
for k, l in lab.items():
    for conf in 'xy':
        out.append(f"| {l}/{conf} | {g(f'gate_A0_ctrl_fresh_val_{k}_{conf}.json')} | {g(f'gate_A2_tail8_fresh_val_{k}_{conf}.json')} | {g(f'gate_A3_2grid_fresh_val_{k}_{conf}.json')} |")
# 3. local checks
pc = json.load(open('p1_checks_cpu.json'))['per_case']
out.append('\n### Local checks (fixed retained displacement, 32 val directions): mean energy excess (%)\n\n| cell | class | B | A3 | B+8/Q1/8 | harmonic+8/Q1/8 | zero+8/Q1/8 | B+32/Q1/32 | harmonic+32 | zero+32 |\n|---|---|---|---|---|---|---|---|---|---|')
inv = {('fresh_val_' + k): v for k, v in lab.items()}
for c, r in pc.items():
    for cls in ('force_c', 'force'):
        if cls not in r: continue
        e = r[cls]['eps']; m = lambda k: P(e[k]['mean']) if k in e else '—'
        out.append(f"| {inv.get(c,c)} | {cls} | {m('B')} | {m('A3')} | {m('B+tail8+Q1_17+tail8')} | {m('harmonic+tail8+Q1_17+tail8')} | {m('zero+tail8+Q1_17+tail8')} | {m('B+tail32+Q1_17+tail32')} | {m('harmonic+tail32+Q1_17+tail32')} | {m('zero+tail32+Q1_17+tail32')} |")
out.append('\n| cell | lambda_max (Lanczos) | b = 1.05 x power | margin | Gershgorin | sym | action-energy | fastnet-trainlib | rigid energy | ghost share force_c / force | delta,kappa B (force_c) | delta,kappa A3 |\n|---|---|---|---|---|---|---|---|---|---|---|---|')
for c, r in pc.items():
    L = r['lam']; o = r['force_c']['ops']; dk = r['force_c']['delta_kappa']
    gs = lambda cls: f"{r[cls]['ghost']['ghost_share']['mean']:.2g}" if cls in r else '—'
    out.append(f"| {inv.get(c,c)} | {L['lanczos_lmax']:.4g} | {L['b']:.4g} | {100*L['margin']:.1f}% | {L['gershgorin']:.3g} | {o['sym_rel_max']:.1e} | {o['action_vs_field_energy_rel_max']:.1e} | {o['fastnet_vs_trainlib_field_rel']:.1e} | {o['rigid_energy_ratio_max']:.1e} | {gs('force_c')} / {gs('force')} | {100*dk['B']['delta']['mean']:.2f}%, {dk['B']['kappa']['mean']:.0f} | {100*dk['A3']['delta']['mean']:.3f}%, {dk['A3']['kappa']['mean']:.0f} |")
# 4. reference validation
out.append('\n### Reference discretisation (clamp / load faces with material; compliance per load direction)\n')
for f in ('ref_valid.json', 'ref_valid_h1.json'):
    for c, r in json.load(open(f))['per_case'].items():
        n = r.get('n', {})
        ok = sorted(int(k) for k, v in n.items() if 'compliance' in v and np.all(np.isfinite(v['compliance'])))
        if not ok: continue
        out.append(f"\n**{inv.get(c,c)}** faces {r.get('faces',{}).get('clamp','z0')}→{r.get('faces',{}).get('load','z1')}; " +
                   '; '.join(f"n={k}: C=" + ', '.join(f'{x:.5g}' for x in n[str(k)]['compliance']) + f" (dofs {n[str(k)]['dofs']}, ghost {max(n[str(k)]['ghost_share']):.1e})" for k in ok))
        if len(ok) > 1:
            fin = n[str(ok[-1])]['compliance']
            out.append('  rel. change vs n=%d: ' % ok[-1] + '; '.join(f"n={k}: " + ', '.join(f'{100*abs(a/b-1):.3g}%' for a, b in zip(n[str(k)]['compliance'], fin)) for k in ok[:-1]))
        for key in ('fd', 'gamma', 'integ'):
            if key in r: out.append(f'  {key}: ' + json.dumps({k: v for k, v in r[key].items() if k in ('rel_to_h1e_5', 'direct_vs_sens')} if key == 'fd' else {k: [round(100*x, 4) for x in v['compliance_rel']] for k, v in r[key].items()}))
# 5. PIML
out.append('\n### PIML-style boundary restriction (exact interiors), config x: gate compliance / gate sensitivity / test-face sensitivity (%)\n')
for f in sorted(glob.glob('piml4_*.json')) + ['piml_gate_all.json']:
    try: d = json.load(open(f))
    except Exception: continue
    for r in d['results']:
        out.append(f"- {f.split('_')[1] if f.startswith('piml4') else 'all'} {inv.get(r['case'], r['case'])}: " + '; '.join(f"p={p}: {100*o['gate_compliance_max']:.3g}/{100*o['gate_sens_max']:.3g}/{100*o['test_face_sens_max']:.3g}" for p, o in r['orders'].items()))
open('RESULTS_TABLES.md', 'w').write('\n'.join(out))
print('\n'.join(out))
