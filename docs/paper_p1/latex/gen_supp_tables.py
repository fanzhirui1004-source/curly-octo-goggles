"""Regenerate supplementary table blocks (R1 key, ST01, ST01b, ST01c, ST07a, ST07b, ST12, ST13) from evidence/.

Usage (from docs/paper_p1):  python3 latex/gen_supp_tables.py [--write]

Without --write the blocks are printed; with --write they replace the corresponding pipe tables
in SUPPLEMENTARY_EN.md. Sources:
  newval2_<run>.json   identity-view validation (per-geometry directional means)   -> ST01, ST01b
  newval_v2L1.json     views 0 and 17 of B on the earlier evaluation                -> ST01c
  newval_c_oh.json     views 0 and 17 of P0 (five original classes only)           -> P0 columns of ST01, ST01b, ST01c
  valmeta.json         geometry strata                                              -> ST01b
  piml4_<mode>_<cell>.json  Bernstein-restricted retained space, x assemblies      -> ST07a, ST07b
  meta_p1.json, meta_c_oh.json  training configuration and checkpoint selection    -> ST12
  gate_<run>_*.json    two-cell continuous-neighbour assemblies                     -> ST13
  MANUSCRIPT_EN.md     Table 2 (training-geometry counts)                           -> ST12
Every entry now comes from an evidence file (meta_c_oh.json holds P0's archived configuration). gate_v2L1_*.json are the per-configuration splits of the archived gate_cont_v2L1_<cell>.json.
The H2/y assembly (fresh_val_2010_d0_v0, configuration y) is not part of the reported comparison.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent
EV = SRC / 'evidence'
SUPP = SRC / 'SUPPLEMENTARY_EN.md'

ARMS = [('B', 'v2L1'), ('C', 'A0_ctrl'), ('S8', 'A2_tail8'), ('A2b', 'A2b_tail8'), ('B+W', 'B2grid'), ('A3', 'A3_2grid')]
CLASSES = ['force', 'support', 'face', 'macro', 'grf', 'force_c', 'face_c', 'support_k', 'glued']
CELLS = [('2000_full', 'U1'), ('2001_full', 'U2'), ('2003_d1_v1', 'M1'), ('2005_d1_v0', 'H1'),
         ('2006_d0_v1', 'M2'), ('2002_d0_v0', 'H3'), ('2004_d0_v2', 'L1')]
EXCLUDED = {('2010_d0_v0', 'y')}


def load(name):
    return json.load(open(EV / name))


def p0_geo(view='0'):
    return {c: g[view] for c, g in load('newval_c_oh.json')['per_geo'].items() if view in g}


def stat(v):
    return f'{v.mean():.3f} / {np.percentile(v, 90):.3f} / {v.max():.3f}' if len(v) else '—'


def per_geo(run, view='0'):
    return {c: g[view] for c, g in load(f'newval2_{run}.json')['per_geo'].items() if view in g}


def vals(pg, cls, keep=lambda c: True):
    v = [g[cls] for c, g in pg.items() if keep(c) and cls in g and np.isfinite(g[cls])]
    return 100 * np.array(v)


def row(cells):
    return '| ' + ' | '.join(cells) + ' |'


def column(rows, name):
    """Map first cell -> cell of the named header column."""
    k = rows[0].index(name)
    return {r[0]: r[k] for r in rows[2:]}


def table_rows(text, heading):
    """Rows (lists of cell strings) of the first pipe table after the heading line matching `heading`."""
    lines = text.split('\n')
    i = next(k for k, l in enumerate(lines) if re.match(heading, l))
    while not lines[i].startswith('|'):
        i += 1
    out = []
    while i < len(lines) and lines[i].startswith('|'):
        out.append([c.strip() for c in lines[i].strip().strip('|').split('|')])
        i += 1
    return out


def replace_table(text, heading, new_lines):
    lines = text.split('\n')
    i = next(k for k, l in enumerate(lines) if re.match(heading, l))
    while not lines[i].startswith('|'):
        i += 1
    j = i
    while j < len(lines) and lines[j].startswith('|'):
        j += 1
    return '\n'.join(lines[:i] + new_lines + lines[j:])


def st01(text):
    pgs = {a: per_geo(r) for a, r in ARMS}
    p0 = p0_geo()
    names = ['P0'] + [a for a, _ in ARMS]
    out = [row(['Class', 'Geometries per evaluated arm'] + names), row(['---'] * (len(names) + 2))]
    for cls in CLASSES:
        n = {len(vals(pg, cls)) for pg in pgs.values()}
        assert len(n) == 1, (cls, n)
        cells = [cls, str(n.pop()), stat(vals(p0, cls))]
        for a, _ in ARMS:
            cells.append(stat(vals(pgs[a], cls)))
        out.append(row(cells))
    return out


def stratum(c, meta):
    m = meta[c]
    if m['kind'] == 'FULL':
        return 'FULL'
    return 'Light cut (v2)' if m['vol'] > 2 / 3 else ('Middle cut (v1)' if m['vol'] > 1 / 3 else 'Heavy cut (v0)')


def st01b(text):
    meta = load('valmeta.json')
    pgs = {a: per_geo(r) for a, r in ARMS}
    p0 = p0_geo()
    out = [row(['Stratum', 'Geometries', 'P0 force/support'] + [f'{a} force/support' for a, _ in ARMS]),
           row(['---'] * (len(ARMS) + 3))]
    for s in ('FULL', 'Light cut (v2)', 'Middle cut (v1)', 'Heavy cut (v0)'):
        keep = lambda c, s=s: c in meta and stratum(c, meta) == s
        n = len(vals(pgs['B'], 'force', keep))
        cells = [s, str(n), f"{vals(p0, 'force', keep).mean():.3f} / {vals(p0, 'support', keep).mean():.3f}"]
        for a, _ in ARMS:
            cells.append(f"{vals(pgs[a], 'force', keep).mean():.3f} / {vals(pgs[a], 'support', keep).mean():.3f}")
        out.append(row(cells))
    return out


def st01c(text):
    pg = load('newval_v2L1.json')['per_geo']
    p0a, p0b = p0_geo('0'), p0_geo('17')
    v0 = {c: g['0'] for c, g in pg.items()}
    v17 = {c: g['17'] for c, g in pg.items() if '17' in g}
    out = [row(['Class', 'Geometries', 'P0', 'B']), row(['---'] * 4)]
    for cls in CLASSES:
        a, b = vals(v0, cls), vals(v17, cls)
        assert len(a) == len(b)
        pa, pb = vals(p0a, cls), vals(p0b, cls)
        p0 = f'{pa.mean():.3f} / {pb.mean():.3f}' if len(pa) else '—'
        out.append(row([cls, str(len(a)), p0, f'{a.mean():.3f} / {b.mean():.3f}']))
    return out


def table2_geometries():
    rows = table_rows((SRC / 'MANUSCRIPT_EN.md').read_text(), r'\*\*Table 2\.')[2:]
    return {r[0].strip('*'): r[2] for r in rows}


def st12(text):
    old = table_rows(text, r'## Table ST12\.')
    runs = dict(load('meta_p1.json')['runs'], **load('meta_c_oh.json')['runs'])
    geo = table2_geometries()
    nv = {a: load(f'newval2_{r}.json') for a, r in ARMS}
    nv['P0'] = load('newval_c_oh.json')
    nv['B'] = load('newval_v2L1.json')                                      # the earlier evaluation carries both views
    fmt = lambda x: f'{x:,}'

    def new_row(arm, train_val, budget, corr, sel=None):
        d = nv[arm]
        step, weights = sel if sel else (d['ckpt_step'], d['ckpt_weights'])
        return row([arm, geo[arm], train_val, budget, f"{fmt(step)} / {weights.upper()}", corr,
                    ', '.join(str(v) for v in d['views']), str(len(d['per_geo']))])

    cfg = {a: runs[r]['cfg'] for a, r in [('P0', 'c_oh'), ('B', 'v2L1'), ('C', 'A0_ctrl'), ('S8', 'A2_tail8'),
                                          ('A3', 'A3_2grid')]}
    p0 = runs['c_oh']
    b_sel = (nv2 := load('newval2_v2L1.json'))['ckpt_step'], nv2['ckpt_weights']
    out = [
        new_row('P0', str(p0['split_event']['val']), fmt(cfg['P0']['steps']), 'None', (p0['step'], p0['weights'])),
        new_row('B', str(cfg['B']['val_max']), fmt(cfg['B']['steps']), 'None', b_sel),
        new_row('C', str(cfg['C']['val_max']), fmt(cfg['C']['steps']), 'None'),
        new_row('S8', str(cfg['S8']['val_max']), fmt(cfg['S8']['steps']), 'Eight-step smoothing'),
        # A2b has no configuration record in meta_p1.json: the 40 training-time validation geometries
        # (Section 6.1) and the 15,000-step budget (Section 6.1, after Table 2) are stated in the main text
        # for all continued predictors; step and weights come from newval2_A2b_tail8.json.
        new_row('A2b', '40', '15,000', 'Eight-step smoothing'),
        new_row('B+W', str(cfg['B']['val_max']), '—', '8 / Q1(17) / 8', b_sel),
        new_row('A3', str(cfg['A3']['val_max']), fmt(cfg['A3']['steps']), '8 / Q1(17) / 8'),
    ]
    return [row(old[0]), row(old[1])] + out


def gate(run, key, cfg):
    f = EV / f'gate_{run}_fresh_val_{key}_{cfg}.json'
    if not f.exists() or (key, cfg) in EXCLUDED:
        return None
    return json.load(open(f))['results'][0]['test']


def num(x):
    x *= 100
    return f'{x:.3f}' if x >= 0.01 else f'{x:.3g}'


def st13(text):
    old = table_rows(text, r'## Table ST13\.')
    out = [row(old[0]), row(old[1])]
    for arm, run in ARMS:
        for key, lab in CELLS:
            for cfg in 'xy':
                t = gate(run, key, cfg)
                if t is None:
                    continue
                out.append(row([lab, cfg, arm, num(t['gate_compliance_max']), num(t['gate_sens_max']),
                                str(t['pcg_iterations']), 'Pass' if t['gate_pass'] else 'Above criterion']))
    return out


def r1(text):
    old = table_rows(text, r'## R1\.')
    extra = [['A2b', 'Continued weights trained through eight smoothing steps', 'A2b_tail8'],
             ['B+W', "B's weights evaluated with A3's correction", 'B2grid'],
             ['A3', 'Principal predictor, trained through the complete correction', 'A3_2grid']]
    keep = [r for r in old[2:] if r[0] not in ('A2b', 'B+W', 'A3')]
    return [row(r) for r in old[:2] + keep + extra]


def piml(mode, heading, text):
    old = table_rows(text, heading)
    out = [row(old[0]), row(old[1])]
    for key, lab in [('2000_full', 'U1'), ('2003_d1_v1', 'M1'), ('2006_d0_v1', 'M2'), ('2005_d1_v0', 'H1')]:
        f = EV / f'piml4_{mode}_fresh_val_{key}.json'
        if not f.exists():
            continue
        r = json.load(open(f))['results'][0]
        loads = r['loads']
        sets = [[i for i, n in enumerate(loads) if n.startswith('test_face')],
                [i for i, g in enumerate(r['gate']) if g], list(range(len(loads)))]
        for o in sorted(r['orders'], key=int):
            v = r['orders'][o]
            cells = [lab, o, f"{v['ctrl_dofs']:,}"]
            for k, s in enumerate(sets):
                if k == 2 and len(s) == len(sets[1]):
                    cells += ['—', '—']          # no macro-cut loads (uncut cell)
                    continue
                cells += [f"{100 * max(v['compliance_rel_err'][i] for i in s):.3f}",
                          f"{100 * max(v['sens_vec_rel_err_test'][i] for i in s):.3f}"]
            out.append(row(cells))
    return out


def st07a(text):
    return piml('all', r'### ST07a\.', text)


def st07b(text):
    return piml('interface', r'### ST07b\.', text)


BLOCKS = [(r'## R1\.', r1), (r'## Table ST01\.', st01), (r'### ST01b\.', st01b), (r'### ST01c\.', st01c),
          (r'### ST07a\.', st07a), (r'### ST07b\.', st07b),
          (r'## Table ST12\.', st12), (r'## Table ST13\.', st13)]

if __name__ == '__main__':
    text = SUPP.read_text()
    for heading, fn in BLOCKS:
        new = fn(text)
        if '--write' in sys.argv:
            text = replace_table(text, heading, new)
        else:
            print(heading, *new, '', sep='\n')
    if '--write' in sys.argv:
        SUPP.write_text(text)
        print('updated', SUPP.name)
