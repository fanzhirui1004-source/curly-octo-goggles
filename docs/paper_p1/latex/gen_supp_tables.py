"""Regenerate supplementary table blocks (R1 key, ST01, ST03, ST03b, ST03c, ST09, ST10, ST14a, ST14b) from evidence/.

Numbering and labels follow revision 1 (review_r1/RENUMBER_MAP.json; decision D6): old ST12 -> ST01, ST01 -> ST03,
ST13 -> ST09, ST06 -> ST10, ST07 -> ST14. Labels: B -> Base network, C -> Uncorrected, A2b -> Smoothing-trained,
B+W -> NICE-post, A3 -> NICE; S8 and P0 keep their names (supplement only).

Usage (from docs/paper_p1):  python3 latex/gen_supp_tables.py [--write]

Without --write the blocks are printed; with --write they replace the corresponding pipe tables
in SUPPLEMENTARY_EN.md. Sources:
  newval2_<run>.json   identity-view validation (per-geometry directional means)   -> ST03, ST03b
  newval_v2L1.json     views 0 and 17 of the base network on the earlier evaluation -> ST03c
  newval_c_oh.json     views 0 and 17 of P0 (five original classes only)           -> P0 columns of ST03, ST03b, ST03c
  valmeta.json         geometry strata                                              -> ST03b
  piml4_<mode>_<cell>.json  Bernstein-restricted retained space, x assemblies      -> ST14a, ST14b (file names are historical)
  meta_p1.json, meta_c_oh.json  training configuration and checkpoint selection    -> ST01
  gate_<run>_*.json    two-cell continuous-neighbour assemblies                     -> ST09 (face loads), ST10 (cut loads)
Training-pool counts (ST01): the SPLIT events of the training logs (server logs, read 2026-09-28): v2L1 305, A0_ctrl,
A2b_tail8 and A3_2grid 591; P0 148 from meta_c_oh.json. They are constants below (POOL) because the logs are not in
evidence/; meta_p1.json records the split file as regenerated later (591 for every arm) and is not used for this column.
gate_v2L1_*.json are the per-configuration splits of the archived gate_cont_v2L1_<cell>.json.
The H2/y assembly (fresh_val_2010_d0_v0, configuration y) is ill-posed (negative exact energy share, PCG at its cap)
and is excluded for every predictor.
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
LABEL = {'P0': 'P0', 'B': 'Base network', 'C': 'Uncorrected', 'S8': 'S8', 'A2b': 'Smoothing-trained', 'B+W': 'NICE-post',
         'A3': 'NICE'}
POOL = {'P0': 148, 'B': 305, 'C': 591, 'S8': 591, 'A2b': 591, 'B+W': 305, 'A3': 591}   # SPLIT events (see docstring)
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
    names = [LABEL[a] for a in ['P0'] + [a for a, _ in ARMS]]
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
    out = [row(['Stratum', 'Geometries', 'P0 force/support'] + [f'{LABEL[a]} force/support' for a, _ in ARMS]),
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
    out = [row(['Class', 'Geometries', 'P0', 'Base network']), row(['---'] * 4)]
    for cls in CLASSES:
        a, b = vals(v0, cls), vals(v17, cls)
        assert len(a) == len(b)
        pa, pb = vals(p0a, cls), vals(p0b, cls)
        p0 = f'{pa.mean():.3f} / {pb.mean():.3f}' if len(pa) else '—'
        out.append(row([cls, str(len(a)), p0, f'{a.mean():.3f} / {b.mean():.3f}']))
    return out


def st12(text):
    """Table ST01 (old ST12): training and evaluation settings."""
    runs = dict(load('meta_p1.json')['runs'], **load('meta_c_oh.json')['runs'])
    nv = {a: load(f'newval2_{r}.json') for a, r in ARMS}
    nv['P0'] = load('newval_c_oh.json')
    nv['B'] = load('newval_v2L1.json')                                      # the earlier evaluation carries both views
    fmt = lambda x: f'{x:,}'

    def new_row(arm, train_val, budget, corr, sel=None):
        d = nv[arm]
        step, weights = sel if sel else (d['ckpt_step'], d['ckpt_weights'])
        return row([LABEL[arm], fmt(POOL[arm]), train_val, budget, f"{fmt(step)} / {weights.upper()}", corr,
                    ', '.join(str(v) for v in d['views']), str(len(d['per_geo']))])

    cfg = {a: runs[r]['cfg'] for a, r in [('P0', 'c_oh'), ('B', 'v2L1'), ('C', 'A0_ctrl'), ('S8', 'A2_tail8'),
                                          ('A3', 'A3_2grid')]}
    p0 = runs['c_oh']
    assert p0['split_event']['train'] == POOL['P0']
    b_sel = (nv2 := load('newval2_v2L1.json'))['ckpt_step'], nv2['ckpt_weights']
    head = ['Arm', 'Training pool (geometries)', 'Training-time validation geometries', 'Run budget (updates)',
            'Evaluated update / weights', 'Evaluation correction', 'New-validation views', 'New-validation geometries']
    out = [row(head), row(['---'] * len(head)),
        new_row('P0', str(p0['split_event']['val']), fmt(cfg['P0']['steps']), 'None', (p0['step'], p0['weights'])),
        new_row('B', str(cfg['B']['val_max']), fmt(cfg['B']['steps']), 'None', b_sel),
        new_row('C', str(cfg['C']['val_max']), fmt(cfg['C']['steps']), 'None'),
        new_row('S8', str(cfg['S8']['val_max']), fmt(cfg['S8']['steps']), 'Eight-step smoothing'),
        # A2b has no configuration record in meta_p1.json; its run configuration (prod/chain_a3r.sh cfg()) sets
        # val_max 40 and 15,000 steps as for the other continuations; step and weights come from newval2_A2b_tail8.json.
        new_row('A2b', '40', '15,000', 'Eight-step smoothing'),
        new_row('B+W', str(cfg['B']['val_max']), '—', '8 / Q1(17) / 8', b_sel),
        new_row('A3', str(cfg['A3']['val_max']), fmt(cfg['A3']['steps']), '8 / Q1(17) / 8'),
    ]
    return out


def gate(run, key, cfg):
    f = EV / f'gate_{run}_fresh_val_{key}_{cfg}.json'
    if not f.exists() or (key, cfg) in EXCLUDED:
        return None
    return json.load(open(f))['results'][0]['test']


def num(x):
    x *= 100
    return f'{x:.3f}' if x >= 0.01 else f'{x:.3g}'


def st13(text):
    old = table_rows(text, r'## Table ST09\.')
    out = [row(old[0]), row(old[1])]
    for arm, run in ARMS:
        for key, lab in CELLS:
            for cfg in 'xy':
                t = gate(run, key, cfg)
                if t is None:
                    continue
                out.append(row([lab, cfg, LABEL[arm], num(t['gate_compliance_max']), num(t['gate_sens_max']),
                                str(t['pcg_iterations']), 'Pass' if t['gate_pass'] else 'Above criterion']))
    return out


def st06(text):
    """Table ST10 (old ST06): cut-traction maxima for every recorded arm/configuration with a cut target."""
    head = ['Arm', 'Cell', 'Configuration', 'Compliance error (%)', 'Sensitivity error (%)']
    out = [row(head), row(['---'] * len(head))]
    cells = [(k, l) for k, l in CELLS if k in ('2003_d1_v1', '2005_d1_v0', '2006_d0_v1', '2002_d0_v0', '2004_d0_v2')]
    order = ['M1', 'H1', 'M2', 'H3', 'L1']
    cells.sort(key=lambda kl: order.index(kl[1]))
    for arm, run in ARMS:
        for key, lab in cells:
            for cfg in 'xy':
                t = gate(run, key, cfg)
                if t is None or t.get('cut_compliance_max') is None:
                    continue
                c, s_ = num(t['cut_compliance_max']), num(t['cut_sens_max'])
                if 100 * t['cut_compliance_max'] > 3:
                    c = f'**{c}**'
                if 100 * t['cut_sens_max'] > 3:
                    s_ = f'**{s_}**'
                out.append(row([LABEL[arm], lab, cfg, c, s_]))
    return out


def r1(text):
    head = ['Label', 'Former label', 'Numerical role', 'Archived run identifier']
    rows = [
        ['P0 (supplement only)', 'P0', 'Earlier uncorrected predictor; separate training lineage (148 legacy geometries)', 'c_oh'],
        ['Base network', 'B', 'Baseline predictor; starting weights of every continuation and of the fixed-weight corrections', 'v2L1'],
        ['Uncorrected', 'C', 'Continued weights, no correction in training or evaluation', 'A0_ctrl'],
        ['S8 (supplement only)', 'S8', 'Continued weights with eight smoothing steps in training on identity-view samples only (rotated training views bypassed the smoothing); evaluated with eight steps', 'A2_tail8'],
        ['Smoothing-trained', 'A2b', 'Continued weights trained through eight smoothing steps (all training views)', 'A2b_tail8'],
        ['NICE-post', 'B+W', "Base network's weights evaluated with NICE's correction; no training through it", 'B2grid'],
        ['NICE', 'A3', 'Principal predictor, trained through the complete correction (8 / Q1(17) / 8)', 'A3_2grid'],
    ]
    return [row(head), row(['---'] * 4)] + [row(r) for r in rows]


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
    return piml('all', r'#### ST14a\.', text)


def st07b(text):
    return piml('interface', r'#### ST14b\.', text)


BLOCKS = [(r'## R1\.', r1), (r'## Table ST01\.', st12), (r'## Table ST03\.', st01), (r'### ST03b\.', st01b),
          (r'### ST03c\.', st01c), (r'## Table ST09\.', st13), (r'## Table ST10\.', st06),
          (r'#### ST14a\.', st07a), (r'#### ST14b\.', st07b)]

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
