"""Figure 5: directional energy error across the 80 validation geometries (identity view). Source: evidence/newval2_<run>.json
(eval_views.py --data S2/data_v2, all classes) and evidence/valmeta.json (cut stratum, weight-selection membership)."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from figstyle import MODEL, C, MUTED, GRID, MM, panel, save

EV = Path(__file__).resolve().parent.parent / 'evidence'
# S8 is reported in the supplement only (Fig. S02, Table ST03); the main-text figure shows the five labelled predictors.
RUNS = [('B', 'v2L1') + MODEL['B'], ('C', 'A0_ctrl') + MODEL['C'], ('A2b', 'A2b_tail8') + MODEL['A2b'],
        ('B+W', 'B2grid') + MODEL['B+W'], ('A3', 'A3_2grid') + MODEL['A3']]
STRATA = ['FULL', 'light', 'moderate', 'heavy']


def main():
    meta = json.loads((EV / 'valmeta.json').read_text())
    strat = lambda c: 'FULL' if meta[c]['kind'] == 'FULL' else ('light' if meta[c]['vol'] > 2 / 3 else ('moderate' if meta[c]['vol'] > 1 / 3 else 'heavy'))
    data = {}
    for key, run, col, mk, lab in RUNS:
        f = EV / f'newval2_{run}.json'
        if f.exists():
            data[key] = (json.loads(f.read_text())['per_geo'], col, mk, lab)
    fig = plt.figure(figsize=(178 * MM, 128 * MM))
    gs = fig.add_gridspec(2, 3, hspace=.55, wspace=.38)
    ax = fig.add_subplot(gs[0, :2])
    keys = list(data)
    for k, key in enumerate(keys):
        d, col, mk, lab = data[key]
        for j, st in enumerate(STRATA):
            v = np.array([g['0']['force_c'] for c, g in d.items() if c in meta and strat(c) == st and 'force_c' in g['0']]) * 100
            x = j + (k - (len(keys) - 1) / 2) * .13
            ax.plot(x, v.mean(), mk, color=col, ms=5, label=lab if j == 0 else None)
            ax.plot([x, x], [np.quantile(v, .1), v.max()], color=col, lw=.8, alpha=.7)
    ax.set_yscale('log'); ax.set_xticks(range(4), ['uncut', 'lightly cut', 'moderately cut', 'heavily cut'])
    ax.set_ylabel('Energy excess (%)'); ax.grid(axis='y', color=GRID, lw=.4)
    ax.legend(frameon=False, ncol=3, fontsize=6.5, loc='upper left', bbox_to_anchor=(0, -0.12))
    panel(ax, 'a', 'Consistent tractions, by cut stratum')
    ax = fig.add_subplot(gs[0, 2])
    if 'B' in data and 'A3' in data:
        b, a = data['B'][0], data['A3'][0]
        cs = [c for c in b if c in a and c in meta and 'force_c' in b[c]['0']]
        xb = np.array([b[c]['0']['force_c'] for c in cs]) * 100; ya = np.array([a[c]['0']['force_c'] for c in cs]) * 100
        full = np.array([meta[c]['kind'] == 'FULL' for c in cs])
        ax.plot(xb[full], ya[full], 'o', mfc='white', color=C['corrected'], ms=3.5, label='uncut')
        ax.plot(xb[~full], ya[~full], 's', color=C['corrected'], ms=3.5, label='cut')
        ax.set_xscale('log'); ax.set_yscale('log')
        lo, hi = 1e-3, 1e2
        ax.plot([lo, hi], [lo, hi], ':', color=MUTED, lw=.6); ax.set_xlim(.1, 100); ax.set_ylim(.005, 2)
        ax.set_xlabel('Base network (%)'); ax.set_ylabel('NICE (%)'); ax.legend(frameon=False, fontsize=6.5, loc='upper left')
        ax.grid(color=GRID, lw=.4)
    panel(ax, 'b', 'Per geometry: base network vs NICE')
    classes = [('glued', 'neighbour-induced'), ('support_k', 'spring-supported'), ('face_c', 'single-face traction'), ('force', 'nodal forces*')]
    ax = fig.add_subplot(gs[1, :])
    for k, key in enumerate(keys):
        d, col, mk, lab = data[key]
        for j, (cls, _) in enumerate(classes):
            v = np.array([g['0'][cls] for c, g in d.items() if c in meta and cls in g['0'] and np.isfinite(g['0'][cls])]) * 100
            x = j + (k - (len(keys) - 1) / 2) * .13
            ax.plot(x, v.mean(), mk, color=col, ms=5)
            ax.plot([x, x], [np.quantile(v, .1), v.max()], color=col, lw=.8, alpha=.7)
    ax.set_yscale('log'); ax.set_xticks(range(len(classes)), [l for _, l in classes]); ax.set_ylabel('Energy excess (%)')
    ax.grid(axis='y', color=GRID, lw=.4)
    panel(ax, 'c', 'Other loading classes, all geometries')
    save(fig, 'F02_validation_A3')


if __name__ == '__main__':
    main()
