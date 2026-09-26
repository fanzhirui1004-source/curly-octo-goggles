"""Figure 8: compliance and eight-corner sensitivity in assembled two-cell configurations (learned target, exact
neighbour). Source: evidence/gate_<run>_fresh_val_<case>_<config>.json (lat_full.py, --sets test)."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from figstyle import MODEL, MUTED, GRID, MM, panel, save

EV = Path(__file__).resolve().parent.parent / 'evidence'
RUNS = {'C': 'A0_ctrl', 'S8': 'A2_tail8', 'B+W': 'B2grid', 'A2b': 'A2b_tail8', 'A3': 'A3_2grid'}
CASES = [('2000_full', 'U1'), ('2001_full', 'U2'), ('2003_d1_v1', 'M1'), ('2006_d0_v1', 'M2'), ('2005_d1_v0', 'H1'),
         ('2002_d0_v0', 'C2'), ('2004_d0_v2', 'C4')]
LOADS = ['T/x', 'T/y', 'T/z', 'N/x', 'N/y', 'N/z']


def rec(run, case, conf):
    f = EV / f'gate_{run}_fresh_val_{case}_{conf}.json'
    if not f.exists():
        return None
    x = json.loads(f.read_text())['results'][0]
    return x['test'], x['loads']


def main():
    rows = [(c, l, conf) for c, l in CASES for conf in 'xy' if any(rec(r, c, conf) for r in RUNS.values())]
    models = [m for m in RUNS if any(rec(RUNS[m], c, conf) for c, _, conf in rows)]
    fig = plt.figure(figsize=(178 * MM, 150 * MM))
    gs = fig.add_gridspec(2, 2, hspace=.45, wspace=.35, height_ratios=[1.25, 1])
    off = {m: (k - (len(models) - 1) / 2) * .16 for k, m in enumerate(models)}
    for j, (key, title) in enumerate((('gate_compliance_max', 'Compliance'), ('gate_sens_max', 'Sensitivity, both cells'))):
        ax = fig.add_subplot(gs[0, j])
        for m in models:
            col, mk, lab = MODEL[m]
            for i, (c, l, conf) in enumerate(rows):
                r = rec(RUNS[m], c, conf)
                if r is None:
                    continue
                ax.plot(100 * r[0][key], i + off[m], mk, color=col, ms=4, label=lab if i == 0 else None)
        ax.set_xscale('log'); ax.axvline(3, ls='--', lw=.75, color=MUTED)
        ax.set_yticks(range(len(rows)), [f'{l}/{conf}' for _, l, conf in rows]); ax.set_ylim(len(rows) - .5, -.6)
        ax.set_xlabel('Maximum error over six face loads (%)'); ax.grid(axis='x', color=GRID, lw=.4)
        panel(ax, 'ab'[j], title)
    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=len(models), frameon=False, bbox_to_anchor=(.5, .99))
    for j, (case, lab) in enumerate((('2003_d1_v1', 'M1/x'), ('2000_full', 'U1/x'))):
        ax = fig.add_subplot(gs[1, j])
        for m in models:
            r = rec(RUNS[m], case, 'x')
            if r is None:
                continue
            col, mk, _ = MODEL[m]
            ce = 100 * np.asarray(r[0]['compliance_rel_err'])[:6]; se = 100 * np.asarray(r[0]['sens_vec_rel_err'])[0][:6]
            ax.plot(ce[:3], se[:3], mk, color=col, ms=4.5)                                  # target-face loads: filled
            ax.plot(ce[3:], se[3:], mk, color=col, ms=4.5, mfc='white', mew=.9)             # neighbour-face loads: open
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1e-5, 30); ax.set_ylim(1e-2, 30)
        ax.axhline(3, ls='--', lw=.75, color=MUTED); ax.axvline(3, ls='--', lw=.75, color=MUTED)
        ax.set_xlabel('Compliance error (%)'); ax.set_ylabel('Target-cell sensitivity error (%)')
        ax.grid(color=GRID, lw=.4); panel(ax, 'cd'[j], f'{lab}: individual face loads')
    fig.text(.5, .005, 'Filled: target-face loads; open: neighbour-face loads. Dashed lines: 3%.', ha='center', fontsize=6.5, color=MUTED)
    save(fig, 'F05_assembly_A3')


if __name__ == '__main__':
    main()
