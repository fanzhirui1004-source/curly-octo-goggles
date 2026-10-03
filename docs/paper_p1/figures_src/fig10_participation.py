"""Figure 10: share-weighted compliance error and local sensitivity for the five variants (Base network, Uncorrected,
Smoothing-trained, Base network, corrected, NICE). Writes F10_energy_share.*.

Source: evidence/gate_<run>_fresh_val_<case>_<config>.json (lat_full.py --sets test). Per load: compliance_rel_err,
sens_vec_rel_err[0] (target cell), bound (beta = sum_m w_m eps_m at the exact assembled trace) and energy_share[0]
(target energy share w). Face loads are the six loads of the joint criterion (gate flag), cut loads the three
cut-surface tractions. All seven cells of Table ST08 are included.
The observation and combination counts are printed and must match the caption."""
import json
from pathlib import Path
import numpy as np
from figstyle import MODEL, MUTED, GRID, TEXT, MM, panel, save, plt

EV = Path(__file__).resolve().parent.parent / 'evidence'
RUNS = [('B', 'v2L1'), ('C', 'A0_ctrl'), ('A2b', 'A2b_tail8'), ('B+W', 'B2grid'), ('A3', 'A3_2grid')]
CASES = [('2000_full', 'U1'), ('2001_full', 'U2'), ('2003_d1_v1', 'M1'), ('2005_d1_v0', 'H1'), ('2006_d0_v1', 'M2'),
         ('2002_d0_v0', 'H3'), ('2004_d0_v2', 'L1')]


def load(run):
    obs, combos = [], 0
    for key, lab in CASES:
        for cfg in 'xy':
            f = EV / f'gate_{run}_fresh_val_{key}_{cfg}.json'
            if not f.exists():
                continue
            r = json.loads(f.read_text())['results'][0]
            t = r['test']
            combos += 1
            for j, name in enumerate(r['loads']):
                obs.append(dict(cell=f'{lab}/{cfg}', load=name, face=bool(r['gate'][j]),
                                ce=100 * t['compliance_rel_err'][j], se=100 * t['sens_vec_rel_err'][0][j],
                                beta=100 * t['bound'][j], w=100 * r['energy_share'][0][j]))
    return obs, combos


def main():
    data = {m: load(run) for m, run in RUNS}
    tot_o = sum(len(o) for o, _ in data.values()); tot_c = sum(c for _, c in data.values())
    for m, (o, c) in data.items():
        print(f'{m}: {len(o)} observations, {c} combinations')
    print(f'total: {tot_o} observations, {tot_c} combinations')
    for m, (o, _) in data.items():
        assert all(np.isfinite([x['ce'], x['se'], x['beta'], x['w']]).all() for x in o), m

    fig = plt.figure(figsize=(178 * MM, 140 * MM))
    gs = fig.add_gridspec(2, 2, hspace=.5, wspace=.32)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    spec = [('beta', 'ce', r'$\beta=\sum_m w_m\varepsilon_m$ (%)', 'Compliance error (%)', 'a', 'Share-weighted local error'),
            ('ce', 'se', 'Compliance error (%)', 'Target sensitivity error (%)', 'b', 'Global and local response'),
            ('w', 'ce', 'Target energy share (%)', 'Compliance error (%)', 'c', 'Energy share and compliance'),
            ('w', 'se', 'Target energy share (%)', 'Target sensitivity error (%)', 'd', 'Energy share and sensitivity')]
    for ax, (kx, ky, lx, ly, letter, title) in zip(axes, spec):
        for m, (o, _) in data.items():
            col, mk, lab = MODEL[m]
            for face in (True, False):
                x = np.array([d[kx] for d in o if d['face'] == face]); y = np.array([d[ky] for d in o if d['face'] == face])
                kw = dict(ls='none', marker=mk, color=col, ms=3.2, alpha=.85)
                if not face:
                    kw.update(mfc='white', mew=.8)
                ax.plot(x, y, **kw)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(color=GRID, lw=.4); ax.minorticks_off()
        ax.set_xlabel(lx); ax.set_ylabel(ly); panel(ax, letter, title)
    for ax in axes[:2]:                                                 # equality lines in (a,b)
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0]); hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        xl, yl = ax.get_xlim(), ax.get_ylim()
        ax.plot([lo, hi], [lo, hi], '--', color=MUTED, lw=.7, zorder=0); ax.set_xlim(xl); ax.set_ylim(yl)
    b = [d for d in data['B'][0] if d['cell'] == 'U1/x' and d['load'] == 'nbr_facey0_z_cons'][0]
    axes[1].annotate('U1/x, base network, N-z', (b['ce'], b['se']), (b['ce'] * 3e-2, b['se'] * 2.5), fontsize=6.5, color=TEXT,
                     arrowprops=dict(arrowstyle='-', color=MUTED, lw=.6))
    h = [plt.Line2D([], [], ls='none', marker=MODEL[m][1], color=MODEL[m][0], ms=4, label=MODEL[m][2]) for m, _ in RUNS]
    fig.legend(handles=h, loc='upper center', ncol=len(h), frameon=False, bbox_to_anchor=(.5, 1.0), fontsize=6.5,
               handletextpad=.3, columnspacing=1.4)                     # variants only; marker fill convention below
    fig.text(.5, .005, 'Filled markers: face loads; open markers: cut-surface loads. Dashed lines: equality.',
             ha='center', fontsize=6.5, color=MUTED)
    save(fig, 'F10_energy_share')


if __name__ == '__main__':
    main()
