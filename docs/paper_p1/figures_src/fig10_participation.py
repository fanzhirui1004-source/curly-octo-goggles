"""Figure 11 (file F10_energy_share.*) and Supplementary Figure S04 (file S07_energy_share_variants.*): share-weighted
compliance error and target-cell sensitivity error in the two-cell configurations of Section 5.6.

Figure 11 shows NICE alone. Figure S04 shows the other four variants (Base network, Uncorrected, Smoothing-trained,
Base network, corrected), one column each, so that variants whose errors differ by orders of magnitude never share
a plotting area. Both figures have the same three relations: (a) compliance error against beta = sum_m w_m eps_m,
(b) compliance error and (c) target sensitivity error against the target's exact energy share w.

Source: evidence/gate_<run>_fresh_val_<case>_<config>.json (lat_full.py --sets test). Per load: compliance_rel_err,
sens_vec_rel_err[0] (target cell), bound (beta at the exact assembled trace) and energy_share[0] (target energy
share w). Face loads are the six loads of the joint criterion (gate flag), cut loads the three cut-surface tractions.
All seven cells of Table ST08 are included. The counts, ratio ranges and fit slopes quoted in the two captions are
printed and must match them."""
import json
from pathlib import Path
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
from figstyle import MODEL, MUTED, GRID, TEXT, MM, panel, save, plt

EV = Path(__file__).resolve().parent.parent / 'evidence'
RUNS = [('B', 'v2L1'), ('C', 'A0_ctrl'), ('A2b', 'A2b_tail8'), ('B+W', 'B2grid'), ('A3', 'A3_2grid')]
CASES = [('2000_full', 'U1'), ('2001_full', 'U2'), ('2003_d1_v1', 'M1'), ('2005_d1_v0', 'H1'), ('2006_d0_v1', 'M2'),
         ('2002_d0_v0', 'H3'), ('2004_d0_v2', 'L1')]
EXAMPLE = ('U1/x', 'nbr_facey0_z_cons')        # neighbour-face z traction on U1/x (Table 5; Section 5.6 text)
OTHERS = ['B', 'C', 'A2b', 'B+W']              # columns of Figure S04, largest error first


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


def stats(m, o, c):
    """Print the numbers quoted in the captions; return the log-log fit (slope, intercept) of ce and se against w."""
    A = lambda k: np.array([d[k] for d in o])
    ce, se, beta, w = A('ce'), A('se'), A('beta'), A('w')
    assert np.isfinite(np.c_[ce, se, beta, w]).all() and min(ce.min(), beta.min(), w.min()) > 0, m
    nf = sum(d['face'] for d in o)
    q = ce / beta; over = q > 1
    fc = np.polyfit(np.log10(w), np.log10(ce), 1); fs = np.polyfit(np.log10(w), np.log10(se), 1)
    print(f'{MODEL[m][2]}: {len(o)} loads ({nf} face, {len(o) - nf} cut-surface), {c} combinations; '
          f'ce/beta {q.min():.3f}-{q.max():.4f}, {over.sum()} above 1'
          + (f' (beta <= {beta[over].max():.1e}%, excess <= {(ce - beta)[over].max() / 100:.2e} of C)' if over.any() else '')
          + f'; slopes ce~w {fc[0]:.3f}, se~w {fs[0]:.3f}')
    return fc, fs


def log_ticks(axis, lo, hi, labelled):
    """Gridline every decade from 10^lo to 10^hi; labels only at the exponents in `labelled`; no minor ticks."""
    ex = list(range(lo, hi + 1))
    axis.set_major_locator(FixedLocator([10. ** e for e in ex]))
    axis.set_major_formatter(FixedFormatter([rf'$10^{{{e}}}$' if e in labelled else '' for e in ex]))
    axis.set_minor_locator(NullLocator())


def scatter(ax, o, kx, ky, m, ms):
    col, mk, _ = MODEL[m]
    for face in (False, True):                  # open cut-surface markers below the filled face markers
        sel = [d for d in o if d['face'] == face]
        kw = dict(ls='none', marker=mk, ms=ms, color=col, alpha=.9, zorder=3 + face)
        kw.update(dict(mec='white', mew=.25) if face else dict(mfc='white', mec=col, mew=.7))
        ax.plot([d[kx] for d in sel], [d[ky] for d in sel], **kw)


def frame(ax):
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.grid(color=GRID, lw=.4); ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=.6, color=MUTED, pad=2)


def fit_line(ax, o, f, x0, y0, ha='right'):
    w = np.array([d['w'] for d in o]); x = np.array([w.min(), w.max()])
    ax.plot(x, 10 ** f[1] * x ** f[0], color=TEXT, lw=.8, zorder=5)
    ax.text(x0, y0, f'fit, slope {f[0]:.2f}'.replace('-', '−'), fontsize=6.5, color=TEXT, ha=ha, va='center')


def circle(ax, d, kx, ky, ms=7):
    ax.plot(d[kx], d[ky], 'o', ms=ms, mfc='none', mec=TEXT, mew=.8, zorder=6)


def example(o):
    return [d for d in o if (d['cell'], d['load']) == EXAMPLE][0]


def mm_axes(fig, W, H, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def face_cut_legend(fig, m, y):
    col, mk, _ = MODEL[m] if m else (MUTED, 's', '')
    h = [Line2D([], [], ls='none', marker=mk, ms=4, color=col, mec=col, label='Face loads'),
         Line2D([], [], ls='none', marker=mk, ms=4, mfc='white', mec=col, mew=.7, color=col, label='Cut-surface loads')]
    fig.legend(handles=h, ncol=2, frameon=False, fontsize=7, handletextpad=.3, columnspacing=2.2, borderaxespad=0,
               loc='upper center', bbox_to_anchor=(.5, y))


def figure_nice(o, fc, fs):
    """Figure 11: NICE alone, three panels on one row; (b) and (c) share the error axis."""
    W, H = 178., 71.
    S, y0 = 44., 15.
    xa = 13.
    xb = xa + S + 16.
    xc = xb + S + 16.
    fig = plt.figure(figsize=(W * MM, H * MM))
    axa, axb, axc = mm_axes(fig, W, H, xa, y0, S, S), mm_axes(fig, W, H, xb, y0, S, S), mm_axes(fig, W, H, xc, y0, S, S)
    ELO, EHI, WLO, WHI = 1e-7, 1., 3e-3, 200.
    ex = example(o)

    frame(axa)
    scatter(axa, o, 'beta', 'ce', 'A3', 3.)
    axa.plot([ELO, EHI], [ELO, EHI], color=TEXT, lw=.7, zorder=2)
    axa.text(1.2e-6, 1.2e-6 * 2.6, 'equality', rotation=45, rotation_mode='anchor', fontsize=6.5, color=TEXT,
             ha='left', va='bottom')
    axa.set_xlim(ELO, EHI); axa.set_ylim(ELO, EHI)
    log_ticks(axa.xaxis, -7, 0, (-6, -4, -2, 0)); log_ticks(axa.yaxis, -7, 0, (-6, -4, -2, 0))
    axa.set_xlabel(r'$\beta=\sum_m w_m\varepsilon_m$ (%)', labelpad=2)
    axa.set_ylabel('Compliance error (%)')
    panel(axa, 'a', 'Compliance error follows β')
    circle(axa, ex, 'beta', 'ce')

    for ax, key, f, letter, title in ((axb, 'ce', fc, 'b', r'Compliance error follows $w$'),
                                      (axc, 'se', fs, 'c', 'Sensitivity error does not')):
        frame(ax)
        scatter(ax, o, 'w', key, 'A3', 3.)
        ax.set_xlim(WLO, WHI); ax.set_ylim(ELO, EHI)
        log_ticks(ax.xaxis, -2, 2, (-2, -1, 0, 1, 2)); log_ticks(ax.yaxis, -7, 0, (-6, -4, -2, 0))
        fit_line(ax, o, f, 150., 2.5e-7)
        circle(ax, ex, 'w', key)
        ax.text(ex['w'] / 1.7, ex[key] * (1.25 if key == 'ce' else 1.), 'U1/x, N-z', fontsize=6.5, color=TEXT,
                ha='right', va='bottom' if key == 'ce' else 'center')
        panel(ax, letter, title)
    axb.set_ylabel('Compliance error (%)')
    axc.set_ylabel('Test-cell sensitivity error (%)')
    fig.text((xb + xc + S) / 2 / W, (y0 - 7.) / H, r'Test-cell energy fraction $w$ (%)', ha='center', va='top', fontsize=8,
             color=TEXT)
    face_cut_legend(fig, 'A3', 1.)
    save(fig, 'F10_energy_share')


def figure_others(data, fits):
    """Figure S04: one column per variant (OTHERS), rows = the three relations of Figure 11; letters column-major."""
    W, H = 178., 155.
    xl, S, gx = 17., 37., 5.                 # left edge of column 1, panel side, horizontal gap
    ytop, gy = 141., 15.                     # top of row 1, vertical gap between rows (holds the row's x label)
    ELO, EHI, WLO, WHI = 1e-7, 1e2, 3e-3, 200.
    fig = plt.figure(figsize=(W * MM, H * MM))
    rows = [('beta', 'ce', r'$\beta=\sum_m w_m\varepsilon_m$ (%)', 'Compliance error (%)'),
            ('w', 'ce', r'Test-cell energy fraction $w$ (%)', 'Compliance error (%)'),
            ('w', 'se', r'Test-cell energy fraction $w$ (%)', 'Test-cell sensitivity error (%)')]
    for j, m in enumerate(OTHERS):
        o, _ = data[m]; ex = example(o) if m == 'B' else None
        x = xl + j * (S + gx)
        for i, (kx, ky, lx, ly) in enumerate(rows):
            ax = mm_axes(fig, W, H, x, ytop - S - i * (S + gy), S, S)
            frame(ax)
            scatter(ax, o, kx, ky, m, 3.)
            ax.set_ylim(ELO, EHI)
            log_ticks(ax.yaxis, -7, 2, (-6, -3, 0))
            if kx == 'beta':
                ax.plot([ELO, EHI], [ELO, EHI], color=TEXT, lw=.7, zorder=2)
                ax.set_xlim(ELO, EHI); log_ticks(ax.xaxis, -7, 2, (-6, -3, 0))
            else:
                ax.set_xlim(WLO, WHI); log_ticks(ax.xaxis, -2, 2, (-2, 0, 2))
                fit_line(ax, o, fits[m][0 if ky == 'ce' else 1], 150., 3e-7)
            if j == 0:
                ax.set_ylabel(ly)
            else:
                ax.tick_params(labelleft=False)
            if ex is not None:
                circle(ax, ex, kx, ky, ms=6)
            ax.set_title(f'({"abcdefghijkl"[3 * j + i]})', loc='left', fontweight='bold', fontsize=8, pad=3)
            if i == 0:
                fig.text((x + S / 2) / W, (ytop + 6.) / H, MODEL[m][2], ha='center', va='bottom', fontsize=8,
                         fontweight='bold', color=MODEL[m][0])
    for i, (_, _, lx, _) in enumerate(rows):
        fig.text((xl + (4 * S + 3 * gx) / 2) / W, (ytop - S - i * (S + gy) - 6.) / H, lx, ha='center', va='top',
                 fontsize=8, color=TEXT)
    face_cut_legend(fig, None, 1.)
    save(fig, 'S07_energy_share_variants')


def main():
    data = {m: load(run) for m, run in RUNS}
    fits = {m: stats(m, o, c) for m, (o, c) in data.items()}
    print(f'total: {sum(len(o) for o, _ in data.values())} loads, {sum(c for _, c in data.values())} combinations')
    for m in ('B', 'A3'):
        d = example(data[m][0])
        print(f'example {MODEL[m][2]} {EXAMPLE}: w {d["w"]:.4f}%, eps {100 * d["beta"] / d["w"]:.4f}%, '
              f'beta {d["beta"]:.3e}%, ce {d["ce"]:.3e}%, se {d["se"]:.3f}%')
    figure_nice(data['A3'][0], *fits['A3'])
    figure_others(data, fits)


if __name__ == '__main__':
    main()
