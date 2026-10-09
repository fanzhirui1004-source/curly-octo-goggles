"""Supplementary Figure S02: geometry-level mean energy error by load class (identity view) for the five
variants of Table ST03 (Base network, Uncorrected continuation, Smoothing-trained, Base network + correction, NICE).
Sources: evidence/newval2_<run>.json (80 geometries; 75 for support_k and glued). Each point is one geometry's
directional mean; bars are medians. Colours, markers and legend labels from figstyle.MODEL."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from figstyle import MODEL, MUTED, GRID, MM, TEXT, panel, save

EV = Path(__file__).resolve().parent.parent / 'evidence'
VARIANTS = [('B', 'newval2_v2L1.json'), ('C', 'newval2_A0_ctrl.json'), ('A2b', 'newval2_A2b_tail8.json'),
            ('B+W', 'newval2_B2grid.json'), ('A3', 'newval2_A3_2grid.json')]
CLASSES = [('force', 'Nodal point loads'), ('support', 'Spring supports'), ('face', 'Single-face point loads'),
           ('macro', 'Polynomial'), ('grf', 'Multiscale'), ('force_c', 'Traction loads'),
           ('face_c', 'Single-face traction'), ('support_k', 'Scaled spring supports'),
           ('glued', 'Imposed by a neighbour')]
LO, HI = 5e-6, 3e2
WIDTH, BAR = .5, .3                      # jitter width and median-bar half-length, in category units


def offsets(n, width=WIDTH):
    """Deterministic horizontal offsets (golden-ratio sequence) so that overlapping points separate."""
    return ((np.arange(n) * 0.6180339887) % 1 - .5) * width


def main():
    data = {m: json.loads((EV / f).read_text())['per_geo'] for m, f in VARIANTS}
    fig, axs = plt.subplots(3, 3, figsize=(178 * MM, 180 * MM), sharey=True,
                            gridspec_kw=dict(hspace=.32, wspace=.1, top=.93))
    for k, ((cls, title), ax) in enumerate(zip(CLASSES, axs.flat)):
        for j, (m, _) in enumerate(VARIANTS):
            col, mk, _ = MODEL[m]
            v = np.array([g['0'][cls] for g in data[m].values() if cls in g['0'] and np.isfinite(g['0'][cls])]) * 100
            if not len(v):
                ax.text(j, LO * 1.6, 'n/a', ha='center', va='bottom', color=MUTED, fontsize=6.5)
                continue
            assert v.min() >= LO and v.max() <= HI, (m, cls, v.min(), v.max())
            ax.plot(j + offsets(len(v)), v, mk, color=col, ms=2.4, alpha=.7, mew=0, ls='none')
            ax.plot([j - BAR, j + BAR], [np.median(v)] * 2, color=TEXT, lw=1.3, solid_capstyle='butt')
        ax.set_yscale('log'); ax.set_ylim(LO, HI); ax.set_xlim(-.6, len(VARIANTS) - .4)
        ax.set_xticks(range(len(VARIANTS)), [''] * len(VARIANTS)); ax.tick_params(axis='x', length=0)
        ax.grid(axis='y', color=GRID, lw=.4); ax.minorticks_off(); ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
        if k % 3 == 0:
            ax.set_ylabel('Energy error (%)')
        panel(ax, 'abcdefghi'[k], title)
    h = [plt.Line2D([], [], ls='none', marker=MODEL[m][1], color=MODEL[m][0], ms=4.5, label=MODEL[m][2]) for m, _ in VARIANTS]
    h.append(plt.Line2D([], [], color=TEXT, lw=1.3, label='Median'))
    fig.legend(handles=h, loc='upper center', ncol=len(h), frameon=False, bbox_to_anchor=(.5, .995), fontsize=7,
               handletextpad=.45, columnspacing=1.2)
    save(fig, 'S01_distributions')


if __name__ == '__main__':
    main()
