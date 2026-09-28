"""Supplementary Figure S02 (S01 before the revision-1 renumbering): geometry-level directional mean energy excess by direction class (identity view).
Sources: evidence/newval_c_oh.json (P0, five original classes), evidence/newval2_<run>.json (B, C, S8; 80 geometries,
75 for support_k and glued). Each point is one geometry's directional mean; bars are medians."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from figstyle import MODEL, C, MUTED, GRID, MM, TEXT, panel, save

EV = Path(__file__).resolve().parent.parent / 'evidence'
ARMS = [('P0', 'newval_c_oh.json', '#98A3AE', 'v'), ('Base\nnetwork', 'newval2_v2L1.json', MODEL['B'][0], MODEL['B'][1]),
        ('Uncor-\nrected', 'newval2_A0_ctrl.json', MODEL['C'][0], MODEL['C'][1]), ('S8', 'newval2_A2_tail8.json', MODEL['S8'][0], MODEL['S8'][1])]
CLASSES = [('force', 'Nodal force'), ('support', 'Spring support'), ('face', 'Single-face force'),
           ('macro', 'Polynomial'), ('grf', 'Multiscale'), ('force_c', 'Traction'),
           ('face_c', 'Face traction'), ('support_k', 'Stiffness support'), ('glued', 'Neighbour-induced')]
LO, HI = 3e-4, 3e2


def offsets(n, width=.34):
    """Deterministic horizontal offsets (golden-ratio sequence) so that overlapping points separate."""
    return ((np.arange(n) * 0.6180339887) % 1 - .5) * width


def main():
    data = {a: json.loads((EV / f).read_text())['per_geo'] for a, f, _, _ in ARMS}
    fig, axs = plt.subplots(3, 3, figsize=(178 * MM, 180 * MM), sharey=True, gridspec_kw=dict(hspace=.45, wspace=.12))
    for k, ((cls, title), ax) in enumerate(zip(CLASSES, axs.flat)):
        for j, (a, _, col, mk) in enumerate(ARMS):
            v = np.array([g['0'][cls] for g in data[a].values() if cls in g['0'] and np.isfinite(g['0'][cls])]) * 100
            if not len(v):
                ax.text(j, LO * 1.6, 'n/a', ha='center', va='bottom', color=MUTED, fontsize=6.5)
                continue
            assert v.min() >= LO and v.max() <= HI, (a, cls, v.min(), v.max())
            ax.plot(j + offsets(len(v)), v, mk, color=col, ms=2.6, alpha=.7, mew=0, ls='none')
            ax.plot([j - .24, j + .24], [np.median(v)] * 2, color=TEXT, lw=1.3, solid_capstyle='butt')
        ax.set_yscale('log'); ax.set_ylim(LO, HI); ax.set_xlim(-.6, len(ARMS) - .4)
        ax.set_xticks(range(len(ARMS)), [a for a, *_ in ARMS])
        ax.grid(axis='y', color=GRID, lw=.4); ax.minorticks_off(); ax.set_yticks([1e-3, 1e-2, 1e-1, 1, 10, 100])
        if k % 3 == 0:
            ax.set_ylabel('Energy excess (%)')
        panel(ax, 'abcdefghi'[k], title)
    save(fig, 'S01_distributions')


if __name__ == '__main__':
    main()
