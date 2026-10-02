"""Figure 7: retained / interior partition and the spatial distribution of the extension error on M1 (fresh_val_2003_d1_v1)
under one consistent-traction direction. Source: evidence/p1_field_M1_small.npz, reduced from p1_checks.py --dump
(bulk element energies x_e^T K_e x_e of the exact field and of the errors of B and A3, 32 validation directions' first 4).

Revision 1 (presentation only, data unchanged): one sequential colour map for the three energy panels, one colour bar per
row (the error panels (c,d) share their normalisation 'element error energy / total' and therefore one bar), the legend of
(a) below its panel, short panel titles, identical viewing angle in all panels, and rasterised scatters (PDF/SVG small)."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from figstyle import GREY, CELL, MUTED, TEXT, MM, panel, save

EV = Path(__file__).resolve().parent.parent / 'evidence'
plt.rcParams['savefig.dpi'] = 300                       # resolution of the rasterised scatters inside the PDF and SVG
VIEW = dict(elev=24, azim=-58)                           # the same viewing angle in all four panels
# one sequential map for all energy panels; the lightest 8% of magma_r is dropped so that the smallest values stay
# visible on the white background (light = small, dark = large)
CMAP = ListedColormap(plt.get_cmap('magma_r')(np.linspace(.08, 1, 256)), name='magma_r_trunc')
PT = dict(s=1.2, linewidths=0, rasterized=True)
# panel (a) categories (layer 3): greys plus one blue-ramp mid tone for the cut band
INTERIOR, BOXFACE, CUTBAND = GREY[2], GREY[0], CELL['M1'][0]


def ax3(fig, pos):
    ax = fig.add_subplot(pos, projection='3d')
    ax.view_init(**VIEW); ax.set_box_aspect((1, 1, 1), zoom=1.12)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((1, 1, 1, 0)); a.line.set_color(MUTED)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_zticks([0, 1])
    ax.tick_params(labelsize=6, pad=0)
    ax.set_xlabel('x', labelpad=-8, fontsize=7); ax.set_ylabel('y', labelpad=-8, fontsize=7); ax.set_zlabel('z', labelpad=-8, fontsize=7)
    return ax


def colourbar(fig, slot, mappable, label):
    """One colour bar in its own grid column (so that the 3D panels keep identical sizes)."""
    host = fig.add_subplot(slot); host.axis('off')
    cax = host.inset_axes([-.6, .2, 1, .6])                 # pulled slightly towards the panel's whitespace
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(label, fontsize=6.5); cb.ax.tick_params(labelsize=6, length=2, width=.5); cb.outline.set_linewidth(.5)
    return cb


def main(k=3):
    z = np.load(EV / 'p1_field_M1_small.npz')
    ctr, E = z['ctr'], z['ee_exact'][:, k]
    tot = E.sum()
    eB, eA = z['ee_err_B'][:, k] / tot, z['ee_err_A3'][:, k] / tot
    mask = E > 0
    fig = plt.figure(figsize=(178 * MM, 150 * MM))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, .04], hspace=.36, wspace=.02, left=.01, right=.93, top=.95, bottom=.02)

    ax = ax3(fig, gs[0, 0])                                             # (a) partition of the DOFs
    P, box = z['port_xyz'], z['port_is_box']
    I = z['int_xyz']
    ax.scatter(*I.T, s=.15, c=INTERIOR, alpha=.35, linewidths=0, label='interior (I)', rasterized=True)
    ax.scatter(*P[box].T, s=.5, c=BOXFACE, linewidths=0, label='retained: box face', rasterized=True)
    ax.scatter(*P[~box].T, s=.6, c=CUTBAND, marker='s', linewidths=0, alpha=.8, label='retained: cut band', rasterized=True)
    handles = [plt.Line2D([], [], ls='none', marker=mk, color=col, ms=4, label=lab) for col, mk, lab in
               ((INTERIOR, 'o', 'interior (I)'), (BOXFACE, 'o', 'retained: box face'), (CUTBAND, 's', 'retained: cut band'))]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.12), ncol=3, frameon=False, fontsize=6.5,
              handletextpad=.3, columnspacing=1.2, borderaxespad=0)      # below the panel (3D axes draw their labels inside)
    panel(ax, 'a', 'Retained and interior DOFs')

    ax = ax3(fig, gs[0, 1])                                             # (b) exact field
    sc = ax.scatter(*ctr[mask].T, c=E[mask] / tot, norm=LogNorm(1e-7, 1e-2), cmap=CMAP, **PT)
    colourbar(fig, gs[0, 2], sc, 'element energy / total')
    panel(ax, 'b', 'Exact field: element energy')

    for pos, e, letter, title in ((gs[1, 0], eB, 'c', 'Base network error energy'), (gs[1, 1], eA, 'd', 'NICE error energy')):
        ax = ax3(fig, pos)                                              # (c,d) error energies, common scale
        sc = ax.scatter(*ctr[mask].T, c=np.maximum(e[mask], 1e-12), norm=LogNorm(1e-9, 1e-3), cmap=CMAP, **PT)
        panel(ax, letter, title)
        ax.text2D(.02, .9, f'total {100 * e.sum():.3g}% of exact energy', transform=ax.transAxes, fontsize=6.5, color=MUTED)
    colourbar(fig, gs[1, 2], sc, 'element error energy / total')
    save(fig, 'F12_field_error_M1')


if __name__ == '__main__':
    main()
