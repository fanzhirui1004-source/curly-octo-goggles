"""New figure: retained / internal partition and the spatial distribution of the extension error on M1 (fresh_val_2003_d1_v1)
under one consistent-traction direction. Source: evidence/p1_field_M1_small.npz, reduced from p1_checks.py --dump
(bulk element energies x_e^T K_e x_e of the exact field and of the errors of B and A3, 32 validation directions' first 4)."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from figstyle import C, MUTED, TEXT, MM, panel, save

EV = Path(__file__).resolve().parent.parent / 'evidence'


def ax3(fig, pos):
    ax = fig.add_subplot(pos, projection='3d')
    ax.view_init(elev=24, azim=-58); ax.set_box_aspect((1, 1, 1))
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((1, 1, 1, 0)); a.line.set_color(MUTED)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_zticks([0, 1])
    ax.tick_params(labelsize=6, pad=0)
    ax.set_xlabel('x', labelpad=-8, fontsize=7); ax.set_ylabel('y', labelpad=-8, fontsize=7); ax.set_zlabel('z', labelpad=-8, fontsize=7)
    return ax


def main(k=3):
    z = np.load(EV / 'p1_field_M1_small.npz')
    ctr, E = z['ctr'], z['ee_exact'][:, k]
    tot = E.sum()
    eB, eA = z['ee_err_B'][:, k] / tot, z['ee_err_A3'][:, k] / tot
    fig = plt.figure(figsize=(178 * MM, 150 * MM))
    gs = fig.add_gridspec(2, 2, hspace=.18, wspace=.12)
    ax = ax3(fig, gs[0, 0])
    P, box = z['port_xyz'], z['port_is_box']
    I = z['int_xyz']
    ax.scatter(*I.T, s=.15, c='#B8C2CC', alpha=.35, linewidths=0, label='internal (I)')
    ax.scatter(*P[box].T, s=.5, c=C['exact'], linewidths=0, label='retained: box face')
    ax.scatter(*P[~box].T, s=.6, c=C['cut'], marker='s', linewidths=0, alpha=.8, label='retained: cut band')
    ax.legend(loc='upper left', bbox_to_anchor=(0, .98), frameon=False, markerscale=6, fontsize=6.5)
    panel(ax, 'a', 'Retained and internal coordinates')
    mask = E > 0
    ax = ax3(fig, gs[0, 1])
    sc = ax.scatter(*ctr[mask].T, s=1.2, c=E[mask] / tot, norm=LogNorm(1e-7, 1e-2), cmap='cividis', linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=.6, pad=.02); cb.set_label('element energy / total', fontsize=6.5); cb.ax.tick_params(labelsize=6)
    panel(ax, 'b', 'Exact field: element energy')
    for pos, e, letter, name, share in ((gs[1, 0], eB, 'c', 'B (uncorrected)', eB.sum()), (gs[1, 1], eA, 'd', 'A3 (corrected)', eA.sum())):
        ax = ax3(fig, pos)
        sc = ax.scatter(*ctr[mask].T, s=1.2, c=np.maximum(e[mask], 1e-12), norm=LogNorm(1e-9, 1e-3), cmap='magma_r', linewidths=0)
        cb = fig.colorbar(sc, ax=ax, shrink=.6, pad=.02); cb.set_label('element error energy / total', fontsize=6.5); cb.ax.tick_params(labelsize=6)
        panel(ax, letter, f'{name}: error energy')
        ax.text2D(.02, .92, f'total {100 * share:.3g}% of exact energy', transform=ax.transAxes, fontsize=6.5, color=MUTED)
    save(fig, 'F12_field_error_M1')


if __name__ == '__main__':
    main()
