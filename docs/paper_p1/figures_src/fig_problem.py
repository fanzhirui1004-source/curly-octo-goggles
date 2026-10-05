"""Figure 1 (problem figure) -> figures/F00_problem.{svg,pdf,png}
(a) Plan view, looking down the plate normal, of the plate supported on its cut of Section 5.10 (8 x 4 x 1 Schwarz-P cells
    at the uniform tau = 0.40, planar cut; 16 uncut and 8 cut cells): the walls, orthographically projected (drawn bottom-up,
    light from above), uncut cells light blue and cut cells dark blue (the cell ramp of figstyle.CELL: light = uncut,
    dark = cut), the part of the clamped cut band seen from above in dark sand; the oblique cut line with the support
    hatched on the removed side; the end-face traction (arrows); the removed part of the 8 x 4 lattice (dashed outline);
    axes and a one-cell scale cue.  Long side y horizontal, short side x vertical (as Figure 13 and F14_designs_scale).
(b) Close-up of the cut cell (3, 2, 0) next to the support, closest to the loaded end: translucent walls, cut plane, and the
    retained nodes of the Q2 model (n = 32): box-face nodes (blue) and cut-band nodes (sand); all other nodes of the
    active elements are interior and are eliminated by condensation.  Oblique view from the cut side (elevation 18 deg,
    azimuth -40 deg; Figure 13(c) uses 28 deg, -118 deg).
Data: data_elev/plate_design.json, plate_k000.npz, cell320_k000.npz, nodes_320_o000.npz (elev_prep_meshes.py).
"""
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle
from mpl_toolkits.mplot3d import proj3d
import figstyle as FS
import elev3d as E
plt = FS.plt

UNCUT, CUTC = FS.CELL['U1'][0], FS.CELL['M2'][0]                         # cell layer: light = uncut, dark = cut
UNCUT_CAP, CUTC_CAP = '#5F86B3', '#1F4573'                             # their wall sections on the top face
B_ELEV, B_AZIM = 18, -40


def rgb(c):
    return np.array(plt.matplotlib.colors.to_rgb(c))


def text3(ax, p, s, **kw):
    M = ax.get_proj()
    x, y, _ = proj3d.proj_transform(*E.to_plot(p), M)
    return ax.annotate(s, (x, y), xycoords='data', annotation_clip=False, **kw)


def panel_a(ax, D):
    m = E.load('plate_k000.npz')
    n, b = np.asarray(D['normal']), D['b_global']
    nx, ny, nz = D['shape']
    v, f = m['v'], m['f']
    tri = v[f]
    cen = tri.mean(1)
    cut_cells = {tuple(c['position'][:2]) for c in D['cells'] if c['kind'] != 'FULL'}
    ci = np.clip(np.floor(cen[:, 0]).astype(int), 0, nx - 1); cj = np.clip(np.floor(cen[:, 1]).astype(int), 0, ny - 1)
    in_cut = np.array([(i, j) in cut_cells for i, j in zip(ci, cj)])
    top = m['on_box'] & np.all(np.abs(tri[:, :, 2] - 1.0) < 1e-3, axis=1)
    col = np.where(in_cut[:, None], rgb(CUTC), rgb(UNCUT))
    col = np.where((top & in_cut)[:, None], rgb(CUTC_CAP), np.where(top[:, None], rgb(UNCUT_CAP), col))
    col = np.where((m['in_band'] | m['on_cut'])[:, None] & ~top[:, None], rgb(E.CLAMP), col)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30
    L = np.array([-.25, -.35, .9]); L /= np.linalg.norm(L)
    s = (.50 + .50 * np.abs(nrm @ L))[:, None]
    col = np.clip(col * s + .12 * (1 - s), 0, 1)
    order = np.argsort(cen[:, 2])                                         # plan view: lower faces first
    P2 = tri[order][:, :, [1, 0]]                                         # plot (X, Y) = (y, x)
    pc = PolyCollection(P2, facecolors=col[order], edgecolors=col[order], linewidths=.05, zorder=2)
    pc.set_rasterized(True)
    ax.add_collection(pc)
    a2 = np.array([n[1], n[0]])                                          # cut normal in plot coordinates
    p0, p1 = E.cut_polygon(n, b, (0, 0), (nx, ny))
    q0, q1 = p0[[1, 0]], p1[[1, 0]]
    # removed part of the 8 x 4 lattice: light fill, dashed outline
    corner = [(ny, 0), (ny, nx)]
    rem = [tuple(q) for q in (q0, q1)] + [c for c in [(ny, 0), (ny, nx), (0, nx)] if a2 @ np.array(c) > b + 1e-9]
    cx = np.mean(np.array(rem), 0)
    rem = sorted(rem, key=lambda p: np.arctan2(p[1] - cx[1], p[0] - cx[0]))
    ax.add_patch(Polygon(rem, closed=True, fc='#F1F3F5', ec='none', zorder=0))
    ax.add_patch(Rectangle((0, 0), ny, nx, fc='none', ec='#A3ADB8', lw=.6, ls=(0, (3, 2.5)), zorder=1))
    for k in range(1, ny):
        ax.plot([k, k], [0, nx], color='#C5CDD4', lw=.3, zorder=1)
    for k in range(1, nx):
        ax.plot([0, ny], [k, k], color='#C5CDD4', lw=.3, zorder=1)
    # cut line and support hatching on the removed side
    ax.plot([q0[0], q1[0]], [q0[1], q1[1]], color=E.CLAMP, lw=1.6, zorder=5, solid_capstyle='butt')
    t = (q1 - q0) / np.linalg.norm(q1 - q0); Lc = np.linalg.norm(q1 - q0)
    for s_ in np.linspace(.12, Lc - .02, 34):
        a0 = q0 + s_ * t; a1 = a0 + .24 * a2 - .16 * t
        ax.plot([a0[0], a1[0]], [a0[1], a1[1]], color=E.CLAMP, lw=.6, zorder=5)
    # loaded end face y = 0 (plot X = 0): in-plane traction of unit resultant, direction +x (plot +Y)
    ax.plot([0, 0], [0, nx], color=FS.TEXT, lw=1.8, zorder=5, solid_capstyle='butt')
    for ys in np.arange(.5, nx, 1.0):
        ax.annotate('', (-.3, ys + .38), (-.3, ys - .38), arrowprops=dict(arrowstyle='-|>', color=FS.TEXT, lw=.9,
                    mutation_scale=7, shrinkA=0, shrinkB=0), annotation_clip=False, zorder=6)
    # cell of panel (b)
    c = D['cell']
    ax.add_patch(Rectangle((c[1], c[0]), 1, 1, fc='none', ec=FS.TEXT, lw=1.1, zorder=6))
    ax.text(c[1] + .5, c[0] + 1.08, '(b)', ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=FS.TEXT, zorder=7)
    # labels
    ax.text(-.62, nx / 2, 'traction on the end face', rotation=90, ha='center', va='center', fontsize=7, color=FS.TEXT)
    mid = (q0 + q1) / 2
    ax.annotate('clamped cut band\n(support on the cut)', (6.25, 1.42), (7.25, 1.95), fontsize=7,
                bbox=dict(fc='#F1F3F5', ec='none', pad=1.0),
                color=FS.TEXT, ha='center', va='center', zorder=7,
                arrowprops=dict(arrowstyle='-', color=FS.MUTED, lw=.5, shrinkA=2, shrinkB=0))
    ax.text(6.75, 3.3, 'part removed\nby the cut', ha='center', va='center', fontsize=7, color=FS.MUTED, zorder=7)
    # axes and scale cue (bottom right, outside the plate)
    ox, oy = ny + .45, -.05
    for dx, dy, lab in ((.8, 0, '$y$'), (0, .8, '$x$')):
        ax.annotate('', (ox + dx, oy + dy), (ox, oy), arrowprops=dict(arrowstyle='-|>', color=FS.TEXT, lw=.7,
                    mutation_scale=6, shrinkA=0, shrinkB=0), annotation_clip=False)
        ax.text(ox + dx + (.12 if dx else 0), oy + dy + (.12 if dy else 0), lab, fontsize=7.5, ha='left' if dx else 'center',
                va='center' if dx else 'bottom', color=FS.TEXT)
    ax.plot([ox, ox + 1], [oy - .32, oy - .32], color=FS.TEXT, lw=1.0, solid_capstyle='butt', clip_on=False)
    ax.text(ox + .5, oy - .42, '1 cell', ha='center', va='top', fontsize=7, color=FS.TEXT)
    ax.set_xlim(-.85, ny + 1.5); ax.set_ylim(-.95, nx + .25); ax.set_aspect('equal'); ax.axis('off')


def panel_b(ax, D):
    m = E.load('cell320_k000.npz'); z = E.load('nodes_320_o000.npz')
    n, b = m['normal'], float(m['offset'])
    tri = m['v'][m['f']]
    col = np.where(m['on_cut'][:, None], rgb(E.SECTION), np.where(m['on_box'][:, None], rgb(E.BOXCAP), rgb(E.BAND)))
    col = np.where(m['in_band'][:, None] & ~m['on_box'][:, None], rgb(E.SECTION), col)
    alpha = np.where(m['on_cut'], .55, np.where(m['on_box'], .30, np.where(m['in_band'], .35, .20)))
    parts = [(E.to_plot(tri), np.c_[E.shade(col, tri), alpha])]
    rt = m['v_removed'][m['f_removed']]
    parts.append((E.to_plot(rt), np.c_[E.shade(rgb(E.GHOST), rt), np.full(len(rt), .07)]))
    p0, p1 = E.cut_polygon(n, b, (0, 0), (1, 1))
    E.collection(ax, parts, lw=.03)
    E.lines(ax, [[np.r_[p0, 0], np.r_[p1, 0]], [np.r_[p0, 1], np.r_[p1, 1]], [np.r_[p0, 0], np.r_[p0, 1]],
                 [np.r_[p1, 0], np.r_[p1, 1]]], colors=E.CLAMP, linewidths=.8)
    E.lines(ax, E.box_edges((0, 0, 0), (1, 1, 1)), colors=FS.TEXT, linewidths=.6)
    g = np.stack(np.unravel_index(z['nodes'], (65,) * 3), 1) / 64.0
    box, cut = z['is_box'], z['is_cut']
    P = E.to_plot(g)
    ax.scatter(*P[box & ~cut].T, s=.9, c=E.BOXCAP, marker='.', linewidths=0, depthshade=False, rasterized=True)
    ax.scatter(*P[cut].T, s=.9, c=E.CLAMP, marker='.', linewidths=0, depthshade=False, rasterized=True)
    E.setup(ax, (-.02, -.02, -.02), (1.02, 1.02, 1.02), elev=B_ELEV, azim=B_AZIM, zoom=1.05)
    ax.text(*E.to_plot(np.r_[p1, 0.0] + np.array([.04, -.04, -.06])), 'cut plane', fontsize=7, color=E.CLAMP, ha='left',
            va='top')
    nret = int((box | cut).sum())
    print(f'cell {tuple(D["cell"])}: {len(g)} nodes, {int(box.sum())} box-face, {int(cut.sum())} cut-band, '
          f'{int((box & cut).sum())} both, retained {nret} ({100 * nret / len(g):.1f}%), interior {len(g) - nret}')
    return nret, len(g)


def main():
    D = E.design()
    fig = plt.figure(figsize=(178 * FS.MM, 74 * FS.MM))
    axa = fig.add_axes([.0, .15, .64, .74])
    axb = fig.add_axes([.655, .16, .335, .74], projection='3d', computed_zorder=False)
    panel_a(axa, D)
    nret, nall = panel_b(axb, D)
    fig.text(.01, .975, '(a) A lattice trimmed to a part: cut cells carry the support', fontweight='bold', fontsize=8.5, va='top')
    fig.text(.665, .975, '(b) Cut cell next to the support', fontweight='bold', fontsize=8.5, va='top')
    fig.legend(handles=[Patch(fc=UNCUT, ec='none', label='uncut cell (16)'),
                        Patch(fc=CUTC, ec='none', label='cut cell (8)'),
                        Patch(fc=E.CLAMP, ec='none', label='clamped cut band'),
                        Line2D([], [], ls='none', marker='o', ms=3, mfc=E.BOXCAP, mec='none', label='retained box-face nodes'),
                        Line2D([], [], ls='none', marker='o', ms=3, mfc=E.CLAMP, mec='none', label='retained cut-band nodes')],
               loc='lower center', bbox_to_anchor=(.5, -.01), ncol=5, frameon=False, fontsize=7, columnspacing=1.3,
               handlelength=1.2, handletextpad=.5)
    fig.text(.83, .155, f'{nret:,} of {nall:,} nodes retained;\nthe others are interior (condensed)', ha='center', va='top',
             fontsize=7, color=FS.MUTED)
    FS.save(fig, 'F00_problem')


if __name__ == '__main__':
    main()
