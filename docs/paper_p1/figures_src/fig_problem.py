"""Figure 1 (problem figure) -> figures/F00_problem.{svg,pdf,png}
(a) The plate supported on its cut of Section 5.10 (8 x 4 x 1 Schwarz-P cells, planar cut, 16 uncut + 8 cut cells) at the
    uniform starting design tau = 0.40: material surface, wall sections on the outer faces, the wall section on the cut plane
    (sand: clamped, every cut-band DOF of every cut cell fixed), the support drawn as a hatched plane on the cut, the part of
    the 8 x 4 lattice removed by the cut (faint), and the in-plane traction on the end face y = 0 (arrows, direction +x).
(b) Close-up of the cut cell (3, 2, 0) next to the clamp, closest to the loaded end: translucent walls, cut plane, and the
    retained nodes of the Q2 model (n = 32): box-face nodes and cut-band nodes; all other nodes of the active elements are
    interior and are eliminated by condensation.
Data: data_elev/plate_design.json, plate_k000.npz, plate_ghost.npz, cell320_k000.npz, nodes_320_o000.npz
(elev_prep_meshes.py).  View: orthographic, elevation 30 deg, azimuth -122 deg, plot axes (y, -x, z) (elev3d.py).
"""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import proj3d
import figstyle as FS
import elev3d as E
plt = FS.plt

ELEV, AZIM = 34, -118


def arrow2d(ax, p0, p1, **kw):
    """Arrow between two 3D points (plot coordinates), drawn in the projected 2D plane of the axes."""
    M = ax.get_proj()
    x0, y0, _ = proj3d.proj_transform(*p0, M); x1, y1, _ = proj3d.proj_transform(*p1, M)
    ax.annotate('', (x1, y1), (x0, y0), xycoords='data', textcoords='data', annotation_clip=False,
                arrowprops=dict(arrowstyle='-|>', shrinkA=0, shrinkB=0, **kw))


def text3(ax, p, s, **kw):
    M = ax.get_proj()
    x, y, _ = proj3d.proj_transform(*E.to_plot(p), M)
    return ax.annotate(s, (x, y), xycoords='data', annotation_clip=False, **kw)


def wall_quads(p0, p1, z0, z1, nu=40, nv=4):
    """Quads of the vertical support plane between the points p0, p1 (x, y) for z0 <= z <= z1 (internal coordinates)."""
    q = []
    for i in range(nu):
        for j in range(nv):
            a, b_ = i / nu, (i + 1) / nu; c, d = z0 + (z1 - z0) * j / nv, z0 + (z1 - z0) * (j + 1) / nv
            P = [np.r_[p0 + a * (p1 - p0), c], np.r_[p0 + b_ * (p1 - p0), c], np.r_[p0 + b_ * (p1 - p0), d], np.r_[p0 + a * (p1 - p0), d]]
            q.append(P)
    return np.asarray(q)


def quads_to_tris(Q):
    return np.concatenate([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])


def panel_a(ax, D):
    m, g = E.load('plate_k000.npz'), E.load('plate_ghost.npz')
    n, b = np.asarray(D['normal']), D['b_global']
    nx, ny, nz = D['shape']
    tri = m['v'][m['f']]
    rgb_ = lambda c: np.array(plt.matplotlib.colors.to_rgb(c))
    col = np.where((m['on_cut'] | m['in_band'])[:, None], rgb_(E.CLAMP),
                   np.where(m['on_box'][:, None], rgb_(E.BOXCAP), rgb_(E.BAND)))
    rgb = E.shade(col, tri, amb=.55)
    parts = [(E.to_plot(tri), np.c_[rgb, np.ones(len(tri))])]
    gt = g['v'][g['f']]
    parts.append((E.to_plot(gt), np.c_[E.shade(np.array(plt.matplotlib.colors.to_rgb(E.GHOST)), gt), np.full(len(gt), .10)]))
    p0, p1 = E.cut_polygon(n, b, (0, 0), (nx, ny))
    W = quads_to_tris(wall_quads(p0, p1, -.18, 1.18))
    parts.append((E.to_plot(W), E.rgba(E.SECTION, len(W), .30)))
    E.collection(ax, parts, lw=.04)
    # support symbol: hatching on the support plane (removed side), and its outline
    t = (p1 - p0) / np.linalg.norm(p1 - p0); L_ = np.linalg.norm(p1 - p0)
    hs = []
    for s in np.linspace(-.9, L_ - .05, 44):                             # hatching of the support plane, clipped to it
        a, z0 = max(s, 0.0), 1.18 - (max(s, 0.0) - s)
        e_ = min(s + 1.0, L_); z1 = 1.18 - (e_ - s)
        if z0 > -.18 and e_ > a:
            z1c = max(z1, -.18); e_ = s + (1.18 - z1c)
            hs.append([np.r_[p0 + a * t, z0], np.r_[p0 + e_ * t, z1c]])
    E.lines(ax, hs, colors=E.CLAMP, linewidths=.45)
    E.lines(ax, [[np.r_[p0, z], np.r_[p1, z]] for z in (-.18, 1.18)] + [[np.r_[p, -.18], np.r_[p, 1.18]] for p in (p0, p1)],
            colors=E.CLAMP, linewidths=.7)
    # outline of the full 8 x 4 lattice before the cut (dashed) and of the plate's outer faces
    E.lines(ax, E.box_edges((0, 0, 0), (nx, ny, nz)), colors='#A3ADB8', linewidths=.5, linestyles=(0, (3, 2.5)))
    # cell of panel (b)
    c = np.asarray(D['cell'], float)
    E.lines(ax, E.box_edges(c, c + 1), colors=FS.TEXT, linewidths=.9)
    E.setup(ax, (-.3, -.6, -.25), (nx + .3, ny + .3, nz + .3), elev=ELEV, azim=AZIM, zoom=1.35)
    # load: consistent traction of unit resultant on the end face y = 0, in-plane, direction +x (towards the viewer)
    for x in (.5, 1.5, 2.5, 3.5):
        arrow2d(ax, E.to_plot((x - .45, -.32, .5)), E.to_plot((x + .45, -.32, .5)), color=FS.TEXT, lw=1.0, mutation_scale=7)
    return p0, p1


def panel_b(ax, D):
    m = E.load('cell320_k000.npz'); z = E.load('nodes_320_o000.npz')
    n, b = m['normal'], float(m['offset'])
    tri = m['v'][m['f']]
    col = np.where(m['on_cut'][:, None], np.array(plt.matplotlib.colors.to_rgb(E.SECTION)),
                   np.where(m['on_box'][:, None], np.array(plt.matplotlib.colors.to_rgb(E.BOXCAP)),
                            np.array(plt.matplotlib.colors.to_rgb(E.BAND))))
    col = np.where(m['in_band'][:, None] & ~m['on_box'][:, None], np.array(plt.matplotlib.colors.to_rgb(E.SECTION)), col)
    alpha = np.where(m['on_cut'], .55, np.where(m['on_box'], .30, np.where(m['in_band'], .35, .20)))
    parts = [(E.to_plot(tri), np.c_[E.shade(col, tri), alpha])]
    rt = m['v_removed'][m['f_removed']]
    parts.append((E.to_plot(rt), np.c_[E.shade(np.array(plt.matplotlib.colors.to_rgb(E.GHOST)), rt), np.full(len(rt), .07)]))
    p0, p1 = E.cut_polygon(n, b, (0, 0), (1, 1))
    W = quads_to_tris(wall_quads(p0, p1, 0, 1, 8, 4))
    parts.append((E.to_plot(W), E.rgba(E.SECTION, len(W), .13)))
    E.collection(ax, parts, lw=.03)
    E.lines(ax, [[np.r_[p0, 0], np.r_[p1, 0]], [np.r_[p0, 1], np.r_[p1, 1]], [np.r_[p0, 0], np.r_[p0, 1]],
                 [np.r_[p1, 0], np.r_[p1, 1]]], colors=E.CLAMP, linewidths=.7)
    E.lines(ax, E.box_edges((0, 0, 0), (1, 1, 1)), colors=FS.TEXT, linewidths=.6)
    g = np.stack(np.unravel_index(z['nodes'], (65,) * 3), 1) / 64.0
    box, cut = z['is_box'], z['is_cut']
    P = E.to_plot(g)
    ax.scatter(*P[box & ~cut].T, s=.9, c=E.BOXCAP, marker='.', linewidths=0, depthshade=False, rasterized=True, clip_on=False)
    ax.scatter(*P[cut].T, s=.9, c=E.CLAMP, marker='.', linewidths=0, depthshade=False, rasterized=True)
    E.setup(ax, (-.02, -.02, -.02), (1.02, 1.02, 1.02), elev=ELEV, azim=AZIM, zoom=1.0)
    nret = int((box | cut).sum())
    print(f'cell {tuple(D["cell"])}: {len(g)} nodes, {int(box.sum())} box-face, {int(cut.sum())} cut-band, '
          f'{int((box & cut).sum())} both, retained {nret} ({100 * nret / len(g):.1f}%), interior {len(g) - nret}')
    return nret, len(g)


def main():
    D = E.design()
    fig = plt.figure(figsize=(178 * FS.MM, 78 * FS.MM))
    axa = fig.add_axes([-.03, .06, .72, .88], projection='3d', computed_zorder=False)
    axb = fig.add_axes([.67, .17, .33, .76], projection='3d', computed_zorder=False)
    p0, p1 = panel_a(axa, D)
    nret, nall = panel_b(axb, D)
    fig.text(.01, .965, '(a) Schwarz-P lattice trimmed by a plane: 16 uncut and 8 cut cells', fontweight='bold', fontsize=8.5, va='top')
    fig.text(.665, .965, '(b) Cut cell next to the support', fontweight='bold', fontsize=8.5, va='top')
    # annotations of (a)
    kw = dict(fontsize=7, color=FS.TEXT, textcoords='offset points', ha='center', va='center')
    text3(axa, (4.0, -.35, .5), 'in-plane traction\non the end face', xytext=(-8, -18), **kw)
    mid = (p0 + p1) / 2
    text3(axa, (mid[0], mid[1], -.18), 'clamped cut band\n(support on the cut)', xytext=(30, -26),
          arrowprops=dict(arrowstyle='-', color=FS.MUTED, lw=.5, shrinkA=1, shrinkB=0), **kw)
    text3(axa, (1.0, 7.6, 1.0), 'part of the 8 × 4 lattice\nremoved by the cut', xytext=(0, 26), color=FS.MUTED,
          **{k: v for k, v in kw.items() if k != 'color'})
    c = np.asarray(D['cell'], float)
    text3(axa, (c[0] + 1, c[1], 0.0), '(b)', xytext=(-10, -8), fontweight='bold', **kw)
    fig.legend(handles=[Patch(fc=E.BAND, ec='none', label='material surface (walls)'),
                        Patch(fc=E.BOXCAP, ec='none', label='wall section on box faces'),
                        Patch(fc=E.CLAMP, ec='none', label='cut band (clamped) and support plane'),
                        Line2D([], [], ls='none', marker='o', ms=3, mfc=E.BOXCAP, mec='none', label='retained box-face nodes'),
                        Line2D([], [], ls='none', marker='o', ms=3, mfc=E.CLAMP, mec='none', label='retained cut-band nodes')],
               loc='lower center', bbox_to_anchor=(.5, -.02), ncol=3, frameon=False, fontsize=7, columnspacing=1.2,
               handlelength=1.2, handletextpad=.5)
    fig.text(.835, .175, f'{nret:,} of {nall:,} nodes retained;\nthe others are interior (condensed)', ha='center', va='top',
             fontsize=7, color=FS.MUTED)
    FS.save(fig, 'F00_problem')


if __name__ == '__main__':
    main()
