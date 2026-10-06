"""Graphical abstract -> figures/GA_nice.{svg,pdf,png} (Elsevier: landscape, >= 1328 x 531 px, readable at 13 x 5 cm).
Canvas 130 x 52 mm; PNG at 300 dpi (1535 x 614 px); saved without a tight bounding box so the aspect ratio is kept.
  left    the plate supported on its cut (Section 5.10) in 3D at the uniform start: walls (light blue), wall sections on
          the outer faces (dark blue), clamped cut band and support plane (sand), end-face traction (arrows)
          (data_elev/plate_design.json, plate_k000.npz; same scene as the 3D view of the plate in elev3d.py)
  middle  the NICE division of tasks of Figure 2(a) (fig02_overview.py colours): learned extension -> two-grid
          correction -> energy form F^T K F; learning: trial field / correction: improvability / variational form: structure
  right   headline: clamped cut layer, homogenised model underestimates the compliance by 27% (Section 5.10: 27.0% of the
          exact compliance of the uniform start); NICE matches exact condensation to 0.03% in compliance and 0.3% in gradient
          (plate checks of the NICE run: compliance 0.011% and 0.030% below exact, gradient errors 0.029% and 0.30%).
"""
import numpy as np
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import proj3d
import figstyle as FS
import elev3d as E
plt = FS.plt

W, H = 130.0, 52.0
LEARN, CORR, VAR = '#0072B2', '#D55E00', '#53616F'                      # as fig02_overview.py
FILL = {'learn': ('#EAF3FA', LEARN), 'corr': ('#FDF1E8', CORR), 'var': ('#F3F5F7', VAR)}
HOM = FS.C['extra']


def rgb(c):
    return np.array(plt.matplotlib.colors.to_rgb(c))


def arrow2d(ax, p0, p1, **kw):
    M = ax.get_proj()
    x0, y0, _ = proj3d.proj_transform(*p0, M); x1, y1, _ = proj3d.proj_transform(*p1, M)
    ax.annotate('', (x1, y1), (x0, y0), xycoords='data', annotation_clip=False,
                arrowprops=dict(arrowstyle='-|>', shrinkA=0, shrinkB=0, **kw))


def plate3d(ax, D):
    m = E.load('plate_k000.npz')
    n, b = np.asarray(D['normal']), D['b_global']
    nx, ny, nz = D['shape']
    tri = m['v'][m['f']]
    col = np.where((m['on_cut'] | m['in_band'])[:, None], rgb(E.CLAMP),
                   np.where(m['on_box'][:, None], rgb(E.BOXCAP), rgb(E.BAND)))
    parts = [(E.to_plot(tri), np.c_[E.shade(col, tri, amb=.55), np.ones(len(tri))])]
    p0, p1 = E.cut_polygon(n, b, (0, 0), (nx, ny))
    q = []
    for i in range(30):                                                  # support plane on the cut
        for j in range(3):
            a, c = i / 30, (i + 1) / 30; z0, z1 = -.2 + 1.4 * j / 3, -.2 + 1.4 * (j + 1) / 3
            Q = [np.r_[p0 + a * (p1 - p0), z0], np.r_[p0 + c * (p1 - p0), z0], np.r_[p0 + c * (p1 - p0), z1],
                 np.r_[p0 + a * (p1 - p0), z1]]
            q += [[Q[0], Q[1], Q[2]], [Q[0], Q[2], Q[3]]]
    q = np.asarray(q)
    parts.append((E.to_plot(q), E.rgba(E.SECTION, len(q), .35)))
    E.collection(ax, parts, lw=.04)
    t = (p1 - p0) / np.linalg.norm(p1 - p0); L_ = np.linalg.norm(p1 - p0)
    hs = [[np.r_[p0 + s * t, 1.2], np.r_[p0 + min(s + .8, L_) * t, 1.2 - (min(s + .8, L_) - s)]]
          for s in np.linspace(0, L_ - .1, 20)]
    E.lines(ax, hs, colors=E.CLAMP, linewidths=.5)
    E.lines(ax, [[np.r_[p0, z], np.r_[p1, z]] for z in (-.2, 1.2)] + [[np.r_[p, -.2], np.r_[p, 1.2]] for p in (p0, p1)],
            colors=E.CLAMP, linewidths=.8)
    E.setup(ax, (-.3, -.6, -.25), (nx + .3, ny + .3, nz + .3), elev=34, azim=-118, zoom=1.4)
    for x in (.5, 1.5, 2.5, 3.5):
        arrow2d(ax, E.to_plot((x - .5, -.35, .5)), E.to_plot((x + .5, -.35, .5)), color=FS.TEXT, lw=1.0, mutation_scale=7)


def box(ax, x, y, w, h, title, sub, kind):
    fc, ec = FILL[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2', fc=fc, ec=ec, lw=1.0, zorder=2))
    ax.text(x + w / 2, y + h * .66, title, ha='center', va='center', fontsize=7.5, fontweight='bold', color=FS.TEXT, zorder=3)
    ax.text(x + w / 2, y + h * .30, sub, ha='center', va='center', fontsize=7.5, color=FS.TEXT, zorder=3)


def main():
    D = E.design()
    fig = plt.figure(figsize=(W * FS.MM, H * FS.MM))
    a3 = fig.add_axes([.06, .16, .25, .64], projection='3d', computed_zorder=False)
    plate3d(a3, D)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    ax.text(2.5, 49.0, 'Cut TPMS lattice,\nclamped on its cut', fontsize=8, linespacing=1.1, fontweight='bold', color=FS.TEXT, va='top')
    ax.text(2.5, 2.5, 'per cell: box-face and cut-band\nDOFs retained, interior condensed', fontsize=6.8,
            color=FS.MUTED, va='bottom', linespacing=1.15)
    # middle: NICE division of tasks (Figure 2(a))
    x0, bw, gap, y, h = 39.0, 14.0, 2.8, 23.0, 12.0
    ax.text(x0 + 1.5 * bw + gap, 49.0, 'NICE: one condensed\noperator per cell', ha='center', va='top', fontsize=8,
            linespacing=1.1, fontweight='bold',
            color=FS.TEXT)
    spec = [('learned', 'extension', 'learn', 'learning:', 'trial field', LEARN),
            ('two-grid', 'correction', 'corr', 'correction:', 'improvability', CORR),
            ('energy', r'form $F^TKF$', 'var', 'variational\nform:', 'structure', VAR)]
    for i, (t1, t2, kind, r1, r2, col) in enumerate(spec):
        x = x0 + i * (bw + gap)
        box(ax, x, y, bw, h, t1, t2, kind)
        ax.plot([x, x, x + bw, x + bw], [y - 1.6, y - 2.4, y - 2.4, y - 1.6], color=col, lw=.8)
        ax.text(x + bw / 2, y - 3.4, f'{r1}\n{r2}', ha='center', va='top', fontsize=6.8, fontweight='bold', color=col,
                linespacing=1.15)
        if i < 2:
            ax.annotate('', (x + bw + gap - .2, y + h / 2), (x + bw + .2, y + h / 2),
                        arrowprops=dict(arrowstyle='-|>', color=FS.TEXT, lw=.9, mutation_scale=8, shrinkA=0, shrinkB=0))
    # right: headline
    xr = 91.0
    ax.plot([88.5, 88.5], [5, 47], color='#DFE5E9', lw=.8)
    ax.text(xr, 49.0, 'Clamped cut layer', fontsize=8, fontweight='bold', color=FS.TEXT, va='top')
    ax.text(xr, 41.5, 'homogenisation under-\nestimates compliance by', fontsize=7, color=FS.TEXT, va='top', linespacing=1.15)
    ax.text(xr + 18.0, 29.0, '27%', fontsize=14, fontweight='bold', color=HOM, va='center', ha='center')
    ax.text(xr, 21.0, 'NICE matches exact to', fontsize=7, color=FS.TEXT, va='top')
    ax.text(xr + 18.0, 12.5, '0.03%', fontsize=14, fontweight='bold', color=FS.C['corrected'], va='center', ha='center')
    ax.text(xr + 18.0, 5.0, '(gradient 0.3%)', fontsize=7, color=FS.TEXT, va='center', ha='center')
    out = FS.OUT
    for ext, kw in (('svg', {}), ('pdf', {}), ('png', dict(dpi=300))):
        fig.savefig(out / f'GA_nice.{ext}', **kw)
    plt.close(fig)


if __name__ == '__main__':
    main()
