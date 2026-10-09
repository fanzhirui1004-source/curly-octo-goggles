"""Scale-demonstration figure -> figures/F16_scale_plates.{svg,pdf,png} (oblique 3D views, equal panels; timing below)
(a-d) Plan views, all at the same scale, of the plates of the scale demonstration (Section 5.10, Table ST20): 24, 51, 88
      and 110 Schwarz-P cells at the uniform start tau = 0.40, one cell thick, short side : long side 1 : 2, trimmed by a
      planar cut that scales with the plate. Walls are drawn from the level set of Eq. (1) as seen from above (height-map
      shading); uncut cells light blue, cut cells dark blue, wall sections on the top face darker. The uncut long side
      x = min is clamped (hatched); the opposite face x = max, which the cut shortens, carries an in-plane traction of unit
      resultant along the long side (arrows). The part removed by the cut is drawn pale with a dashed outline.
(e)   Time per design iteration against the number of cells (mean and range over the timed iterations), with the
      line through the origin at the mean time per cell (the former Figure 15(c), same data and fit).
Data: evidence/opt/scale/runs/layouts/plateS{24,51,88,110}.json (cell positions, kinds, cut plane),
      review_r1/results/X6_final/scale/scale_summary.json (timings); DOF counts and times as in Table ST20.
Usage: python3 fig_scale_plates.py [--preview]
"""
import json
import sys
from pathlib import Path
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import figstyle as FS
import elev3d as E
import elev_prep_meshes as EP
plt = FS.plt
ELEV, AZIM, ZOOM = 55, -90, 1.25                                      # clamped side nearest, acute corner bottom right

HERE = Path(__file__).resolve().parent
EV = HERE.parent / 'evidence' / 'opt' / 'scale'                            # layouts of the four plates
DC = HERE.parent / 'review_r1' / 'results' / 'X6_final'                   # final-route timings (as fig_opt.py)
PLATES = [('plateS24', 'plateS24r4', 'a'), ('plateS51', 'plateS51r4', 'b'), ('plateS88', 'plateS88', 'c'),
          ('plateS110', 'plateS110', 'd')]
ST20 = {24: ('6.51 M', '0.388 M', 242), 51: ('14.57 M', '0.780 M', 547), 88: ('25.85 M', '1.305 M', 962),
        110: ('32.70 M', '1.618 M', 1164)}                         # Table ST20: cut-model DOFs, free retained DOFs, s
UNCUT, CUTC = FS.CELL['U1'][0], FS.CELL['M2'][0]
UNCUT_CAP, CUTC_CAP = '#5F86B3', '#1F4573'
TAU = 0.40                                                         # uniform start


def hex2rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0



def P2(P):
    """Plot coordinates (X, Y, Z) = (y, x, z): long side to the right, clamped side x = 0 nearest to the viewer. This is
    the mirror of elev3d.to_plot; the uniform-start plate is symmetric about its mid-plane z = 1/2, so the view equals a
    view of the plate from below."""
    P = np.asarray(P, float)
    return np.stack([P[..., 1], P[..., 0], P[..., 2]], -1)


def setup(ax, lo, hi):
    a, b = np.minimum(P2(lo), P2(hi)), np.maximum(P2(lo), P2(hi))
    ax.set_xlim(a[0], b[0]); ax.set_ylim(a[1], b[1]); ax.set_zlim(a[2], b[2])
    ax.set_box_aspect(tuple(b - a), zoom=ZOOM)
    ax.set_proj_type('ortho'); ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off(); ax.patch.set_alpha(0)


def lines(ax, segs, **kw):
    lc = Line3DCollection([P2(np.asarray(x)) for x in segs], **kw)
    lc.set_clip_on(False)
    ax.add_collection3d(lc)


def plate_mesh(lay, R=12, target=60000):
    """Marching-cubes surface of the plate at the uniform start (|phi| <= tau, kept side of the cut, layout cells only),
    with per-triangle flags: cut cell, on an outer face of the plate, on the cut plane."""
    nx, ny, nz = lay['shape']
    n, b = np.asarray(lay['normal'], float), float(lay['b_global'])
    xs, ys, zs = (np.linspace(0, m, m * R + 1) for m in (nx, ny, nz))
    h = xs[1] - xs[0]
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    G = np.abs(np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y) + np.cos(2 * np.pi * Z)) - TAU
    G = np.maximum(G, n[0] * X + n[1] * Y - b)
    have = np.zeros((nx, ny, nz), bool)
    for c in lay['cells']:
        have[tuple(c['position'])] = True
    ci = [np.clip(np.floor(A - 1e-12).astype(int), 0, m - 1) for A, m in ((X, nx), (Y, ny), (Z, nz))]
    cj = [np.clip(np.floor(A + 1e-12).astype(int), 0, m - 1) for A, m in ((X, nx), (Y, ny), (Z, nz))]
    G = np.where(have[ci[0], ci[1], ci[2]] | have[cj[0], cj[1], cj[2]], G, np.maximum(G, 1.0))
    v, f = EP.mc(G, h, (0, 0, 0))
    v = np.clip(v, 0, [nx, ny, nz])
    v, f = EP.decimate(v, f, target)
    tri = v[f]
    on_cut = np.all(np.abs(tri @ n - b) < 2e-3, axis=1)
    on_box = np.zeros(len(f), bool)
    for a, hi in enumerate((nx, ny, nz)):
        for w in (0, hi):
            on_box |= np.all(np.abs(tri[:, :, a] - w) < 2e-3, axis=1)
    cen = tri.mean(1)
    cut_cells = {tuple(c['position']) for c in lay['cells'] if c['kind'] != 'FULL'}
    kc = np.floor(np.clip(cen, 0, np.array([nx, ny, nz]) - 1e-9)).astype(int)
    is_cut = np.array([tuple(k) in cut_cells for k in kc])
    return tri, is_cut, on_box & ~on_cut, on_cut


def arrow2d(ax, p0, p1, zorder=30, **kw):
    M = ax.get_proj()
    x0, y0, _ = proj3d.proj_transform(*p0, M); x1, y1, _ = proj3d.proj_transform(*p1, M)
    ax.annotate('', (x1, y1), (x0, y0), xycoords='data', annotation_clip=False, zorder=zorder,
                arrowprops=dict(arrowstyle='-|>', shrinkA=0, shrinkB=0, **kw))


def text2d(ax, p, s, **kw):
    M = ax.get_proj()
    x, y, _ = proj3d.proj_transform(*p, M)
    ax.annotate(s, (x, y), xycoords='data', annotation_clip=False, **kw)


def draw_plate(ax, name, run, letter):
    lay = json.loads((EV / 'runs' / 'layouts' / f'{name}.json').read_text())
    nx, ny, nz = lay['shape']
    n, b = np.asarray(lay['normal'], float), float(lay['b_global'])
    ncell = len(lay['cells']); ncut = sum(c['kind'] != 'FULL' for c in lay['cells'])
    tri, is_cut, on_box, on_cut = plate_mesh(lay)
    rgb = np.where(is_cut[:, None], hex2rgb(CUTC), hex2rgb(UNCUT))
    rgb[on_box] = np.where(is_cut[on_box, None], hex2rgb(CUTC_CAP), hex2rgb(UNCUT_CAP))
    rgb[on_cut] = hex2rgb(E.SECTION)
    rgb = E.shade(rgb, tri, amb=.55)
    E.collection(ax, [(P2(tri), np.c_[rgb, np.ones(len(tri))])], lw=.03)
    # clamped long side x = 0 (nearest): ground hatching along its lower edge, pointing away from the plate
    hs = [[(0, y, 0), (-.5, y - .25, 0)] for y in np.linspace(.15, ny - .05, 3 * ny)]
    lines(ax, hs, colors=FS.MUTED, linewidths=.45)
    lines(ax, [[(0, 0, 0), (0, ny, 0)]], colors=FS.TEXT, linewidths=1.0)
    setup(ax, (-.7, -.2, -.2), (nx + .9, ny + .2, nz + .2))
    # load face x = nx (front), in-plane traction along +y: arrows in front of the face
    ytop = min(ny, (b - n[0] * nx) / n[1])
    for y in np.linspace(.2, ytop - .9, max(2, int(round(ytop * .8)))):
        arrow2d(ax, P2((nx + .55, y, .5)), P2((nx + .55, y + .75, .5)), color=FS.TEXT, lw=.7,
                mutation_scale=5, zorder=30)
    dofs, ret, secs = ST20[ncell]
    ax.set_title(f'({letter}) {ncell} cells ({ncut} cut)', loc='left', fontsize=8, fontweight='bold', pad=0, y=1.02)
    ax.text2D(0.04, 0.15, f'{dofs} DOFs, {ret} retained\n{secs / 60:.1f} min per design iteration',
              transform=ax.transAxes, fontsize=6.2, color=FS.MUTED, va='top', ha='left', linespacing=1.25)
    print(f'{name}: {len(tri)} triangles, {is_cut.mean():.2f} on cut cells')


def draw_timing(ax):
    """As the former Figure 15(c) (fig_opt.draw_scale): mean and range of the timed iterations, dashed line through the
    origin at the mean time per cell over the four plates."""
    S = json.loads((DC / 'scale' / 'scale_summary.json').read_text())
    runs = ['plateS24', 'plateS51', 'plateS88', 'plateS110']
    n = np.array([S[r]['cells'] for r in runs])
    t = [np.array(S[r]['iter_s']) / 60 for r in runs]
    tm = np.array([x.mean() for x in t])
    err = np.array([[m - x.min() for m, x in zip(tm, t)], [x.max() - m for m, x in zip(tm, t)]])
    spc = float(np.sum(tm * 60) / np.sum(n))
    ax.plot([0, 125], np.array([0, 125]) * spc / 60, color=FS.MUTED, lw=.6, ls='--', zorder=2)
    ax.errorbar(n, tm, yerr=err, color=FS.GREY[0], marker='s', ms=3.2, lw=.9, capsize=1.5, elinewidth=.6, zorder=4)
    ax.text(62, 62 * spc / 60 - 2.5, f'{spc:.1f} s per cell', fontsize=6, color=FS.MUTED, ha='left', va='top')
    ax.set_xlabel('cells'); ax.set_ylabel('time per design iteration (min)', fontsize=7)
    ax.set_xlim(0, 125); ax.set_ylim(0, 25); ax.set_xticks([0, 24, 51, 88, 110])
    ax.tick_params(labelsize=6.5)
    ax.grid(True, color=FS.GRID, lw=.4)
    FS.panel(ax, 'e', 'Time per iteration')
    print('timing (min)', np.round(tm, 2).tolist(), 's per cell', round(spc, 2))


def main():
    W, H = 178.0, 92.0
    fig = plt.figure(figsize=(W * FS.MM, H * FS.MM))
    w, h, gap, y0 = 42.5, 38.0, 2.0, 48.0
    for k, P in enumerate(PLATES):
        ax = fig.add_axes([(2.0 + k * (w + gap)) / W, y0 / H, w / W, h / H], projection='3d')
        draw_plate(ax, *P)
    draw_timing(fig.add_axes([16 / W, 9 / H, 158 / W, 26 / H]))
    if '--preview' in sys.argv:
        fig.savefig(HERE / '_preview_F16.png', dpi=220)
        print('preview written')
    else:
        FS.save(fig, 'F16_scale_plates')
        print('wrote figures/F16_scale_plates.{svg,pdf,png}')


if __name__ == '__main__':
    main()
