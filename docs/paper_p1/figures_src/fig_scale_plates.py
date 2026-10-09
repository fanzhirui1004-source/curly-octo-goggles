"""Scale-demonstration figure -> figures/F16_scale_plates.{svg,pdf,png}
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
import figstyle as FS
plt = FS.plt

HERE = Path(__file__).resolve().parent
EV = HERE.parent / 'evidence' / 'opt' / 'scale'                            # layouts of the four plates
DC = HERE.parent / 'review_r1' / 'results' / 'X6_final'                   # final-route timings (as fig_opt.py)
PLATES = [('plateS24', 'plateS24r4', 'a'), ('plateS51', 'plateS51r4', 'b'), ('plateS88', 'plateS88', 'c'),
          ('plateS110', 'plateS110', 'd')]
ST20 = {24: ('6.51 M', '0.388 M', 242), 51: ('14.57 M', '0.780 M', 547), 88: ('25.85 M', '1.305 M', 962),
        110: ('32.70 M', '1.618 M', 1164)}                         # Table ST20: cut-model DOFs, free retained DOFs, s
UNCUT, CUTC = FS.CELL['U1'][0], FS.CELL['M2'][0]
UNCUT_CAP, CUTC_CAP = '#5F86B3', '#1F4573'
RES, NZ, TAU = 40, 40, 0.40                                        # pixels per cell, z samples, uniform start


def hex2rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0


def clip(poly, f):
    """Part of a convex polygon where f(P) >= 0 (one Sutherland-Hodgman pass)."""
    out = []
    for k in range(len(poly)):
        P, Q = np.asarray(poly[k], float), np.asarray(poly[(k + 1) % len(poly)], float)
        fp, fq = f(P), f(Q)
        if fp >= 0:
            out.append(tuple(P))
        if (fp >= 0) != (fq >= 0):
            t = fp / (fp - fq)
            out.append(tuple(P + t * (Q - P)))
    return out


def line_in_rect(n, b, nx, ny):
    """End points (plot coordinates X = y, Y = x) of the cut line n . x = b inside the plate rectangle."""
    pts = []
    for Y in (0.0, float(nx)):
        X = (b - n[0] * Y) / n[1]
        if 0 <= X <= ny:
            pts.append((X, Y))
    for X in (0.0, float(ny)):
        Y = (b - n[1] * X) / n[0]
        if 0 < Y < nx:
            pts.append((X, Y))
    return pts[:2]


def render(lay):
    """RGBA plan image (rows = x, columns = y) of the plate walls seen from +z."""
    nx, ny, _ = lay['shape']
    n = np.asarray(lay['normal'], float); b = float(lay['b_global'])
    kind = {(c['position'][0], c['position'][1]): c['kind'] for c in lay['cells']}
    gx = (np.arange(nx * RES) + .5) / RES
    gy = (np.arange(ny * RES) + .5) / RES
    z = (np.arange(NZ) + .5) / NZ
    X, Y = np.meshgrid(gx, gy, indexing='ij')
    cxy = np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y)
    phi = cxy[..., None] + np.cos(2 * np.pi * z)[None, None, :]
    mat = np.abs(phi) <= TAU
    keep = (n[0] * X + n[1] * Y <= b)
    ci, cj = np.floor(X).astype(int), np.floor(Y).astype(int)
    present = np.zeros_like(keep)
    cutk = np.zeros_like(keep)
    for (i, j), k in kind.items():
        m = (ci == i) & (cj == j)
        present |= m
        if k != 'FULL':
            cutk |= m
    mat &= (keep & present)[..., None]
    has = mat.any(-1)
    top = np.where(has, (NZ - 1 - np.argmax(mat[..., ::-1], axis=-1)) / (NZ - 1), 0.0)
    gyh, gxh = np.gradient(top)                                      # hill shading from the height map
    light = np.clip(.78 + 2.2 * (-.55 * gxh - .35 * gyh), .45, 1.12)
    shade = (.62 + .38 * top) * light
    rgb = np.ones(top.shape + (3,))
    base = np.where(cutk[..., None], hex2rgb(CUTC), hex2rgb(UNCUT))
    cap = np.where(cutk[..., None], hex2rgb(CUTC_CAP), hex2rgb(UNCUT_CAP))
    capm = has & (top >= 1 - 1e-9)
    col = np.clip(base * shade[..., None], 0, 1)
    col[capm] = cap[capm]
    rgb[has] = col[has]
    alpha = has.astype(float)
    return np.dstack([rgb, alpha]), nx, ny, n, b


def draw_plate(ax, name, run, letter):
    lay = json.loads((EV / 'runs' / 'layouts' / f'{name}.json').read_text())
    img, nx, ny, n, b = render(lay)
    ncell = len(lay['cells']); ncut = sum(c['kind'] != 'FULL' for c in lay['cells'])
    # removed part of the nominal rectangle (pale, dashed outline); plot X = y (long side), plot Y = x (short side)
    rect = [(0, 0), (ny, 0), (ny, nx), (0, nx)]
    removed = clip(rect, lambda P: n[1] * P[0] + n[0] * P[1] - b)         # where n . x > b
    if removed:
        ax.add_patch(Polygon(removed, closed=True, fc='#F1F3F5', ec='none', zorder=0))
    ax.add_patch(Rectangle((0, 0), ny, nx, fc='none', ec='#AEB6BF', lw=.6, ls=(0, (3, 2)), zorder=1))
    ax.imshow(img, extent=(0, ny, 0, nx), origin='lower', interpolation='bilinear', zorder=2)
    for k in range(1, ny):                                             # faint cell grid
        ax.plot([k, k], [0, nx], color='#C9D0D7', lw=.25, zorder=1)
    for k in range(1, nx):
        ax.plot([0, ny], [k, k], color='#C9D0D7', lw=.25, zorder=1)
    seg = line_in_rect(n, b, nx, ny)                                   # cut line
    ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color='#8C6A2F', lw=.9, zorder=4)
    ycut_top = min(ny, (b - n[0] * nx) / n[1])                         # end of the load face (x = nx)
    # clamped long side x = 0 (bottom), hatched below
    ax.plot([0, ny], [0, 0], color=FS.TEXT, lw=1.4, zorder=5, solid_capstyle='butt')
    for t in np.arange(.15, ny, .35):
        ax.plot([t, t - .22], [0, -.32], color=FS.MUTED, lw=.45, zorder=5)
    # load face x = nx (top), in-plane traction along +y
    ax.plot([0, ycut_top], [nx, nx], color=FS.TEXT, lw=1.0, zorder=5)
    for t in np.linspace(.15, ycut_top - .65, max(2, int(round(ycut_top)))):
        ax.annotate('', (t + .55, nx + .32), (t, nx + .32), arrowprops=dict(arrowstyle='-|>', color=FS.TEXT, lw=.6,
                    mutation_scale=5, shrinkA=0, shrinkB=0), zorder=6)
    ax.set_xlim(-.3, ny + .3); ax.set_ylim(-.6, nx + .9); ax.set_aspect('equal'); ax.axis('off')
    dofs, ret, secs = ST20[ncell]
    ax.text(-.3, nx + 1.05, f'({letter}) {ncell} cells ({ncut} cut)', fontsize=8, fontweight='bold', color=FS.TEXT,
            va='bottom', ha='left')
    ax.text(-.3, -.75, f'{dofs} DOFs, {ret} free retained\n{secs / 60:.1f} min per design iteration', fontsize=6.3,
            color=FS.MUTED, va='top', ha='left', linespacing=1.25)


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
    W, H, s = 178.0, 118.0, 4.8                                         # mm; s = mm per cell, the same for all plates
    fig = plt.figure(figsize=(W * FS.MM, H * FS.MM))

    def plate_axes(x0, y0, nx, ny):                                    # axes in mm, data limits fixed in draw_plate
        return fig.add_axes([x0 / W, y0 / H, (ny + .6) * s / W, (nx + 1.5) * s / H])
    draw_plate(plate_axes(1.5, 82.0, 4, 8), *PLATES[0])
    draw_plate(plate_axes(52.0, 72.5, 6, 12), *PLATES[1])
    draw_plate(plate_axes(1.5, 10.0, 8, 16), *PLATES[2])
    draw_plate(plate_axes(84.5, 10.0, 9, 18), *PLATES[3])
    draw_timing(fig.add_axes([131 / W, 79 / H, 42 / W, 32 / H]))
    if '--preview' in sys.argv:
        fig.savefig(HERE / '_preview_F16.png', dpi=220)
        print('preview written')
    else:
        FS.save(fig, 'F16_scale_plates')
        print('wrote figures/F16_scale_plates.{svg,pdf,png}')


if __name__ == '__main__':
    main()
