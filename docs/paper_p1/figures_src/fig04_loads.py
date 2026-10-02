"""Figure 4: supports and loading of the two-cell assembly examples (schematic, no data). Two oblique projections of
the neighbour cell N (exact condensation, grey) and the target cell T (learned, light blue):
(a) configuration x: N translated by (-1,0,0), face x=-1 clamped, face tractions on the y=0 faces of both cells;
(b) configuration y: N translated by (0,-1,0), face y=-1 clamped, face tractions on the x=0 faces of both cells.
Each loaded face carries separate x-, y- and z-directed consistent tractions (three arrows); the shared box face
(dashed) carries the coincident box-node DOFs shared across the interface. Writes F09_assembly_loads.*"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from figstyle import C, MUTED, TEXT, MM, panel, save

A, B = .5, .33                                               # oblique projection: depth (+y) runs to the upper left
FILL = {'N': '#E3E7EB', 'T': '#D6E7F4'}                     # exact neighbour: grey; learned target: light blue
LOAD, CLAMP, SHARED, ARROW = '#FFF0CC', '#C5CDD4', MUTED, TEXT      # neutral schematic: no variant colours
plt.rcParams['hatch.linewidth'] = .5


def P(x, y, z):
    """Projection of a 3D point (cabinet-like oblique view, x to the right, z up, y into the depth)."""
    return np.array([x - A * y, z + B * y])


def face(x0, y0, which):
    """Corner coordinates of one visible face of the unit box [x0,x0+1] x [y0,y0+1] x [0,1]."""
    x1, y1 = x0 + 1, y0 + 1
    return {'front': [(x0, y0, 0), (x1, y0, 0), (x1, y0, 1), (x0, y0, 1)],      # y = y0
            'left': [(x0, y0, 0), (x0, y1, 0), (x0, y1, 1), (x0, y0, 1)],       # x = x0
            'top': [(x0, y0, 1), (x1, y0, 1), (x1, y1, 1), (x0, y1, 1)]}[which]  # z = 1


def poly(ax, pts, **kw):
    ax.add_patch(Polygon([P(*p) for p in pts], closed=True, **kw))


def draw_face(ax, pts, cell, role=None):
    kw = dict(edgecolor=TEXT, lw=.7, joinstyle='round', zorder=2)
    if role == 'clamp':                                       # hatch takes the edge colour
        poly(ax, pts, facecolor=CLAMP, hatch='////', edgecolor=MUTED, lw=0, zorder=2)
        poly(ax, pts, facecolor='none', **kw)
    else:
        poly(ax, pts, facecolor=LOAD if role == 'load' else FILL[cell], **kw)


def arrow(ax, base, vec, label, off=(0, 0), color=ARROW, lw=.8, size=6, fs=6.5):
    b, t = P(*base), P(*(np.add(base, vec)))
    ax.annotate('', xy=t, xytext=b, zorder=6,
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=size, shrinkA=0, shrinkB=0))
    d = (t - b) / np.linalg.norm(t - b)
    ax.text(*(t + .09 * d + np.asarray(off)), label, fontsize=fs, color=color, ha='center', va='center', zorder=7)


def cell_label(ax, centre3d, letter, word):
    cx, cy = P(*centre3d)
    ax.text(cx - .03, cy, letter, fontsize=8.5, fontweight='bold', ha='right', va='center', color=TEXT, zorder=8)
    ax.text(cx + .04, cy, word, fontsize=6.5, ha='left', va='center', color=TEXT, zorder=8)


def shared_face(ax, pts):
    poly(ax, pts, facecolor=SHARED, alpha=.12, edgecolor='none', zorder=4)
    poly(ax, pts, facecolor='none', edgecolor=SHARED, lw=1.0, ls=(0, (3, 2)), zorder=5)


def shared_label(ax, xy, anchor, ha):
    x, y = xy
    ax.text(x, y, 'shared box face', fontsize=7, color=SHARED, ha=ha, va='bottom', zorder=8)
    ax.text(x, y + .14, 'coincident box-node\nDOFs shared', fontsize=6, color=MUTED, ha=ha, va='bottom',
            linespacing=1.1, zorder=8)
    ax.plot([anchor[0], x if ha == 'left' else x], [anchor[1], y - .02], color=SHARED, lw=.6, zorder=5)


def triad(ax, origin, L=.3):
    for vec, lab, off in (((L, 0, 0), 'x', (0, 0)), ((0, L, 0), 'y', (0, 0)), ((0, 0, L), 'z', (0, 0))):
        arrow(ax, origin, vec, lab, off, color=MUTED, lw=.7, size=5, fs=7)


def config_x(ax):
    """(a) N at x in [-1,0], T at x in [0,1]; clamp x=-1; loads on the y=0 (front) faces."""
    draw_face(ax, face(0, 0, 'top'), 'T')                              # T: top and loaded front face
    draw_face(ax, face(0, 0, 'front'), 'T', 'load')
    draw_face(ax, face(-1, 0, 'top'), 'N')                             # N: top, clamped left face, loaded front face
    draw_face(ax, face(-1, 0, 'left'), 'N', 'clamp')
    draw_face(ax, face(-1, 0, 'front'), 'N', 'load')
    shared_face(ax, [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)])     # x = 0
    for x0 in (-1, 0):                                                 # three traction directions on each loaded face
        base, L = (x0 + .32, 0, .3), .3
        arrow(ax, base, (L, 0, 0), 'x')
        arrow(ax, base, (0, 0, L), 'z')
        arrow(ax, (base[0], -.8 * L, base[2]), (0, .8 * L, 0), 'y', off=(.03, 0))   # face normal, +y, ending on the face
    cell_label(ax, (-.5, .5, 1), 'N', 'exact')
    cell_label(ax, (.42, .5, 1), 'T', 'learned')
    ax.text(-1 - A - .08, .5 + B / 2, 'clamped\nx = −1', fontsize=7, ha='right', va='center', color=TEXT)
    ax.text(0, -.1, 'loaded faces, y = 0', fontsize=7, ha='center', va='top', color=TEXT)
    ax.text(0, -.27, 'separate x-, y-, z-tractions', fontsize=6, ha='center', va='top', color=MUTED)
    shared_label(ax, (-A, 1 + B + .17), P(0, 1, 1), 'center')
    triad(ax, (1.5, 0, .05))


def config_y(ax):
    """(b) N at y in [-1,0] (in front), T at y in [0,1]; clamp y=-1; loads on the x=0 (left) faces."""
    draw_face(ax, face(0, 0, 'top'), 'T')                              # T behind N: top and loaded left face
    draw_face(ax, face(0, 0, 'left'), 'T', 'load')
    draw_face(ax, face(0, -1, 'top'), 'N')                             # N: top, loaded left face, clamped front face
    draw_face(ax, face(0, -1, 'left'), 'N', 'load')
    draw_face(ax, face(0, -1, 'front'), 'N', 'clamp')
    shared_face(ax, [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)])     # y = 0 (hidden behind N: dashed)
    for base in ((0, -.84, .42), (0, .1, .45)):                        # N face, T face
        L = .26
        arrow(ax, (-.85 * L, base[1], base[2]), (.85 * L, 0, 0), 'x', off=(0, -.08))   # face normal, +x, ending on the face
        arrow(ax, base, (0, L, 0), 'y', off=(-.04, .05))
        arrow(ax, base, (0, 0, L), 'z')
    cell_label(ax, (.5, -.5, 1), 'N', 'exact')
    cell_label(ax, (.25, .5, 1), 'T', 'learned')
    ax.text(.5 + A, -B - .1, 'clamped, y = −1', fontsize=7, ha='center', va='top', color=TEXT)
    ax.text(-A - .1, .62, 'loaded faces, x = 0', fontsize=7, ha='right', va='center', color=TEXT)
    ax.text(-A - .1, .45, 'separate x-, y-, z-tractions', fontsize=6, ha='right', va='center', color=MUTED)
    shared_label(ax, (1.42, 1.1), P(1, 0, 1), 'left')
    triad(ax, (1.85, 0, -.3))


def main():
    fig = plt.figure(figsize=(178 * MM, 66 * MM))
    gs = fig.add_gridspec(1, 2, wspace=.04, left=.01, right=.99, top=.9, bottom=.08)
    W, H = 4.6, 2.55                                                   # identical scale in both panels
    for pos, draw, letter, title, (cx, cy) in ((gs[0], config_x, 'a', 'Configuration x', (-.1, .75)),
                                               (gs[1], config_y, 'b', 'Configuration y', (.5, .6))):
        ax = fig.add_subplot(pos)
        draw(ax)
        ax.set_xlim(cx - W / 2, cx + W / 2); ax.set_ylim(cy - H / 2, cy + H / 2)
        ax.set_aspect('equal'); ax.axis('off')
        panel(ax, letter, title)
    fig.text(.5, .0, 'Hatched: clamped face. Light fill: loaded faces, each with separate x-, y- and z-directed '
             'consistent tractions (arrows). Dashed: shared box face.', ha='center', va='bottom', fontsize=6.5, color=MUTED)
    save(fig, 'F09_assembly_loads')


if __name__ == '__main__':
    main()
