"""Shared 3D drawing helpers of Figures F00_problem and F14_plate_design3d (matplotlib only; meshes from data_elev/).

Plot coordinates: (X, Y, Z) = (y, -x, z) of the internal lattice coordinates (right-handed): the long side y runs to the
right, the short side x towards the viewer, z up.  All surfaces of a panel go into ONE Poly3DCollection so that matplotlib
sorts the triangles individually (separate collections are sorted as wholes).
"""
from pathlib import Path
import json
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import figstyle as FS
plt = FS.plt

DATA = Path(__file__).resolve().parent / 'data_elev'
BAND, SECTION, BOXCAP, GHOST = '#7FA6C4', '#C9B48A', '#3F6E96', '#B8BEC6'   # as Figure F08_geometry
CLAMP = '#8C6A2F'                                                             # dark sand: clamped cut band / its nodes
LIGHT = np.array([-.35, -.55, .76])


def to_plot(P):
    P = np.asarray(P, float)
    return np.stack([P[..., 1], -P[..., 0], P[..., 2]], -1)


def lambert(tri):
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30
    l_ = LIGHT / np.linalg.norm(LIGHT)
    return np.abs(nrm @ l_)


def shade(rgb, tri, amb=.42):
    s = (amb + (1 - amb) * lambert(tri))[:, None]
    return np.clip(rgb * s + .10 * (1 - s), 0, 1)


def rgba(c, n=1, a=1.0):
    r = np.array(plt.matplotlib.colors.to_rgba(c, a))
    return np.tile(r, (n, 1))


def load(name):
    z = np.load(DATA / name)
    return {k: z[k] for k in z.files}


def design():
    return json.loads((DATA / 'plate_design.json').read_text())


def collection(ax, parts, lw=.05):
    """parts: list of (triangles (n, 3, 3) in plot coordinates, rgba (n, 4)); one sorted collection, rasterised."""
    tri = np.concatenate([p[0] for p in parts]); col = np.concatenate([p[1] for p in parts])
    c = Poly3DCollection(tri, facecolors=col, edgecolors=col, linewidths=lw, zsort='average')
    c.set_rasterized(True); c.set_clip_on(False)                      # 3D axes are square: no clipping to the box
    ax.add_collection3d(c)
    return c


def lines(ax, segs, **kw):
    lc = Line3DCollection([to_plot(np.asarray(s)) for s in segs], **kw)
    lc.set_clip_on(False)
    ax.add_collection3d(lc)
    return lc


def setup(ax, lo, hi, elev=32, azim=-122, zoom=1.0):
    lo, hi = to_plot(np.asarray(lo, float)), to_plot(np.asarray(hi, float))
    a, b = np.minimum(lo, hi), np.maximum(lo, hi)
    ax.set_xlim(a[0], b[0]); ax.set_ylim(a[1], b[1]); ax.set_zlim(a[2], b[2])
    ax.set_box_aspect(tuple(b - a), zoom=zoom)
    ax.set_proj_type('ortho'); ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.patch.set_alpha(0)


def box_edges(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    out = []
    for a in np.ndindex(2, 2, 2):
        for b in np.ndindex(2, 2, 2):
            if a < b and sum(x != y for x, y in zip(a, b)) == 1:
                out.append([lo + (hi - lo) * np.array(a), lo + (hi - lo) * np.array(b)])
    return out


def cut_polygon(n, b, lo, hi):
    """Corners of the plane n . x = b (vertical: n_z = 0) inside the box [lo, hi], as a quad (bottom-top order)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    pts = []
    for x in (lo[0], hi[0]):                                             # intersections with the vertical box sides
        y = (b - n[0] * x) / n[1]
        if lo[1] - 1e-9 <= y <= hi[1] + 1e-9:
            pts.append((x, y))
    for y in (lo[1], hi[1]):
        x = (b - n[1] * y) / n[0]
        if lo[0] - 1e-9 <= x <= hi[0] + 1e-9:
            pts.append((x, y))
    pts = sorted(set((round(p[0], 12), round(p[1], 12)) for p in pts))
    p0, p1 = np.array(pts[0]), np.array(pts[-1])
    return p0, p1
