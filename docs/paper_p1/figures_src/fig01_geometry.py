"""Figure 1 -> figures/F08_geometry.{svg,pdf,png}: the validation cells U1, M1, H1 and H2 of Eq. (1), rendered by marching
cubes on a 129^3 grid of the implicit material function g(x) = max(|phi(x)| - tau(x), n.x - b_cut) (material: g <= 0),
light-shaded, with the cut-plane section (where n.x = b_cut is active) in sand; the same orthographic view for all cells.

Parameters.  The eight corner thickness parameters tau_c (corner index 4x + 2y + z) are archived in this repository:
docs/data/newmachine_20260924/gcell/gval.json, key 'tau_corners_P' of the P-twin of every G cell (U1 = fresh_val_2000_full,
M1 = fresh_val_2003_d1_v1, H1 = fresh_val_2005_d1_v0, H2 = fresh_val_2010_d0_v0); for U1, M1 and H1 they equal 'taus0' of
review_r1/results/X3/pairs/pair_<case>.json.  The cut normal n and offset b_cut of M1, H1 and H2 are NOT archived here: they
are the keys 'normal' and 'offset' of <packets>/<case>/FRESH_CONTEXT.json ('case' record) on the production machine.
Fill CUT below with these values (normal as three floats, offset as a float).  While any cut is missing the script does not
touch figures/F08_geometry.*: `--test-planes` renders a PREVIEW with placeholder planes (approximate, for layout only) to
figures_src/_preview_F08_geometry.png.
Checks printed: the remaining box volume of every cut against the figure percentages (valmeta.json: 0.6478, 0.08604,
0.02666) and the material fraction of the box face x = 0 of H2 against the archived face weight 0.1552463.
Usage: python3 fig01_geometry.py [--test-planes]
"""
import sys
from pathlib import Path
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import figstyle as FS
plt = FS.plt

N = 129                                                                 # grid positions per axis
CELLS = ['U1', 'M1', 'H1', 'H2']
TAU = {'U1': [0.404705721799, 0.186542452434, 0.259790447271, 0.350197316864, 0.269476167484, 0.28996898988, 0.313362990829, 0.248899627657],
       'M1': [0.358345310902] * 8,
       'H1': [0.427344527906, 0.430917143525, 0.39037314392, 0.466222363336, 0.374529786043, 0.45710116601, 0.375188930106, 0.444833544485],
       'H2': [0.278376827296, 0.273152959929, 0.42946932661, 0.424712724681, 0.549415003545, 0.545571157306, 0.268164855166, 0.21064664523]}
CUT = {'U1': None,                                                      # uncut
       'M1': ([0.8136594653387408, 0.5813417879910875, 0.0], 0.8177872088372765),   # FRESH_CONTEXT.json of fresh_val_2003_d1_v1
       'H1': ([0.74637223045224, 0.6655287323697966, 0.0], 0.292372729558999),      # fresh_val_2005_d1_v0
       'H2': ([0.9984316431248911, 0.05598440860570193, 0.0], 0.05459404962921295)} # fresh_val_2010_d0_v0
PCT = {'U1': '100%', 'M1': '64.8%', 'H1': '8.6%', 'H2': '2.67%'}       # remaining box volume, as in the figure
VOL = {'M1': 0.6478160496334634, 'H1': 0.08604416938804589, 'H2': 0.026660923780561247}   # evidence/valmeta.json 'vol'
BAND, SECTION, BOX = '#7FA6C4', '#C9B48A', '#9AA5B1'               # section: neutral sand (no vermillion / gold)
BOXCAP, GHOST = '#3F6E96', '#B8BEC6'                                # wall section on the box faces; part removed by the cut
BOX_EDGES = [(a, b) for a in np.ndindex(2, 2, 2) for b in np.ndindex(2, 2, 2) if a < b and sum(x != y for x, y in zip(a, b)) == 1]


def plane_from_volume(theta, v):
    """Offset b of the vertical plane (cos t, sin t, 0).x <= b retaining the box fraction v (bisection on the exact area)."""
    c, s = np.cos(theta), np.sin(theta)
    g = (np.arange(4000) + .5) / 4000
    X, Y = np.meshgrid(g, g, indexing='ij')
    area = lambda b: float((c * X + s * Y <= b).mean())
    lo, hi = 0.0, c + s
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if area(mid) < v else (lo, mid)
    return np.array([c, s, 0.0]), (lo + hi) / 2


# placeholder planes for --test-planes only (M1: plane fitted to the archived cut-band nodes of evidence/p1_field_M1_small.npz,
# theta = 35.56 deg, b = 0.8186; H1, H2: angles inside their generator strata with the archived retained volumes)
TEST = {'M1': (np.array([np.cos(np.radians(35.56)), np.sin(np.radians(35.56)), 0.0]), 0.8186),
        'H1': plane_from_volume(np.radians(30.0), VOL['H1']),
        'H2': plane_from_volume(np.radians(2.0), VOL['H2'])}


def tau_field(X, Y, Z, t):
    t = np.asarray(t, float)
    out = np.zeros_like(X)
    for c in range(8):
        bx, by, bz = (c >> 2) & 1, (c >> 1) & 1, c & 1
        out += t[c] * (bx * X + (1 - bx) * (1 - X)) * (by * Y + (1 - by) * (1 - Y)) * (bz * Z + (1 - bz) * (1 - Z))
    return out


def surface(t, cut, keep=+1):
    """Closed marching-cubes surface of the material domain (walls capped on the box faces by a padding layer outside the
    box); keep = -1 gives the part removed by the cut instead.  Returns the triangles and per-face flags of the cut-plane
    section and of the box-face section."""
    g = np.linspace(0, 1, N)
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    g1 = np.abs(np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y) + np.cos(2 * np.pi * Z)) - tau_field(X, Y, Z, t)
    if cut is None:
        G = g1
    else:
        n, b = cut
        G = np.maximum(g1, keep * (n[0] * X + n[1] * Y + n[2] * Z - b))
    h = g[1]
    Gp = np.pad(G, 1, constant_values=50.0)                             # outside the box: no material -> walls are capped
    verts, faces, _, _ = measure.marching_cubes(Gp, level=0.0, spacing=(h,) * 3)
    verts = np.clip(verts - h, 0.0, 1.0)                                # caps land on the box faces
    tri = verts[faces]; cen = tri.mean(1)
    if cut is None:
        is_cut = np.zeros(len(faces), bool)
    else:
        n, b = cut
        tc = tau_field(cen[:, 0], cen[:, 1], cen[:, 2], t)
        g1c = np.abs(np.cos(2 * np.pi * cen).sum(1)) - tc
        is_cut = keep * (cen @ n - b) > g1c
    on_box = np.zeros(len(faces), bool)
    for a in range(3):
        for v in (0.0, 1.0):
            on_box |= np.all(np.abs(tri[:, :, a] - v) < 1e-9, axis=1)
    return verts, faces, is_cut & ~on_box, on_box


def shaded(tri, base, light=(-.35, -.8, .55)):
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30
    l = np.asarray(light, float); l /= np.linalg.norm(l)
    shade = (.45 + .55 * np.abs(nrm @ l))[:, None]
    return np.clip(base * shade + .12 * (1 - shade), 0, 1)


def draw(ax, verts, faces, is_cut, on_box, ghost=None):
    rgb = lambda c: np.array(plt.matplotlib.colors.to_rgb(c))
    tri = verts[faces]
    if ghost is not None:                                              # part removed by the cut: faint, for context
        gv, gf = ghost
        gt = gv[gf]
        gc = np.concatenate([shaded(gt, rgb(GHOST)), np.full((len(gt), 1), .12)], 1)
        cg = Poly3DCollection(gt, facecolors=gc, edgecolors='none', linewidths=0, zsort='average')
        cg.set_rasterized(True)
        ax.add_collection3d(cg)
    base = np.where(is_cut[:, None], rgb(SECTION), np.where(on_box[:, None], rgb(BOXCAP), rgb(BAND)))
    cols = shaded(tri, base)
    coll = Poly3DCollection(tri, facecolors=cols, edgecolors=cols, linewidths=.08, zsort='average')
    coll.set_rasterized(True)
    ax.add_collection3d(coll)
    # unit box: all twelve edges; the three meeting at the corner hidden from the view (0, 0, 0) are dashed
    for e0, e1 in BOX_EDGES:
        hidden = (0, 0, 0) in (e0, e1)
        ax.plot(*zip(e0, e1), color=BOX, lw=.6, ls=(0, (2.5, 2)) if hidden else '-', zorder=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.set_proj_type('ortho'); ax.view_init(elev=24, azim=40)              # cut sections (normals towards +x, +y) in front
    ax.set_box_aspect((1, 1, 1), zoom=1.0); ax.set_axis_off()


def retained_fraction(cut):
    g = (np.arange(1200) + .5) / 1200
    X, Y, Z = np.meshgrid(g, g, g[::8], indexing='ij')                   # vertical planes: coarse in z
    n, b = cut
    return float((X * n[0] + Y * n[1] + Z * n[2] <= b).mean())


def main():
    test = '--test-planes' in sys.argv[1:]
    cuts = dict(CUT)
    missing = [c for c in CELLS if c != 'U1' and cuts[c] is None]
    if missing and not test:
        raise SystemExit(f"cut planes missing for {missing}: fill CUT from FRESH_CONTEXT.json, or run with --test-planes for a preview")
    if test:
        for c in missing:
            cuts[c] = TEST[c]
    for c in CELLS:
        if cuts[c] is not None:
            n, b = cuts[c]
            print(f"{c}: retained box volume {retained_fraction((np.asarray(n, float), float(b))):.4f} (figure: {PCT[c]}, valmeta {VOL[c]:.5f})")
    # archived face weight of H2 (x = 0 face fully retained): material fraction of that face from tau alone
    g = (np.arange(2000) + .5) / 2000
    Y, Z = np.meshgrid(g, g, indexing='ij')
    t = np.asarray(TAU['H2'])
    tau0 = tau_field(np.zeros_like(Y), Y, Z, t)
    print(f"H2: material fraction of the face x = 0: {(np.abs(1 + np.cos(2 * np.pi * Y) + np.cos(2 * np.pi * Z)) <= tau0).mean():.5f} (archived 0.15525)")

    fig = plt.figure(figsize=(178 * FS.MM, 62 * FS.MM))
    for i, c in enumerate(CELLS):
        ax = fig.add_subplot(1, 4, i + 1, projection='3d')
        cut = None if cuts[c] is None else (np.asarray(cuts[c][0], float), float(cuts[c][1]))
        verts, faces, is_cut, on_box = surface(TAU[c], cut)
        ghost = None
        if cut is not None:
            gv, gf, _, _ = surface(TAU[c], cut, keep=-1)
            ghost = (gv, gf)
        print(f"{c}: {len(faces)} triangles, {int(is_cut.sum())} on the cut section, {int(on_box.sum())} on the box faces")
        draw(ax, verts, faces, is_cut, on_box, ghost)
        ax.text2D(.5, 1.0, f'({"abcd"[i]}) {c}', transform=ax.transAxes, fontweight='bold', fontsize=8, ha='center', va='bottom')
        ax.text2D(.5, -.02, f'remaining box volume {PCT[c]}', transform=ax.transAxes, fontsize=7, ha='center', va='top')
    fig.legend(handles=[Patch(fc=BAND, ec='none', label='material surface'), Patch(fc=BOXCAP, ec='none', label='wall section on the box faces'),
                        Patch(fc=SECTION, ec='none', label='cut-plane section'), Patch(fc=GHOST, ec='none', alpha=.4, label='part removed by the cut')],
               loc='upper center', bbox_to_anchor=(.5, 1.0), ncol=4, frameon=False, fontsize=7, columnspacing=2.0)
    fig.subplots_adjust(left=.0, right=1.0, bottom=.1, top=.84, wspace=.0)
    if test:
        fig.text(.5, .5, 'PREVIEW: cut planes of M1, H1, H2 are placeholders', ha='center', va='center', fontsize=11,
                 color='#C0392B', alpha=.7, rotation=20, zorder=10)
        out = Path(__file__).resolve().parent / '_preview_F08_geometry.png'
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print('wrote', out)
    else:
        FS.save(fig, 'F08_geometry')
        print('wrote figures/F08_geometry.{svg,pdf,png}')


if __name__ == '__main__':
    main()
