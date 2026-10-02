"""Section 6.11 figures: thickness design optimisation.

Main Figure 12 -> figures/F13_optimisation.{svg,pdf,png}   (the prefix F12_ belongs to Figure 7, F12_field_error_M1)
  (a) case A (2x2x2 block): NICE and exact-condensation (twin) compliance per design iteration, exact compliance of the
      NICE designs at iterations 0, 12, 23 (open circles); lower strip: NICE compliance relative to the twin run at the same
      design iteration (line; the designs of the two runs differ) and to the exact compliance of the same design (circles);
      the perturbed designs of the geometry-generation fallback (iterations 16 and 19) are marked.
  (b) plate supported on its cut: NICE compliance per design iteration from the uniform start, the compliance of the
      homogenised macroscale model along its own optimisation, the homogenisation design evaluated with NICE, and the exact
      compliance of the checked designs (open circles); shading: V > V*; lower strip: the range marked by the bracket,
      enlarged.  The legends sit inside the panels.
Main Figure 13 -> figures/F14_designs_scale.{svg,pdf,png}
  (a,b) corner thickness parameters of the final NICE design and of the homogenisation design on the plate, drawn with the
      long side horizontal: internal y horizontal, internal x vertical; layer z = 0 (the z = 1 difference is printed);
      vertices in the removed region are corners of cut cells; clamp along the cut, in-plane traction on the end face.
  (c) scale demonstration (X6_opt/scale/scale_summary.json): time per design iteration (mean, range bars), peak CPU memory
      of the main process and peak device memory in use (nvidia-smi, includes the cuDSS factor) against the number of
      cells, for plates of 24, 51, 88 and 110 cells; 135 cells failed (K_PP factor on the GPU).
Supplementary Figure S06 -> figures/S06_homogenised_law.{svg,pdf,png}
  C11, C12, C44 and the material volume fraction (symbol V^H; rho is the recomputed residual of Eq. (18)) of the
  uniform-thickness cell against tau, with the cubic splines of homog_macro.Material.

Colours and markers: NICE keeps its house identity (figstyle.MODEL['A3']: vermillion, squares). The exact-condensation
twin is ink (figstyle.C['exact']) with marker 'x'; the homogenisation series use C['extra'] (purple, a colour no variant
uses) with markers no variant uses ('*', 'h'); the resource curves of the scale demonstration are neutral greys
(figstyle.GREY); the load face and Figure S06 are drawn in neutral colours.

Data only from review_r1/results: X6_opt/optA/optA, X6_opt/optA/optAx (histories, check_*.json), X6_opt/scale,
X6_opt/homog/homog_cells.json ('cells'), X6_opt/FACTS_6_11.json (case A, cross-checked against the histories); and
X6_cplate: cplateN (NICE run: history, meta 'vkeys'/'fixed', check_exact_*.json), HC_x (homogenised macroscale run),
hevalHC (homogenisation design evaluated with NICE, check_exact_*.json), plate841.json ('shape', 'normal', 'b_global',
'cells').
Usage: python3 fig_opt.py            Figures 12, 13 and S06
"""
import json
import sys
from pathlib import Path
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.transforms import blended_transform_factory
import figstyle as FS
plt = FS.plt

HERE = Path(__file__).resolve().parent
D = HERE.parent / 'review_r1' / 'results' / 'X6_opt'
DC = HERE.parent / 'review_r1' / 'results' / 'X6_final'          # final-route runs (case A NICE, plate)
SRC = HERE.parents[1] / 'data' / 'newmachine_20260924' / 'src_v2_wip'   # homog_macro.py (spline of the material law)
F = json.loads((D / 'FACTS_6_11.json').read_text())

NICE, NICE_MK = FS.MODEL['A3'][0], FS.MODEL['A3'][1]                    # NICE: vermillion squares, as in Figures 5-10
EXACT, EXACT_MK = FS.C['exact'], 'x'                                    # exact condensation: ink, no variant marker
HOM, HOM_MK, HMAC_MK = FS.C['extra'], '*', 'h'                          # homogenisation design / macroscale model
SHADE = '#EEF1F4'
TAU_LO, TAU_HI = 0.18, 0.69
CMAP = 'cividis'


def hist(p, root=D):
    return [json.loads(l) for l in open(root / p)]


def ctrace(H):
    return np.array([h['C'] for h in H])


def check(run, k):
    f = DC / run / f'check_exact_{k:03d}.json'
    return json.loads(f.read_text()) if f.exists() else None


# ------------------------------------------------------------------------------------------------ data
A, AX = hist('optA/history.jsonl', DC), hist('optA/optAx/history.jsonl')
CHK = {k: c for k in (0, 12, len(A) - 1) if (c := check('optA', k)) is not None}
CN, CH, CE = hist('cplateN/history.jsonl', DC), hist('HC_x/history.jsonl', DC), hist('hevalHC/history.jsonl', DC)
LAY = json.loads((DC / 'plate841.json').read_text())
CHN = {k: c for k in (0, len(CN) - 1) if (c := check('cplateN', k)) is not None}   # exact checks of the NICE run
CHH = check('hevalHC', 0)                                                          # exact check of the homogenisation design

# consistency with the fact sheet (the single source of numbers for case A) and with the records of the plate
FF = json.loads((DC / 'FACTS_FINAL.json').read_text())
assert np.allclose(ctrace(A), FF['A']['C_trace']) and np.allclose(ctrace(AX), F['A_exact_twin']['C_trace'])
for k, c in CHK.items():
    assert np.isclose(c['C_hat'], A[k]['C'], rtol=1e-12), k
assert [h['k'] for h in A] == list(range(len(A))) and [h['k'] for h in CN] == list(range(len(CN)))
for k, c in CHN.items():
    assert np.isclose(c['C_hat'], CN[k]['C'], rtol=1e-12), k
if CHH is not None:
    assert np.isclose(CHH['C_hat'], CE[0]['C'], rtol=1e-12)
assert np.allclose(np.load(DC / 'HC_x' / 'final_tv.npy'), CE[0]['tv'])           # the evaluated design is the macroscale result


# ------------------------------------------------------------------------------------------------ plate geometry
def clip_halfplane(poly, a, b, keep_le=True):
    """Sutherland-Hodgman: clip polygon (list of (X, Y)) by a . p <= b (or >= b)."""
    s = 1.0 if keep_le else -1.0
    inside = lambda p: s * (a[0] * p[0] + a[1] * p[1] - b) <= 1e-12
    out = []
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        ip, iq = inside(p), inside(q)
        if ip:
            out.append(p)
        if ip != iq:
            dp, dq = a[0] * p[0] + a[1] * p[1] - b, a[0] * q[0] + a[1] * q[1] - b
            t = dp / (dp - dq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def plate_geometry():
    """Plot coordinates: X = internal y (long side, 8 cells), Y = internal x (short side, 4 cells).  The cut n . x <= b
    is given in internal coordinates, so in plot coordinates a = (n_y, n_x)."""
    nx, ny, nz = LAY['shape']
    assert nz == 1
    n, b = LAY['normal'], LAY['b_global']
    a = (n[1], n[0])
    rect = [(0, 0), (ny, 0), (ny, nx), (0, nx)]
    kept = clip_halfplane(rect, a, b, True)
    removed = clip_halfplane(rect, a, b, False)
    cut = sorted({(round(p[0], 12), round(p[1], 12)) for p in kept if abs(a[0] * p[0] + a[1] * p[1] - b) < 1e-9})
    cut = [cut[0], cut[-1]]                                              # end points of the cut line on the plate
    cells = [(c['position'][1], c['position'][0], c['kind'], c['retained']) for c in LAY['cells']]   # (X0, Y0, kind, retained)
    return dict(W=ny, Hh=nx, kept=kept, removed=removed, cut=cut, cells=cells, a=a, b=b)


def vertex_field(tv, C):
    """Corner parameters of a plate design on the (x, y) vertex grid, layer z = 0; also the z = 0/1 difference."""
    meta = json.loads((DC / 'cplateN' / 'meta.json').read_text())
    vk, tv, fixed = np.array(meta['vkeys']), np.asarray(tv), np.array(meta['fixed'])
    nx, ny, _ = LAY['shape']
    T = np.full((nx + 1, ny + 1), np.nan)
    FX = np.zeros((nx + 1, ny + 1), bool)
    d = {tuple(k): v for k, v in zip(vk.tolist(), tv)}
    for (i, j, k), v, f in zip(vk.tolist(), tv, fixed):
        if k == 0:
            T[i, j] = v; FX[i, j] = f
    dz = max(abs(d[(i, j, 0)] - d[(i, j, 1)]) for (i, j, k) in d if k == 0)
    return T, FX, dz, C


def draw_plate(ax, G, T, FX, note):
    W, Hh = G['W'], G['Hh']
    norm = Normalize(TAU_LO, TAU_HI)
    # bilinear interpolation of the vertex field inside every existing cell (tau is trilinear in each cell, Eq. (1))
    r = 60
    X = (np.arange(W * r) + .5) / r; Y = (np.arange(Hh * r) + .5) / r
    XX, YY = np.meshgrid(X, Y)
    j0 = np.clip(np.floor(XX).astype(int), 0, W - 1); i0 = np.clip(np.floor(YY).astype(int), 0, Hh - 1)
    fx, fy = XX - j0, YY - i0
    V = (T[i0, j0] * (1 - fx) * (1 - fy) + T[i0, j0 + 1] * fx * (1 - fy) + T[i0 + 1, j0] * (1 - fx) * fy
         + T[i0 + 1, j0 + 1] * fx * fy)
    im = ax.imshow(np.ma.masked_invalid(V), origin='lower', extent=(0, W, 0, Hh), cmap=CMAP, norm=norm,
                   interpolation='bilinear', zorder=1)
    kept = Polygon(G['kept'], closed=True, fc='none', ec='none')
    ax.add_patch(kept); im.set_clip_path(kept)
    ax.add_patch(Polygon(G['removed'], closed=True, fc='#F3F5F7', ec='none', zorder=0.5))
    for X0, Y0, kind, ret in G['cells']:
        ax.add_patch(Rectangle((X0, Y0), 1, 1, fc='none', ec='white' if kind == 'FULL' else '#B8C2CC', lw=.35, zorder=2))
    ax.add_patch(Polygon(G['kept'], closed=True, fc='none', ec=FS.TEXT, lw=.8, zorder=3))
    # vertices coloured by tau; fixed load-face vertices as squares
    I, J = np.nonzero(np.isfinite(T))
    for mk, sel in (('o', ~FX[I, J]), ('s', FX[I, J])):
        ax.scatter(J[sel], I[sel], c=T[I[sel], J[sel]], cmap=CMAP, norm=norm, s=15 if mk == 'o' else 13, marker=mk,
                   edgecolors=FS.TEXT, linewidths=.45, zorder=5, clip_on=False)
    # clamp along the cut: every cut-band DOF fixed; hatching on the removed side of the cut line
    (x1, y1), (x2, y2) = G['cut']
    ax.plot([x1, x2], [y1, y2], color=FS.TEXT, lw=1.4, zorder=4, solid_capstyle='butt')
    a = np.asarray(G['a']); t_ = np.array([x2 - x1, y2 - y1]); L = np.hypot(*t_); t_ /= L
    for s_ in np.linspace(.12, L - .05, 30):
        p0 = np.array([x1, y1]) + s_ * t_
        p1 = p0 + .2 * a - .14 * t_
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=FS.MUTED, lw=.5, zorder=4)
    ax.text(W - .15, Hh - .3, 'clamped\ncut band', ha='right', va='top', fontsize=6.5, color=FS.MUTED, zorder=6,
            linespacing=1.0)
    # load face: the end face internal y = min (plot X = 0); in-plane traction in internal +x (plot +Y)
    ax.plot([0, 0], [0, Hh], color=FS.TEXT, lw=2.4, zorder=4, solid_capstyle='butt')
    for ys in np.arange(.5, Hh, 1.0):
        ax.annotate('', (-.28, ys + .36), (-.28, ys - .36), arrowprops=dict(arrowstyle='-|>', color=FS.TEXT, lw=.8,
                    mutation_scale=6, shrinkA=0, shrinkB=0), annotation_clip=False)
    ax.text(-.55, Hh / 2, 'load face, in-plane traction', ha='center', va='center', rotation=90, fontsize=6.5,
            color=FS.TEXT)
    ax.text(W / 2, -.3, note, ha='center', va='top', fontsize=6.5, color=FS.TEXT)
    ax.set_xlim(-.85, W + .15); ax.set_ylim(-.85, Hh + .25); ax.set_aspect('equal'); ax.axis('off')
    return im


def removed_vertices(G, T, FX):
    """Vertices strictly inside the removed half-plane (corners of cut cells only), for the caption."""
    a, b = G['a'], G['b']
    I, J = np.nonzero(np.isfinite(T))
    return [(int(j), int(i), round(float(T[i, j]), 3), bool(FX[i, j])) for i, j in zip(I, J) if a[0] * j + a[1] * i > b + 1e-9]


def fmt_thousands(x, nd):
    return f'{x:,.{nd}f}'


def zoom_bracket(ax, lo, hi):
    """Thin solid bracket just right of the axes marking the range [lo, hi] enlarged in the strip below."""
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    ax.plot([1.005, 1.03, 1.03, 1.005], [lo, lo, hi, hi], color=FS.TEXT, lw=.7, transform=tr, clip_on=False,
            solid_joinstyle='miter', zorder=6)


# ------------------------------------------------------------------------------------------------ panel drawing
def legend_handles():
    """Legend entries of the histories; 'a' belongs to case A, 'b' to the plates (in-panel legends of the split figure)."""
    L2 = plt.Line2D
    a = [L2([], [], color=NICE, marker=NICE_MK, ms=3, lw=1.0, label='NICE'),
         L2([], [], color=EXACT, ls='--', marker=EXACT_MK, ms=4, mew=.8, lw=.8, label='exact condensation (twin)'),
         L2([], [], ls='none', marker='o', ms=6, mfc='none', mec=EXACT, mew=1.0, label='exact $C$, NICE design')]
    b = [L2([], [], color=NICE, marker=NICE_MK, ms=3, lw=1.0, label='NICE'),
         L2([], [], color=HOM, ls='--', marker=HMAC_MK, ms=3, lw=.8, label='homogenised model'),
         L2([], [], ls='none', marker=HOM_MK, ms=7.5, mfc=HOM, mec='white', mew=.5, label='homogenisation design, NICE'),
         L2([], [], ls='none', marker='o', ms=6, mfc='none', mec=EXACT, mew=1.0, label='exact $C$ of the design'),
         Patch(fc=SHADE, ec='none', label=r'$V>V^*$'),
         L2([], [], ls='none', marker=r'$]$', ms=6, color=FS.TEXT, label='range enlarged below')]
    return a, b


def draw_caseA(fig, gs_up, gs_lo, legend=False):
    """(a) case A: NICE and exact-twin compliance histories with the relative-difference strip."""
    ax, axs = fig.add_subplot(gs_up), fig.add_subplot(gs_lo)
    kA, kX = np.array([h['k'] for h in A]), np.array([h['k'] for h in AX])
    CA, CX = ctrace(A), ctrace(AX)
    vA = np.array([h['V_rel'] for h in A])
    k_on = int(np.argmax(vA <= 1.0 + 1e-3))                               # first iteration at V* (volume phase before it)
    for a_ in (ax, axs):
        a_.axvspan(0, k_on, color=SHADE, lw=0, zorder=0)
    ax.text(.75, 23.3, r'$V>V^*$', ha='left', va='bottom', fontsize=6.3, color=FS.MUTED)
    # twin underneath (thin dashed line, open crosses larger than the NICE squares), NICE on top
    ax.plot(kX, CX, '--', color=EXACT, lw=.8, marker=EXACT_MK, ms=5.4, mew=.75, zorder=3)
    ax.plot(kA, CA, '-', color=NICE, lw=.9, marker=NICE_MK, ms=2.1, zorder=4)
    kc = sorted(CHK)
    ax.plot(kc, [CHK[k]['C_exact'] for k in kc], ls='none', marker='o', ms=6.5, mfc='none', mec=EXACT, mew=1.0, zorder=5)
    ax.set_ylabel('compliance $C$'); ax.set_ylim(23, 42.5 if legend else 39.5)
    FS.panel(ax, 'a', 'Case A: 2×2×2 block')
    if legend:
        ax.legend(handles=legend_handles()[0], loc='upper right', bbox_to_anchor=(1.03, 1.0), frameon=False, fontsize=6,
                  handlelength=1.8, handletextpad=.5, labelspacing=.35, borderaxespad=.1)
    n = min(len(CA), len(CX))
    rel_twin = 100 * (CA[:n] / CX[:n] - 1)
    rel_chk = [100 * (CHK[k]['C_hat'] / CHK[k]['C_exact'] - 1) for k in kc]
    axs.axhline(0, color=FS.GRID, lw=.8, zorder=1)
    axs.plot(kA[:n], rel_twin, '-', color=EXACT, lw=.8, marker=EXACT_MK, ms=3.0, mew=.7, zorder=3)
    axs.plot(kc, rel_chk, ls='none', marker='o', ms=5, mfc='none', mec=EXACT, mew=.9, zorder=4)
    pert = [p['k'] for p in FF['A']['perturbations']]
    for kp in pert:
        axs.annotate('', (kp, rel_twin[kp]), (12.3, -.083), arrowprops=dict(arrowstyle='-', color=FS.MUTED, lw=.5,
                     shrinkA=0, shrinkB=2.5))
    axs.text(12.2, -.083, 'perturbed\ndesigns', ha='right', va='center', fontsize=6, color=FS.MUTED, linespacing=1.0)
    axs.set_ylim(-.12, .015); axs.set_yticks([-.1, -.05, 0])
    axs.set_ylabel('NICE vs exact (%)', fontsize=7)
    axs.set_xlabel('design iteration')
    plt.setp(ax.get_xticklabels(), visible=False)
    for a_ in (ax, axs):
        a_.grid(True, color=FS.GRID, lw=.4); a_.set_xlim(-.8, kA[-1] + .8)
    print(f"A: NICE/twin rel diff (%) {np.round(rel_twin, 4).tolist()}; checks {np.round(rel_chk, 4).tolist()}; volume phase k < {k_on}")
    for kp in pert:
        print(f"A: perturbed design k={kp}: NICE vs twin {rel_twin[kp]:.4f}%; V_rel NICE/twin {A[kp]['V_rel']:.6f}/{AX[kp]['V_rel']:.6f}")
    return ax, axs


def draw_cplate(fig, gs_up, gs_lo, legend=False, zoom=(77.4, 81.4)):
    """(b) plate supported on its cut: NICE history, homogenised-model history, homogenisation design evaluated with NICE,
    exact compliance of the checked designs; lower strip: the bracketed range enlarged."""
    ax, axz = fig.add_subplot(gs_up), fig.add_subplot(gs_lo)
    cn, ch = ctrace(CN), ctrace(CH)
    kn, kh = np.arange(len(cn)), np.arange(len(ch))
    vn = np.array([h['V_rel'] for h in CN])
    kon = int(np.argmax(vn <= 1.0 + 1e-2))
    k_h = kh[-1]                                                         # the homogenisation design: last macroscale iterate
    tail = kn >= 15                                                       # the lower strip shows the approach to the end
    for a_, sel in ((ax, slice(None)), (axz, tail)):
        a_.axvspan(0, kon, color=SHADE, lw=0, zorder=0)
        a_.plot(kh, ch, '--', color=HOM, lw=.8, marker=HMAC_MK, ms=2.6, zorder=2)
        a_.plot(kn[sel], cn[sel], '-', color=NICE, lw=1.0, marker=NICE_MK, ms=2.4, zorder=3)
        a_.plot([k_h], [CE[0]['C']], ls='none', marker=HOM_MK, ms=8, mfc=HOM, mec='white', mew=.5, zorder=5)
        ex = [(k, c['C_exact']) for k, c in CHN.items()] + ([(k_h, CHH['C_exact'])] if CHH is not None else [])
        a_.plot([e[0] for e in ex], [e[1] for e in ex], ls='none', marker='o', ms=6.5, mfc='none', mec=EXACT, mew=1.0,
                zorder=6)
        a_.grid(True, color=FS.GRID, lw=.4); a_.set_xlim(-.8, len(cn) - .2)
    ax.set_ylabel('compliance $C$'); ax.set_ylim(48, 128 if legend else 116)
    axz.set_ylabel('$C$ (enlarged)', fontsize=7); axz.set_ylim(*zoom)
    axz.yaxis.set_major_locator(plt.MultipleLocator(1))
    axz.set_xlabel('design iteration')
    zoom_bracket(ax, *zoom)
    plt.setp(ax.get_xticklabels(), visible=False)
    FS.panel(ax, 'b', 'Plate supported on its cut')
    if legend:
        ax.legend(handles=legend_handles()[1], loc='upper right', bbox_to_anchor=(1.03, 1.0), frameon=False,
                  fontsize=6, handlelength=1.9, handletextpad=.5, labelspacing=.3, borderaxespad=.1)
    print(f"plate: NICE C0 {cn[0]:.4f} -> {cn[-1]:.4f} (k={kn[-1]}), peak {cn.max():.3f} at k={int(cn.argmax())}; "
          f"macroscale {ch[0]:.4f} -> {ch[-1]:.4f} (k={kh[-1]}); Hom design with NICE {CE[0]['C']:.4f}, V/V* {CE[0]['V'] / CN[-1]['V'] * CN[-1]['V_rel']:.5f}; "
          f"exact NICE run {dict((k, round(c['C_exact'], 5)) for k, c in CHN.items())}; exact Hom "
          f"{None if CHH is None else round(CHH['C_exact'], 5)}; volume phase k < {kon}")
    return ax, axz


def draw_fields(fig, gs_c, gs_d, gs_cbar, letters=('a', 'b')):
    """Final NICE design and homogenisation design on the plate with a shared horizontal colour bar."""
    G = plate_geometry()
    runs = {'NICE': vertex_field(CN[-1]['tv'], CN[-1]['C']), 'Hom': vertex_field(CE[0]['tv'], CE[0]['C'])}
    for run, (T, FX, dz, C) in runs.items():
        print(f"{run}: z=0/z=1 max |diff| {dz:.2e}, NICE C {C:.3f}, tau range {np.nanmin(T):.4f}-{np.nanmax(T):.4f}; "
              f"vertices in the removed region (X, Y, tau, fixed): {removed_vertices(G, T, FX)}")
    axc, axd = fig.add_subplot(gs_c), fig.add_subplot(gs_d)
    T, FX, _, C = runs['NICE']
    ex = CHN.get(len(CN) - 1)
    im = draw_plate(axc, G, T, FX, f'NICE $C$ = {C:.2f}' + ('' if ex is None else f", exact $C$ = {ex['C_exact']:.2f}"))
    FS.panel(axc, letters[0], 'Final NICE design')
    T, FX, _, C = runs['Hom']
    draw_plate(axd, G, T, FX, f'NICE $C$ = {C:.2f}' + ('' if CHH is None else f", exact $C$ = {CHH['C_exact']:.2f}"))
    FS.panel(axd, letters[1], 'Homogenisation design')
    cax = fig.add_subplot(gs_cbar)
    pos = cax.get_position(); cax.set_position([pos.x0 + .18 * pos.width, pos.y0, .64 * pos.width, pos.height])
    cb_ = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[.18, .3, .4, .5, .6, .69])
    cb_.set_label(r'corner thickness parameter $\tau$, layer $z=0$ (circles: free, squares: fixed)', fontsize=7)
    cb_.ax.tick_params(labelsize=6.5); cb_.outline.set_linewidth(.5)
    return axc, axd


def draw_scale(fig, gs_e, letter='e', wide=False):
    """Scale demonstration: time and CPU/GPU memory per design iteration against the number of cells.  `wide`: the
    panel of the split figure (labels placed where nothing else is drawn)."""
    axe = fig.add_subplot(gs_e)
    S = json.loads((D / 'scale' / 'scale_summary.json').read_text())
    runs = ['plateS24r4', 'plateS51r4', 'plateS88', 'plateS110']
    n = np.array([S[r]['cells'] for r in runs])
    t = [np.array(S[r]['iter_s']) / 60 for r in runs]
    tm = np.array([x.mean() for x in t])
    err = np.array([[m - x.min() for m, x in zip(tm, t)], [x.max() - m for m, x in zip(tm, t)]])
    TIME, MEM = FS.GREY[0], FS.GREY[1]                                  # resource curves: neutral greys (layer 3)
    axe.errorbar(n, tm, yerr=err, color=TIME, marker=NICE_MK, ms=3.2, lw=.9, capsize=1.5, elinewidth=.6, zorder=4)
    axe.set_xlabel('cells'); axe.set_ylabel('time per design iteration (min)', color=TIME, fontsize=7)
    axe.tick_params(axis='y', colors=TIME)
    axe.set_xlim(0, 150); axe.set_ylim(0, 55); axe.set_xticks([0, 24, 51, 88, 110, 135])
    axe.tick_params(axis='x', labelsize=6)
    axm = axe.twinx()
    hp = np.array([S[r]['host_peak_gb'] for r in runs])                  # GiB (ru_maxrss / 2**20, main process)
    gp = np.array([S[r]['sampler_gpu_used_max_gib'] for r in runs])     # GiB in use on the device (nvidia-smi, 30-s samples)
    axm.plot(n, hp, '--', color=MEM, lw=.8, marker='o', ms=3.0, mfc='white', mew=.8, zorder=3)
    axm.plot(n, gp, ':', color=MEM, lw=.9, marker='^', ms=3.0, mfc=MEM, mew=.6, zorder=3)
    cap = 32607 / 1024
    axm.axhline(cap, color=MEM, lw=.5, ls='-', alpha=.5)
    axm.set_ylim(0, 110); axm.set_ylabel('peak memory (GiB)', color=MEM, fontsize=7)
    axm.tick_params(axis='y', colors=MEM, labelsize=6.5)
    axe.axvline(135, color=FS.MUTED, lw=.6, ls=':')
    if wide:
        axm.text(132.5, cap + 1.2, 'GPU capacity', fontsize=5.8, color=MEM, ha='right', va='bottom')
        axm.text(n[-1] + 3.5, hp[-1], 'CPU', fontsize=6, color=MEM, ha='left', va='center')
        axm.text(n[-1] + 3.5, gp[-1] - .5, 'GPU', fontsize=6, color=MEM, ha='left', va='top')
        axe.text(132.5, 11.0, 'out of GPU\nmemory at\n135 cells', ha='right', va='top', fontsize=5.8, color=FS.MUTED,
                 linespacing=1.05)
    else:
        axm.text(62, cap + 1.2, 'GPU capacity', fontsize=5.6, color=MEM, va='bottom')
        axm.text(n[-1] + 3, hp[-1], 'CPU', fontsize=5.8, color=MEM, ha='left', va='center')
        axm.text(n[-1] + 3, gp[-1] - 1, 'GPU', fontsize=5.8, color=MEM, ha='left', va='top')
        axe.text(142.5, 27.5, '135 cells: out of GPU memory', ha='center', va='center', rotation=90, fontsize=5.6,
                 color=FS.MUTED)
    axe.grid(True, color=FS.GRID, lw=.4)
    print('scale: cells', n.tolist(), 'time/iter (min)', np.round(tm, 2).tolist(), 'CPU peak GiB', np.round(hp, 1).tolist(), 'GPU in use GiB', np.round(gp, 1).tolist())
    FS.panel(axe, letter, 'Scale demonstration')
    return axe


# ------------------------------------------------------------------------------------------------ Figure 12
def fig_main():
    """Figure 12: (a) case A and (b) the plate supported on its cut, each with its lower strip, legends inside."""
    fig = plt.figure(figsize=(178 * FS.MM, 84 * FS.MM))
    top = fig.add_gridspec(2, 2, height_ratios=[2.3, 1], hspace=.10, wspace=.22, top=.925, bottom=.105, left=.065, right=.975)
    draw_caseA(fig, top[0, 0], top[1, 0], legend=True)
    draw_cplate(fig, top[0, 1], top[1, 1], legend=True)
    FS.save(fig, 'F13_optimisation')


# ------------------------------------------------------------------------------------------------ Figure 13
def fig_designs():
    """Figure 13: (a, b) final NICE design and homogenisation design with the colour bar, (c) scale demonstration."""
    fig = plt.figure(figsize=(178 * FS.MM, 64 * FS.MM))
    bot = fig.add_gridspec(2, 3, height_ratios=[1, .06], width_ratios=[1, 1, .78], hspace=.05, wspace=.10, top=.90,
                           bottom=.15, left=.02, right=.975)
    axc, _ = draw_fields(fig, bot[0, 0], bot[0, 1], bot[1, 0:2], letters=('a', 'b'))
    axe = draw_scale(fig, bot[0, 2], letter='c', wide=True)
    fig.canvas.draw()                                                   # align (c) with the aspect-constrained plates
    pc, pe = axc.get_position(), axe.get_position()
    axe.set_position([pe.x0 + .04, pc.y0, pe.width - .07, pc.height])
    FS.save(fig, 'F14_designs_scale')


# ------------------------------------------------------------------------------------------------ Figure S06
def fig_law():
    sys.path.insert(0, str(SRC)); sys.dont_write_bytecode = True
    from homog_macro import Material                                    # the spline used by the macroscale model
    mat = Material.from_json(D / 'homog' / 'homog_cells.json')
    cells = json.loads((D / 'homog' / 'homog_cells.json').read_text())['cells']
    t = np.array([c['tau'] for c in cells])
    CH = [np.asarray(c['CH']) for c in cells]
    pts = dict(C11=[np.mean(np.diag(M)[:3]) for M in CH], C12=[np.mean([M[0, 1], M[0, 2], M[1, 2]]) for M in CH],
               C44=[np.mean(np.diag(M)[3:]) for M in CH], rho=[c['rho'] for c in cells])
    for k in ('C11', 'C12', 'C44', 'rho'):
        assert np.allclose(pts[k], F['homog_law'][k], rtol=1e-11), k
    tt = np.linspace(t.min(), t.max(), 400)
    fig, axs = plt.subplots(1, 2, figsize=(178 * FS.MM, 66 * FS.MM), constrained_layout=True)
    mk = dict(ls='none', marker='h', ms=4.2, mfc='white', mew=.8, zorder=3)
    sty = [('C11', r'$C^H_{11}$', '-', 0), ('C12', r'$C^H_{12}$', '--', .004), ('C44', r'$C^H_{44}$', '-.', -.005)]
    for key, lab, ls, dy in sty:
        axs[0].plot(tt, mat.s[key](tt), ls=ls, color=FS.TEXT, lw=.9, zorder=2)
        axs[0].plot(t, pts[key], color=FS.TEXT, **mk)
        axs[0].text(t[-1] + .012, pts[key][-1] + dy, lab, ha='left', va='center', fontsize=7.5, color=FS.TEXT)
    axs[1].plot(tt, mat.s['rho'](tt), '-', color=FS.TEXT, lw=.9, zorder=2)
    axs[1].plot(t, pts['rho'], color=FS.TEXT, **mk)
    for ax in axs:
        ax.grid(True, color=FS.GRID, lw=.4); ax.set_xlabel(r'uniform corner thickness parameter $\tau$')
    axs[0].set_xlim(.16, .77); axs[1].set_xlim(.16, .72)
    axs[0].set_ylabel(r'effective stiffness $C^H_{ij}\,/\,E_Y$')
    axs[1].set_ylabel(r'material volume fraction $V^H$')
    FS.panel(axs[0], 'a', 'Effective elasticity tensor (cubic)')
    FS.panel(axs[1], 'b', 'Material volume fraction')
    L2 = plt.Line2D
    fig.legend(handles=[L2([], [], color=FS.TEXT, **{k: v for k, v in mk.items() if k != 'zorder'},
                           label=f'periodic homogenisation (markers, {len(t)} thicknesses)'),
                        L2([], [], color=FS.TEXT, lw=.9, label='cubic splines of the macroscale model (lines)')],
               loc='outside upper center', ncol=2, frameon=False, fontsize=7, handlelength=2.2, columnspacing=2.0)
    FS.save(fig, 'S06_homogenised_law')
    print(f"law: {len(t)} thicknesses {t.min()}-{t.max()}; C11 {pts['C11'][0]:.5f}-{pts['C11'][-1]:.5f}, "
          f"C12 {pts['C12'][-1]:.5f}, C44 {pts['C44'][-1]:.5f} at the thickest; V^H {pts['rho'][0]:.4f}-{pts['rho'][-1]:.4f}")

if __name__ == '__main__':
    fig_main()
    fig_designs()
    fig_law()

