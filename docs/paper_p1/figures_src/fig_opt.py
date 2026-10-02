"""Section 6.11 figures: thickness design optimisation.

Main Figure 12 -> figures/F13_optimisation.{svg,pdf,png}   (the prefix F12_ belongs to Figure 7, F12_field_error_M1)
  (a) case A (2x2x2 block): NICE and exact-condensation (twin) compliance per design iteration, exact compliance of the
      NICE designs at iterations 0, 12, 23 (open circles); lower strip: NICE compliance relative to the twin run at the same
      design iteration (line; the designs of the two runs differ) and to the exact compliance of the same design (circles);
      the perturbed designs of the geometry-generation fallback (iterations 16 and 19) are marked.
  (b) plates, B1 (in-plane load, left) and B2 (bending, right): compliance divided by C0, the initial NICE compliance of B1 /
      B2 at the uniform start; the homogenisation design (Hom-y / Hom-z) evaluated with NICE and the NICE continuation from it
      (X-y / X-z) are divided by the same C0 and drawn over their own design iterations; shading: design iterations with
      V > V* of the run from the uniform start; the bracket in the upper plots marks the range enlarged in the lower strips.
  The legends sit inside the panels: the entries of (a) in panel (a), those of (b) in the B1 panel.
Main Figure 13 -> figures/F14_designs_scale.{svg,pdf,png}   (formerly panels (c), (d), (e) of Figure 12, same data)
  (a,b) final corner thickness parameters of B2 and X-z on the plate, drawn in the original orientation (long side
      horizontal): internal y horizontal, internal x vertical; layer z = 0 (the z = 1 difference is printed); vertices in
      the removed region are corners of cut cells.
  (c) scale demonstration (scale/scale_summary.json): time per design iteration (mean, range bars), peak CPU memory of
      the main process and peak device memory in use (nvidia-smi, includes the cuDSS factor) against the number of cells, for plates of 24, 51, 88 and 110 cells; 135 cells failed (K_PP factor on the GPU).
Supplementary Figure S06 -> figures/S06_homogenised_law.{svg,pdf,png}
  C11, C12, C44 and the material volume fraction (symbol V^H; rho is the recomputed residual of Eq. (18)) of the
  uniform-thickness cell against tau, with the cubic splines of homog_macro.Material.

Colours and markers: NICE keeps its house identity (figstyle.MODEL['A3']: vermillion, squares). The exact-condensation
twin is ink (figstyle.C['exact']) with marker 'x'; the homogenisation series use C['extra'] (purple, a colour no variant
uses) with markers no variant uses ('*', 'h'); the resource curves of the scale demonstration are neutral greys
(figstyle.GREY); the load face and Figure S06 are drawn in neutral colours.

Data only from review_r1/results/X6_opt: optA/optA, optA/optAx (histories, check_*.json), plates/* (histories, meta.json
'vkeys' and 'fixed'), homog/plate841.json ('shape', 'normal', 'b_global', 'cells'), homog/homog_cells.json ('cells'),
FACTS_6_11.json (cross-checked against the histories below).
Usage: python3 fig_opt.py            Figures 12, 13 and S06
       python3 fig_opt.py --single   additionally the pre-split five-panel figure -> figures/F13_optimisation_single.*
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
SRC = HERE.parents[1] / 'data' / 'newmachine_20260924' / 'src_v2_wip'   # homog_macro.py (spline of the material law)
F = json.loads((D / 'FACTS_6_11.json').read_text())

NICE, NICE_MK = FS.MODEL['A3'][0], FS.MODEL['A3'][1]                    # NICE: vermillion squares, as in Figures 5-10
EXACT, EXACT_MK = FS.C['exact'], 'x'                                    # exact condensation: ink, no variant marker
HOM, HOM_MK, XH_MK = FS.C['extra'], '*', 'h'                            # homogenisation design / continuation from it
SHADE = '#EEF1F4'
TAU_LO, TAU_HI = 0.18, 0.69
CMAP = 'cividis'
NAME = {'H_y': r'Hom-$y$', 'H_z': r'Hom-$z$', 'XH_y': r'X-$y$', 'XH_z': r'X-$z$'}   # display names (H_y etc. are record keys)


def hist(p):
    return [json.loads(l) for l in open(D / p)]


def ctrace(H):
    return np.array([h['C'] for h in H])


# ------------------------------------------------------------------------------------------------ data
A, AX = hist('optA/optA/history.jsonl'), hist('optA/optAx/history.jsonl')
CHK = {k: json.loads((D / f'optA/optA/check_{k:03d}.json').read_text()) for k in (0, 12, 23)}
PL = {t: hist(f'plates/{r}/history.jsonl') for t, r in
      (('B1', 'plateB1'), ('B2', 'plateB2'), ('XH_y', 'xstartH_y'), ('XH_z', 'xstartH_z'), ('H_y', 'hevalH_y'), ('H_z', 'hevalH_z'))}
LAY = json.loads((D / 'homog' / 'plate841.json').read_text())

# consistency with the fact sheet (the single source of numbers)
assert np.allclose(ctrace(A), F['A']['C_trace']) and np.allclose(ctrace(AX), F['A_exact_twin']['C_trace'])
for k, c in CHK.items():
    assert np.isclose(c['C_exact'], F['A_checks'][str(k)]['C_exact'])
for t in ('B1', 'B2', 'XH_y', 'XH_z'):
    assert np.allclose(ctrace(PL[t]), F[t]['C_trace'])
assert np.isclose(PL['H_y'][0]['C'], F['Hfine_y']['C_fine']) and np.isclose(PL['H_z'][0]['C'], F['Hfine_z']['C_fine'])
assert [h['k'] for h in A] == list(range(len(A)))                      # history index = design iteration


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


def vertex_field(run):
    """Final corner parameters of a plate run on the (x, y) vertex grid, layer z = 0; also the z = 0/1 difference."""
    meta = json.loads((D / 'plates' / run / 'meta.json').read_text())
    H = hist(f'plates/{run}/history.jsonl')
    vk, tv, fixed = np.array(meta['vkeys']), np.array(H[-1]['tv']), np.array(meta['fixed'])
    nx, ny, _ = LAY['shape']
    T = np.full((nx + 1, ny + 1), np.nan)
    FX = np.zeros((nx + 1, ny + 1), bool)
    d = {tuple(k): v for k, v in zip(vk.tolist(), tv)}
    for (i, j, k), v, f in zip(vk.tolist(), tv, fixed):
        if k == 0:
            T[i, j] = v; FX[i, j] = f
    dz = max(abs(d[(i, j, 0)] - d[(i, j, 1)]) for (i, j, k) in d if k == 0)
    return T, FX, dz, H[-1]['C']


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
    ax.add_patch(Polygon(G['removed'], closed=True, fc='none', ec=FS.GRID, hatch='////', lw=0, zorder=0.6))
    for X0, Y0, kind, ret in G['cells']:
        ax.add_patch(Rectangle((X0, Y0), 1, 1, fc='none', ec='white' if kind == 'FULL' else '#B8C2CC', lw=.35, zorder=2))
    ax.add_patch(Polygon(G['kept'], closed=True, fc='none', ec=FS.TEXT, lw=.8, zorder=3))
    (x1, y1), (x2, y2) = G['cut']
    ax.plot([x1, x2], [y1, y2], color=FS.TEXT, lw=1.3, zorder=3.5, solid_capstyle='butt')
    # vertices coloured by tau; fixed load-face vertices as squares
    I, J = np.nonzero(np.isfinite(T))
    for mk, sel in (('o', ~FX[I, J]), ('s', FX[I, J])):
        ax.scatter(J[sel], I[sel], c=T[I[sel], J[sel]], cmap=CMAP, norm=norm, s=15 if mk == 'o' else 13, marker=mk,
                   edgecolors=FS.TEXT, linewidths=.45, zorder=5, clip_on=False)
    # clamp along the lower long edge
    ax.plot([0, W], [0, 0], color=FS.TEXT, lw=1.4, zorder=4, solid_capstyle='butt')
    for xh in np.linspace(0.1, W, 33):
        ax.plot([xh, xh - .16], [0, -.2], color=FS.MUTED, lw=.5, zorder=4)
    ax.text(W / 2, -.34, 'clamped', ha='center', va='top', fontsize=6.3, color=FS.MUTED)
    # load face: material part of the upper face (internal x = max).  The traction is +z; the plot axes (internal y, x)
    # are a mirror image of the view from +z, so +z points into the page: circled crosses.
    top = [p for p in G['kept'] if abs(p[1] - Hh) < 1e-9]
    xa, xb = min(p[0] for p in top), max(p[0] for p in top)
    ax.plot([xa, xb], [Hh, Hh], color=FS.TEXT, lw=2.4, zorder=4, solid_capstyle='butt')
    ys = Hh + .33
    for xs in np.arange(xa + .5, xb, 1.0):
        ax.plot([xs], [ys], ls='none', marker='o', ms=5.2, mfc='white', mec=FS.TEXT, mew=.7, zorder=6, clip_on=False)
        ax.plot([xs], [ys], ls='none', marker='x', ms=3.0, mec=FS.TEXT, mew=.7, zorder=7, clip_on=False)
    ax.text(xb + .15, ys, 'load face, out-of-plane traction', ha='left', va='center', fontsize=6.3,
            color=FS.TEXT)
    ax.text(W - .1, Hh - .5, note, ha='right', va='center', fontsize=6.3, color=FS.TEXT, zorder=6,
            bbox=dict(fc='#F3F5F7', ec='none', pad=.6))
    ax.text(W - .15, Hh - 1.5, 'removed', ha='right', va='center', fontsize=6.3, color=FS.MUTED, zorder=6,
            bbox=dict(fc='#F3F5F7', ec='none', pad=.6))
    ax.set_xlim(-.15, W + .15); ax.set_ylim(-1.0, Hh + .75); ax.set_aspect('equal'); ax.axis('off')
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
    b = [L2([], [], color=NICE, marker=NICE_MK, ms=3, lw=1.0, label='NICE, uniform start'),
         L2([], [], ls='none', marker=HOM_MK, ms=7.5, mfc=HOM, mec='white', mew=.5, label='Hom-$y$, Hom-$z$ with NICE'),
         L2([], [], color=HOM, marker=XH_MK, ms=3.2, lw=1.0, label='X-$y$, X-$z$ from Hom'),
         L2([], [], color=NICE, ls=':', lw=.8, label='final, uniform start'),
         Patch(fc=SHADE, ec='none', label=r'$V>V^*$, uniform start'),
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
    pert = [p['k'] for p in F['A']['body_perturbations_applied']]
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


def draw_plates(fig, cells, legend=False):
    """(b) plates B1 and B2: one letter for both columns, each column (gs_up, gs_lo) with its own C0 and y-label."""
    specs = [('B1', 'XH_y', 'H_y', 'B1: in-plane load', (.9195, .9375)),
             ('B2', 'XH_z', 'H_z', 'B2: bending', (.9055, .9325))]
    b_axes = []
    for (gs_up, gs_lo), (rb, rx, rh, title, zoom) in zip(cells, specs):
        ax, axz = fig.add_subplot(gs_up), fig.add_subplot(gs_lo)
        b_axes.append(ax)
        C0 = PL[rb][0]['C']
        cb, cx, ch = ctrace(PL[rb]) / C0, ctrace(PL[rx]) / C0, PL[rh][0]['C'] / C0
        kb, kx = np.arange(len(cb)), np.arange(len(cx))
        vb = np.array([h['V_rel'] for h in PL[rb]])
        vx = np.array([h['V_rel'] for h in PL[rx]])
        kon = int(np.argmax(vb <= 1.0 + 1e-2))
        for a_ in (ax, axz):
            a_.axvspan(0, kon, color=SHADE, lw=0, zorder=0)
            a_.axhline(cb[-1], color=NICE, lw=.6, ls=':', zorder=1)
            a_.plot(kb, cb, '-', color=NICE, lw=1.0, marker=NICE_MK, ms=2.4, zorder=3)
            a_.plot(kx, cx, '-', color=HOM, lw=1.0, marker=XH_MK, ms=2.8, zorder=3)
            a_.plot([0], [ch], ls='none', marker=HOM_MK, ms=8, mfc=HOM, mec='white', mew=.5, zorder=5)
            a_.grid(True, color=FS.GRID, lw=.4); a_.set_xlim(-.8, len(cb) - .2)
            a_.set_ylabel(r'$C\,/\,C_0$', fontsize=7 if a_ is axz else 8)
        zoom_bracket(ax, *zoom)
        axz.set_ylim(*zoom)
        axz.yaxis.set_major_locator(plt.MultipleLocator(.01))
        axz.set_xlabel('design iteration')
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.text(.97, .95, title, ha='right', va='top', fontsize=7.5, color=FS.TEXT, transform=ax.transAxes)
        if legend and rb == 'B1':
            ax.set_ylim(top=1.5)                                            # headroom for the in-panel legend
            ax.legend(handles=legend_handles()[1], loc='upper right', bbox_to_anchor=(1.03, .89), frameon=False,
                      fontsize=5.8, handlelength=1.9, handletextpad=.5, labelspacing=.3, borderaxespad=.1)
        print(f"{rb}: C0 {C0:.4f}; final {cb[-1]:.5f}, {rh} {ch:.5f}, {rx} final {cx[-1]:.5f}; V_rel of {rx} "
              f"{vx[0]:.5f} at its start, range {vx.min():.5f}-{vx.max():.5f}; volume phase of {rb} k < {kon}")
    FS.panel(b_axes[0], 'b', 'Plates B1 and B2')
    return b_axes


def draw_fields(fig, gs_c, gs_d, gs_cbar, letters=('c', 'd')):
    """Final corner thickness parameters of B2 and X-z on the plate with a shared horizontal colour bar."""
    G = plate_geometry()
    fields = {}
    for run in ('plateB2', 'xstartH_z'):
        fields[run] = vertex_field(run)
        T, FX, dz, C = fields[run]
        print(f"{run}: z=0/z=1 max |diff| {dz:.2e}, final C {C:.3f}, tau range {np.nanmin(T):.4f}-{np.nanmax(T):.4f}; "
              f"vertices in the removed region (X, Y, tau, fixed): {removed_vertices(G, T, FX)}")
    assert np.isclose(fields['plateB2'][3], F['B2']['C_final']) and np.isclose(fields['xstartH_z'][3], F['XH_z']['C_final'])
    axc, axd = fig.add_subplot(gs_c), fig.add_subplot(gs_d)
    T, FX, _, C = fields['plateB2']
    im = draw_plate(axc, G, T, FX, f'NICE $C$ = {fmt_thousands(C, 1)}')
    FS.panel(axc, letters[0], 'Final design B2')
    T, FX, _, C = fields['xstartH_z']
    draw_plate(axd, G, T, FX, f'NICE $C$ = {fmt_thousands(C, 1)}')
    FS.panel(axd, letters[1], f"Final design {NAME['XH_z']}")
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


# ------------------------------------------------------------------------------------------------ Figure 12 (split)
def fig_main():
    """Figure 12: (a) case A and (b) plates B1 / B2 with their lower strips, legends inside the panels."""
    fig = plt.figure(figsize=(178 * FS.MM, 84 * FS.MM))
    top = fig.add_gridspec(2, 3, height_ratios=[2.3, 1], hspace=.10, wspace=.40, top=.925, bottom=.105, left=.075, right=.98)
    draw_caseA(fig, top[0, 0], top[1, 0], legend=True)
    draw_plates(fig, [(top[0, 1], top[1, 1]), (top[0, 2], top[1, 2])], legend=True)
    FS.save(fig, 'F13_optimisation')


# ------------------------------------------------------------------------------------------------ Figure 13 (new)
def fig_designs():
    """Figure 13: (a, b) final designs B2 and X-z with the colour bar, (c) scale demonstration (formerly Figure 12c-e)."""
    fig = plt.figure(figsize=(178 * FS.MM, 64 * FS.MM))
    bot = fig.add_gridspec(2, 3, height_ratios=[1, .06], width_ratios=[1, 1, .78], hspace=.05, wspace=.10, top=.90,
                           bottom=.15, left=.02, right=.975)
    axc, _ = draw_fields(fig, bot[0, 0], bot[0, 1], bot[1, 0:2], letters=('a', 'b'))
    axe = draw_scale(fig, bot[0, 2], letter='c', wide=True)
    fig.canvas.draw()                                                   # align (c) with the aspect-constrained plates
    pc, pe = axc.get_position(), axe.get_position()
    axe.set_position([pe.x0 + .04, pc.y0, pe.width - .07, pc.height])
    FS.save(fig, 'F14_designs_scale')


# ------------------------------------------------------------------------------------------------ Figure 12 (pre-split)
def fig_main_single():
    """The five-panel figure before the split (a-e with the three-column legend on top) -> F13_optimisation_single."""
    fig = plt.figure(figsize=(178 * FS.MM, 142 * FS.MM))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.3, 1], hspace=.25, top=.885, bottom=.04, left=.08, right=.975)
    top = outer[0].subgridspec(2, 3, height_ratios=[2.3, 1], hspace=.10, wspace=.42)
    bot = outer[1].subgridspec(2, 3, height_ratios=[1, .06], width_ratios=[1, 1, .78], hspace=.05, wspace=.10)
    draw_caseA(fig, top[0, 0], top[1, 0])
    draw_plates(fig, [(top[0, 1], top[1, 1]), (top[0, 2], top[1, 2])])
    L2 = plt.Line2D
    hl = [L2([], [], color=NICE, marker=NICE_MK, ms=3, lw=1.0, label='NICE (a), NICE from the uniform start (b)'),
          L2([], [], color=EXACT, ls='--', marker=EXACT_MK, ms=4, mew=.8, lw=.8, label='twin run, exact condensation (a)'),
          L2([], [], ls='none', marker='o', ms=6, mfc='none', mec=EXACT, mew=1.0, label='exact compliance of the NICE design (a)'),
          L2([], [], ls='none', marker=HOM_MK, ms=7.5, mfc=HOM, mec='white', mew=.5, label='homogenisation design, NICE (b)'),
          L2([], [], color=HOM, marker=XH_MK, ms=3.2, lw=1.0, label='NICE from the homogenisation design (b)'),
          L2([], [], color=NICE, ls=':', lw=.8, label='final NICE value from the uniform start (b)'),
          Patch(fc=SHADE, ec='none', label=r'$V>V^*$ (b: run from the uniform start)'),
          L2([], [], ls='none', marker=r'$]$', ms=6, color=FS.TEXT, label='range enlarged in the lower strip (b)')]
    fig.legend(handles=hl, loc='lower center', bbox_to_anchor=(.5, .917), ncol=3, frameon=False, fontsize=6.5,
               handlelength=2.2, columnspacing=1.3, handletextpad=.5)
    axc, _ = draw_fields(fig, bot[0, 0], bot[0, 1], bot[1, 0:2])
    axe = draw_scale(fig, bot[0, 2])
    fig.canvas.draw()                                                   # align (e) with the aspect-constrained plates
    pc, pe = axc.get_position(), axe.get_position()
    axe.set_position([pe.x0 + .035, pc.y0, pe.width - .06, pc.height])
    FS.save(fig, 'F13_optimisation_single')


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
    if '--single' in sys.argv[1:]:
        fig_main_single()
