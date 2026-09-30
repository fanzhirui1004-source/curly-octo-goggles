"""Section 6.11 figures: thickness design optimisation.

Main Figure 12 -> figures/F13_optimisation.{svg,pdf,png}
  (a) case A (2x2x2 block): NICE and exact-condensation (twin) compliance per design iteration, exact compliance of the
      NICE designs at iterations 0, 12, 23 (open markers); lower strip: NICE relative to the twin at the same iteration and
      to the exact compliance of the same design.
  (b) plates, in-plane load (B1) and bending (B2): compliance normalised by the initial NICE value of B1 / B2, NICE from the
      uniform start, the homogenisation design evaluated with NICE and the NICE continuation started from it (drawn over its
      own design iterations); lower strips: zoom on the final values.
  (c,d) final corner thickness parameters of B2 and XH_z on the plate, drawn in the original orientation (long side
      horizontal): internal y horizontal, internal x vertical; layer z = 0 (z = 1 differs by < 1e-3, printed).
  (e) placeholder for the scale demonstration (pending).
Supplementary Figure S06 -> figures/S06_homogenised_law.{svg,pdf,png}
  C11, C12, C44 and rho of the uniform-thickness cell against tau, with the cubic splines of homog_macro.Material.

Data only from review_r1/results/X6_opt: optA/optA, optA/optAx (histories, check_*.json), plates/* (histories, meta.json),
homog/plate841.json, homog/homog_cells.json, FACTS_6_11.json (cross-checked against the histories below).
Usage: python3 fig_opt.py
"""
import json
import sys
from pathlib import Path
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon, Rectangle
import figstyle as FS
plt = FS.plt

HERE = Path(__file__).resolve().parent
D = HERE.parent / 'review_r1' / 'results' / 'X6_opt'
SRC = HERE.parents[1] / 'data' / 'newmachine_20260924' / 'src_v2_wip'   # homog_macro.py (spline of the material law)
F = json.loads((D / 'FACTS_6_11.json').read_text())

NICE, EXACT, HOM = FS.C['corrected'], FS.C['exact'], FS.C['uncorrected']
TAU_LO, TAU_HI = 0.18, 0.69
CMAP = 'cividis'


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
    return dict(W=ny, Hh=nx, kept=kept, removed=removed, cut=cut, cells=cells)


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


def draw_plate(ax, G, T, FX, load_label, note):
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
    # load face: material part of the upper face (internal x = max)
    top = [p for p in G['kept'] if abs(p[1] - Hh) < 1e-9]
    xa, xb = min(p[0] for p in top), max(p[0] for p in top)
    ax.plot([xa, xb], [Hh, Hh], color=NICE, lw=2.2, zorder=4, solid_capstyle='butt')
    ax.text(xa, Hh + .2, load_label, ha='left', va='bottom', fontsize=6.3, color=FS.TEXT)
    ax.text(W - .1, Hh - .5, note, ha='right', va='center', fontsize=6.3, color=FS.TEXT, zorder=6,
            bbox=dict(fc='#F3F5F7', ec='none', pad=.6))
    ax.text(W - .15, Hh - 1.5, 'removed', ha='right', va='center', fontsize=6.3, color=FS.MUTED, zorder=6,
            bbox=dict(fc='#F3F5F7', ec='none', pad=.6))
    ax.set_xlim(-.15, W + .15); ax.set_ylim(-1.0, Hh + .75); ax.set_aspect('equal'); ax.axis('off')
    return im


def fmt_thousands(x, nd):
    return f'{x:,.{nd}f}'


# ------------------------------------------------------------------------------------------------ Figure 12
def fig_main():
    fig = plt.figure(figsize=(178 * FS.MM, 138 * FS.MM))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.3, 1], hspace=.27, top=.90, bottom=.04, left=.08, right=.985)
    top = outer[0].subgridspec(2, 3, height_ratios=[2.3, 1], hspace=.10, wspace=.36)
    bot = outer[1].subgridspec(2, 3, height_ratios=[1, .06], width_ratios=[1, 1, .78], hspace=.05, wspace=.10)

    # ---------------- (a) case A
    ax, axs = fig.add_subplot(top[0, 0]), fig.add_subplot(top[1, 0])
    kA, kX = np.array([h['k'] for h in A]), np.array([h['k'] for h in AX])
    CA, CX = ctrace(A), ctrace(AX)
    vA = np.array([h['V_rel'] for h in A])
    k_on = int(np.argmax(vA <= 1.0 + 1e-3))                               # first iteration at V* (volume phase before it)
    for a_ in (ax, axs):
        a_.axvspan(0, k_on, color='#EEF1F4', lw=0, zorder=0)
    ax.text(.75, 23.3, r'$V>V^*$', ha='left', va='bottom', fontsize=6.3, color=FS.MUTED)
    ax.plot(kA, CA, '-', color=NICE, lw=1.1, marker='s', ms=2.4, label='NICE', zorder=3)
    ax.plot(kX, CX, '--', color=EXACT, lw=1.0, marker='P', ms=2.6, label='exact condensation', zorder=4)
    kc = sorted(CHK)
    ax.plot(kc, [CHK[k]['C_exact'] for k in kc], ls='none', marker='o', ms=6.5, mfc='none', mec=EXACT, mew=1.0,
            label='exact, NICE design', zorder=5)
    ax.set_ylabel('compliance $C$'); ax.set_ylim(23, 39.5)
    FS.panel(ax, 'a', 'Case A: 2×2×2 block')
    n = min(len(CA), len(CX))
    rel_twin = 100 * (CA[:n] / CX[:n] - 1)
    rel_chk = [100 * (CHK[k]['C_hat'] / CHK[k]['C_exact'] - 1) for k in kc]
    axs.axhline(0, color=FS.GRID, lw=.8, zorder=1)
    axs.plot(kA[:n], rel_twin, '-', color=EXACT, lw=.9, marker='P', ms=2.4, zorder=3)
    axs.plot(kc, rel_chk, ls='none', marker='o', ms=5, mfc='none', mec=EXACT, mew=.9, zorder=4)
    pert = [p['k'] for p in F['A']['body_perturbations_applied']]
    k19 = max(pert, key=lambda k: abs(rel_twin[k]))
    axs.annotate('perturbed\ndesign', (k19, rel_twin[k19]), (k19 - 9.0, rel_twin[k19] + .004), fontsize=6, color=FS.MUTED,
                 va='center', arrowprops=dict(arrowstyle='-', color=FS.MUTED, lw=.5))
    axs.set_ylim(-.12, .015); axs.set_yticks([-.1, -.05, 0])
    axs.set_ylabel('rel. to exact (%)', fontsize=7)
    axs.set_xlabel('design iteration')
    plt.setp(ax.get_xticklabels(), visible=False)
    for a_ in (ax, axs):
        a_.grid(True, color=FS.GRID, lw=.4); a_.set_xlim(-.8, kA[-1] + .8)
    print(f"A: NICE/twin rel diff (%) {np.round(rel_twin, 4).tolist()}; checks {np.round(rel_chk, 4).tolist()}; volume phase k < {k_on}")
    print(f"A: perturbed designs at k {pert}; V_rel NICE/twin at k={k19}: {A[k19]['V_rel']:.6f}/{AX[k19]['V_rel']:.6f}")

    # ---------------- (b) plates
    specs = [('y', 'B1', 'XH_y', 'H_y', 'in-plane load', (.9195, .9375), 'b'),
             ('z', 'B2', 'XH_z', 'H_z', 'bending', (.9055, .9325), None)]
    for col, (tag, rb, rx, rh, title, zoom, letter) in enumerate(specs, start=1):
        ax, axz = fig.add_subplot(top[0, col]), fig.add_subplot(top[1, col])
        C0 = PL[rb][0]['C']
        cb, cx, ch = ctrace(PL[rb]) / C0, ctrace(PL[rx]) / C0, PL[rh][0]['C'] / C0
        kb, kx = np.arange(len(cb)), np.arange(len(cx))
        vb = np.array([h['V_rel'] for h in PL[rb]])
        kon = int(np.argmax(vb <= 1.0 + 1e-2))
        for a_ in (ax, axz):
            a_.axvspan(0, kon, color='#EEF1F4', lw=0, zorder=0)
            a_.axhline(cb[-1], color=NICE, lw=.6, ls=':', zorder=1)
            a_.plot(kb, cb, '-', color=NICE, lw=1.1, marker='s', ms=2.4, zorder=3,
                    label=f'NICE from uniform start ({rb})')
            a_.plot(kx, cx, '-', color=HOM, lw=1.1, marker='o', ms=2.3, zorder=3,
                    label=f'NICE from homogenisation design (X{rh})')
            a_.plot([0], [ch], ls='none', marker='D', ms=5.2, mfc=HOM, mec='white', mew=.6, zorder=5,
                    label=f'homogenisation design ({rh})')
            a_.grid(True, color=FS.GRID, lw=.4); a_.set_xlim(-.8, len(cb) - .2)
        ax.axhspan(*zoom, fc='none', ec=FS.MUTED, lw=.5, ls='--', zorder=2)
        axz.set_ylim(*zoom)
        axz.yaxis.set_major_locator(plt.MultipleLocator(.01))
        axz.set_xlabel('design iteration')
        plt.setp(ax.get_xticklabels(), visible=False)
        if letter:
            FS.panel(ax, letter, f'Plate {rb}: {title}')
            ax.set_ylabel(r'$C\,/\,C_0$')
            axz.set_ylabel(r'$C\,/\,C_0$ (zoom)', fontsize=7)
        else:
            ax.set_title(f'Plate {rb}: {title}', loc='left', fontweight='bold', fontsize=8.5, pad=6)
        print(f"{rb}: final {cb[-1]:.5f}, H {ch:.5f}, X final {cx[-1]:.5f} (C0 {C0:.4f}); volume phase k < {kon}")

    L2 = plt.Line2D
    hl = [L2([], [], color=NICE, marker='s', ms=3, lw=1.1, label='NICE (a), NICE from the uniform start (b)'),
          L2([], [], color=EXACT, ls='--', marker='P', ms=3.2, lw=1.0, label='exact condensation (a)'),
          L2([], [], ls='none', marker='o', ms=6, mfc='none', mec=EXACT, mew=1.0, label='exact compliance of the NICE design (a)'),
          L2([], [], ls='none', marker='D', ms=5, mfc=HOM, mec='white', mew=.6, label='homogenisation design, NICE (b)'),
          L2([], [], color=HOM, marker='o', ms=3, lw=1.1, label='NICE from the homogenisation design (b)'),
          L2([], [], color=NICE, ls=':', lw=.8, label='final NICE value from the uniform start (b)')]
    fig.legend(handles=hl, loc='lower center', bbox_to_anchor=(.5, .935), ncol=3, frameon=False, fontsize=6.5,
               handlelength=2.2, columnspacing=1.4, handletextpad=.5)

    # ---------------- (c, d) thickness fields
    G = plate_geometry()
    fields = {}
    for run in ('plateB2', 'xstartH_z'):
        fields[run] = vertex_field(run)
        print(f"{run}: z=0/z=1 max |diff| {fields[run][2]:.2e}, final C {fields[run][3]:.3f}, "
              f"tau range {np.nanmin(fields[run][0]):.4f}-{np.nanmax(fields[run][0]):.4f}")
    assert np.isclose(fields['plateB2'][3], F['B2']['C_final']) and np.isclose(fields['xstartH_z'][3], F['XH_z']['C_final'])
    axc, axd = fig.add_subplot(bot[0, 0]), fig.add_subplot(bot[0, 1])
    T, FX, _, C = fields['plateB2']
    im = draw_plate(axc, G, T, FX, 'load face (out of plane)', f'NICE $C$ = {fmt_thousands(C, 1)}')
    FS.panel(axc, 'c', 'Final design B2')
    T, FX, _, C = fields['xstartH_z']
    draw_plate(axd, G, T, FX, 'load face (out of plane)', f'NICE $C$ = {fmt_thousands(C, 1)}')
    FS.panel(axd, 'd', 'Final design XH_z')
    cax = fig.add_subplot(bot[1, 0:2])
    pos = cax.get_position(); cax.set_position([pos.x0 + .18 * pos.width, pos.y0, .64 * pos.width, pos.height])
    cb_ = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[.18, .3, .4, .5, .6, .69])
    cb_.set_label(r'corner thickness parameter $\tau$ (circles: free, squares: fixed)', fontsize=7)
    cb_.ax.tick_params(labelsize=6.5); cb_.outline.set_linewidth(.5)

    # ---------------- (e) placeholder
    axe = fig.add_subplot(bot[0, 2])
    axe.set_xticks([]); axe.set_yticks([])
    for s in axe.spines.values():
        s.set_visible(True); s.set_linestyle('--'); s.set_color(FS.GRID)
    axe.text(.5, .5, 'scale demonstration:\npending', ha='center', va='center', fontsize=7.5, color=FS.MUTED,
             transform=axe.transAxes)
    FS.panel(axe, 'e', 'Scale demonstration')
    fig.canvas.draw()                                                   # align (e) with the aspect-constrained plates
    pc, pe = axc.get_position(), axe.get_position()
    axe.set_position([pe.x0, pc.y0, pe.width, pc.height])
    FS.save(fig, 'F13_optimisation')


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
    for k, fk in (('C11', 'C11'), ('C12', 'C12'), ('C44', 'C44'), ('rho', 'rho')):
        assert np.allclose(pts[k], F['homog_law'][fk], rtol=1e-11), k
    tt = np.linspace(t.min(), t.max(), 400)
    fig, axs = plt.subplots(1, 2, figsize=(178 * FS.MM, 62 * FS.MM), constrained_layout=True)
    sty = [('C11', r'$C^H_{11}$', FS.C['exact'], 'o'), ('C12', r'$C^H_{12}$', FS.C['uncorrected'], 's'),
           ('C44', r'$C^H_{44}$', FS.C['corrected'], '^')]
    for key, lab, col, mk in sty:
        axs[0].plot(tt, mat.s[key](tt), '-', color=col, lw=1.0, zorder=2)
        axs[0].plot(t, pts[key], ls='none', marker=mk, color=col, ms=4, mfc='white', mew=.9, label=lab, zorder=3)
    axs[1].plot(tt, mat.s['rho'](tt), '-', color=FS.C['assembly'], lw=1.0, zorder=2)
    axs[1].plot(t, pts['rho'], ls='none', marker='D', color=FS.C['assembly'], ms=3.6, mfc='white', mew=.9, zorder=3)
    for ax in axs:
        ax.grid(True, color=FS.GRID, lw=.4); ax.set_xlabel(r'uniform corner thickness parameter $\tau$')
        ax.set_xlim(.16, .72)
    axs[0].set_ylabel(r'effective stiffness $C^H_{ij}\,/\,E_Y$')
    axs[0].legend(loc='upper left', frameon=False, fontsize=7, handlelength=1.5)
    axs[1].set_ylabel(r'material volume fraction $\rho$')
    FS.panel(axs[0], 'a', 'Effective elasticity tensor (cubic)')
    FS.panel(axs[1], 'b', 'Material volume fraction')
    axs[1].text(.705, .115, 'markers: periodic homogenisation\nlines: cubic splines', ha='right', va='bottom', fontsize=6.3,
                color=FS.MUTED)
    FS.save(fig, 'S06_homogenised_law')
    print(f"law: {len(t)} thicknesses {t.min()}-{t.max()}; C11 {pts['C11'][0]:.5f}-{pts['C11'][-1]:.5f}, "
          f"rho {pts['rho'][0]:.4f}-{pts['rho'][-1]:.4f}")


if __name__ == '__main__':
    fig_main()
    fig_law()
