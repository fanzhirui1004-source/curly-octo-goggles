"""Figure 2 -> figures/F01_method_overview.{svg,pdf,png}, two-row layout of the same content as fig02_overview.py: method overview of NICE (neural-initialised condensation with
equilibrium correction), drawn with matplotlib boxes and arrows in the house style (figstyle), 178 mm wide.

  (a) Condensed operator of one cell: the chain q -> rigid split -> learned extension (trial interior field) -> rigid field
      added back, q restored on P -> E^ q -> fixed equilibrium correction W (Chebyshev, Q1 coarse solve, Chebyshev at fixed
      q) -> u^ = F q -> K -> F^T (the transpose of the complete extension) -> S^ q, with the three roles under the
      blocks (learning: trial field; correction: improvability; variational form: structure), the by-construction
      properties of the extension (Eq. 14) and the properties of S^ = F^T K F (Sections 4.1, 3.1, 4.4).
  (b) Assembly and design: cell operators -> assembled K^ -> global solve -> field recovery -> compliance and field-based
      thickness sensitivity, with the design update closing the loop (Section 6.11).
No data; mathtext only (no LaTeX installation needed).  Usage: python3 fig02_overview_v2.py [--preview]
"""
from matplotlib.patches import FancyBboxPatch, Rectangle
import sys
import figstyle as FS
plt = FS.plt

W, H = 178.0, 140.0                                                     # figure size in mm (data units of the canvas axes)
LEARN, CORR, VAR = '#0072B2', '#D55E00', '#53616F'                      # learned block, correction block, variational blocks
FILL = {'plain': ('#F3F5F7', '#8E99A4'), 'learn': ('#EAF3FA', LEARN), 'corr': ('#FDF1E8', CORR), 'var': ('#F3F5F7', VAR),
        'note': ('#FBFCFD', '#C5CDD4')}
PT = 25.4 / 72                                                         # mm per point

fig = plt.figure(figsize=(W * FS.MM, H * FS.MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect('equal'); ax.axis('off')
ax.add_patch(Rectangle((0, 0), W, H, fc='none', ec='none'))           # pins the tight bounding box to the full canvas


def box(x, y, w, h, lines, kind='plain', lw=.7, pitch=1.3, dy=0.0):
    """Rounded box with centred text lines; lines are (text, size_pt[, colour]) tuples.  dy shifts the text block."""
    fc, ec = FILL[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2', fc=fc, ec=ec, lw=lw, zorder=2))
    hs = [l[1] * PT * pitch for l in lines]
    yy = y + h / 2 + sum(hs) / 2 + dy                                   # running cursor from the top of the text block
    for (s, sz, *c), hh in zip(lines, hs):
        ax.text(x + w / 2, yy - hh / 2, s, ha='center', va='center', fontsize=sz, color=c[0] if c else FS.TEXT, zorder=3)
        yy -= hh
    return x, y, w, h


def arrow(x0, x1, y, label=None, lw=.7, color=FS.TEXT, size=6.5, ms=6, zorder=4):
    ax.annotate('', (x1, y), (x0, y), arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, shrinkA=0, shrinkB=0,
                mutation_scale=ms, joinstyle='miter'), zorder=zorder)
    if label:
        ax.text((x0 + x1) / 2, y + 1.3, label, ha='center', va='bottom', fontsize=size, color=color, zorder=5)


def bracket(x0, x1, y, color, d=1.0):
    ax.plot([x0, x0, x1, x1], [y + d, y, y, y + d], color=color, lw=.6, solid_joinstyle='miter', zorder=4)


# ------------------------------------------------------------------------------------------------ (a) one cell
fig.text(3.5 / W, 135.0 / H, '(a) Condensed operator of one cell', fontweight='bold', fontsize=8.5,
         color=FS.TEXT, va='bottom', ha='left')
# row 1: the extension E^ (rigid split, learned extension, rigid field restored)
Y1, H1 = 107.0, 18.0
y1 = Y1 + H1 / 2
ax.text(6.5, y1, '$q$', ha='center', va='center', fontsize=9)
arrow(9.0, 14.0, y1)
bS = box(14.0, Y1, 34.0, H1, [('rigid split', 7.5), (r'$C_Rq\ \,|\,\ \Pi_Pq$', 7.5),
                              ('rigid part | deformation', 7.0, FS.MUTED)])
arrow(48.0, 56.0, y1)
bN = box(56.0, Y1, 50.0, H1, [('learned extension', 7.5), (r'$\mathcal{N}_\theta(\eta)\,\Pi_Pq$', 8.0),
                              ('trial interior field', 7.0), ('geometry-conditioned, linear in $q$', 7.0, FS.MUTED)],
         kind='learn', lw=.9)
arrow(106.0, 114.0, y1)
bR = box(114.0, Y1, 40.0, H1, [('rigid field $+\\,RC_Rq$', 7.5), ('restore $q$ on $P$', 7.5)])
arrow(154.0, 164.0, y1)
ax.text(170.0, y1, r'$\widehat{E}q$', ha='center', va='center', fontsize=9)
# rigid bypass above row 1
yb = Y1 + H1 + 3.0
ax.plot([31.0, 31.0, 134.0, 134.0], [Y1 + H1, yb, yb, Y1 + H1], color=FS.MUTED, lw=.6, zorder=1)
ax.annotate('', (134.0, Y1 + H1 + .05), (134.0, yb), arrowprops=dict(arrowstyle='-|>', color=FS.MUTED, lw=.6, shrinkA=0,
            shrinkB=0, mutation_scale=5))
ax.text(82.5, yb + .7, 'rigid coefficients $C_Rq$ and prescribed $q$ bypass the network', ha='center', va='bottom',
        fontsize=7.0, color=FS.MUTED)
bracket(bN[0], bN[0] + bN[2], Y1 - 1.4, LEARN)
ax.text(bN[0] + bN[2] / 2, Y1 - 2.2, 'learning: trial field', ha='center', va='top', fontsize=7.5, color=LEARN,
        fontweight='bold')
box(14.0, 86.0, 160.0, 12.0, [('by construction (Eq. 14)', 7.0, FS.MUTED),
                              (r'$J_P\widehat{E}=I_p$ (admissible),   $\widehat{E}R_P=R$ (rigid-body motion reproduced)', 7.0),
                              (r'$\widehat{E}$ linear in $q$ at fixed geometry;   network parameters $\theta$ shared by all cells', 7.0)],
    kind='note', lw=.6)

# row 2: the correction W and the variational form F^T K
Y2, H2 = 56.0, 22.0
y2 = Y2 + H2 / 2
ax.text(6.5, y2, r'$\widehat{E}q$', ha='center', va='center', fontsize=9)
arrow(11.0, 16.0, y2)
xC, wC = 16.0, 72.0
box(xC, Y2, wC, H2, [], kind='corr', lw=.9)
ax.text(xC + wC / 2, Y2 + H2 - 3.2, r'equilibrium correction $\mathcal{W}$', ha='center', va='center', fontsize=7.5, zorder=3)
sub = [('Chebyshev', 19.0), ('$Q_1$ coarse', 19.0), ('Chebyshev', 19.0)]
gap = 4.0
xs, ys, hs_ = xC + (wC - sum(w for _, w in sub) - 2 * gap) / 2, Y2 + 8.4, 6.0
for i, (s, w_) in enumerate(sub):
    ax.add_patch(FancyBboxPatch((xs, ys), w_, hs_, boxstyle='round,pad=0,rounding_size=.8', fc='white', ec=CORR, lw=.6,
                                zorder=3))
    ax.text(xs + w_ / 2, ys + hs_ / 2, s, ha='center', va='center', fontsize=7.0, color=CORR, zorder=4)
    if i < 2:
        arrow(xs + w_, xs + w_ + gap, ys + hs_ / 2, lw=.6, color=CORR, ms=5, zorder=5)
    xs += w_ + gap
ax.text(xC + wC / 2, Y2 + 4.6, '$q$ held fixed; fixed, linear;', ha='center', va='center', fontsize=7.0, zorder=3)
ax.text(xC + wC / 2, Y2 + 2.0, 'coefficients set once per geometry', ha='center', va='center', fontsize=7.0, zorder=3)
arrow(88.0, 102.0, y2, label=r'$\hat{u} = Fq$', size=8)
bK = box(102.0, Y2, 14.0, H2, [('$K$', 9.0)], kind='var', lw=.9)
arrow(116.0, 122.0, y2)
bT = box(122.0, Y2, 32.0, H2, [('$F^{T}$', 9.0), ('transpose of the', 7.0), ('complete extension', 7.0)],
         kind='var', lw=.9)
arrow(154.0, 164.0, y2)
ax.text(170.0, y2, r'$\widehat{S}q$', ha='center', va='center', fontsize=9)
for (x0, x1, lab, col) in ((xC, xC + wC, 'correction: improvability', CORR),
                           (bK[0], bT[0] + bT[2], 'variational form: structure', VAR)):
    bracket(x0, x1, Y2 - 1.4, col)
    ax.text((x0 + x1) / 2, Y2 - 2.2, lab, ha='center', va='top', fontsize=7.5, color=col, fontweight='bold')
box(14.0, 34.0, 160.0, 12.0, [(r'$\widehat{S}=F^{T}KF$: symmetric, $\succeq0$, rigid-body kernel', 7.0),
                              (r'$\widehat{S}-S=H^{T}AH\succeq0$ (error quadratic in the field error $H$)', 7.0),
                              (r'$S\preceq\widehat{S}\preceq\widehat{S}_{\rm net}$ (exact coarse solve, spectrum in $(0,b]$)', 7.0)],
    kind='note', lw=.6)
ax.plot([170.0, 170.0], [y2 - 3.0, 46.0], color=FS.MUTED, lw=.6, zorder=1)
ax.plot([170.0], [46.0], marker='o', ms=2.2, color=FS.MUTED, zorder=1)

# ------------------------------------------------------------------------------------------------ (b) assembly and design
YB, HB = 5.5, 16.0
ymb = YB + HB / 2
fig.text(3.5 / W, 25.5 / H, '(b) Assembly and design', fontweight='bold', fontsize=8.5, color=FS.TEXT, va='bottom', ha='left')
# stack of cell operators
for k in (2, 1):
    ax.add_patch(FancyBboxPatch((6.0 + 1.2 * k, YB + 1.2 * k), 30.0, HB, boxstyle='round,pad=0,rounding_size=1.2',
                                fc='#F3F5F7', ec='#8E99A4', lw=.6, zorder=1))
box(6.0, YB, 30.0, HB, [('cells $m$', 7.0), (r'$\widehat{S}_m=F_m^{T}K_mF_m$', 7.5), ('operators of row (a)', 7.0, FS.MUTED)])
arrow(38.4, 43.0, ymb)
box(43.0, YB, 36.0, HB, [('assembly', 7.0), (r'$\widehat{\mathbb{K}}=\Sigma_m\,B_m^{T}\,\widehat{S}_m B_m$', 7.5),
                         ('shared box-face DOFs', 7.0, FS.MUTED)])
arrow(79.0, 83.5, ymb)
box(83.5, YB, 25.0, HB, [('global solve', 7.0), (r'$\widehat{\mathbb{K}}\,\widehat{U}=f_g$', 7.5),
                         ('actions of row (a)', 7.0, FS.MUTED)])
arrow(108.5, 113.0, ymb)
box(113.0, YB, 26.0, HB, [('field recovery', 7.0), (r'$\hat{u}_m=F_mB_m\widehat{U}$', 7.5), ('every cell', 7.0, FS.MUTED)])
arrow(139.0, 143.5, ymb)
box(143.5, YB, 30.5, HB, [(r'compliance $\widehat{C}=f_g^{T}\widehat{U}$', 7.0), ('thickness sensitivity', 7.0),
                          (r'$\tilde{s}_c=-\hat{u}_m^{T}K_{,c}\,\hat{u}_m$', 7.5)])
# design loop
yd = YB - 3.4
ax.plot([158.75, 158.75, 21.0, 21.0], [YB, yd, yd, YB - .05], color=FS.MUTED, lw=.6, zorder=1)
ax.annotate('', (21.0, YB - .05), (21.0, yd), arrowprops=dict(arrowstyle='-|>', color=FS.MUTED, lw=.6, shrinkA=0, shrinkB=0,
            mutation_scale=5))
ax.text(90.0, yd - .7, r'design iteration: update of the corner parameters $\tau_c$ (new geometry $\eta$, same network '
        r'parameters $\theta$; correction coefficients recomputed)', ha='center', va='top', fontsize=7.0, color=FS.MUTED)

if '--preview' in sys.argv:
    fig.savefig(FS.Path(__file__).resolve().parent / '_preview_F01.png', dpi=220, bbox_inches='tight')
    print('preview written')
else:
    FS.save(fig, 'F01_method_overview')
    print('wrote figures/F01_method_overview.{svg,pdf,png}')
