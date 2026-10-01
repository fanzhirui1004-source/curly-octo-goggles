"""Figure 2 -> figures/F01_method_overview.{svg,pdf,png}: method overview of NICE (neural-initialised condensation with
equilibrium correction), drawn with matplotlib boxes and arrows in the house style (figstyle), 178 mm wide.

  (a) Condensed operator of one cell: the chain q -> rigid split -> learned extension (trial interior field) -> rigid field
      added back, q restored on P -> E^ q -> fixed equilibrium correction W (Chebyshev, Q1 coarse solve, Chebyshev at fixed
      q) -> u^ = F q -> K -> F^T (the transpose of the complete extension) -> S^ q, with the three roles under the
      blocks (learning: trial field; correction: improvability; variational form: structure), the by-construction
      properties of the extension (Eq. 17) and the properties of S^ = F^T K F (Sections 4.1, 3.1, 4.3).
  (b) Assembly and design: cell operators -> assembled K^ -> global solve -> field recovery -> compliance and field-based
      thickness sensitivity, with the design update closing the loop (Section 6.11).
No data; mathtext only (no LaTeX installation needed).  Usage: python3 fig02_overview.py
"""
from matplotlib.patches import FancyBboxPatch, Rectangle
import figstyle as FS
plt = FS.plt

W, H = 178.0, 92.0                                                     # figure size in mm (data units of the canvas axes)
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
YA, HA = 60.5, 18.5                                                    # chain boxes: bottom and height
ym = YA + HA / 2                                                       # arrow height
fig.text(3.5 / W, 85.0 / H, '(a) Condensed operator of one cell', fontweight='bold', fontsize=8.5,
         color=FS.TEXT, va='bottom', ha='left')
ax.text(6.0, ym, '$q$', ha='center', va='center', fontsize=8)
arrow(8.0, 11.0, ym)
bS = box(11.0, YA, 17.0, HA, [('rigid split', 6.5), (r'$C_Rq\ \,|\,\ \Pi_Pq$', 6.5), ('rigid part |', 5.8, FS.MUTED),
                              ('deformation', 5.8, FS.MUTED)])
arrow(28.0, 31.0, ym)
bN = box(31.0, YA, 29.0, HA, [('learned extension', 6.5), (r'$\mathcal{N}_\theta(\eta)\,\Pi_Pq$', 7.0),
                              ('trial interior field', 5.8), ('geometry-conditioned,', 5.8), ('linear in $q$', 5.8)],
         kind='learn', lw=.9)
arrow(60.0, 63.0, ym)
bR = box(63.0, YA, 21.0, HA, [('rigid field', 6.5), ('$+\\,RC_Rq$', 6.5), ('restore $q$ on $P$', 5.8),
                              (r'$\Rightarrow\ \widehat{E}q$', 6.5)])
arrow(84.0, 92.0, ym, label=r'$\widehat{E}q$')
# correction block with the nested sequence
xC, wC = 92.0, 36.0
box(xC, YA, wC, HA, [], kind='corr', lw=.9)
ax.text(xC + wC / 2, YA + HA - 2.6, r'equilibrium correction $\mathcal{W}$', ha='center', va='center', fontsize=6.5, zorder=3)
sub = [('Chebyshev', 10.9), ('$Q_1$ coarse', 10.9), ('Chebyshev', 10.9)]
gap = 0.9
xs, ys, hs_ = xC + (wC - sum(w for _, w in sub) - 2 * gap) / 2, YA + 6.4, 4.4
for i, (s, w_) in enumerate(sub):
    ax.add_patch(FancyBboxPatch((xs, ys), w_, hs_, boxstyle='round,pad=0,rounding_size=.7', fc='white', ec=CORR, lw=.6,
                                zorder=3))
    ax.text(xs + w_ / 2, ys + hs_ / 2, s, ha='center', va='center', fontsize=5.2, color=CORR, zorder=4)
    if i < 2:
        arrow(xs + w_, xs + w_ + gap, ys + hs_ / 2, lw=.6, color=CORR, ms=4, zorder=5)
    xs += w_ + gap
ax.text(xC + wC / 2, YA + 4.1, '$q$ held fixed; fixed, linear;', ha='center', va='center', fontsize=5.6, zorder=3)
ax.text(xC + wC / 2, YA + 1.9, 'coefficients set once per geometry', ha='center', va='center', fontsize=5.6, zorder=3)
arrow(128.0, 137.0, ym, label=r'$\hat{u} = Fq$')
bK = box(137.0, YA, 8.0, HA, [('$K$', 8.0)], kind='var', lw=.9)
arrow(145.0, 148.0, ym)
bT = box(148.0, YA, 17.0, HA, [('$F^{T}$', 8.0), ('transpose of', 5.8), ('the complete', 5.8), ('extension', 5.8)],
         kind='var', lw=.9)
arrow(165.0, 168.0, ym)
ax.text(171.5, ym, r'$\widehat{S}q$', ha='center', va='center', fontsize=8)

# rigid bypass above the chain
yb = YA + HA + 2.6
ax.plot([19.5, 19.5, 73.5, 73.5], [YA + HA, yb, yb, YA + HA], color=FS.MUTED, lw=.6, zorder=1)
ax.annotate('', (73.5, YA + HA + .05), (73.5, yb), arrowprops=dict(arrowstyle='-|>', color=FS.MUTED, lw=.6, shrinkA=0,
            shrinkB=0, mutation_scale=5))
ax.text(46.5, yb + .6, 'rigid coefficients $C_Rq$ and prescribed $q$ bypass the network', ha='center', va='bottom',
        fontsize=5.8, color=FS.MUTED)

# roles under the three blocks
yl = YA - 1.2
for (x0, x1, lab, col) in ((bN[0], bN[0] + bN[2], 'learning: trial field', LEARN),
                           (xC, xC + wC, 'correction: improvability', CORR),
                           (bK[0], bT[0] + bT[2], 'variational form: structure', VAR)):
    bracket(x0, x1, yl, col)
    ax.text((x0 + x1) / 2, yl - .7, lab, ha='center', va='top', fontsize=6.5, color=col, fontweight='bold')

# call-outs: by-construction properties of the extension (left), properties of the condensed operator (right)
YC, HC = 39.0, 13.0
box(11.0, YC, 73.0, HC, [('by construction (Eq. 17)', 6.3, FS.MUTED),
                         (r'$J_P\widehat{E}=I_p$ (admissible), $\widehat{E}R_P=R$ (rigid-body motion reproduced)', 6.3),
                         (r'$\widehat{E}$ linear in $q$ at fixed geometry;', 6.3), (r'network parameters $\theta$ shared by all cells', 6.3)],
    kind='note', lw=.6)
box(92.0, YC, 82.0, HC, [(r'$\widehat{S}=F^{T}KF$: symmetric, $\succeq0$, rigid-body kernel', 6.3),
                         (r'$\widehat{S}-S=H^{T}AH\succeq0$ (error quadratic in the field error $H$)', 6.3),
                         (r'$S\preceq\widehat{S}_{\rm tg}\preceq\widehat{S}_0$ (correction cannot increase the error)', 6.3)],
    kind='note', lw=.6)
ax.plot([171.5, 171.5], [ym - 2.4, YC + HC], color=FS.MUTED, lw=.6, zorder=1)
ax.plot([171.5], [YC + HC], marker='o', ms=2.2, color=FS.MUTED, zorder=1)

# ------------------------------------------------------------------------------------------------ (b) assembly and design
YB, HB = 9.5, 16.0
ymb = YB + HB / 2
fig.text(3.5 / W, 30.0 / H, '(b) Assembly and design', fontweight='bold', fontsize=8.5, color=FS.TEXT, va='bottom', ha='left')
# stack of cell operators
for k in (2, 1):
    ax.add_patch(FancyBboxPatch((6.0 + 1.2 * k, YB + 1.2 * k), 28.0, HB, boxstyle='round,pad=0,rounding_size=1.2',
                                fc='#F3F5F7', ec='#8E99A4', lw=.6, zorder=1))
box(6.0, YB, 28.0, HB, [('cells $m$', 6.5), (r'$\widehat{S}_m=F_m^{T}K_mF_m$', 6.8), ('operators of row (a)', 5.8, FS.MUTED)])
arrow(36.4, 40.0, ymb)
box(40.0, YB, 36.0, HB, [('assembly', 6.5), (r'$\widehat{\mathbb{K}}=\Sigma_m\,B_m^{T}\,\widehat{S}_m B_m$', 7.0),
                         ('shared box-face DOFs', 5.8, FS.MUTED)])
arrow(76.0, 80.0, ymb)
box(80.0, YB, 24.0, HB, [('global solve', 6.5), (r'$\widehat{\mathbb{K}}\,\widehat{U}=f_g$', 7.0),
                         ('actions of row (a)', 5.8, FS.MUTED)])
arrow(104.0, 108.0, ymb)
box(108.0, YB, 26.0, HB, [('field recovery', 6.5), (r'$\hat{u}_m=F_mB_m\widehat{U}$', 7.0), ('every cell', 5.8, FS.MUTED)])
arrow(134.0, 138.0, ymb)
box(138.0, YB, 36.0, HB, [(r'compliance $\widehat{C}=f_g^{T}\widehat{U}$', 6.5), ('thickness sensitivity', 6.5),
                          (r'$\tilde{s}_c=-\hat{u}_m^{T}K_{,c}\,\hat{u}_m$', 7.0)])
# design loop
yd = YB - 3.6
ax.plot([156.0, 156.0, 20.0, 20.0], [YB, yd, yd, YB - .05], color=FS.MUTED, lw=.6, zorder=1)
ax.annotate('', (20.0, YB - .05), (20.0, yd), arrowprops=dict(arrowstyle='-|>', color=FS.MUTED, lw=.6, shrinkA=0, shrinkB=0,
            mutation_scale=5))
ax.text(88.0, yd - .7, r'design iteration: update of the corner parameters $\tau_c$ (new geometry $\eta$, same network '
        r'parameters $\theta$; correction coefficients recomputed)', ha='center', va='top', fontsize=5.8, color=FS.MUTED)

FS.save(fig, 'F01_method_overview')
print('wrote figures/F01_method_overview.{svg,pdf,png}')
