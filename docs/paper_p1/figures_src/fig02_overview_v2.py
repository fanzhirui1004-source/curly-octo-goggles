"""Figure 2 -> figures/F01_method_overview.{svg,pdf,png}: method overview of NICE (neural-initialised condensation with
equilibrium correction), drawn with matplotlib boxes and arrows in the house style (figstyle), 178 mm wide.

  (a) Condensed operator of one cell: the chain q -> rigid split -> learned extension (trial interior field) -> rigid field
      added back, q restored on P -> E^ q -> fixed equilibrium correction W (Chebyshev, Q1 coarse solve, Chebyshev at fixed
      q) -> u^ = F q -> K -> F^T (the transpose of the complete extension) -> S^ q, with the three roles under the
      blocks (learning: trial field; correction: improvability; variational form: structure), the by-construction
      properties of the extension (Eq. 14) and the properties of S^ = F^T K F (Sections 4.1, 3.1, 4.4).
  (b) Assembly and design: cell operators -> assembled K^ -> global solve -> field recovery -> compliance and field-based
      thickness sensitivity, with the design update closing the loop (Section 6.11).
No data; mathtext only (no LaTeX installation needed).  Usage: python3 fig02_overview.py
"""
from matplotlib.patches import FancyBboxPatch, Rectangle
import figstyle as FS
plt = FS.plt

W, H = 178.0, 66.0                                                     # figure size in mm (data units of the canvas axes)
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
YA, HA = 41.0, 13.0
ym = YA + HA / 2
fig.text(3.5 / W, 61.0 / H, '(a) Condensed operator of one cell', fontweight='bold', fontsize=8.5, color=FS.TEXT,
         va='bottom', ha='left')
ax.text(6.5, ym, '$q$', ha='center', va='center', fontsize=9)
arrow(9.0, 15.0, ym)
bN = box(15.0, YA, 40.0, HA, [(r'learned extension $\widehat{E}$', 8.0), ('geometry-conditioned network,', 7.0, FS.MUTED),
                              ('linear in $q$, admissible', 7.0, FS.MUTED)], kind='learn', lw=1.0)
arrow(55.0, 67.0, ym, label=r'$\widehat{E}q$', size=7.5)
bC = box(67.0, YA, 40.0, HA, [(r'two-grid cycle $\mathcal{W}$', 8.0), (r'8 / $Q_1(17)$ / 8,', 7.0, FS.MUTED),
                              ('linear, fixed per geometry', 7.0, FS.MUTED)], kind='corr', lw=1.0)
arrow(107.0, 119.0, ym, label=r'$\hat{u}=Fq$', size=7.5)
bV = box(119.0, YA, 40.0, HA, [(r'energy form $F^{T}K$', 8.0), ('transpose of the', 7.0, FS.MUTED),
                               ('complete extension', 7.0, FS.MUTED)], kind='var', lw=1.0)
arrow(159.0, 165.0, ym)
ax.text(170.0, ym, r'$\widehat{S}q$', ha='center', va='center', fontsize=9)
yl = YA - 1.4
for (b, lab, sub, col) in ((bN, 'learning: trial field', r'$J_P\widehat{E}=I_p,\ \widehat{E}R_P=R$', LEARN),
                           (bC, 'correction: improvability', r'$S\preceq\widehat{S}\preceq\widehat{S}_{\rm net}$', CORR),
                           (bV, 'variational form: structure', r'$\widehat{S}=F^{T}KF$, $\ \widehat{S}-S=H^{T}AH\succeq0$', VAR)):
    x0, x1 = b[0], b[0] + b[2]
    bracket(x0, x1, yl, col)
    ax.text((x0 + x1) / 2, yl - .9, lab, ha='center', va='top', fontsize=7.5, color=col, fontweight='bold')
    ax.text((x0 + x1) / 2, yl - 5.0, sub, ha='center', va='top', fontsize=7.5, color=FS.TEXT)

# ------------------------------------------------------------------------------------------------ (b) assembly and design
YB, HB = 9.0, 11.0
ymb = YB + HB / 2
fig.text(3.5 / W, 24.5 / H, '(b) Assembly and design', fontweight='bold', fontsize=8.5, color=FS.TEXT, va='bottom', ha='left')
for k in (2, 1):
    ax.add_patch(FancyBboxPatch((6.0 + 1.0 * k, YB + 1.0 * k), 27.0, HB, boxstyle='round,pad=0,rounding_size=1.2',
                                fc='#F3F5F7', ec='#8E99A4', lw=.6, zorder=1))
cols = [(6.0, 27.0, [('cell operators', 7.5), (r'$\widehat{S}_m$', 8.0)]),
        (41.0, 30.0, [('assembly', 7.5), (r'$\widehat{\mathbb{K}}=\Sigma_m B_m^{T}\widehat{S}_mB_m$', 8.0)]),
        (79.0, 26.0, [('solve', 7.5), (r'$\widehat{\mathbb{K}}\,\widehat{U}=f_g$', 8.0)]),
        (113.0, 26.0, [('field recovery', 7.5), (r'$\hat{u}_m=F_mB_m\widehat{U}$', 8.0)]),
        (147.0, 27.0, [(r'compliance $\widehat{C}$', 7.5), (r'sensitivity $\tilde{s}_c$', 7.5)])]
for i, (x, w, lines) in enumerate(cols):
    box(x, YB, w, HB, lines)
    if i < len(cols) - 1:
        arrow(x + w + (2.4 if i == 0 else 0), cols[i + 1][0], ymb)
yd = YB - 3.2
ax.plot([160.5, 160.5, 19.5, 19.5], [YB, yd, yd, YB - .05], color=FS.MUTED, lw=.6, zorder=1)
ax.annotate('', (19.5, YB - .05), (19.5, yd), arrowprops=dict(arrowstyle='-|>', color=FS.MUTED, lw=.6, shrinkA=0, shrinkB=0,
            mutation_scale=5))
ax.text(90.0, yd - .6, r'design iteration: new corner parameters $\tau_c$, same network, $\mathcal{W}$ recomputed',
        ha='center', va='top', fontsize=7.0, color=FS.MUTED)

import sys
if '--preview' in sys.argv:
    fig.savefig('_preview_F01.png', dpi=220)
    print('preview written')
else:
    FS.save(fig, 'F01_method_overview')
    print('wrote figures/F01_method_overview.{svg,pdf,png}')
