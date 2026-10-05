"""Figure F14_plate_design3d (Section 5.10): designs of the plate supported on its cut in 3D, and the recovered field of one cut
cell at the final design.
(a) uniform start tau = 0.40 and (b) final NICE design (design iteration 23, checked with exact condensation in Table ST21):
    walls coloured by the local thickness parameter tau(x) of Eq. (1) (trilinear in each cell from the corner parameters;
    cividis on [0.18, 0.69], the bounds of the design, as in Figure 13); same view as Figure 1 (F00_problem); the clamped cut
    band is outlined by the support plane, the loaded end face by arrows.
(c) cut cell (3, 2, 0), next to the support and closest to the loaded end, at the final design under the design load (unit
    in-plane traction on the end face): displacement magnitude |u| on its walls from NICE (recovered field F_m B_m U_hat of
    the NICE lattice solve) and from exact condensation (E_m B_m U, exact lattice solve of the exact check), on one colour
    scale, and the magnitude of their difference; the relative differences are those of data_elev/cellfield_summary.json
    (energy norm of the cell's stiffness, Euclidean norm of the nodal displacements), copied to data_elev/F14_caption.json.
Data: data_elev/plate_design.json, plate_k000.npz, plate_k023.npz, cell320_k023.npz, cellfield_summary.json
(elev_prep_meshes.py; server scripts data_elev/nice_cell_field.py and exact_cell_field.py).
"""
import json
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import proj3d
import figstyle as FS
import elev3d as E
plt = FS.plt

ELEV, AZIM = 34, -118
TAU_LO, TAU_HI = .18, .69
TAU_CMAP = 'cividis'                                                     # as Figure 13 (fig_opt.py)
U_CMAP, D_CMAP = 'viridis', 'Greys'
CELL_VIEW = (28, -118)


def arrow2d(ax, p0, p1, **kw):
    M = ax.get_proj()
    x0, y0, _ = proj3d.proj_transform(*p0, M); x1, y1, _ = proj3d.proj_transform(*p1, M)
    ax.annotate('', (x1, y1), (x0, y0), xycoords='data', annotation_clip=False,
                arrowprops=dict(arrowstyle='-|>', shrinkA=0, shrinkB=0, **kw))


def plate(ax, D, name):
    m = E.load(name)
    tri = m['v'][m['f']]
    tf = m['tau'][m['f']].mean(1)
    cm = plt.get_cmap(TAU_CMAP)
    rgb = E.shade(cm(Normalize(TAU_LO, TAU_HI)(tf))[:, :3], tri, amb=.62)
    parts = [(E.to_plot(tri), np.c_[rgb, np.ones(len(tri))])]
    n, b = np.asarray(D['normal']), D['b_global']
    nx, ny, nz = D['shape']
    p0, p1 = E.cut_polygon(n, b, (0, 0), (nx, ny))
    E.collection(ax, parts, lw=.04)
    t = (p1 - p0) / np.linalg.norm(p1 - p0); L_ = np.linalg.norm(p1 - p0)
    hs = []
    for s in np.linspace(.3, L_ - .1, 24):                                    # support symbol on the cut, top edge
        a0 = p0 + s * t
        hs.append([np.r_[a0, 1.0], np.r_[a0 + .22 * n[:2] - .12 * t, 1.0]])
    E.lines(ax, hs, colors=E.CLAMP, linewidths=.5)
    E.lines(ax, [[np.r_[p0, 1.0], np.r_[p1, 1.0]], [np.r_[p0, 0.0], np.r_[p1, 0.0]], [np.r_[p0, 0.0], np.r_[p0, 1.0]],
                 [np.r_[p1, 0.0], np.r_[p1, 1.0]]], colors=E.CLAMP, linewidths=.9)
    c = np.asarray(D['cell'], float)
    E.lines(ax, E.box_edges(c, c + 1), colors=FS.TEXT, linewidths=.6, linestyles=(0, (2, 1.5)))
    E.setup(ax, (-.3, -.6, -.25), (nx + .3, ny + .3, nz + .3), elev=ELEV, azim=AZIM, zoom=1.6)
    for x in (.5, 1.5, 2.5, 3.5):
        arrow2d(ax, E.to_plot((x - .4, -.3, .5)), E.to_plot((x + .4, -.3, .5)), color=FS.TEXT, lw=.8, mutation_scale=6)
    print(f'{name}: tau on the surface {m["tau"].min():.3f}-{m["tau"].max():.3f}')


def cell(ax, m, val, norm, cmap, ghost=True):
    tri = m['v'][m['f']]
    vf = val[m['f']].mean(1)
    rgb = E.shade(plt.get_cmap(cmap)(norm(vf))[:, :3], tri, amb=.62)
    parts = [(E.to_plot(tri), np.c_[rgb, np.ones(len(tri))])]
    if ghost:
        rt = m['v_removed'][m['f_removed']]
        parts.append((E.to_plot(rt), np.c_[E.shade(np.array(plt.matplotlib.colors.to_rgb(E.GHOST)), rt), np.full(len(rt), .06)]))
    E.collection(ax, parts, lw=.03)
    n, b = m['normal'], float(m['offset'])
    p0, p1 = E.cut_polygon(n, b, (0, 0), (1, 1))
    E.lines(ax, [[np.r_[p0, 0], np.r_[p1, 0]], [np.r_[p0, 1], np.r_[p1, 1]], [np.r_[p0, 0], np.r_[p0, 1]],
                 [np.r_[p1, 0], np.r_[p1, 1]]], colors=E.CLAMP, linewidths=.8)
    E.lines(ax, E.box_edges((0, 0, 0), (1, 1, 1)), colors='#A3ADB8', linewidths=.5)
    E.setup(ax, (-.02, -.02, -.02), (1.02, 1.02, 1.02), elev=CELL_VIEW[0], azim=CELL_VIEW[1], zoom=1.12)


def main():
    D = E.design()
    S = json.loads((E.DATA / 'cellfield_summary.json').read_text())
    case = f"{D['case']}_o{D['k_final']:03d}"
    R = S['cells'][case]
    m = E.load('cell320_k023.npz')
    un, ue = np.linalg.norm(m['u_nice'], axis=1), np.linalg.norm(m['u_exact'], axis=1)
    du = np.linalg.norm(m['u_nice'] - m['u_exact'], axis=1)
    fig = plt.figure(figsize=(178 * FS.MM, 132 * FS.MM))
    # (a), (b): plates
    axa = fig.add_axes([.0, .55, .49, .40], projection='3d', computed_zorder=False)
    axb = fig.add_axes([.50, .55, .49, .40], projection='3d', computed_zorder=False)
    plate(axa, D, 'plate_k000.npz'); plate(axb, D, 'plate_k023.npz')
    chk = {0: (D['C0'], D['C0_exact']), D['k_final']: (D['C_final'], D['C_final_exact'])}   # exact C: Table ST21
    fig.text(.01, .975, '(a) Uniform start, τ = 0.40', fontweight='bold', fontsize=8.5, va='top')
    fig.text(.51, .975, f'(b) Final NICE design (iteration {D["k_final"]})', fontweight='bold', fontsize=8.5, va='top')
    for x, k in ((.25, 0), (.75, D['k_final'])):
        fig.text(x, .575, f'NICE $C$ = {chk[k][0]:.2f}, exact $C$ = {chk[k][1]:.2f}', ha='center', fontsize=7, color=FS.TEXT)
    cax = fig.add_axes([.30, .545, .40, .012])
    cb = fig.colorbar(ScalarMappable(Normalize(TAU_LO, TAU_HI), TAU_CMAP), cax=cax, orientation='horizontal',
                      ticks=[.18, .3, .4, .5, .6, .69])
    cb.set_label(r'local thickness parameter $\tau(x)$ on the walls', fontsize=7, labelpad=2)
    cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(.5)
    # (c): cut cell (3, 2, 0) at the final design
    vmax = float(max(un.max(), ue.max()))
    nu = Normalize(0, vmax)
    w = .27
    axs = [fig.add_axes([.03 + i * .325, .125, w, .28], projection='3d', computed_zorder=False) for i in range(3)]
    cell(axs[0], m, un, nu, U_CMAP); cell(axs[1], m, ue, nu, U_CMAP)
    nd = Normalize(0, float(du.max()))
    cell(axs[2], m, du, nd, 'magma_r')
    fig.text(.01, .455, f'(c) Cut cell {tuple(D["cell"])} at the final design, design load: displacement magnitude on the walls',
             fontweight='bold', fontsize=8.5, va='top')
    for x_, t in zip((.03 + w / 2, .355 + w / 2, .68 + w / 2), ('NICE, $|F_m B_m \\widehat U|$', 'exact condensation, $|E_m B_m U|$',
                           'difference $|F_m B_m \\widehat U - E_m B_m U|$')):
        fig.text(x_, .415, t, ha='center', va='center', fontsize=7.5)
    c1 = fig.add_axes([.12, .108, .44, .011])
    cb1 = fig.colorbar(ScalarMappable(nu, U_CMAP), cax=c1, orientation='horizontal')
    cb1.set_label('displacement magnitude $|u|$ (shared scale)', fontsize=7, labelpad=2)
    c2 = fig.add_axes([.72, .108, .21, .011])
    cb2 = fig.colorbar(ScalarMappable(nd, 'magma_r'), cax=c2, orientation='horizontal',
                       format=plt.matplotlib.ticker.FormatStrFormatter('%.2f'))
    cb2.set_label('$|u_{\\rm NICE}-u_{\\rm exact}|$ (own scale)', fontsize=7, labelpad=2)
    for c_ in (cb1, cb2):
        c_.ax.tick_params(labelsize=6.5, length=2); c_.outline.set_linewidth(.5)
    fig.text(.5, .0, f'Relative difference of the cell field, NICE vs exact: {100 * R["rel_energy_norm"]:.2f}% in the energy norm, '
             f'{100 * R["rel_l2"]:.2f}% in the nodal $\\ell_2$ norm;\nlargest difference on the walls '
             f'{100 * du.max() / ue.max():.2f}% of the largest $|u|$ on the walls', ha='center', va='bottom', fontsize=7,
             color=FS.TEXT, linespacing=1.3)
    FS.save(fig, 'F14_plate_design3d')
    cap = dict(cell=list(D['cell']), case=case, design=f"final NICE design, iteration {D['k_final']}",
               load=S['load'], clamp=S['clamp'], nice_solve=dict(C_hat=S['C_hat'], pcg=S['pcg'], true_residual=S['true_residual']),
               rel_energy_norm=R['rel_energy_norm'], rel_l2=R['rel_l2'], rel_l2_interior=R['rel_l2_interior'],
               rel_l2_retained=R['rel_l2_retained'], rel_max_magnitude=R['rel_max_magnitude'],
               local_extension_rel_energy_norm=R['local_rel_energy_norm'], local_extension_rel_l2=R['local_rel_l2'],
               max_magnitude_exact=R['max_magnitude_exact'], max_magnitude_nice=R['max_magnitude_nice'],
               surface_max_magnitude_nice=float(un.max()), surface_max_magnitude_exact=float(ue.max()),
               surface_max_difference=float(du.max()), plate=S.get('plate'),
               plate_energy_identity=dict(rel_energy_norm_squared=S['plate']['rel_energy_norm'] ** 2,
                                          compliance_gap_rel=(D['C_final_exact'] - S['C_hat']) / D['C_final_exact'],
                                          note='sum over cells of the error energy relative to the exact compliance, against '
                                               '(C - C_hat)/C of the NICE solve of this script'),
               cut_cells={c: dict(position=r['position'], rel_energy_norm=r['rel_energy_norm'], rel_l2=r['rel_l2'])
                          for c, r in S['cells'].items() if r['kind'] != 'FULL'},
               definitions=dict(rel_energy_norm='sqrt((u_hat-u)^T K_m (u_hat-u) / u^T K_m u), K_m the stabilised cell stiffness',
                                rel_l2='||u_hat-u||_2/||u||_2 over all nodal displacement DOFs of the cell',
                                local_extension='the same with u_hat replaced by F_m q_m, q_m the exact retained displacements'))
    (E.DATA / 'F14_caption.json').write_text(json.dumps(cap, indent=1, default=float))
    print(json.dumps({k: cap[k] for k in ('rel_energy_norm', 'rel_l2', 'rel_l2_interior', 'rel_l2_retained',
                                          'local_extension_rel_energy_norm', 'plate')}, indent=1, default=float))


if __name__ == '__main__':
    main()
