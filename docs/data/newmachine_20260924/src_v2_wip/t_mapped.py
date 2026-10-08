"""Unit tests of mapped_cell.py on real cells (GPU server; no factorisations).

U1 identity map: mapped K = teacher K (products with random vectors), the rule's plain moments = cell_moments, and the
   physical-coordinate ghost matrix = the cached unit ghost matrix (GP_UPPER.npz).
U2 rigid rotation R (general axis): K~ (R x) = R (K x).
U3 uniform scaling s: K~ = s K (elastic and ghost parts scale alike).
U4 affine shear/stretch: element matrices of the point-wise route = the constant-A route sum_c A_c sum_m M_m Tt_cm built
   from the geometric moments (independent of the point-wise Jacobians).
U5 rigid kernel of the mapped K~ under twist, bend, grade and Bezier maps: ||K~ Q_phys|| against the reference-coordinate
   rigid modes and against random vectors; element PSD sample; min det J.
Usage: t_mapped.py <body_dir> <out_json> <case> [<case> ...]
"""
import json, sys, time, gc
from pathlib import Path
import numpy as np
import torch
import teacher as TE
import mapped_cell as MC

dev, dt = TE.dev, TE.dt


def rel(a, b):
    return float((a - b).norm() / b.norm())


def node_rotate(x, A):
    """Apply the 3 x 3 matrix A to every node block of x (nb, k)."""
    return torch.einsum('ij,njk->nik', A, x.reshape(-1, 3, x.shape[1])).reshape(x.shape)


def rot_matrix(axis, deg):
    a = np.asarray(axis, float); a /= np.linalg.norm(a); t = np.deg2rad(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def ghost_compare(C0, Cm, body_dir, case):
    z = np.load(Path(body_dir) / case / 'GP_UPPER.npz')
    k0 = torch.as_tensor(z['keys'], device=dev); v0 = torch.as_tensor(z['vals'], device=dev)
    k1, v1 = Cm.ghost_unit
    U = torch.unique(torch.cat([k0, k1]))
    a = torch.zeros(len(U), dtype=dt, device=dev).index_add_(0, torch.searchsorted(U, k0), v0)
    b = torch.zeros(len(U), dtype=dt, device=dev).index_add_(0, torch.searchsorted(U, k1), v1)
    ratio = float((b @ a) / (a @ a))
    return dict(ghost_rel=rel(b, a), ghost_ls_ratio=ratio, ghost_keys_ref=int(len(k0)), ghost_keys_new=int(len(k1)))


def one(case, body_dir, log):
    rec = dict(case=case)
    gen = torch.Generator(device=dev).manual_seed(0)
    t = time.perf_counter()
    C0 = TE.Cell(case, body_dir, log=lambda s_: None).assemble()
    rec['teacher_assemble_s'] = time.perf_counter() - t
    X = torch.randn((C0.nb, 4), dtype=dt, device=dev, generator=gen)
    KX = C0.K @ X
    # ---- U1 identity
    t = time.perf_counter()
    Cm = MC.MappedCell(case, body_dir, {'kind': 'identity'}, log=lambda s_: None).assemble()
    rec['mapped_assemble_s'] = time.perf_counter() - t
    rec['U1_K_rel'] = rel(Cm.K @ X, KX)
    rec['U1_moments_rel'] = rel(Cm.M, C0.M)
    rec.update({'U1_' + k: v for k, v in ghost_compare(C0, Cm, body_dir, case).items()})
    rec['U1_rigid_Q_rel'] = rel(Cm.Q, TE.rigid_basis(C0.port_node_ids, C0.n)) if Cm.Q.shape == C0.Q.shape else None
    rec['map_stats_identity'] = Cm.map_stats
    del Cm; gc.collect(); torch.cuda.empty_cache()
    log(json.dumps(dict(event='U1', **{k: v for k, v in rec.items() if k.startswith('U1')})))
    # ---- U2 rotation
    A = rot_matrix((1.0, 2.0, 0.5), 37.0)
    Cm = MC.MappedCell(case, body_dir, {'kind': 'affine', 'A': A.tolist()}, log=lambda s_: None).assemble()
    At = torch.as_tensor(A, dtype=dt, device=dev)
    rec['U2_rotation_rel'] = rel(Cm.K @ node_rotate(X, At), node_rotate(KX, At))
    del Cm; gc.collect(); torch.cuda.empty_cache()
    # ---- U3 scaling
    Cm = MC.MappedCell(case, body_dir, {'kind': 'scale', 's': 1.7}, log=lambda s_: None).assemble()
    rec['U3_scale_rel'] = rel(Cm.K @ X, 1.7 * KX)
    del Cm; gc.collect(); torch.cuda.empty_cache()
    log(json.dumps(dict(event='U2U3', U2=rec['U2_rotation_rel'], U3=rec['U3_scale_rel'])))
    # ---- U4 affine (shear + stretch): point-wise route vs constant-A route on the elements
    A = np.array([[1.3, 0.25, 0.1], [0.0, 0.8, 0.2], [0.05, 0.0, 1.1]])
    Cm = MC.MappedCell(case, body_dir, {'kind': 'affine', 'A': A.tolist()}, log=lambda s_: None)
    Mt = MC.mapped_moments(Cm.cells, Cm.n, Cm.taus0, Cm.normal, Cm.offset, Cm.s, Cm.levels, Cm.surface, Cm.xe, Cm.xi_nodes,
                           Cm.lam, Cm.mu).reshape(len(Cm.cells), -1)
    Jt = torch.as_tensor(A / (2 * Cm.n), dtype=dt, device=dev)
    Ac, _ = MC.a_tensor(Jt, Cm.lam, Cm.mu)
    pi = [p for p, _ in MC.PAIRS]; qi = [q for _, q in MC.PAIRS]
    Mg = C0.M * (2 * C0.n) ** 3                                                 # t-measure geometric moments
    Mt_alt = (Ac[pi, qi][None, :, None] * Mg[:, None, :]).reshape(len(Mg), -1)
    sel = torch.randperm(len(Mg), generator=torch.Generator().manual_seed(1))[:2000].to(dev)
    Ke = Mt[sel] @ Cm.Tt_up; Ke_alt = Mt_alt[sel] @ Cm.Tt_up
    rec['U4_affine_element_rel'] = rel(Ke, Ke_alt)
    del Cm, Mt, Mt_alt; gc.collect(); torch.cuda.empty_cache()
    log(json.dumps(dict(event='U4', U4=rec['U4_affine_element_rel'])))
    # ---- U5 rigid kernel under non-affine maps
    rec['U5'] = {}
    for spec in ({'kind': 'twist', 'deg': 30}, {'kind': 'bend', 'R': 3}, {'kind': 'grade', 'ratio': 1.5},
                 {'kind': 'bezier', 'amp': 0.1, 'seed': 1}):
        t = time.perf_counter()
        Cm = MC.MappedCell(case, body_dir, spec, log=lambda s_: None).assemble()
        sec = time.perf_counter() - t
        R = Cm.Qall
        Qref = TE.rigid_basis(Cm.nodes, Cm.n)
        Xr = torch.linalg.qr(torch.randn((Cm.nb, 6), dtype=dt, device=dev, generator=gen))[0]
        nr = float((Cm.K @ R).norm()); nref = float((Cm.K @ Qref).norm()); nx = float((Cm.K @ Xr).norm())
        # element PSD sample (2000 elements)
        Mt = MC.mapped_moments(Cm.cells[:2000], Cm.n, Cm.taus0, Cm.normal, Cm.offset, Cm.s, Cm.levels, Cm.surface,
                               Cm.xe[:2000], Cm.xi_nodes, Cm.lam, Cm.mu).reshape(min(2000, len(Cm.cells)), -1)
        iu = Cm.iu
        Ku = Mt @ Cm.Tt_up
        Kf = torch.zeros((len(Ku), 81, 81), dtype=dt, device=dev); Kf[:, iu[0], iu[1]] = Ku; Kf[:, iu[1], iu[0]] = Ku
        ev = torch.linalg.eigvalsh(Kf)
        r = dict(rigid_phys_over_random=nr / nx, rigid_ref_over_random=nref / nx,
                 element_min_eig_over_max=float((ev[:, 0] / ev[:, -1].clamp_min(1e-300)).min()),
                 assemble_s=sec, **Cm.map_stats)
        rec['U5'][spec['kind']] = r
        log(json.dumps(dict(event='U5', spec=spec, **r)))
        del Cm, Mt, Ku, Kf; gc.collect(); torch.cuda.empty_cache()
    rec['gpu_peak_gb'] = torch.cuda.max_memory_allocated() / 2 ** 30
    return rec


if __name__ == '__main__':
    body_dir, out = sys.argv[1], sys.argv[2]
    recs = []
    for case in sys.argv[3:]:
        torch.cuda.reset_peak_memory_stats()
        r = one(case, body_dir, log=lambda s_: print(s_, flush=True))
        print(json.dumps(r), flush=True)
        recs.append(r)
        Path(out).write_text(json.dumps(recs, indent=2))
    print('DONE', flush=True)
