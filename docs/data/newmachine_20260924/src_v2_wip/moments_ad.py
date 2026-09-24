"""Differentiable element moments (torch) and design sensitivities by reverse mode (task 54).

Same geometry and arithmetic as polyref_torch_fast.cell_moments (piecewise-linear psi on Kuhn tetrahedra of an s^3
sub-grid with `levels` octree refinements near the boundary, clipped by psi1 = tau - f, psi2 = tau + f,
psi3 = offset - n.x; closed-form moments on full sub-cubes, collapsed Gauss rule on clipped tetrahedra), with two
changes:
  - every row (element) carries its own 8 corner values tau and its own plane (normal 0 / offset 1 = no plane), so the
    elements of many lattice cells go through one call;
  - the result stays a torch tensor that is differentiable in tau at fixed clip topology (the clip points are linear
    interpolations t = v_i / (v_i - v_j) of the psi values, which are linear in tau).
The derivative is the exact derivative of the discrete moments within the fixed clip topology; the central differences
of Cell.dmoments approximate the same quantity.

Design sensitivities: for an energy density g[e, m, k] = u_e^T Tm_m u_e (fixed fields u_k),
    s[c, k] = -u_k^T (dK/dtau_c) u_k = -d/dtau_c sum_{e,m} M_em(tau) g[e, m, k]
is one reverse pass per load (or one pass for a weighted sum of loads, which is what a design objective needs),
instead of 16 moment evaluations for the full dM/dtau.
"""
from itertools import product
import numpy as np
import torch
import torch.utils.checkpoint

from element_polyref import KUHN, CUBE, tet_rule
from polyref_torch_fast import _clip

dt = torch.float64


class _Rule:
    cache = {}

    @classmethod
    def get(cls, device, rule_order):
        key = (str(device), rule_order)
        if key not in cls.cache:
            trule = tet_rule(rule_order)
            cls.cache[key] = dict(
                tref=torch.tensor(trule[0], dtype=dt, device=device), tw=torch.tensor(trule[1], dtype=dt, device=device),
                kuhn=torch.tensor(KUHN, device=device), cube=torch.tensor(CUBE, dtype=dt, device=device),
                kp1=torch.arange(1, 6, dtype=dt, device=device))
        return cls.cache[key]


def _jac(tets):
    E_ = torch.stack([tets[:, 1] - tets[:, 0], tets[:, 2] - tets[:, 0], tets[:, 3] - tets[:, 0]], 1)
    return torch.linalg.det(E_).abs()


def _quad(tets, tref, tw):
    """125 moments of each tetrahedron (tensor-product form, as polyref_torch_fast). Checkpointed under autograd: only
    the vertices are kept for the backward pass, the quadrature intermediates are recomputed."""
    E_ = torch.stack([tets[:, 1] - tets[:, 0], tets[:, 2] - tets[:, 0], tets[:, 3] - tets[:, 0]], 1)
    J = torch.linalg.det(E_).abs()
    P = tets[:, None, 0, :] + torch.einsum('qk,tkd->tqd', tref, E_)
    P2 = P * P
    pw = torch.stack([torch.ones_like(P), P, P2, P2 * P, P2 * P2], -1)
    X = pw[:, :, 0, :] * (J[:, None] * tw[None])[..., None]
    YZ = (pw[:, :, 1, :, None] * pw[:, :, 2, None, :]).reshape(len(P), -1, 25)
    return torch.bmm(X.transpose(1, 2), YZ).reshape(len(P), 125)


def plane_rows(normal, offset, E, device):
    """(E,3) normals and (E,) offsets; no plane -> normal 0, offset 1 (psi3 = 1 everywhere)."""
    if normal is None:
        return torch.zeros((E, 3), dtype=dt, device=device), torch.ones(E, dtype=dt, device=device)
    nrm = torch.as_tensor(np.asarray(normal, float), dtype=dt, device=device)[None].expand(E, 3).contiguous()
    return nrm, torch.full((E,), float(offset), dtype=dt, device=device)


def moments_rows(cells, n, taus, nrm, off, s=4, levels=1, rule_order=4, batch=2048):
    """125 moments per row, scaled like cell_moments (physical measure, local xi coordinates).
    cells (E,3) integer cell indices; taus (E,8) torch float64 (may require grad); nrm (E,3), off (E,) torch."""
    device = taus.device
    R = _Rule.get(device, rule_order)
    tref, tw, kuhn, cube, kp1 = R['tref'], R['tw'], R['kuhn'], R['cube'], R['kp1']
    cells_t = torch.as_tensor(np.asarray(cells), dtype=dt, device=device)
    idx = torch.tensor(list(product(range(s), repeat=3)), dtype=dt, device=device)
    cmask = cube.bool().view(8, 1, 1, 3)
    outs = []
    for b0 in range(0, len(cells_t), batch):
        cb = cells_t[b0:b0 + batch]; nb = len(cb)
        tb, nb_, ob = taus[b0:b0 + batch], nrm[b0:b0 + batch], off[b0:b0 + batch]

        def psi_at(own, xi):                                     # xi: C x 8 x 3 local in [-1, 1]
            phys = (cb[own][:, None, :] + (xi + 1) / 2) / n
            f = torch.cos(2 * torch.pi * phys).sum(-1)
            w8 = torch.where(cmask, phys[None], 1 - phys[None]).prod(-1)          # 8(corner) x C x 8
            tau = torch.einsum('Cc,cCk->Ck', tb[own], w8)
            p3 = ob[own][:, None] - (phys * nb_[own][:, None, :]).sum(-1)
            return torch.stack([tau - f, tau + f, p3], -1)

        parts = []                                               # (owner, 125 moments) pieces, summed at the end
        h = 2 / s
        owner = torch.arange(nb, device=device).repeat_interleave(len(idx))
        lo = (-1 + h * idx).repeat(nb, 1)
        for level in range(levels + 1):
            cx = lo[:, None, :] + h * cube[None]
            cp = psi_at(owner, cx)
            cpd = cp.detach()
            full = (cpd >= 0).all(-1).all(-1); empty = (cpd < 0).all(1).any(-1)
            part = ~full & ~empty
            if full.any():
                lf = lo[full]; hf = lf + h
                m1 = (hf[:, :, None] ** kp1 - lf[:, :, None] ** kp1) / kp1
                mono = (m1[:, 0, :, None, None] * m1[:, 1, None, :, None] * m1[:, 2, None, None, :]).reshape(-1, 125)
                parts.append((owner[full], mono))
            if not part.any():
                break
            if level < levels:
                lo = (lo[part][:, None, :] + (h / 2) * cube[None]).reshape(-1, 3)
                owner = owner[part].repeat_interleave(8)
                h = h / 2
                continue
            tets = cx[part][:, kuhn].reshape(-1, 4, 3)
            attr = cp[part][:, kuhn].reshape(-1, 4, 3)
            town = owner[part].repeat_interleave(6)
            for k in range(3):
                tets, attr, town = _clip(tets, attr, town, k)
            if len(tets):
                with torch.no_grad():
                    keep = _jac(tets) > 0
                tets, town = tets[keep], town[keep]
                ck = torch.is_grad_enabled() and tets.requires_grad
                for lo_t in range(0, len(tets), 65536):
                    tc = tets[lo_t:lo_t + 65536]
                    mom = torch.utils.checkpoint.checkpoint(_quad, tc, tref, tw, use_reentrant=False) if ck else _quad(tc, tref, tw)
                    parts.append((town[lo_t:lo_t + 65536], mom))
        M = torch.zeros((nb, 125), dtype=dt, device=device)
        for o, v in parts:
            M = M.index_add(0, o, v)
        outs.append(M)
    return torch.cat(outs) * (1 / (2 * n)) ** 3


def moments_vjp(cells, n, taus, nrm, off, G, s=4, levels=1, batch=512, rule_order=4):
    """Moments M (E,125) and the per-row vector-Jacobian products
        V[e, c, k] = sum_m (dM_em / dtau_{e,c}) G[e, m, k]        (G: (E,125,K) float64)
    Each row's moments depend only on its own tau row, so summing V over the rows of one lattice cell gives
    d/dtau_c sum_{e,m} M_em G_emk for that cell. One forward (with graph) and K backward passes per row batch."""
    E, K = len(cells), G.shape[2]
    M = torch.empty((E, 125), dtype=dt, device=taus.device)
    V = torch.zeros((E, 8, K), dtype=dt, device=taus.device)
    for b0 in range(0, E, batch):
        b1 = min(E, b0 + batch)
        tb = taus[b0:b1].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            Mb = moments_rows(cells[b0:b1], n, tb, nrm[b0:b1], off[b0:b1], s=s, levels=levels, rule_order=rule_order,
                              batch=b1 - b0)
            for k in range(K):
                gk, = torch.autograd.grad((Mb * G[b0:b1, :, k]).sum(), tb, retain_graph=k < K - 1, allow_unused=True)
                if gk is not None:
                    V[b0:b1, :, k] = gk
        M[b0:b1] = Mb.detach()
        del Mb, tb
    return M, V


def cell_rows(cell):
    """Row inputs of one teacher Cell at its current taus."""
    E = len(cell.cells)
    taus = torch.as_tensor(np.asarray(cell.taus, float), dtype=dt, device=cell.Tm.device)[None].expand(E, 8).contiguous()
    nrm, off = plane_rows(cell.normal, cell.offset, E, cell.Tm.device)
    return cell.cells, taus, nrm, off


def cell_sens(cell, g, batch=512):
    """-u^T dK/dtau u for the fields behind g = cell.energy_density(u): (8, K), by reverse mode."""
    cells, taus, nrm, off = cell_rows(cell)
    _, V = moments_vjp(cells, cell.n, taus, nrm, off, g, s=cell.s, levels=cell.levels, batch=batch)
    return -V.sum(0)
