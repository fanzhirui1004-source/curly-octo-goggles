"""Step 0: operators in one interface, for the lattice harness and for training.

An operator for one cell provides, on that cell's port DOFs (teacher.Cell order: port nodes in NODES order, node-major xyz):
  field(q)  -> full displacement on all active DOFs (nb, k)  (the extension / solution operator)
  apply(q)  -> reaction on the ports (np, k)
ExtensionOp turns any LINEAR extension into an operator with the variational readout S_hat = E^T K E (exact K, fp64):
  - the rigid part of q is extended exactly as the rigid motion (least squares on the ports), only the rest goes to ext;
  - apply uses an explicit transpose if given, otherwise autograd (vjp of the linear map).
GraphLift is the zero-parameter baseline: per component, the graph-harmonic extension on the active-node graph (all
node pairs within an element, weights 1 or the element's material volume), ports held.
"""
import torch
import numpy as np
import teacher as TE

dev, dt = TE.dev, TE.dt


def rigid_raw(node_ids, n, center):
    g = np.stack(np.unravel_index(node_ids, (2 * n + 1,) * 3), 1) / (2 * n) - center
    R = np.zeros((len(g), 3, 6))
    for a in range(3):
        R[:, a, a] = 1.0
        e = np.zeros(3); e[a] = 1.0
        R[:, :, 3 + a] = np.cross(e[None, :], g)
    return torch.as_tensor(R.reshape(-1, 6), dtype=dt, device=dev)


class ExactOp:
    """Teacher: dense port Schur complement (host or device) for apply, sparse interior factor for field."""

    def __init__(self, cell, T):
        self.cell, self.T = cell, T

    def apply(self, q):
        return torch.cat([self.T[r0:r0 + 4096].to(dev) @ q for r0 in range(0, self.T.shape[0], 4096)])

    def field(self, q):
        if self.cell.sol_I is None:
            self.cell.factor(neumann=False, fp32=True)                       # + fp64 refinement in extend
        return self.cell.extend(q)


class ExtensionOp:
    def __init__(self, cell, ext, ext_T=None, rigid_exact=True):
        self.cell, self.ext, self.ext_T = cell, ext, ext_T
        self.rigid_exact = rigid_exact
        if rigid_exact:
            g = np.stack(np.unravel_index(cell.port_node_ids, (2 * cell.n + 1,) * 3), 1) / (2 * cell.n)
            c = g.mean(0)
            self.RP = rigid_raw(cell.port_node_ids, cell.n, c)
            self.RA = rigid_raw(cell.nodes, cell.n, c)
            self.RPpinv = torch.linalg.pinv(self.RP)

    def split(self, q):
        c = self.RPpinv @ q
        return c, q - self.RP @ c

    def field(self, q):
        if not self.rigid_exact:
            return self.ext(q)
        c, qd = self.split(q)
        return self.RA @ c + self.ext(qd)

    def apply(self, q):
        """(E^T K E) q with the rigid split: E q = RA c + ext(qd), c = RP^+ q, qd = (I - RP RP^+) q."""
        u = self.field(q)
        y = self.cell.K @ u                                        # nb x k, fp64
        if self.ext_T is not None:
            yd = self.ext_T(y)                                     # ports
        else:
            qq = q.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                _, qd = self.split(qq) if self.rigid_exact else (None, qq)
                ud = self.ext(qd)
                yd = torch.autograd.grad(ud, qq, grad_outputs=y.to(ud.dtype))[0].to(dt)
            if self.rigid_exact:
                return yd + self.RPpinv.T @ (self.RA.T @ y)
            return yd
        if not self.rigid_exact:
            return yd
        # adjoint of q -> (c, qd): c = RP^+ q ; qd = q - RP RP^+ q
        return (yd - self.RPpinv.T @ (self.RP.T @ yd)) + self.RPpinv.T @ (self.RA.T @ y)


class GraphLift:
    """Zero-parameter baseline: graph-harmonic extension (Laplacian over element node pairs), per component."""

    def __init__(self, cell, weight='volume'):
        self.cell = cell
        en = (cell.dofs[:, ::3] // 3)                                              # E x 27 local node ids
        w = cell.M[:, 0].abs() if weight == 'volume' else torch.ones(len(en), dtype=dt, device=dev)
        iu = torch.triu_indices(27, 27, 1, device=dev)
        a, b = en[:, iu[0]].reshape(-1), en[:, iu[1]].reshape(-1)
        ww = w[:, None].expand(-1, iu.shape[1]).reshape(-1)
        N = len(cell.nodes)
        W = torch.sparse_coo_tensor(torch.stack([torch.cat([a, b]), torch.cat([b, a])]), torch.cat([ww, ww]), (N, N)).coalesce()
        deg = torch.sparse.sum(W, 1).to_dense()
        ii = torch.arange(N, device=dev)
        L = torch.sparse_coo_tensor(torch.cat([W.indices(), torch.stack([ii, ii])], 1),
                                    torch.cat([-W.values(), deg]), (N, N)).coalesce()
        onport = torch.as_tensor(np.isin(cell.nodes, cell.port_node_ids), device=dev)
        self.Pn = torch.nonzero(onport).squeeze(1); self.In = torch.nonzero(~onport).squeeze(1)
        new = torch.full((N,), -1, dtype=torch.long, device=dev); new[self.In] = torch.arange(len(self.In), device=dev)
        r, c = L.indices(); v = L.values()
        selI = (new[r] >= 0) & (new[c] >= 0)
        up = selI & (r <= c)
        rI, cI, vI = new[r[up]], new[c[up]], v[up]
        order = torch.argsort(rI * len(self.In) + cI)
        rI, cI, vI = rI[order], cI[order], vI[order]
        dI = vI[rI == cI]
        s = torch.zeros(len(self.In), dtype=dt, device=dev); s[rI[rI == cI]] = 1 / torch.sqrt(dI)
        crow = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(torch.bincount(rI, minlength=len(self.In)), 0)])
        self.s = s
        self.sol = TE.SPDSolver(crow.int(), cI.int(), (vI * s[rI] * s[cI]).contiguous(), len(self.In))
        selIP = (new[r] >= 0) & onport[c]
        pnew = torch.full((N,), -1, dtype=torch.long, device=dev); pnew[self.Pn] = torch.arange(len(self.Pn), device=dev)
        self.LIP = torch.sparse_coo_tensor(torch.stack([new[r[selIP]], pnew[c[selIP]]]), v[selIP], (len(self.In), len(self.Pn))).coalesce()
        self.LPI = self.LIP.t().coalesce()
        self.N = N

    def _solve(self, r):
        return self.s[:, None] * self.sol.solve(self.s[:, None] * r)

    def ext(self, q):
        """q: port DOFs (np, k) node-major xyz -> full field (nb, k)."""
        k = q.shape[1]
        qn = q.reshape(-1, 3 * k)                                   # (port nodes, 3k): rows = nodes, cols = (xyz, k)
        uI = self._solve(-(self.LIP @ qn))
        u = torch.zeros((self.N, 3 * k), dtype=dt, device=dev)
        u[self.Pn] = qn; u[self.In] = uI
        return u.reshape(self.N * 3, k)

    def ext_T(self, y):
        """Adjoint: y (nb, k) -> ports (np, k)."""
        k = y.shape[1]
        yn = y.reshape(self.N, 3 * k)
        z = self._solve(yn[self.In])
        out = yn[self.Pn] - self.LPI @ z
        return out.reshape(-1, k)
