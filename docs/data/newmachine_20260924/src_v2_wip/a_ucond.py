"""Direction A: stretch-conditioned network inputs for mapped cells (new module; models.py unchanged).

In V0R the network works in each node's pulled-back frame (J = R U, port values rotated by R^T), so what the map leaves for
the network to see is the right stretch U = R^T J in reference coordinates. Uniform scaling is exact (K~ = s K) and is removed:
    Ut = U / det(U)^(1/3),  L = log Ut (symmetric, traceless),  ufeat = (L11, L22, L33, sqrt2 L12, sqrt2 L23, sqrt2 L13)
per node (zero for every similarity map). attach() adds a zero-initialised linear term u_in(ufeat) to the first layer of
the node encoder, so a freshly attached model is exactly the checkpoint it was loaded from; geometries without ufeat
behave as before. prepare() also divides the nodal diag3 inputs by det(J)^(1/3) (a0_eval.write_data(det_norm=True)), so that
local volume changes do not shift the stiffness-magnitude feature."""
import numpy as np
import torch

import a0_eval as AE

dev, dt = AE.dev, AE.dt
NU = 6


def u_features(C):
    """(N, 6) float32 log of the unimodular right stretch at every node, reference frame."""
    J = AE.nodal_J(C)
    _, S, Vh = torch.linalg.svd(J)
    ls = torch.log(S); ls = ls - ls.mean(1, keepdim=True)
    V = Vh.transpose(1, 2)
    L = V @ torch.diag_embed(ls) @ Vh
    r2 = 2 ** 0.5
    return torch.stack([L[:, 0, 0], L[:, 1, 1], L[:, 2, 2], r2 * L[:, 0, 1], r2 * L[:, 1, 2], r2 * L[:, 0, 2]], 1).to(torch.float32)


def prepare(g, C, det_norm=True):
    """Before model.add_geo(g): det-normalised diag3 in g.nd (copy)."""
    if det_norm:
        det = torch.linalg.det(AE.nodal_J(C)).clamp_min(1e-300).pow(1 / 3).cpu().numpy()
        g.nd = dict(g.nd); g.nd['diag3'] = np.asarray(g.nd['diag3']) / det[:, None, None]


def set_ufeat(model, g, C):
    """After model.add_geo(g)."""
    model.caches[g.case].ufeat = u_features(C)


def attach(model):
    if hasattr(model, 'u_in'):
        return model
    Cg = model.node_in[0].out_features
    model.u_in = torch.nn.Linear(NU, Cg, bias=False).to(dev)
    torch.nn.init.zeros_(model.u_in.weight)
    NF2 = 5

    def _node_in(c, agg, m=model):
        lin = m.node_in[0]
        if not m.feat_v2:
            h = torch.nn.functional.linear(torch.cat([c.nfeat, agg], 1), lin.weight, lin.bias)
        else:
            k = lin.in_features - NF2
            h = torch.nn.functional.linear(torch.cat([c.nfeat, agg], 1), lin.weight[:, :k].contiguous(), lin.bias) + \
                torch.nn.functional.linear(c.nfeat2, lin.weight[:, k:].contiguous())
        uf = getattr(c, 'ufeat', None)
        if uf is not None:
            h = h + m.u_in(uf)
        return m.node_in[1:](h)
    model._node_in = _node_in
    return model
