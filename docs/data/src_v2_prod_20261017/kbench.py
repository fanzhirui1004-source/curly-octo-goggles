"""Micro-benchmark of the lean stiffness product K x (ghost CSR parts, element bmm, scatter) and alternatives."""
import sys, time, json
import models as MD  # noqa: F401
import torch
import teacher as TE
dev, dt = TE.dev, TE.dt


def T(fn, rep=10):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep):
        out = fn()
    torch.cuda.synchronize(); return out, 1000 * (time.perf_counter() - t) / rep


body, case = sys.argv[1].split(':')
C = TE.Cell(case, body, log=lambda s_: None); C.assemble(); C.lean()
Gf = (C.G.to_sparse_coo() + C.Gt.to_sparse_coo()).coalesce()
d = Gf.indices(); keep = torch.ones(d.shape[1], dtype=torch.bool, device=dev)
Gfull = torch.sparse_coo_tensor(d, Gf.values(), Gf.shape)
Gfull = (Gfull - torch.sparse_coo_tensor(torch.stack([torch.arange(C.nb, device=dev)] * 2), C.dG, Gf.shape)).coalesce().to_sparse_csr()
G32 = torch.sparse_csr_tensor(Gfull.crow_indices().int(), Gfull.col_indices().int(), Gfull.values(), Gfull.shape)
out = {}
for B in (1, 6, 16, 64):
    x = torch.randn((C.nb, B), dtype=dt, device=dev)
    xc = x.t().contiguous().t()
    r = {}
    y0, r['K_total'] = T(lambda: C.K @ x)
    _, r['ghost_G+Gt'] = T(lambda: C.G @ x + C.Gt @ x - C.dG[:, None] * x)
    yg, r['ghost_full_csr64'] = T(lambda: Gfull @ x)
    _, r['ghost_full_csr32idx'] = T(lambda: G32 @ x)
    _, r['ghost_full_colmajor'] = T(lambda: Gfull @ xc)
    _, r['ghost_percol'] = T(lambda: torch.stack([Gfull @ x[:, j] for j in range(B)], 1))

    def elem():
        y = torch.zeros_like(x)
        for lo in range(0, len(C.Ke), C._lean_chunk):
            de = C.dofs[lo:lo + C._lean_chunk]
            y.index_add_(0, de.reshape(-1), torch.bmm(C.Ke[lo:lo + C._lean_chunk], x[de]).reshape(-1, B))
        return y
    ye, r['elem_bmm_scatter'] = T(elem)
    _, r['elem_gather_only'] = T(lambda: x[C.dofs])
    xe = x[C.dofs]
    _, r['elem_bmm_only'] = T(lambda: torch.bmm(C.Ke, xe))
    r['check_rel'] = float(((yg + ye) - y0).norm() / y0.norm())
    out[B] = r; print(json.dumps(dict(B=B, **{k: round(v, 3) for k, v in r.items()})), flush=True)
print(json.dumps(dict(ghost_nnz_full=int(Gfull.values().numel()), E=len(C.Ke))))
