import sys, time, json
import models as MD  # noqa: F401
import torch
import teacher as TE
dev, dt = TE.dev, TE.dt
f32 = torch.float32


def T(fn, rep=10):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep):
        out = fn()
    torch.cuda.synchronize(); return out, 1000 * (time.perf_counter() - t) / rep


body, case = sys.argv[1].split(':')
C = TE.Cell(case, body, log=lambda s_: None); C.assemble(); C.lean()
Gf = (C.G.to_sparse_coo() + C.Gt.to_sparse_coo()).coalesce()
Gfull = (torch.sparse_coo_tensor(Gf.indices(), Gf.values(), Gf.shape)
         - torch.sparse_coo_tensor(torch.stack([torch.arange(C.nb, device=dev)] * 2), C.dG, Gf.shape)).coalesce().to_sparse_csr()
G32 = torch.sparse_csr_tensor(Gfull.crow_indices().int(), Gfull.col_indices().int(), Gfull.values().to(f32), Gfull.shape)
Ke32 = C.Ke.to(f32)
for B in (1, 6, 12, 16, 64):
    x = torch.randn((C.nb, B), dtype=dt, device=dev); x32 = x.to(f32)
    r = {}
    y0 = C.K @ x
    _, r['ghost32_spmm'] = T(lambda: G32 @ x32)
    _, r['ghost32_percol'] = T(lambda: torch.stack([G32 @ x32[:, j] for j in range(B)], 1))
    xe = x32[C.dofs]
    _, r['elem32_bmm'] = T(lambda: torch.bmm(Ke32, xe))

    def k32():
        y = G32 @ x32
        y.index_add_(0, C.dofs.reshape(-1), torch.bmm(Ke32, x32[C.dofs]).reshape(-1, B))
        return y.to(dt)
    y1, r['K32_total'] = T(k32)
    r['K32_rel'] = float((y1 - y0).norm() / y0.norm())
    print(json.dumps(dict(B=B, **{k: (round(v, 3) if v > 1e-3 else v) for k, v in r.items()})), flush=True)
