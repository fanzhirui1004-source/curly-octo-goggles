"""Training-time sparse hyperedge layer (same map as models.MGNO._fine / MGNO2._hyper, different evaluation):
    Z  = G(a) X              G: (H*Eh x N), G[(h,e), hn[e,s]] = a[e,s,h]
    Z' = Z W_h               per-head channel mix
    dX = S(b) Z'             S: (N x H*Eh), S[hn[e,s], (h,e)] = b[e,s,h] / deg[hn[e,s]]
Backward: dZ' = S^T dY, dW_h = Z_h^T dZ'_h, dZ = dZ' W_h^T, dX = G^T dZ,
          da[e,s,h] = <X[hn[e,s]], dZ[h,e]>,  db[e,s,h] = <dY[hn[e,s]] / deg, Z'[h,e]>   (chunked over hyperedges)
No E x 27 x B x F tensor is kept across the layer (only Z and Z', H/27 of that size), so no checkpointing is needed.
"""
import torch

dev = torch.device('cuda:0')


def _gram(A, B, chunk=4096):
    """sum_m A[h, m, :]^T B[h, m, :] for very long m: split m into chunks (a batched GEMM with many short reductions
    instead of one skinny GEMM with a 1e5-long reduction), then sum the partial products."""
    H, M, F = A.shape
    nc = M // chunk
    out = torch.zeros((H, F, B.shape[2]), dtype=A.dtype, device=A.device)
    if nc:
        a = A[:, :nc * chunk].reshape(H * nc, chunk, F)
        b = B[:, :nc * chunk].reshape(H * nc, chunk, B.shape[2])
        out += torch.bmm(a.transpose(1, 2), b).reshape(H, nc, F, -1).sum(1)
    if nc * chunk < M:
        out += torch.bmm(A[:, nc * chunk:].transpose(1, 2), B[:, nc * chunk:])
    return out


def pattern(hn, H, N):
    """CSR patterns of G / S^T (rows (h,e), cols nodes) and S / G^T (rows nodes, cols (h,e)), with the permutations that
    map the flattened (h, e, s) value arrays onto them. Depends on the geometry only (cache it)."""
    Eh = hn.shape[0]
    he = (torch.arange(H, device=hn.device)[:, None, None] * Eh + torch.arange(Eh, device=hn.device)[None, :, None]).expand(H, Eh, 27).reshape(-1)
    nodes = hn[None].expand(H, Eh, 27).reshape(-1)
    p1 = torch.argsort(he * N + nodes)
    p2 = torch.argsort(nodes * (H * Eh) + he)

    def crow(r, n):
        c = torch.zeros(n + 1, dtype=torch.int64, device=hn.device)
        c[1:] = torch.cumsum(torch.bincount(r, minlength=n), 0)
        return c.to(torch.int32)
    return dict(H=H, Eh=Eh, N=N, hn=hn,
                p1=p1, crow1=crow(he[p1], H * Eh), col1=nodes[p1].to(torch.int32),
                p2=p2, crow2=crow(nodes[p2], N), col2=he[p2].to(torch.int32))


def _mats(P, a, b, deg):
    H, Eh, N = P['H'], P['Eh'], P['N']
    va = a.permute(2, 0, 1).reshape(-1)                                    # (h, e, s) order
    vb = (b.permute(2, 0, 1) / deg[P['hn']][None]).reshape(-1)
    G = torch.sparse_csr_tensor(P['crow1'], P['col1'], va[P['p1']].contiguous(), size=(H * Eh, N))
    St = torch.sparse_csr_tensor(P['crow1'], P['col1'], vb[P['p1']].contiguous(), size=(H * Eh, N))
    S = torch.sparse_csr_tensor(P['crow2'], P['col2'], vb[P['p2']].contiguous(), size=(N, H * Eh))
    Gt = torch.sparse_csr_tensor(P['crow2'], P['col2'], va[P['p2']].contiguous(), size=(N, H * Eh))
    return G, St, S, Gt


class HyperFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, a, b, W, deg, P, chunk):
        N, B, F = X.shape
        H, Eh = P['H'], P['Eh']
        G, St, S, Gt = _mats(P, a.detach(), b.detach(), deg)
        Z = torch.sparse.mm(G, X.reshape(N, B * F)).reshape(H, Eh * B, F)
        Zp = torch.bmm(Z, W)
        dX = torch.sparse.mm(S, Zp.reshape(H * Eh, B * F)).reshape(N, B, F)
        ctx.save_for_backward(X, a, b, W, deg, Z, Zp)
        ctx.P, ctx.chunk, ctx.mats = P, chunk, (St, Gt)
        return dX

    @staticmethod
    def backward(ctx, dY):
        X, a, b, W, deg, Z, Zp = ctx.saved_tensors
        P, chunk = ctx.P, ctx.chunk
        St, Gt = ctx.mats
        N, B, F = X.shape
        H, Eh = P['H'], P['Eh']
        K = B * F
        dY = dY.contiguous()
        dZp = torch.sparse.mm(St, dY.reshape(N, K)).reshape(H, Eh * B, F)
        dW = _gram(Z, dZp)
        dZ = torch.bmm(dZp, W.transpose(1, 2))
        dX = torch.sparse.mm(Gt, dZ.reshape(H * Eh, K)).reshape(N, B, F)
        # da, db by sampled dense-dense products on the G pattern (cuSPARSE SDDMM), then back to (e, s, h)
        p1 = P['p1']
        pat = torch.sparse_csr_tensor(P['crow1'], P['col1'], torch.zeros(p1.numel(), device=X.device, dtype=X.dtype), size=(H * Eh, N))
        va = torch.sparse.sampled_addmm(pat, dZ.reshape(H * Eh, K), X.reshape(N, K).t(), beta=0.0).values()
        vb = torch.sparse.sampled_addmm(pat, Zp.reshape(H * Eh, K), (dY / deg[:, None, None]).reshape(N, K).t(), beta=0.0).values()
        fa = torch.empty_like(va); fa[p1] = va
        fb = torch.empty_like(vb); fb[p1] = vb
        da = fa.reshape(H, Eh, 27).permute(1, 2, 0).contiguous()
        db = fb.reshape(H, Eh, 27).permute(1, 2, 0).contiguous()
        return dX, da, db, dW, None, None, None


def hyper(X, a, b, W, deg, P, chunk=2048):
    return HyperFn.apply(X, a, b, W, deg, P, chunk)
