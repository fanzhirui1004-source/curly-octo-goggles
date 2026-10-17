"""Weight store of the matrix-free products (kmf.wstore: one float64 array of the partial elements, one row of the full
elements, lam and mu applied in the kernels) on a full plate cell: the float32 weights the kernel forms, float(lam w)
and float(mu w), against the arrays stored before ((w lam).to(float32) over all elements, the same chunked GEMM):
bitwise; the float64 ones against the float64 arrays of before (GEMM over the partial elements: last-bit level);
the products: float32 against the float64 product of the moments (as t_kmf), float64 against the stored-matrix float64
product (KMF64 off); bytes kept per cell before / after.
usage: python t_wstore.py BODY_DIR:CASE"""
import os, sys, json
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
os.environ['OPL_TET_TRITON'] = '1'; os.environ.setdefault('OPL_KE_SYMV', '1'); os.environ.setdefault('OPL_KE_SYMV64', '1')
import torch
import teacher as TE, fastidx as FI, fullke as FK, kmf as KMF

dev = 'cuda'
FI.ON = True
body, case = sys.argv[1].split(':')
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
f32, f64 = torch.float32, torch.float64
res = dict(case=case)
with torch.no_grad():
    W = KMF.wstore(C)
    xi, lam, mu = C._kmf_spec
    _, Vi = KMF.tables(xi, C.n, C.M.device)
    E = C.M.shape[0]
    sc = (2.0 * C.n) ** 2
    res.update(E=E, npart=W['npart'], nextra=W['nextra'], nfull=int((W['widx'] < 0).sum()))
    old32l, old32m = torch.empty((E, 125), dtype=f32, device=dev), torch.empty((E, 125), dtype=f32, device=dev)
    for lo in range(0, E, 8192):                                          # the float32 weights as stored before
        w = (C.M[lo:lo + 8192] @ Vi) * sc
        old32l[lo:lo + 8192] = (w * lam).to(f32)
        old32m[lo:lo + 8192] = (w * mu).to(f32)
    rows = W['widx'].long()
    wv = torch.where((rows >= 0)[:, None], W['w'][rows.clamp_min(0)], W['wfull'][None, :])
    new32l, new32m = (W['lam'] * wv).to(f32), (W['mu'] * wv).to(f32)
    res['f32_bitwise'] = bool(torch.equal(new32l, old32l) and torch.equal(new32m, old32m))
    res['f32_mismatch'] = int((new32l != old32l).sum() + (new32m != old32m).sum())
    part = W['part']
    old64 = torch.empty((len(part), 125), dtype=f64, device=dev)
    for lo in range(0, len(part), 8192):                                  # the float64 weights as stored before
        old64[lo:lo + 8192] = (C.M[part[lo:lo + 8192]] @ Vi) * sc
    new64 = W['w'][:W['npart']]
    res['f64_lam_w_rel_max'] = float(((W['lam'] * new64) - (old64 * lam)).abs().max() / (old64 * lam).abs().max())
    res['f64_bitwise'] = bool(torch.equal(W['lam'] * new64, old64 * lam) and torch.equal(W['mu'] * new64, old64 * mu))
    res['MiB'] = dict(before=round((2 * E * 125 * 4 + 2 * len(part) * 125 * 8) / 2 ** 20, 2),
                      after=round((W['w'].numel() * 8 + 125 * 8 + E * 4) / 2 ** 20, 2))
    # products
    st = KMF.state(C)
    g = torch.Generator(device=dev); g.manual_seed(0)
    X = torch.randn((C.nb, 2), dtype=f32, device=dev, generator=g)
    Tf = C.Tm.reshape(125, 81 * 81)
    Y64 = torch.zeros((C.nb, 2), dtype=f64, device=dev)
    for lo in range(0, E, 2048):
        de = C.dofs[lo:lo + 2048]
        Ke = (C.M[lo:lo + 2048] @ Tf).reshape(-1, 81, 81)
        Y64.index_add_(0, de.reshape(-1), torch.bmm(Ke, X.double()[de]).reshape(-1, 2))
    y = KMF.apply_(torch.zeros_like(X), st, X.contiguous())
    res['f32_product_rel_f64'] = float((y.double() - Y64).abs().max() / Y64.abs().max())
    x64 = torch.randn((C.nb, 3), dtype=f64, device=dev, generator=g)
    KMF.ON64 = False
    C.store64(allow=True)
    y0 = C.Kx(x64).clone()
    KMF.ON64 = True
    y1 = C.Kx(x64).clone()
    res['f64_product_rel'] = float((y1 - y0).abs().max() / y0.abs().max())
print(json.dumps(res), flush=True)
