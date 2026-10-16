"""One fused kernel per hyperedge layer application (gather -> channel mix W_h -> scatter, H = 4) vs fused_hyper's three
steps (gather kernel, cuBLAS bmm, scatter kernel), on a real cell's layers: time per application (forward with the
folded residual, adjoint) and the relative difference of the result (the channel mix is summed in another order).
usage: python t_hfuse.py BODY_DIR:CASE [--B 1,3]"""
import os, sys, json, time, argparse, itertools
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
import models as MD  # noqa
import torch
import triton
import triton.language as tl
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI


@triton.jit
def _hfused4(X, hn, WG, WS, deg, W, out, Eh, NBF, S: tl.constexpr, F: tl.constexpr, NB: tl.constexpr, EB: tl.constexpr,
             DEG_G: tl.constexpr, DEG_S: tl.constexpr):
    """out[hn[e, s], b, :] += sum_h ws[e, s, h] (sum_s' wg[e, s', h] X[hn[e, s'], b, :]) W_h, H = 4 (W: 4 x F x F)."""
    e = tl.program_id(0) * EB + tl.arange(0, EB)
    me = e < Eh
    f = tl.arange(0, F)
    W0 = tl.load(W + 0 * F * F + f[:, None] * F + f[None, :])
    W1 = tl.load(W + 1 * F * F + f[:, None] * F + f[None, :])
    W2 = tl.load(W + 2 * F * F + f[:, None] * F + f[None, :])
    W3 = tl.load(W + 3 * F * F + f[:, None] * F + f[None, :])
    for bb in tl.static_range(NB):
        a0 = tl.zeros((EB, F), tl.float32)
        a1 = tl.zeros((EB, F), tl.float32)
        a2 = tl.zeros((EB, F), tl.float32)
        a3 = tl.zeros((EB, F), tl.float32)
        for s in tl.static_range(S):
            n = tl.load(hn + e * S + s, mask=me, other=0).to(tl.int64)
            x = tl.load(X + n[:, None] * NBF + bb * F + f[None, :], mask=me[:, None], other=0.0)
            base = (e * S + s) * 4
            w0 = tl.load(WG + base, mask=me, other=0.0)
            w1 = tl.load(WG + base + 1, mask=me, other=0.0)
            w2 = tl.load(WG + base + 2, mask=me, other=0.0)
            w3 = tl.load(WG + base + 3, mask=me, other=0.0)
            if DEG_G:
                dg = tl.load(deg + n, mask=me, other=1.0)
                w0 = w0 / dg; w1 = w1 / dg; w2 = w2 / dg; w3 = w3 / dg
            a0 += w0[:, None] * x
            a1 += w1[:, None] * x
            a2 += w2[:, None] * x
            a3 += w3[:, None] * x
        z0 = tl.dot(a0, W0, input_precision="ieee")
        z1 = tl.dot(a1, W1, input_precision="ieee")
        z2 = tl.dot(a2, W2, input_precision="ieee")
        z3 = tl.dot(a3, W3, input_precision="ieee")
        for s in tl.static_range(S):
            n = tl.load(hn + e * S + s, mask=me, other=0).to(tl.int64)
            base = (e * S + s) * 4
            w0 = tl.load(WS + base, mask=me, other=0.0)
            w1 = tl.load(WS + base + 1, mask=me, other=0.0)
            w2 = tl.load(WS + base + 2, mask=me, other=0.0)
            w3 = tl.load(WS + base + 3, mask=me, other=0.0)
            if DEG_S:
                dg = tl.load(deg + n, mask=me, other=1.0)
                w0 = w0 / dg; w1 = w1 / dg; w2 = w2 / dg; w3 = w3 / dg
            v = w0[:, None] * z0 + w1[:, None] * z1 + w2[:, None] * z2 + w3[:, None] * z3
            tl.atomic_add(out + n[:, None] * NBF + bb * F + f[None, :], v, mask=me[:, None])


def hfused(L, X, adjoint, EB=16, nw=4):
    """X (N, B, F) -> X + layer(X) (forward, fwd_res) or X + adjoint(X) (adj_res)."""
    N, B, F = X.shape
    Xf = X.reshape(N, B * F).contiguous()
    out = Xf.clone()
    if L.Eh:
        WG, WS, Wm = (L.b, L.a, L.Wt) if adjoint else (L.a, L.b, L.W)
        _hfused4[(triton.cdiv(L.Eh, EB),)](Xf, L.hn, WG, WS, L.deg, Wm, out, L.Eh, B * F, S=L.hn.shape[1], F=F, NB=B, EB=EB,
                                          DEG_G=adjoint, DEG_S=not adjoint, num_warps=nw)
    return out.view(N, B, F)


def bench(fn, rep=50):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize(); return 1e6 * (time.perf_counter() - t) / rep


ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--B', default='1,3')
a = ap.parse_args()
FN.FUSED = True; FI.ON = True
body, case = a.spec.split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
g = torch.Generator(device='cuda'); g.manual_seed(0)
rep = {}
tot = dict(orig=0.0, fused=0.0)
for kind in ('elem', 'face', 'fringe'):
    for li, L in enumerate(getattr(f, kind)):
        if L.H != 4 or L.Eh == 0:
            continue
        F = L.W.shape[-1]
        for B in [int(x) for x in a.B.split(',')]:
            X = torch.randn((L.N, B, F), dtype=torch.float32, device='cuda', generator=g)
            r = {}
            for adj in (False, True):
                ref = (L.adj_res if adj else L.fwd_res)(X)
                t0 = bench(lambda: (L.adj_res if adj else L.fwd_res)(X))
                best = None
                for EB, nw in itertools.product((16, 32), (2, 4, 8)):
                    try:
                        y = hfused(L, X, adj, EB, nw)
                    except Exception as ex:                                       # noqa
                        r.setdefault('fail', []).append([EB, nw, str(ex)[:100]]); continue
                    t1 = bench(lambda: hfused(L, X, adj, EB, nw))
                    d = float(((y - ref).abs().max() / ref.abs().max()))
                    if best is None or t1 < best[0]:
                        best = (round(t1, 1), EB, nw, d)
                r['adj' if adj else 'fwd'] = dict(orig_us=round(t0, 1), best=best)
                if B == 1 and best:
                    tot['orig'] += t0; tot['fused'] += best[0]
            rep[f'{kind}{li}_B{B}'] = dict(Eh=L.Eh, **r)
            print(json.dumps({f'{kind}{li}_B{B}': rep[f'{kind}{li}_B{B}']}), flush=True)
print(json.dumps(dict(case=case, total_B1_us=tot)), flush=True)
