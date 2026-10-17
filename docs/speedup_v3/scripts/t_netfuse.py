"""Check of the fused network paths (src_v3 S24, OPL_V3_NETFUSE=1; netfused.py) on a full plate cell:
  - every U-Net convolution (forward and transposed) against the original path: bitwise (no atomics involved);
  - one forward hyperedge layer with the port clamp, one adjoint layer with the masks: against the original (the
    scatters' atomic order differs run to run, so the difference is compared with the original against itself);
  - S_hat (B = 1, 3) against the original path, and the time of each.
usage: python t_netfuse.py BODY_DIR:CASE [--rep 20]"""
import os, sys, json, time, argparse
os.environ['OPL_V3_NETFUSE'] = '1'
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
os.environ['OPL_TET_TRITON'] = '1'
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
import netfused as NF

dev = 'cuda'
ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--rep', type=int, default=20)
a = ap.parse_args()
FN.FUSED = True; FI.ON = True
body, case = a.spec.split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
with torch.no_grad():
    f.s_hat(torch.zeros((C.np_, 1), dtype=torch.float64, device=dev))
if hasattr(C, 'store64'):
    C.store64(allow=True)
g = torch.Generator(device=dev); g.manual_seed(0)
r = dict(case=case)


def timed(fn, rep):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep):
        out = fn()
    torch.cuda.synchronize(); return out, 1000 * (time.perf_counter() - t) / rep


def original(fn):
    """fn() with the fused paths switched off (the FastNet state is the same)."""
    pm8, cinv = f.pm8, f._cinv
    f.pm8, f._cinv = None, None
    try:
        return fn()
    finally:
        f.pm8, f._cinv = pm8, cinv


with torch.no_grad():
    conv = []
    for l in range(f.levels):
        t = f.c.trans[l]
        n_l = len(t['dense_idx']) if 'dense_idx' in t else 0
        for B in (1, 3):
            X = torch.randn((n_l, B, f.W_in.shape[1]), dtype=torch.float32, device=dev, generator=g)
            for k in range(2 * f.cpl):
                for tr in (False, True):
                    y1 = f._conv(X, l, k, transpose=tr)
                    y0 = original(lambda: f._conv(X, l, k, transpose=tr))
                    conv.append(bool(torch.equal(y0, y1)))
    r['conv_bitwise'] = f'{sum(conv)}/{len(conv)}'
    h0 = f.elem[0]
    N, F = f.N, f.W_in.shape[1]
    for B in (1, 3):
        X = torch.randn((N, B, F), dtype=torch.float32, device=dev, generator=g)
        X0 = torch.randn((N, B, F), dtype=torch.float32, device=dev, generator=g)
        ref = f._clamp(h0.fwd_res(X), X0)
        ref2 = f._clamp(h0.fwd_res(X), X0)
        new = h0.fwd_clamped(X, X0, f.pmask, f.pm8)
        r[f'fwd_layer_B{B}'] = dict(new_vs_orig=float((new - ref).abs().max() / ref.abs().max()),
                                     orig_vs_orig=float((ref2 - ref).abs().max() / ref.abs().max()),
                                     port_rows_equal=bool(torch.equal(new[f.pmask.reshape(-1)], X0[f.pmask.reshape(-1)])))
        Xb = torch.randn((N, B, F), dtype=torch.float32, device=dev, generator=g)
        X0b = torch.randn((N, B, F), dtype=torch.float32, device=dev, generator=g)
        zero = torch.zeros((), device=dev)
        X0b_ref = X0b + torch.where(f.pmask, Xb, zero)
        A_ref = torch.where(f.pmask, zero, Xb)
        X0b_new = X0b.clone()
        A_new = NF.mask_split(Xb, X0b_new, f.pm8)
        r[f'mask_split_B{B}'] = bool(torch.equal(A_new, A_ref) and torch.equal(X0b_new, X0b_ref))
        ref = h0.adj_res(A_ref); ref2 = h0.adj_res(A_ref)
        new = h0.adj_inplace(A_ref.clone())
        r[f'adj_layer_B{B}'] = dict(new_vs_orig=float((new - ref).abs().max() / ref.abs().max()),
                                     orig_vs_orig=float((ref2 - ref).abs().max() / ref.abs().max()))
    for B in (1, 3):
        q = torch.randn((C.np_, B), dtype=torch.float64, device=dev, generator=g)
        q = q - C.Q @ (C.Q.T @ q)
        s1, t1 = timed(lambda: f.s_hat(q), a.rep)
        s0, t0 = original(lambda: timed(lambda: f.s_hat(q), a.rep))
        s0b = original(lambda: f.s_hat(q))
        r[f's_hat_B{B}'] = dict(ms_orig=round(t0, 2), ms_fused=round(t1, 2),
                                rel_new=float(((s1 - s0).norm(dim=0) / s0.norm(dim=0)).max()),
                                rel_orig_rerun=float(((s0b - s0).norm(dim=0) / s0.norm(dim=0)).max()))
print(json.dumps(r), flush=True)
