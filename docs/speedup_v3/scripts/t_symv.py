"""Micro-benchmark of the float32 element product (P2): ke_symv._symv (current) against the single-read candidates of
ke_symv2 on one full plate cell's packed float32 element matrices (M T_up, as KMat.hold forms them), B = 1, 3, 16:
microseconds per product, error against a float64 product of the same float32 matrices (the current kernel's error for
comparison), and the difference to the current kernel.
usage: python t_symv.py BODY_DIR:CASE [--rep 50]"""
import os, sys, json, time, argparse
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
import torch
import triton
import teacher as TE
import ke_symv as KS, ke_symv2 as K2

ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--rep', type=int, default=50)
a = ap.parse_args()
body, case = a.spec.split(':')
dev = 'cuda'
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
f32 = torch.float32
Tup = C.Tm_up.reshape(125, -1).to(f32)
ch = C._lean_chunk
E = len(C.M)
P = torch.empty((E, Tup.shape[1]), dtype=f32, device=dev)
for lo in range(0, E, ch):
    torch.matmul(C.M[lo:lo + ch].to(f32), Tup, out=P[lo:lo + ch])
dofs = C.dofs.contiguous()
nb = int(dofs.max()) + 1
iu = torch.triu_indices(81, 81, device=dev)
pos = torch.full((81, 81), -1, dtype=torch.long, device=dev)
pos[iu[0], iu[1]] = torch.arange(iu.shape[1], device=dev)
sym = torch.where(pos >= 0, pos, pos.t()).reshape(-1)
g = torch.Generator(device=dev); g.manual_seed(0)
res = dict(case=case, E=E, nb=nb, dofs_dtype=str(dofs.dtype))


def timed(fn, rep):
    fn(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        fn()
    e.record(); torch.cuda.synchronize()
    return 1000 * s.elapsed_time(e) / rep


for B in (1, 3, 16):
    x = torch.randn((nb, B), dtype=f32, device=dev, generator=g).contiguous()
    y64 = torch.zeros((nb, B), dtype=torch.float64, device=dev)
    for lo in range(0, E, 2048):
        Ke = P[lo:lo + 2048][:, sym].reshape(-1, 81, 81).double()
        de = dofs[lo:lo + 2048]
        y64.index_add_(0, de.reshape(-1).long(), torch.bmm(Ke, x.double()[de.long()]).reshape(-1, B))
    scale = float(y64.abs().max())
    rows = {}

    def run(name, fn):
        y = torch.zeros((nb, B), dtype=f32, device=dev)
        fn(y)
        us = timed(lambda: fn(torch.zeros((nb, B), dtype=f32, device=dev)), a.rep)
        rows[name] = dict(us=round(us, 1), err64=float((y.double() - y64).abs().max()) / scale)
        return y

    y0 = run('v0', lambda y: KS.symv_(y, P, dofs, x))
    for bi in (4, 8):
        for w in (1, 2, 4):
            BP = triton.next_power_of_2(min(B, 16))
            run(f'v0_bi{bi}_w{w}', lambda y, bi=bi, w=w: KS._symv[(E,)](P, dofs, x, y, min(B, 16), B, BP=BP, BI=bi,
                                                                        NBLK=(81 + bi - 1) // bi, num_warps=w) if B <= 16 else None)
    for oh in (True, False):
        for bi in (4, 8, 16):
            for w in (1, 2, 4):
                if B >= 16 and bi == 16:
                    continue
                y = run(f'up_{"oh" if oh else "tw"}_bi{bi}_w{w}', lambda y, bi=bi, w=w, oh=oh: K2.symv_(y, P, dofs, x, BI=bi, warps=w, onehot=oh))
                rows[f'up_{"oh" if oh else "tw"}_bi{bi}_w{w}']['diff_v0'] = float((y - y0).abs().max()) / scale
    best = min((r['us'], k) for k, r in rows.items())
    res[f'B{B}'] = dict(best=best, rows=rows)
    print(json.dumps({f'B{B}': dict(v0=rows['v0'], best=best, best_row=rows[best[1]])}), flush=True)
print(json.dumps(res), flush=True)
