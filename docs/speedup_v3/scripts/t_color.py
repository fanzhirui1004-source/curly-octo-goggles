"""Atomic-add contention in the float32 correction product: the element product (ke_symv) and the ghost-face product
(current GF_CAT path and the fused gffused kernel) scatter into y with float atomics; neighbouring elements / faces share
nodes, and consecutive ones (one program, one warp, or programs running at the same time) collide. Here the work is
reordered by a spatial colouring (elements: parity of the cell coordinates, 8 colours, no two elements of one colour
share a node; faces: cell coordinates mod 3 and template, no two faces of one colour share a node) and timed against the
natural order: cold L2 (256 MB rewritten before each call) and back to back, B = 1 and 3. Results differ only in the
order of the atomic adds.
usage: python t_color.py BODY_DIR:CASE [--rep 30]"""
import os, sys, json, time, argparse
from pathlib import Path
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1'); os.environ.setdefault('OPL_KE_SYMV', '1')
import numpy as np
import torch
import teacher as TE
import ke_symv as KS, gffused as GFF

ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--rep', type=int, default=30)
a = ap.parse_args()
body, case = a.spec.split(':')
dev = 'cuda'
f32 = torch.float32
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
buf = torch.zeros(64 * 2 ** 20, dtype=f32, device=dev)


def ev(fn, rep):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn(); torch.cuda.synchronize()
    s.record()
    for _ in range(rep):
        fn()
    e.record(); torch.cuda.synchronize()
    return 1000 * s.elapsed_time(e) / rep                              # us


def cold(fn, rep):
    return ev(lambda: (buf.add_(1.0), fn()), rep) - ev(lambda: buf.add_(1.0), rep)


res = dict(case=case)
Tup = C.Tm_up.reshape(125, -1).to(f32)
E, ch = len(C.M), C._lean_chunk
P = torch.empty((E, Tup.shape[1]), dtype=f32, device=dev)
for lo in range(0, E, ch):
    torch.matmul(C.M[lo:lo + ch].to(f32), Tup, out=P[lo:lo + ch])
cells = torch.as_tensor(np.asarray(C.cells), device=dev)
col = (cells[:, 0] % 2) + 2 * (cells[:, 1] % 2) + 4 * (cells[:, 2] % 2)
ordE = torch.argsort(col, stable=True)
Pc, dofs = P[ordE].contiguous(), C.dofs.contiguous()
dc = dofs[ordE].contiguous()
nb = int(dofs.max()) + 1
gf = C._Kc.gf
faces = np.load(Path(C._body_dir) / C.case / 'GP_FACES.npy').astype(np.int64)
fc = np.asarray(C.cells)[faces[:, 0]]
fcol = (fc[:, 0] % 3) + 3 * (fc[:, 1] % 3) + 9 * (fc[:, 2] % 3)
perm = []
for k in range(len(gf.cuts) - 1):                                      # within each template block (GF_CAT order)
    idx = np.nonzero(faces[:, 2] == k)[0]
    o = np.argsort(fcol[idx], kind='stable')
    perm.append(gf.cuts[k] + o)
perm = torch.as_tensor(np.concatenate(perm), device=dev)
iT0 = gf.iTc
iTc = iT0[:, perm].contiguous()
g = torch.Generator(device=dev); g.manual_seed(0)
for B in (1, 3):
    x = torch.randn((nb, B), dtype=f32, device=dev, generator=g).contiguous()
    r = {}
    for tag, PP, DD in (('elem_natural', P, dofs), ('elem_colored', Pc, dc)):
        y = torch.zeros_like(x)
        r[tag] = dict(cold_us=round(cold(lambda: KS.symv_(y, PP, DD, x), a.rep), 1), warm_us=round(ev(lambda: KS.symv_(y, PP, DD, x), a.rep), 1))
    y0 = torch.zeros_like(x); KS.symv_(y0, P, dofs, x)
    y1 = torch.zeros_like(x); KS.symv_(y1, Pc, dc, x)
    r['elem_rel'] = float((y1 - y0).abs().max() / y0.abs().max())
    for tag, IT in (('gf_natural', iT0), ('gf_colored', iTc)):
        gf.iTc = IT
        GFF.ON = False
        yc = torch.zeros_like(x)
        r[tag + '_cur'] = dict(cold_us=round(cold(lambda: gf.matvec(x, torch.zeros_like(x)), a.rep), 1))
        r[tag + '_fused'] = dict(cold_us=round(cold(lambda: GFF.matvec(gf, x, torch.zeros_like(x)), a.rep), 1),
                                 warm_us=round(ev(lambda: GFF.matvec(gf, x, torch.zeros_like(x)), a.rep), 1))
    gf.iTc = iT0
    res[f'B{B}'] = r
    print(json.dumps({f'B{B}': r}), flush=True)
res['colors'] = dict(elements=int(col.max()) + 1, faces=int(fcol.max()) + 1)
print(json.dumps(res), flush=True)
