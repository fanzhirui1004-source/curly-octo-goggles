"""Tune the packed symmetric element product: row block BI, warps, index table vs arithmetic (synthetic P, real dofs)."""
import sys, time, json
import numpy as np, torch, triton, triton.language as tl
import ke_symv as KS


@triton.jit
def _symv_tab(P_ptr, tab_ptr, dofs_ptr, x_ptr, y_ptr, B, BP: tl.constexpr, BI: tl.constexpr, NBLK: tl.constexpr):
    e = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, 128)
    mj = j < 81
    c = tl.arange(0, BP)
    mc = c < B
    dj = tl.load(dofs_ptr + e * 81 + j, mask=mj, other=0).to(tl.int64)
    X = tl.load(x_ptr + dj[:, None] * B + c[None, :], mask=mj[:, None] & mc[None, :], other=0.0)
    for blk in tl.static_range(NBLK):
        i = blk * BI + tl.arange(0, BI)
        mi = i < 81
        m2 = mi[:, None] & mj[None, :]
        idx = tl.load(tab_ptr + i[:, None] * 128 + j[None, :], mask=m2, other=0)
        K = tl.load(P_ptr + e * 3321 + idx, mask=m2, other=0.0)
        Y = tl.sum(K[:, :, None] * X[None, :, :], axis=1)
        di = tl.load(dofs_ptr + e * 81 + i, mask=mi, other=0).to(tl.int64)
        tl.atomic_add(y_ptr + di[:, None] * B + c[None, :], Y, mask=mi[:, None] & mc[None, :])


dofs = torch.as_tensor(np.load(sys.argv[1]), device='cuda').long().contiguous()
E = len(dofs); nb = int(dofs.max()) + 1
P = torch.randn((E, 3321), device='cuda')
iu = torch.triu_indices(81, 81)
full = torch.zeros(81, 81, dtype=torch.long); full[iu[0], iu[1]] = torch.arange(3321); full = torch.maximum(full, full.T)
tab = torch.zeros(81, 128, dtype=torch.int32); tab[:, :81] = full.int(); tab = tab.cuda()
for B in (3, 6, 12):
    x = torch.randn((nb, B), device='cuda'); BP = triton.next_power_of_2(B)
    ref = torch.zeros((nb, B), device='cuda'); KS.symv_(ref, P, dofs, x)
    res = []
    for kind in ('arith', 'tab'):
        for BI in (4, 8, 16, 32):
            if BI * 128 * BP > 32768:
                continue
            for w in (2, 4, 8):
                nblk = (81 + BI - 1) // BI
                y = torch.zeros((nb, B), device='cuda')
                def f():
                    if kind == 'arith':
                        KS._symv[(E,)](P, dofs, x, y, B, BP=BP, BI=BI, NBLK=nblk, num_warps=w)
                    else:
                        _symv_tab[(E,)](P, tab, dofs, x, y, B, BP=BP, BI=BI, NBLK=nblk, num_warps=w)
                f(); torch.cuda.synchronize()
                err = float((y - ref).norm() / ref.norm())
                t = time.perf_counter()
                for _ in range(20): f()
                torch.cuda.synchronize(); ms = (time.perf_counter() - t) / 20 * 1000
                res.append((round(ms, 4), kind, BI, w, err))
    res.sort()
    print(json.dumps(dict(E=E, B=B, best=res[:5], default=[r for r in res if r[1] == 'arith' and r[3] == 4 and r[2] == min(32, 16384 // (128 * BP))])), flush=True)
