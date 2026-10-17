"""Check of S25 (stream_ops.StreamedOp.spill / unspill, spill_guard; OPL_V3_SPILL): a StreamedOp over random CUDA
tensors (float64, float32, int64, a sparse CSR matrix) is spilled to OPL_V3_SPILL_DIR and read back; after the next
activation every tensor must be bitwise equal to its copy. Also: the host memory freed while spilled (RSS), the write
and read rates, and spill_guard with a forced amount (OPL_V3_SPILL_FORCE_GB) on two ops.
usage: python t_spill.py [--gb 2]"""
import os, sys, json, time, argparse
os.environ['OPL_V3_SPILL'] = '1'
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
import torch
import stream_ops as SO, hostmem as HM

ap = argparse.ArgumentParser(); ap.add_argument('--gb', type=float, default=2.0)
a = ap.parse_args()
dev = 'cuda'
g = torch.Generator(device=dev); g.manual_seed(0)
GiB = 2 ** 30


class Holder:
    pass


def make(gb):
    h = Holder()
    n = int(gb * GiB / 3 / 8)
    h.x = torch.randn(n, dtype=torch.float64, device=dev, generator=g)
    h.y = torch.randn(n, dtype=torch.float32, device=dev, generator=g)
    h.i = torch.randint(0, 2 ** 40, (n // 2,), device=dev, generator=g)
    m = 20000
    r = torch.randint(0, m, (200000,), device=dev, generator=g)
    c = torch.randint(0, m, (200000,), device=dev, generator=g)
    h.A = torch.sparse_coo_tensor(torch.stack([r, c]), torch.randn(200000, dtype=torch.float64, device=dev, generator=g),
                                  (m, m)).coalesce().to_sparse_csr()
    h.d = {'w': [torch.randn(1000, 7, device=dev, generator=g)]}
    return h


def snapshot(h):
    return dict(x=h.x.clone(), y=h.y.clone(), i=h.i.clone(), crow=h.A.crow_indices().clone(), col=h.A.col_indices().clone(),
                val=h.A.values().clone(), w=h.d['w'][0].clone())


def equal(h, ref):
    now = snapshot(h)
    return all(torch.equal(now[k], ref[k]) for k in ref)


res = {}
h1, h2 = make(a.gb), make(a.gb / 2)
ref1, ref2 = snapshot(h1), snapshot(h2)
op1, op2 = SO.StreamedOp(None, None, [h1]), SO.StreamedOp(None, None, [h2])
SO.link([op1, op2])
torch.cuda.synchronize()
with op1.active():
    res['equal_before'] = equal(h1, ref1)
d = SO.SPILL_DIR or os.path.join(os.environ.get('OPL_V3W', '/root/autodl-tmp/OPL/V3W'), 'spill')
os.makedirs(d, exist_ok=True)
HM.trim(); r0 = HM.rss_bytes()
t = time.perf_counter(); nb = op1.spill(os.path.join(d, 't_spill_op1.bin')); tw = time.perf_counter() - t
HM.trim(); r1 = HM.rss_bytes()
t = time.perf_counter(); op1.unspill(); tr = time.perf_counter() - t
r2 = HM.rss_bytes()
with op1.active():
    eq1 = equal(h1, ref1)
res['spill_one'] = dict(GiB=round(nb / GiB, 3), write_GBps=round(nb / tw / 1e9, 2), read_GBps=round(nb / tr / 1e9, 2),
                        rss_freed_GiB=round((r0 - r1) / GiB, 3), rss_back_GiB=round((r2 - r1) / GiB, 3), equal=eq1,
                        file_removed=not os.path.exists(os.path.join(d, 't_spill_op1.bin')), registered=op1.registered)
SO.SPILL_FORCE = 1                                                   # spill_guard: at least one byte -> the largest op
with SO.spill_guard(0) as info:
    inside = dict(op1_spilled=op1.spilled is not None, op2_spilled=op2.spilled is not None, rss_GiB=round(HM.rss_bytes() / GiB, 3))
SO.SPILL_FORCE = 1.2 * a.gb * GiB                                    # more than the first op: both
with SO.spill_guard(0) as info2:
    inside2 = dict(op1_spilled=op1.spilled is not None, op2_spilled=op2.spilled is not None)
with op1.active():
    e1 = equal(h1, ref1)
with op2.active():
    e2 = equal(h2, ref2)
res['guard'] = dict(info=info, inside=inside, info_both=info2, inside_both=inside2, equal_after=[e1, e2],
                    leftover_files=[f for f in os.listdir(d) if f.startswith('cell_')])
print(json.dumps(res), flush=True)
