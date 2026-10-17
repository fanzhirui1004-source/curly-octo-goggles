"""Items 1a / 2a (src_v3 OPL_V3_KPP_DISK, OPL_V3_KPP_CSR): K_PP of a plate assembled (a) as before
(MultiLattice._assemble_kpp_lean: int64 triplets of the whole matrix, sorted at once), twice (its own rerun difference),
and (b) in row blocks straight to CSR (assemble_kpp_csr), automatic block size and a small one forcing many blocks, from
the cells' triplets in host tensors and in kppstore files. Reports: rows / columns equal (bitwise), values (largest
relative difference per entry and number of entries that differ: the float64 atomics summing DOF pairs shared by cells
leave rounding differences, as two runs of (a) do), device peak above the base and seconds of each assembly; then the
cuDSS factor from both (SparseSPD triplet path / CSR path): Jacobi scales bitwise, solutions of the same right-hand sides
against each other and against the triplet path's own rerun, device peaks of the two factor setups.
usage: python t_kppcsr.py BODY_DIR LAYOUT.json [--limit N] [--small 1048576]"""
import os, sys, json, argparse, time
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
import models as MD  # noqa: F401
import torch
import teacher as TE, lat_multi as LM, lat_precond as PR, kppstore as KS

ap = argparse.ArgumentParser(); ap.add_argument('body'); ap.add_argument('layout')
ap.add_argument('--limit', type=int, default=0); ap.add_argument('--small', type=int, default=2 ** 20)
a = ap.parse_args()
dev = torch.device('cuda')
L = json.load(open(a.layout))
cells = L['cells'][:a.limit] if a.limit else L['cells']
Cs, host, disk = {}, {}, {}
t0 = time.perf_counter()
for c in cells:
    case = c['case'] + '_o000'
    C = TE.Cell(case, a.body, log=lambda s_: None, deploy=True)
    C.assemble_deploy(C.taus0)
    r, cc, v = C._kpp_cache
    host[case] = (r.int().cpu(), cc.int().cpu(), v.cpu())
    disk[case] = KS.put(case, r, cc, v)
    C._kpp_cache = None
    del r, cc, v
    Cs[case] = C
res = dict(cells=len(cells), prep_s=round(time.perf_counter() - t0, 1),
           host_MiB=round(sum(KS.nbytes(e) for e in host.values()) / 2 ** 20, 1),
           disk_equal=all(all(torch.equal(x, y) for x, y in zip(host[k], disk[k].load())) for k in host))
lay = {}
src = dict(which=host)
for c in cells:
    case = c['case'] + '_o000'
    g = LM.from_teacher(Cs[case])
    g._kpp_fn = (lambda cs: (lambda: KS.to_device(src['which'][cs], dev)))(case)
    lay[tuple(c['position'])] = g
lat = LM.MultiLattice(lay, clamp=('x', 'min'), load=('x', 'max'), loads='consistent', n_random=0, device=dev, max_cols=16,
                      log=lambda s_: None)
n = lat.nfree
res['nfree'] = n


def measured(fn):
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated(); t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, dict(s=round(time.perf_counter() - t, 3), peak_MiB=round((torch.cuda.max_memory_allocated() - base) / 2 ** 20, 1),
                     kept_MiB=round((torch.cuda.memory_allocated() - base) / 2 ** 20, 1))


def expand(crow):
    return torch.repeat_interleave(torch.arange(n, device=dev), (crow[1:] - crow[:-1]).long())


with torch.no_grad():
    (r0, c0, v0), res['lean'] = measured(lat._assemble_kpp_lean)
    (r0b, c0b, v0b), res['lean_rerun'] = measured(lat._assemble_kpp_lean)
    res['nnz_upper'] = int(len(v0))
    res['lean_rerun_cmp'] = dict(idx_equal=bool(torch.equal(r0, r0b) and torch.equal(c0, c0b)),
                                 vals_differ=int((v0 != v0b).sum()), rel=float(((v0 - v0b).abs() / v0.abs().clamp_min(1e-300)).max()))
    del r0b, c0b, v0b
    for tag, which, me in (('csr_auto_host', host, None), ('csr_small_disk', disk, a.small), ('csr_auto_disk', disk, None)):
        src['which'] = which
        (crow, cc, vv), res[tag] = measured(lambda: lat.assemble_kpp_csr(me))
        res[tag]['info'] = dict(lat.kpp_csr_info)
        rr = expand(crow)
        res[tag]['cmp_lean'] = dict(idx_equal=bool(len(rr) == len(r0) and torch.equal(rr, r0) and torch.equal(cc.long(), c0)),
                                    vals_differ=int((vv != v0).sum()) if len(vv) == len(v0) else -1,
                                    rel=float(((vv - v0).abs() / v0.abs().clamp_min(1e-300)).max()) if len(vv) == len(v0) else None)
        print(json.dumps({tag: res[tag]}), flush=True)
        del rr
        if tag != 'csr_auto_disk':
            del crow, cc, vv
    # factors: the triplet path (as before) and the CSR path, the same values in, solutions compared
    g = torch.Generator(device=dev); g.manual_seed(0)
    R = torch.randn((n, 3), dtype=torch.float64, device=dev, generator=g)
    sol1, res['spd_triplets'] = measured(lambda: PR.SparseSPD([r0, c0, v0.clone()], None, None, n, 'auto'))
    y1 = sol1(R).clone(); y1b = sol1(R).clone()
    s1 = sol1.s.clone()
    sol1.free(); del sol1
    sol2, res['spd_csr'] = measured(lambda: PR.SparseSPD(['csr', crow, cc, vv], None, None, n, 'auto'))
    y2 = sol2(R).clone()
    res['scales_equal'] = bool(torch.equal(s1, sol2.s))
    res['solve_rel_csr_vs_triplets'] = float((y2 - y1).norm() / y1.norm())
    res['solve_rel_triplets_rerun'] = float((y1b - y1).norm() / y1.norm())
    res['mem_est'] = {k: v for k, v in (sol2.mem_est or {}).items() if isinstance(v, (int, float, bool))}
    sol2.free()
KS.cleanup()
print(json.dumps(res), flush=True)
