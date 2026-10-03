"""Check moments_ad against the teacher: moments equality, sensitivities vs central differences, timings, batching.
Usage: t_mad.py <body_dir> <out_json> <case> [<case> ...]"""
import sys, json, time
from pathlib import Path
from fractions import Fraction
import numpy as np
import torch
import teacher as TE
import moments_ad as MA

dev, dt = TE.dev, TE.dt


def tic():
    TE.sync(); return time.perf_counter()


def toc(t):
    TE.sync(); return time.perf_counter() - t


def fields(C, gen):
    """4 white-noise fields and 4 smooth cubic polynomial fields on all nodes: (nb, 8)."""
    X = torch.as_tensor(np.stack(np.unravel_index(C.nodes, (2 * C.n + 1,) * 3), 1) / (2 * C.n), dtype=dt, device=dev) - 0.5
    u = [torch.randn((C.nb, 4), dtype=dt, device=dev, generator=gen)]
    mons = [X[:, 0] ** a * X[:, 1] ** b * X[:, 2] ** c for a in range(4) for b in range(4) for c in range(4) if 0 < a + b + c <= 3]
    Mn = torch.stack(mons, 1)
    for _ in range(4):
        co = torch.randn((Mn.shape[1], 3), dtype=dt, device=dev, generator=gen)
        u.append((Mn @ co).reshape(-1, 1))
    return torch.cat(u, 1)


def rel(a, b):
    return ((a - b).norm(dim=0) / b.norm(dim=0)).max().item()


def main(body, out, cases):
    rec = {}
    rows = []
    for case in cases:
        r = {}
        C = TE.Cell(case, body, log=lambda s_: None)
        t = tic(); C.assemble(); r['assemble_s'] = toc(t)
        cells, taus, nrm, off = MA.cell_rows(C)
        r['elements'] = int(len(cells))
        t = tic(); M0 = C.moments(C.taus); r['moments_pt_s'] = toc(t)
        with torch.no_grad():
            t = tic(); M = MA.moments_rows(cells, C.n, taus, nrm, off, s=C.s, levels=C.levels); r['moments_ad_nograd_s'] = toc(t)
        r['M_max_rowrel_vs_pt'] = ((M - M0).abs().max(1).values / M0.abs().max(1).values.clamp_min(1e-300)).max().item()
        gen = torch.Generator(device=dev).manual_seed(0)
        u = fields(C, gen)
        g = C.energy_density(u)
        s_fd = {}
        for k in ('1e-3', '1e-4', '1e-5', '1e-6'):
            t = tic(); C.dmoments(float(k))
            if k == '1e-5':
                r['fd_dM_s'] = toc(t)
            s_fd[k] = -torch.einsum('cem,emk->ck', C.dM, g)
        C.dM = None; torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated()
        t = tic(); M1, V = MA.moments_vjp(cells, C.n, taus, nrm, off, g, s=C.s, levels=C.levels); r['vjp_K8_s'] = toc(t); r['vjp_K8_extra_GB'] = (torch.cuda.max_memory_allocated() - m0) / 2 ** 30
        s_ad = -V.sum(0)
        r['M_vjp_vs_pt'] = ((M1 - M0).abs().max(1).values / M0.abs().max(1).values.clamp_min(1e-300)).max().item()
        for k, v in s_fd.items():
            r[f'sens_rel_ad_vs_fd{k}'] = rel(s_ad, v)
            r[f'sens_rel_ad_vs_fd{k}_noise'] = rel(s_ad[:, :4], v[:, :4])
            r[f'sens_rel_ad_vs_fd{k}_poly'] = rel(s_ad[:, 4:], v[:, 4:])
        r['sens_rel_fd1e-4_vs_fd1e-5'] = rel(s_fd['1e-4'], s_fd['1e-5'])
        r['sens_rel_fd1e-6_vs_fd1e-5'] = rel(s_fd['1e-6'], s_fd['1e-5'])
        w = torch.rand(8, dtype=dt, device=dev, generator=gen)
        gw = (g * w).sum(2, keepdim=True)
        t = tic(); _, V1 = MA.moments_vjp(cells, C.n, taus, nrm, off, gw, s=C.s, levels=C.levels); r['vjp_K1_s'] = toc(t)
        r['aggregated_rel_vs_fd1e-5'] = rel(-V1.sum(0), (s_fd['1e-5'] @ w)[:, None])
        # batch sizes
        for bs in (256, 512, 1024):
            torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated()
            t = tic(); MA.moments_vjp(cells, C.n, taus, nrm, off, gw, s=C.s, levels=C.levels, batch=bs); r[f'vjp_K1_batch{bs}_s'] = toc(t)
            r[f'vjp_K1_batch{bs}_extra_GB'] = (torch.cuda.max_memory_allocated() - m0) / 2 ** 30
        r['peak_mem_GB'] = torch.cuda.max_memory_allocated() / 2 ** 30
        rows.append(dict(case=case, cells=np.asarray(C.cells), taus=list(C.taus), normal=C.normal, offset=C.offset, n=C.n,
                         s=C.s, levels=C.levels, M=M0.cpu()))
        rec[case] = r
        print(json.dumps({case: r}), flush=True)
        C._free(); del C, g, V, V1; torch.cuda.empty_cache()
    # one call over the elements of all cells (each row with its own tau and plane) == the per-cell calls
    if len(rows) > 1:
        cells = np.concatenate([x['cells'] for x in rows])
        taus = torch.cat([torch.as_tensor(np.asarray(x['taus'], float), dtype=dt, device=dev)[None].expand(len(x['cells']), 8) for x in rows])
        pl = [MA.plane_rows(x['normal'], x['offset'], len(x['cells']), dev) for x in rows]
        nrm = torch.cat([p[0] for p in pl]); off = torch.cat([p[1] for p in pl])
        n, s, lv = rows[0]['n'], rows[0]['s'], rows[0]['levels']
        with torch.no_grad():
            t = tic(); M = MA.moments_rows(cells, n, taus, nrm, off, s=s, levels=lv); tb = toc(t)
        M0 = torch.cat([x['M'] for x in rows]).to(dev)
        G = torch.randn((len(cells), 125, 1), dtype=dt, device=dev)
        t = tic(); _, V = MA.moments_vjp(cells, n, taus, nrm, off, G, s=s, levels=lv); tv = toc(t)
        lo, per = 0, []
        for x in rows:
            k = len(x['cells'])
            nn_, oo = MA.plane_rows(x['normal'], x['offset'], k, dev)
            _, Vs = MA.moments_vjp(x['cells'], n, taus[lo:lo + k], nn_, oo, G[lo:lo + k], s=s, levels=lv)
            per.append(((V[lo:lo + k].sum(0) - Vs.sum(0)).norm() / Vs.sum(0).norm()).item())
            lo += k
        rec['batched'] = dict(elements=int(len(cells)), moments_s=tb, vjp_K1_s=tv,
                              M_max_rowrel_vs_percell=((M - M0).abs().max(1).values / M0.abs().max(1).values.clamp_min(1e-300)).max().item(),
                              vjp_rel_vs_percell=per)
        print(json.dumps({'batched': rec['batched']}), flush=True)
    Path(out).write_text(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
