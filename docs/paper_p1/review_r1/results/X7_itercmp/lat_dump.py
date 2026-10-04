"""Export one lattice's whole-lattice system for the iterative-solver comparison (AMG-PCG, BDDC).
Same cells, lattice, clamp (y = min), loads (consistent on y = max + 3 random) and global numbering as lat_direct_cpu2.py
(LD.global_matrix), so the system is the one of Table 5 route (a). Writes, into <outdir>:
  K_indptr.npy, K_indices.npy (int32), K_data.npy  full symmetric CSR of the unscaled stiffness over
                                                   [free retained | interior cell 0 | cell 1 | ...]
  b.npy (n, nl), F.npy (nfree, nl), xyz.npy (n, 3) DOF positions in cell units, comp.npy (n,) component 0/1/2
  cell<i>.npz   l2g (global ids of the cell's kept local DOFs), local full CSR (indptr, indices, data) in that order
  meta.json     sizes, timings of cell setup / assembly / global assembly (host, OPL_DEV=cpu)
Usage (env_cpu.sh): python lat_dump.py <layout.json> <outdir> [--body /root/autodl-tmp/OPL/S4/body]"""
import diag_sens as DS                                                   # noqa: F401  first: CPU environment
import sys, json, time, argparse, gc
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import trainlib as TL
import teacher as TE
import box_encode as BX
import lat_multi as LM
import lat_direct_cpu as LD

BX.dev = TL.dev


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('layout'); ap.add_argument('out')
    ap.add_argument('--body', default='/root/autodl-tmp/OPL/S4/body'); ap.add_argument('--n-random', type=int, default=3)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    L = json.loads(Path(a.layout).read_text())
    meta = dict(layout=L['name'], cells={})
    Cs, lay = [], {}
    t0 = time.perf_counter()
    for c in L['cells']:
        t = time.perf_counter(); C = TE.Cell(c['case'], a.body, log=lambda s_: None); ts = time.perf_counter() - t
        t = time.perf_counter(); C.assemble(); ta = time.perf_counter() - t
        meta['cells'][c['case']] = dict(setup_s=ts, assembly_s=ta, dofs=int(C.nb), retained=int(C.np_), interior=int(C.ni))
        Cs.append(C); lay[tuple(c['position'])] = LM.from_teacher(C)
    meta['cells_s'] = time.perf_counter() - t0
    lat = LM.MultiLattice(lay, clamp=('y', 'min'), load=('y', 'max'), loads='consistent', n_random=a.n_random,
                          device='cpu', log=lambda s_: None)
    F = lat.F.cpu().numpy().astype(np.float64)
    t = time.perf_counter()
    Ku, n, offs = LD.global_matrix(Cs, lat)
    meta['global_assembly_s'] = time.perf_counter() - t
    meta.update(global_dofs=int(n), free_retained=int(lat.nfree), nnz_upper=int(Ku.nnz), loads=lat.labels)
    print(json.dumps({k: v for k, v in meta.items() if k != 'cells'}), flush=True)
    # positions and components of every global DOF, and per-cell local systems (same map as LD.global_matrix)
    xyz = np.full((n, 3), np.nan); comp = np.full(n, -1, np.int8)
    for i, (C, c) in enumerate(zip(Cs, L['cells'])):
        g = np.full(int(C.nb), -1, np.int64)
        P, I = C.P.cpu().numpy(), C.I.cpu().numpy()
        g[P] = lat.fmap[lat.idx[i]]
        g[I] = offs[i] + np.arange(len(I))
        loc = np.arange(int(C.nb)); node = np.asarray(C.nodes)[loc // 3]
        gx = np.stack(np.unravel_index(node, (2 * C.n + 1,) * 3), 1) / (2 * C.n) + np.asarray(c['position'], float)[None, :]
        keep = g >= 0
        prev = xyz[g[keep]]
        ok = np.isnan(prev[:, 0]) | (np.abs(prev - gx[keep]).max(1) < 1e-9)
        assert ok.all(), f'cell {i}: inconsistent DOF positions on shared DOFs'
        xyz[g[keep]] = gx[keep]; comp[g[keep]] = (loc % 3)[keep]
        ru, cu, v = C.ru.cpu().numpy().astype(np.int64), C.cu.cpu().numpy().astype(np.int64), C.vals.cpu().numpy().astype(np.float64)
        lid = np.full(int(C.nb), -1, np.int64); lid[keep] = np.arange(int(keep.sum()))
        m = (lid[ru] >= 0) & (lid[cu] >= 0) & (v != 0)
        r_, c_, v_ = lid[ru[m]], lid[cu[m]], v[m]
        nl = int(keep.sum())
        U = sp.coo_matrix((v_, (np.minimum(r_, c_), np.maximum(r_, c_))), shape=(nl, nl)).tocsr(); U.sum_duplicates()
        A = (U + sp.triu(U, 1).T).tocsr(); A.sort_indices()
        np.savez(out / f'cell{i}.npz', l2g=g[keep].astype(np.int64), indptr=A.indptr.astype(np.int64),
                 indices=A.indices.astype(np.int32), data=A.data)
        meta['cells'][c['case']].update(kept=nl, local_nnz=int(A.nnz))
        del U, A; gc.collect()
    assert not np.isnan(xyz).any() and (comp >= 0).all()
    np.save(out / 'xyz.npy', xyz); np.save(out / 'comp.npy', comp)
    for C in Cs:
        C._free()
    Cs = lay = lat = None; gc.collect()
    K = (Ku + sp.triu(Ku, 1).T).tocsr(); K.sort_indices(); del Ku; gc.collect()
    np.save(out / 'K_indptr.npy', K.indptr.astype(np.int64)); np.save(out / 'K_indices.npy', K.indices.astype(np.int32))
    np.save(out / 'K_data.npy', K.data)
    b = np.zeros((n, F.shape[1])); b[:F.shape[0]] = F
    np.save(out / 'b.npy', b); np.save(out / 'F.npy', F)
    meta['nnz_full'] = int(K.nnz)
    (out / 'meta.json').write_text(json.dumps(meta, indent=1, default=float))
    print('DONE', json.dumps(dict(n=n, nnz_full=int(K.nnz))), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
