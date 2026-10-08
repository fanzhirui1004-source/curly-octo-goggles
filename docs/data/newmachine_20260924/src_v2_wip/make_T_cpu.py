"""Dense exact port operator T = S (the lattice3.dense_T cache <body>/<case>_portview/T64.npy) on the host, for cells whose
GPU Schur-mode factorisation exceeds device memory. Columns S e_j = (K E e_j)_P through teacher.Cell.apply with the interior
factor by host PARDISO (diag_sens.cpu_factor), in blocks; T is symmetrised as in dense_T. Same port ordering as dense_T
(teacher.Cell.P; the teacher's self-check compares T q with C.apply(q) directly).
  --check <case>: compare 32 random columns against an existing T64.npy instead of writing (validation).
Run with OPL_DEV=cpu and the PARDISO environment: make_T_cpu.py <body> <case>[,...] [--check] [--block 256]"""
import os, sys, time, json, argparse
import numpy as np
import diag_sens as DS                                                         # noqa: F401  first (CPU patches)
import torch
import teacher as TE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('body'); ap.add_argument('cases'); ap.add_argument('--check', action='store_true')
    ap.add_argument('--block', type=int, default=256)
    a = ap.parse_args()
    from pathlib import Path
    for case in a.cases.split(','):
        pd = Path(a.body) / (case + '_portview')
        t0 = time.perf_counter()
        C = TE.Cell(case, a.body, log=lambda s_: None); C.assemble()
        DS.cpu_factor(C)
        rec = dict(case=case, ports=int(C.np_), interior=int(C.ni), setup_factor_s=time.perf_counter() - t0)
        if a.check:
            T = np.load(pd / 'T64.npy', mmap_mode='r')
            assert np.array_equal(np.load(pd / 'BOX_NODES.npy'), C.port_node_ids)
            idx = np.sort(np.random.default_rng(0).choice(C.np_, 32, replace=False))
            E = torch.zeros((C.np_, 32), dtype=TE.dt); E[idx, np.arange(32)] = 1
            S = C.apply(E).numpy()
            Tc = np.asarray(T[:, idx])
            rec['check_rel'] = float(np.linalg.norm(S - Tc) / np.linalg.norm(Tc))
        else:
            pd.mkdir(exist_ok=True)
            for fn in ('NODES.npy', 'CELL_INDICES.npy', 'dofs.npy', 'GP_FACES.npy'):
                if not (pd / fn).exists():
                    os.symlink(Path(a.body) / case / fn, pd / fn)
            n = C.np_
            T = np.empty((n, n), dtype=np.float64)
            t = time.perf_counter()
            for j0 in range(0, n, a.block):
                k = min(a.block, n - j0)
                E = torch.zeros((n, k), dtype=TE.dt); E[torch.arange(j0, j0 + k), torch.arange(k)] = 1
                T[:, j0:j0 + k] = C.apply(E).numpy()
            rec['columns_s'] = time.perf_counter() - t
            rec['asym_rel'] = float(np.linalg.norm(T[:512, :512] - T[:512, :512].T) / np.linalg.norm(T[:512, :512]))
            T += T.T; T *= 0.5
            np.save(pd / 'BOX_NODES.npy', C.port_node_ids)
            np.save(pd / 'T64.npy', T)
            del T
        C._free()
        print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
