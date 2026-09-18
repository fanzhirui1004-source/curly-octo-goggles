#!/usr/bin/env python3
"""Evaluate a route-5 run: the prediction is L with A_hat^-1 = L^T L.

The spectrum comes from one matmul rather than a triangular solve, because

    X = R_*^-T A_hat R_*^-1,   X^-1 = R_* L^T L R_*^T = (L R_*^T)^T (L R_*^T)

so nu = eig((L R_*^T)^T (L R_*^T)) equals 1/mu.  Note R_*^T, not R_*: the transposed
pair is a different matrix with different eigenvalues, and getting that wrong is how a
first version of this reported g = 7498 for an exact label instead of 1.

A_hat itself is only formed for e_A and for the assembled test; in production it never
is, which is the point of the route.
"""
import argparse, gc, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, default=None)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--inference-chunk', type=int, default=8192)
    ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    sys.path.insert(0, str(a.source))
    from stage_cutfem_m4.run import selected_rows, prepare, predict_full, write
    from stage_cutfem_m4.factors import Conditioning, load_upper
    from stage_cutfem_m4.adapter import model_for

    protocol = json.loads((a.run / 'PROTOCOL.json').read_text())
    if protocol['head'] != 'inverse':
        raise ValueError('NOT_AN_INVERSE_RUN')
    checkpoint = a.checkpoint or sorted(a.run.glob('CHECKPOINT_*.pt'))[-1]
    saved = torch.load(checkpoint, map_location='cuda:0', weights_only=False)
    conditioning = Conditioning(**saved['conditioning'])
    conditioning.validate()

    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    args = SimpleNamespace(
        manifest=Path('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'),
        seats=protocol['seats'], known_328=protocol['known_328_diagnostic'],
        patch_size=32, head='inverse', steps=protocol['steps'],
        inference_chunk=a.inference_chunk, pair_sampling=protocol['pair_sampling'],
        pairs_per_bucket=protocol['same_patch_pairs'], seed=protocol['seed'])
    rows = selected_rows(args.manifest, args.seats, args.known_328)
    samples = [prepare(r, args) for r in rows]
    model = model_for(samples[0]['context']).to('cuda:0')
    model.log_pivot_lower, model.log_pivot_upper = -20., 20.
    model.load_state_dict(saved['model'])

    out = {}
    for s in samples:
        d = s['d']
        seat = int(s['seat'])
        # the head is upper triangular with a positive diagonal, exactly like chol
        L, inference = predict_full(model, s, conditioning, a.inference_chunk, 'chol')
        packed = np.lib.format.open_memmap(a.output / f'L_PRED_UPPER_{seat:04d}.npy',
                                           mode='w+', dtype=np.float64, shape=(d * (d + 1) // 2,))
        host = L.cpu().numpy(); off = 0
        for row in range(d):
            packed[off:off + d - row] = host[row, row:]; off += d - row
        packed.flush(); del packed, host

        label = load_upper(s['factor'], d, L.device)          # G_UPPER.npy
        e_factor = float((L - label).norm() / label.norm())
        del label
        gc.collect(); torch.cuda.empty_cache()
        Rstar = load_upper(s['whitener'], d, L.device)        # R_UPPER.npy

        Q = L @ Rstar.T
        Wq = Q.T @ Q
        del Q
        Wq = (Wq + Wq.T) * .5
        nu, V = torch.linalg.eigh(Wq)
        mu_min, mu_max = 1.0 / float(nu[-1]), 1.0 / float(nu[0])
        ids = torch.cat((torch.arange(8, device=L.device),
                         torch.arange(d - 8, d, device=L.device)))
        extreme = V[:, ids]
        res = (Wq @ extreme - extreme * nu[ids]).norm(dim=0) / nu[ids].abs()
        residual = float(res.max())
        del Wq, V, extreme, res
        gc.collect(); torch.cuda.empty_cache()
        positive = bool((nu > 0).all())
        trace = float((1.0 / nu).sum())
        logdet_gap = float(-nu.log().sum())
        D = trace - logdet_gap - d
        np.save(a.output / f'EIGENVALUES_{seat:04d}.npy', (1.0 / nu).flip(0).cpu().numpy())
        del nu
        gc.collect(); torch.cuda.empty_cache()

        # A_hat = L^-1 L^-T, formed here only for e_A and the assembled test
        eye = torch.eye(d, dtype=torch.float64, device=L.device)
        Linv = torch.linalg.solve_triangular(L, eye, upper=True)
        del eye, L
        gc.collect(); torch.cuda.empty_cache()
        Ahat = Linv @ Linv.T
        del Linv
        Ahat = (Ahat + Ahat.T) * .5
        Astar = Rstar.T @ Rstar
        e_A = float((Ahat - Astar).norm() / Astar.norm())
        inverse_check = float((Ahat @ (Rstar.T @ Rstar) - torch.eye(d, dtype=torch.float64,
                                                                   device=Ahat.device)).norm() / d ** .5)
        del Astar, Rstar
        gc.collect(); torch.cuda.empty_cache()
        chol_of_Ahat, info = torch.linalg.cholesky_ex(Ahat, upper=True)
        spd = int(info) == 0
        if spd:
            cpacked = np.lib.format.open_memmap(a.output / f'A_PRED_UPPER_{seat:04d}.npy',
                                                mode='w+', dtype=np.float64,
                                                shape=(d * (d + 1) // 2,))
            chost = chol_of_Ahat.cpu().numpy(); off = 0
            for row in range(d):
                cpacked[off:off + d - row] = chost[row, row:]; off += d - row
            cpacked.flush(); del cpacked, chost
        del chol_of_Ahat, Ahat
        gc.collect(); torch.cuda.empty_cache()

        row = dict(seat=seat, d=int(d), head='inverse', step=int(saved['step']),
                   diagonal_loss=protocol.get('diagonal_loss'),
                   checkpoint=str(checkpoint), inference=inference,
                   mu_min=mu_min, mu_max=mu_max,
                   eps_op=max(abs(mu_min - 1), abs(mu_max - 1)),
                   under_stiff_factor=1.0 / mu_min if mu_min > 0 else None,
                   g=max(mu_max, 1.0 / mu_min) if mu_min > 0 else None,
                   positive_numerical_spectrum=positive,
                   maximum_extreme_eigenpair_residual=residual,
                   D=D, D_per_mode=D / d, logdet_gap=logdet_gap,
                   e_A=e_A, factor_relative=e_factor,
                   A_hat_times_A_star_minus_I=inverse_check,
                   A_hat_is_spd=spd,
                   spectrum_route='nu = eig((L R_*^T)^T (L R_*^T)); mu = 1/nu',
                   note='factor_relative compares L against the A^-1 factor label, so it is '
                        'not comparable to the chol or sqrt arms entry by entry; g is')
        out[str(seat)] = row
        print(json.dumps({k: row[k] for k in
                          ('seat', 'eps_op', 'g', 'under_stiff_factor', 'mu_max', 'mu_min',
                           'e_A', 'factor_relative', 'D_per_mode', 'A_hat_is_spd',
                           'A_hat_times_A_star_minus_I')}, indent=1), flush=True)
        write(a.output / 'INVERSE_EVAL.json',
              dict(status='INVERSE_EVAL_COMPLETE', evaluations=out, seconds=time.time() - t0))
    print('\ndone (%.0f s)' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
