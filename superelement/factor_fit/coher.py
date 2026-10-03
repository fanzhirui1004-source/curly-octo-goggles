#!/usr/bin/env python3
"""The per-entry requirement depends on how CORRELATED the errors are, so measure that too.

perentry.py used independent zero-mean relative noise and found +-3% needs about 2.7e-4 per entry
on the factor.  Independent errors add incoherently, which is the most forgiving case and is not
how a network fails: a network's error field is smooth in the geometry, so neighbouring entries are
wrong in the same direction, and coherent errors add linearly instead of as a square root.

Model the coherence length directly.  Partition the coordinates into spatial blocks of side B and
give every entry inside one block pair the SAME relative offset, drawn once.  B = 1 is the
independent case; larger B is a longer coherence length; B = d is a single global bias.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
sys.path.insert(0, '/root/autodl-tmp/NEURAL_SCHUR')
import v1_scaled as V
from stage_cutfem_neural_a.elimination_reference import load_upper_factor
from backend import spatial_blocks

EPSS = (1e-5, 1e-4, 1e-3, 1e-2)
BLOCKS = (1, 64, 512)


def main():
    t0 = time.time()
    recs = json.load(open('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'))
    rec = [r for r in recs if int(r['seat']) == 328][0]
    rec.setdefault('reference', '/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_0328')
    label = V.Label(rec, 0.2, 0.03, 10.0)
    g = label.to_gpu(need_A=False, need_Z=False, z_dtype=torch.float32)
    d = label.d
    R = load_upper_factor(Path(rec['reference']) / 'R_UPPER.npy', d, V.DEV)
    Rinv = torch.linalg.solve_triangular(R, torch.eye(d, dtype=torch.float64, device=V.DEV), upper=True)
    order = g['data'].quotient.order.cpu().numpy()
    pts = label.ijk[order[6:] // 3].astype(float) / 64.0
    print('seat 328  d %d   perturbing the FACTOR R*, measuring eps_op on the real spectrum\n' % d, flush=True)

    labels_for = {}
    for B in BLOCKS:
        if B == 1:
            labels_for[B] = None
            continue
        lab = np.empty(d, dtype=np.int64)
        for bi, idx in enumerate(spatial_blocks(pts, B)):
            lab[idx] = bi
        labels_for[B] = torch.as_tensor(lab, device=V.DEV)
        print('coherence block %4d -> %d spatial groups' % (B, int(lab.max()) + 1), flush=True)

    print('\n%-10s %s' % ('per-entry', '  '.join('eps_op @ block %-5d' % B for B in BLOCKS)), flush=True)
    torch.manual_seed(1)
    rows = []
    for eps in EPSS:
        vals = []
        for B in BLOCKS:
            if B == 1:
                Nz = torch.randn(d, d, dtype=torch.float64, device=V.DEV)
            else:
                lab = labels_for[B]; nb = int(lab.max()) + 1
                Gb = torch.randn(nb, nb, dtype=torch.float64, device=V.DEV)
                Nz = Gb[lab][:, lab]                      # one offset per block pair
                del Gb
            Rh = R + eps * R.abs() * Nz
            del Nz; torch.cuda.empty_cache()
            X = Rh @ Rinv; del Rh; torch.cuda.empty_cache()
            H = X.T @ X; del X; torch.cuda.empty_cache()
            mu = torch.linalg.eigvalsh(0.5 * (H + H.T)); del H; torch.cuda.empty_cache()
            vals.append(float((mu - 1).abs().max())); del mu; torch.cuda.empty_cache()
        print('%-10.0e %s' % (eps, '  '.join('%-20.4e' % v for v in vals)), flush=True)
        rows.append(dict(per_entry=eps, **{('block_%d' % B): v for B, v in zip(BLOCKS, vals)}))
    Path('/root/autodl-tmp/NEURAL_SCHUR/COHERENCE.json').write_text(json.dumps(
        dict(seat=328, d=int(d), blocks=list(BLOCKS), rows=rows, seconds=time.time()-t0), indent=1))
    print('\nwritten COHERENCE.json (%.0f s)' % (time.time()-t0), flush=True)


if __name__ == '__main__':
    main()
