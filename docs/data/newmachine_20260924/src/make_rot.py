"""V step, part 1: metadata scan of every packet, and rotated copies of three packets under cube-group elements
(families.rotate_geometry: x' = .5 + R (x - .5), corners permuted, normal rotated, offset shifted), written as new
packet directories <case>_rot<k> next to the originals."""
import json, sys, shutil
from fractions import Fraction
from itertools import product
from pathlib import Path
import numpy as np

P = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/packets')
CORNERS = np.array(list(product((0, 1), repeat=3)), dtype=int)
ROTS = {                                                   # signed axis permutations
    'swapxy': [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
    'mirrorx': [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
    'rotz90': [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    'invert': [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
    'general': [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
}


def rotate(case, r):
    r = np.asarray(r)
    moved = ((CORNERS - .5) @ r.T + .5).astype(int) @ np.array([4, 2, 1])
    corners = [None] * 8
    for i, j in enumerate(moved):
        corners[j] = case['tau_corners'][i]
    nf = [Fraction(v) for v in case['normal']]
    nn = [sum(int(r[a, b]) * nf[b] for b in range(3)) for a in range(3)]
    off = Fraction(case['offset']) + (sum(nn) - sum(nf)) / 2
    out = dict(case)
    out.update(tau_corners=corners, normal=[str(v) for v in nn], offset=str(off))
    return out


def main(cases):
    scan = {}
    for d in sorted(P.iterdir()):
        f = d / 'FRESH_CONTEXT.json'
        if not f.exists() or '_rot' in d.name:
            continue
        c = json.loads(f.read_text()); s = json.loads((d / 'SAMPLE.json').read_text())
        key = (c['n'], c['material']['E'], c['material']['nu'], s['gp']['gamma'], c['case']['kind'])
        scan.setdefault(str(key), []).append(d.name)
    print(json.dumps({'metadata_groups': {k: len(v) for k, v in scan.items()}}), flush=True)
    for case in cases:
        src = P / case
        ctx = json.loads((src / 'FRESH_CONTEXT.json').read_text())
        for name, r in ROTS.items():
            dst = P / f'{case}_rot{name}'
            if dst.exists():
                shutil.rmtree(dst)
            dst.mkdir()
            c2 = dict(ctx); c2['case'] = rotate(ctx['case'], r); c2['case']['case_id'] = dst.name
            (dst / 'FRESH_CONTEXT.json').write_text(json.dumps(c2, indent=2))
            shutil.copy(src / 'SAMPLE.json', dst / 'SAMPLE.json')
    print('ROTATED', ' '.join(f'{c}_rot{k}' for c in cases for k in ROTS), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
