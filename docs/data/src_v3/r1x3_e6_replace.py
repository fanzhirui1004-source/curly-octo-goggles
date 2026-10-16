"""Revision round 1, E6 replacement draw for the heavy stratum (fresh_val_2026_d0_v0 failed with EMPTY_LOAD in x and y).
Reproduces the pre-registered E6 draw of R1/ACC/PREREG_R1ACC.json exactly (numpy default_rng(20260928), one rng.choice
per stratum over the remaining sel=false cells, sorted case ids, order FULL, light, moderate, heavy, the five worst cells
excluded) and checks it against PREREG; then continues the SAME generator over the remaining heavy cells (excluding every
cell already used, 2026 included) and takes the first draw whose two-cell configurations are not empty-loaded in both x
and y (the lattice3 EMPTY_LOAD condition: no non-private box node on the load plane, local y = 0 for x, x = 0 for y, in the
test cell or its neighbour packet), at most 5 draws. Writes <acc_dir>/E6_REPLACEMENT_R1X3.json (PREREG is not edited)
and prints the chosen case. Usage: r1x3_e6_replace.py <acc_dir> <valmeta.json> [<body S0> [<failed replacements, comma list>]]"""
import json, sys
from pathlib import Path
import numpy as np

acc, vm = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
body = Path(sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else '/root/autodl-tmp/OPL/S0')
failed = [c for c in (sys.argv[4] if len(sys.argv) > 4 else '').split(',') if c]      # replacements whose gates both failed
pre = json.loads((acc / 'PREREG_R1ACC.json').read_text())['E6']['cells']
worst = pre['worst5']


def stratum(m):
    if m['kind'] == 'FULL':
        return 'FULL'
    return 'light' if m['vol'] > 2 / 3 else ('moderate' if m['vol'] > 1 / 3 else 'heavy')


def loaded(case, axis):
    """lattice3: a non-private port DOF on the plane x_axis = 0 (local grid coordinate 0) exists."""
    n = int(json.loads(Path(f'/root/autodl-tmp/OPL/S3/packets/{case}/FRESH_CONTEXT.json').read_text())['n'])
    d = body / case
    box = np.load(d / 'BOX_NODES.npy')
    g = np.stack(np.unravel_index(box, (2 * n + 1,) * 3), 1)
    return bool((g[:, axis] == 0).any())


rng = np.random.default_rng(20260928)
used, draw = set(worst), {}
for s in ('FULL', 'light', 'moderate', 'heavy'):
    c = sorted(k for k, m in vm.items() if not m['sel'] and stratum(m) == s and k not in used)
    draw[s] = str(rng.choice(c)); used.add(draw[s])
ok = draw == pre['random']
tries = []
chosen = None
for _ in range(8):
    c = sorted(k for k, m in vm.items() if not m['sel'] and stratum(m) == 'heavy' and k not in used)
    if not c:
        break
    x = str(rng.choice(c)); used.add(x)
    cfg = {conf: loaded(x, 1 if conf == 'x' else 0) and loaded(f'{x}_nbm{conf}', 1 if conf == 'x' else 0) for conf in ('x', 'y')}
    tries.append(dict(case=x, vol=vm[x]['vol'], loaded=cfg, gates_failed=x in failed))
    if all(cfg.values()) and x not in failed:
        chosen = x; break
rec = dict(schema='R1_ACC_E6_REPLACEMENT_V1', written_by='X3 (R1EXT chain)', reason='fresh_val_2026_d0_v0 (heavy stratum, retained volume 0.046) '
           'failed in both configurations: x EMPTY_LOAD (no material on the loaded face), y LATTICE_CHOLESKY_INFO (singular '
           'reference lattice: ill-posed); its failure records are kept',
           procedure='continuation of the pre-registered generator default_rng(20260928) after the four stratum draws, over the remaining '
                     'sel=false heavy cells (sorted), excluding all used cells; the first draw whose x and y configurations are both '
                     'non-empty-loaded (lattice3 EMPTY_LOAD condition on the cell and its neighbour packets) and whose gates did not '
                     'both fail in an earlier attempt is taken (<= 8 draws); PREREG_R1ACC.json unchanged',
           prereg_reproduced=ok, prereg_draw=draw, tries=tries, replacement=chosen)
(acc / 'E6_REPLACEMENT_R1X3.json').write_text(json.dumps(rec, indent=1))
print(chosen if (ok and chosen) else 'NONE')
