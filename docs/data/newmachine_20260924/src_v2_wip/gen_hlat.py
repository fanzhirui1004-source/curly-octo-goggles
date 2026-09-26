"""Heterogeneous lattice for P1 (item 9): nx x ny x nz distinct cells with ONE continuous graded thickness field and one
global vertical cut plane on the +x side (the canonical cut family of the training data).
  thickness  tau at the lattice vertices X (integer coordinates): tau(X) = t0 + gx (X/nx - 1/2) + gy sin(pi Y / ny)
             + gz cos(pi Z / nz) (+ a seeded vertex perturbation of amplitude --jitter); every cell takes its 8 corner values
             (corner index 4x + 2y + z), so neighbouring cells share their face corners exactly
  cut        global half-space n . X <= b, n = (cos theta, sin theta, 0); per cell the local offset b - n . o (o = cell
             origin): >= n_x + n_y -> FULL (normal (1,0,0), offset 1), <= 0 -> cell removed, else CUT (retained box volume
             recorded; cells retaining less than --min-vol are removed as well)
  contract   corner values inside (0.1755, 0.6983) and per-cell span <= 0.3 (checked, error otherwise)
Writes <packets>/<name>_<i><j><k>/{FRESH_CONTEXT.json, SAMPLE.json} (schema of the expansion packets, template = an existing
validation context) and <out>/<name>.json (layout: position -> case, kind, retained volume, corners).
Usage: gen_hlat.py <name> [--shape 2,2,2] [--theta-deg 20] [--b0 0.75] [--t0 0.38] [--gx 0.12] [--gy 0.06] [--gz 0.05]
       [--jitter 0.02] [--seed 0] [--min-vol 0.15] [--packets /root/autodl-tmp/OPL/S4/packets] [--out /root/autodl-tmp/OPL/S4]
       [--template /root/autodl-tmp/OPL/S3/packets/fresh_val_2003_d1_v1]"""
import argparse, json, math
from pathlib import Path
import numpy as np


def retained(nrm, b, m=200):
    s = (np.arange(m) + .5) / m
    X, Y = np.meshgrid(s, s, indexing='ij')
    return float(((nrm[0] * X + nrm[1] * Y) <= b).mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('name'); ap.add_argument('--shape', default='2,2,2')
    ap.add_argument('--theta-deg', type=float, default=20.0); ap.add_argument('--b0', type=float, default=0.75)
    ap.add_argument('--t0', type=float, default=0.38); ap.add_argument('--gx', type=float, default=0.12)
    ap.add_argument('--gy', type=float, default=0.06); ap.add_argument('--gz', type=float, default=0.05)
    ap.add_argument('--jitter', type=float, default=0.02); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min-vol', type=float, default=0.15)
    ap.add_argument('--packets', default='/root/autodl-tmp/OPL/S4/packets'); ap.add_argument('--out', default='/root/autodl-tmp/OPL/S4')
    ap.add_argument('--template', default='/root/autodl-tmp/OPL/S3/packets/fresh_val_2003_d1_v1')
    a = ap.parse_args(argv)
    nx, ny, nz = (int(x) for x in a.shape.split(','))
    rng = np.random.default_rng(a.seed)
    V = np.zeros((nx + 1, ny + 1, nz + 1))
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                V[i, j, k] = a.t0 + a.gx * (i / nx - .5) + a.gy * math.sin(math.pi * j / ny) + a.gz * math.cos(math.pi * k / nz)
    V += a.jitter * rng.uniform(-1, 1, V.shape)
    th = math.radians(a.theta_deg); nrm = (math.cos(th), math.sin(th), 0.0)
    bg = a.b0 + nrm[0] * (nx - 1)                                       # local offset b0 for the cut cell at (nx-1, 0, *)
    tpl = json.loads((Path(a.template) / 'FRESH_CONTEXT.json').read_text())
    sample = (Path(a.template) / 'SAMPLE.json').read_text()
    cells = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                corners = [V[i + ((c >> 2) & 1), j + ((c >> 1) & 1), k + (c & 1)] for c in range(8)]
                if not (min(corners) > 0.1755 and max(corners) < 0.6983 and np.ptp(corners) <= 0.3):
                    raise ValueError(f'CONTRACT {i}{j}{k} {corners}')
                bl = bg - nrm[0] * i - nrm[1] * j
                if bl >= nrm[0] + nrm[1]:
                    kind, vol, normal, offset = 'FULL', 1.0, ['1', '0', '0'], '1'
                elif bl <= 0:
                    continue
                else:
                    vol = retained(nrm, bl)
                    if vol < a.min_vol:
                        continue
                    kind, normal, offset = 'CUT', [repr(nrm[0]), repr(nrm[1]), '0'], repr(bl)
                case = f'{a.name}_{i}{j}{k}'
                ctx = json.loads(json.dumps(tpl))
                cs = dict(tpl['case'])
                cs.update(case_id=case, family_id=a.name, split='lattice', field_kind='LATTICE', kind=kind,
                          tau_corners=[format(x, '.12f') for x in corners], normal=normal, offset=offset,
                          retained_macro_volume_target=vol, retained_macro_volume_check=vol, lattice_position=[i, j, k])
                for key in ('direction_index', 'depth_index', 'generator_case_id', 'generator_family_id', 'geometry_orbit_sha256',
                            'block', 'block_position', 'family_index'):
                    cs.pop(key, None)
                ctx['case'] = cs
                ctx['provenance'] = dict(writer='OPL gen_hlat.py', args=vars(a))
                d = Path(a.packets) / case; d.mkdir(parents=True, exist_ok=True)
                (d / 'FRESH_CONTEXT.json').write_text(json.dumps(ctx, indent=1)); (d / 'SAMPLE.json').write_text(sample)
                cells.append(dict(position=[i, j, k], case=case, kind=kind, retained=vol, tau_corners=corners))
    lay = dict(name=a.name, shape=[nx, ny, nz], normal=nrm, b_global=bg, cells=cells, args=vars(a))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'{a.name}.json').write_text(json.dumps(lay, indent=1))
    print(json.dumps(dict(name=a.name, cells=len(cells), kinds={k: sum(c['kind'] == k for c in cells) for k in ('FULL', 'CUT')},
                          cut_retained=[round(c['retained'], 3) for c in cells if c['kind'] == 'CUT'],
                          tau_range=[round(float(V.min()), 3), round(float(V.max()), 3)])))


if __name__ == '__main__':
    main()
