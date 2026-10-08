"""Direction A: a trimmed block of uniform-thickness Schwarz-P cells for the lattice-scale maps of lat_global.py (new script).

The block nx x ny x nz (reference cell units) is trimmed by ONE global vertical half-space n . X <= b, n = (cos theta,
sin theta, 0) (the canonical cut family of the training data, as gen_hlat.py), placed so that the cell at (nx-1, 0, *)
has local offset b0. Per cell (local offset bl = b - n . (i, j, 0)):
  bl >= n_x + n_y   FULL: the existing uniform-thickness periodic cell (--full_case, e.g. homogP_t02200; no new body)
  bl <= 0 or retained box volume < --min-vol   removed
  else              CUT: packet with eight equal corners --tau, normal n, offset bl
Cut cells with the same local offset (the plane is vertical, so cells along z) share one case. After the global map of
lat_global.py the plane becomes a curved trimming surface of the physical part.
Writes packets (<packets>/<name>_c<i><j>/FRESH_CONTEXT.json, SAMPLE.json; template = the full case's packet), bodies
(fast_prep4 in the frozen CPU environment, as homog_cell.py), a body directory <bodies> that also links the full case's body
and the GP templates, and the layout <out>/<name>.json: cells [{position, case, kind, retained}].
Usage: gen_cutglob.py <name> --shape 2,2,6 --full_case homogP_t02200 --full_body <dir> --full_packets <dir> --tau 0.22
       [--theta-deg 20] [--b0 0.75] [--min-vol 0.15] [--packets dir] [--bodies dir] [--out dir] [--workers 8]"""
import argparse, json, math, os, shutil, subprocess
from pathlib import Path
import numpy as np


def retained(nrm, b, m=200):
    s = (np.arange(m) + .5) / m
    X, Y = np.meshgrid(s, s, indexing='ij')
    return float(((nrm[0] * X + nrm[1] * Y) <= b).mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('name'); ap.add_argument('--shape', default='2,2,6')
    ap.add_argument('--full_case', required=True); ap.add_argument('--full_body', required=True)
    ap.add_argument('--full_packets', required=True); ap.add_argument('--tau', type=float, required=True)
    ap.add_argument('--theta-deg', type=float, default=20.0); ap.add_argument('--b0', type=float, default=0.75)
    ap.add_argument('--min-vol', type=float, default=0.15)
    ap.add_argument('--packets', default='/root/autodl-tmp/OPL/A0/packets_cut')
    ap.add_argument('--bodies', default='/root/autodl-tmp/OPL/A0/bodies_cut')
    ap.add_argument('--out', default='/root/autodl-tmp/OPL/A0/layouts')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--frozen', default='/root/autodl-tmp/CUTFEM_DEPENDENCIES_20260924/run_frozen_python.sh')
    ap.add_argument('--frozen-cwd', default='/root/autodl-tmp/CUTFEM_DEPENDENCIES_20260924/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/COVER_G/src')
    ap.add_argument('--prep', default='/root/autodl-tmp/OPL/src_v2/fast_prep4.py')
    a = ap.parse_args(argv)
    nx, ny, nz = (int(x) for x in a.shape.split(','))
    if not 0.1755 < a.tau < 0.6983:
        raise ValueError('TAU_CONTRACT')
    th = math.radians(a.theta_deg); nrm = (math.cos(th), math.sin(th), 0.0)
    bg = a.b0 + nrm[0] * (nx - 1)
    tpl_dir = Path(a.full_packets) / a.full_case
    tpl = json.loads((tpl_dir / 'FRESH_CONTEXT.json').read_text())
    PK, BD = Path(a.packets), Path(a.bodies)
    PK.mkdir(parents=True, exist_ok=True); BD.mkdir(parents=True, exist_ok=True)
    gp = Path(a.full_body) / 'GP_TEMPLATES_n32.npz'
    if not (BD / gp.name).exists():
        shutil.copy(gp, BD / gp.name)
    if not (BD / a.full_case).exists():
        os.symlink(Path(a.full_body) / a.full_case, BD / a.full_case)
    cells, cut_cases = [], {}
    for i in range(nx):
        for j in range(ny):
            bl = bg - nrm[0] * i - nrm[1] * j
            if bl >= nrm[0] + nrm[1]:
                kind, case, vol = 'FULL', a.full_case, 1.0
            elif bl <= 0:
                continue
            else:
                vol = retained(nrm, bl)
                if vol < a.min_vol:
                    continue
                kind, case = 'CUT', f'{a.name}_c{i}{j}'
                cut_cases[case] = (bl, vol, (i, j))
            for k in range(nz):
                cells.append(dict(position=[i, j, k], case=case, kind=kind, retained=vol))
    for case, (bl, vol, ij) in cut_cases.items():
        d = PK / case
        ctx = json.loads(json.dumps(tpl)); cs = dict(tpl['case'])
        cs.update(case_id=case, family_id=a.name, split='lattice', field_kind='LATTICE', kind='CUT',
                  tau_corners=[format(a.tau, '.12f')] * 8, normal=[repr(nrm[0]), repr(nrm[1]), '0'], offset=repr(bl),
                  retained_macro_volume_target=vol, retained_macro_volume_check=vol, lattice_position=[ij[0], ij[1], 0])
        ctx['case'] = cs
        ctx['provenance'] = dict(writer='OPL gen_cutglob.py', args=vars(a), template=str(tpl_dir))
        d.mkdir(parents=True, exist_ok=True)
        (d / 'FRESH_CONTEXT.json').write_text(json.dumps(ctx, indent=1))
        shutil.copy(tpl_dir / 'SAMPLE.json', d / 'SAMPLE.json')
    todo = [c for c in cut_cases if not (BD / c / 'PREP.json').exists()]
    if todo:
        one = BD / 'body_one.sh'
        one.write_text('#!/bin/bash\n[ -f $1/$2/PREP.json ] && exit 0\n'
                       f'export OPL_PACKETS_EXTRA={PK}\n'
                       f'cd {a.frozen_cwd} && OMP_NUM_THREADS=1 timeout 1800 {a.frozen} {a.prep} 1 $1 $2 > $1/$2.log 2>&1 || touch $1/$2.failed\n')
        one.chmod(0o755)
        (BD / 'todo.txt').write_text('\n'.join(todo) + '\n')
        subprocess.run(['bash', '-c', f'xargs -a {BD}/todo.txt -P {a.workers} -I{{}} {one} {BD} {{}}'], check=False)
    bad = [c for c in cut_cases if not (BD / c / 'PREP.json').exists()]
    lay = dict(name=a.name, shape=[nx, ny, nz], normal=nrm, b_global=bg, tau=a.tau, full_case=a.full_case,
               body=str(BD), packets=str(PK), cells=cells, cut_cases={k: dict(offset=v[0], retained=v[1]) for k, v in cut_cases.items()},
               failed=bad, args=vars(a))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'{a.name}.json').write_text(json.dumps(lay, indent=1))
    print(json.dumps(dict(name=a.name, cells=len(cells), full=sum(c['kind'] == 'FULL' for c in cells),
                          cut=sum(c['kind'] == 'CUT' for c in cells), cut_cases={k: [round(v[0], 3), round(v[1], 3)] for k, v in cut_cases.items()},
                          failed=bad)))


if __name__ == '__main__':
    main()
