"""G (gyroid sheet) twin of a P lattice layout: every cell keeps its position, kind and macro plane; its thickness
corners are mapped P -> G at equal sheet volume fraction (vf_map.json). Writes <packets>/<name>_<ijk>/{FRESH_CONTEXT,
SAMPLE}.json with case['surface'] = 'G' and <out>/<name>.json (layout, cells with the P twin and both corner sets).
Usage: gen_gtwin.py <P_layout.json> <name> [--vf /root/autodl-tmp/OPL/S5/vf_map.json] [--packets S5/packets] [--out S5]"""
import argparse, json, os
from fractions import Fraction
from pathlib import Path
import numpy as np


def packet_dir(case):
    for r_ in [Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/packets')] + \
              [Path(x) for x in os.environ.get('OPL_PACKETS_EXTRA', '').split(':') if x]:
        if (r_ / case).exists():
            return r_ / case
    raise FileNotFoundError(case)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('layout'); ap.add_argument('name')
    ap.add_argument('--vf', default='/root/autodl-tmp/OPL/S5/vf_map.json')
    ap.add_argument('--packets', default='/root/autodl-tmp/OPL/S5/packets'); ap.add_argument('--out', default='/root/autodl-tmp/OPL/S5')
    a = ap.parse_args()
    vf = json.loads(Path(a.vf).read_text())
    tau, vP, vG = (np.asarray(vf[k]) for k in ('tau', 'vf_P', 'vf_G'))
    p2g = lambda t: float(np.interp(np.interp(t, tau, vP), vG, tau))
    L = json.loads(Path(a.layout).read_text())
    cells = []
    for c in L['cells']:
        src = packet_dir(c['case'])
        ctx = json.loads((src / 'FRESH_CONTEXT.json').read_text())
        tp = [float(Fraction(v)) for v in ctx['case']['tau_corners']]
        tg = [p2g(t) for t in tp]
        ijk = ''.join(str(v) for v in c['position'])
        case = f'{a.name}_{ijk}'
        cs = dict(ctx['case'])
        cs.update(case_id=case, family_id=a.name, surface='G', tau_corners=[format(x, '.12f') for x in tg], p_twin=c['case'],
                  tau_map='equal sheet volume fraction P -> G (vf_map.json)')
        ctx['case'] = cs
        ctx['provenance'] = dict(writer='OPL gen_gtwin.py', p_twin=c['case'], vf_map=a.vf)
        d = Path(a.packets) / case; d.mkdir(parents=True, exist_ok=True)
        (d / 'FRESH_CONTEXT.json').write_text(json.dumps(ctx, indent=1))
        (d / 'SAMPLE.json').write_text((src / 'SAMPLE.json').read_text())
        cells.append(dict(position=c['position'], case=case, kind=c['kind'], p_twin=c['case'], tau_corners_P=tp, tau_corners=tg))
    out = dict(name=a.name, surface='G', p_twin_layout=L['name'], shape=L.get('shape'), cells=cells)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'{a.name}.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(name=a.name, cells=len(cells), tau_G_range=[min(min(c['tau_corners']) for c in cells), max(max(c['tau_corners']) for c in cells)])))


if __name__ == '__main__':
    main()
