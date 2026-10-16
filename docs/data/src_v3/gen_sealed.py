"""The registered sealed test set (P1 final evaluation, used ONCE after the model freeze): the 'sealed_test' stream of the
registered generator seed 2026092101 (fresh_gp.families.generate, frozen source, read-only), --families families (the
registered default is 8), every case the generator returns for them (FULL + its cuts), renamed fresh_sealed_<idx>_<suffix>.
Neighbours (continuous thickness, gen_new.neighbour): gate tags mx, my for every cell; one glued face per cell drawn among the
admissible faces (seeded by 'sealed', case). Writes <out>/packets/<case>/{FRESH_CONTEXT.json, SAMPLE.json} and
<out>/PLAN_SEALED.json. Refuses to overwrite a differing context.
Run in the frozen environment: run_frozen_python.sh gen_sealed.py <out> [--families 8]"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_new as GN

SEED = 2026092101


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--families', type=int, default=8)
    a = ap.parse_args()
    sys.path.insert(0, str(GN.SRC))
    from fresh_gp import families as F
    src_sha = hashlib.sha256((GN.SRC / 'fresh_gp' / 'families.py').read_bytes()).hexdigest()
    fams, cases = F.generate(SEED, 8, 8, GN.ceil8(a.families))
    sealed = {f['family_id']: f for f in fams if f['split'] == 'sealed_test' and f['family_index'] < a.families}
    rows = []
    for c in cases:
        fid = next((f for f in sealed if c['case_id'].startswith(f + '_')), None)
        if fid is None:
            continue
        idx = sealed[fid]['family_index']
        suffix = c['case_id'][len(fid):]
        nid = f'fresh_sealed_{idx:04d}'
        rows.append(dict(c, case_id=nid + suffix, family_id=nid, split='sealed_test', generator_seed=SEED,
                         generator_case_id=c['case_id'], generator_family_id=fid, design='registered family design'))
    nbs, want = [], {}
    for r in rows:
        adm = GN.admissible_glue(r)
        h = hashlib.sha256(f"sealed:{r['case_id']}:glued".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], 'little'))
        g = sorted(rng.choice(adm, size=1, replace=False).tolist()) if adm else []
        want[r['case_id']] = dict(glued=g, gate=['mx', 'my'])
        for t in dict.fromkeys(g + ['mx', 'my']):
            nbs.append(GN.neighbour(F, r, t, SEED))
    out = Path(a.out) / 'packets'
    for r in rows + nbs:
        d = out / r['case_id']
        ctx = json.dumps(GN.context(r, SEED), indent=1)
        if (d / 'FRESH_CONTEXT.json').exists():
            if (d / 'FRESH_CONTEXT.json').read_text() != ctx:
                raise SystemExit(f"CONTEXT_DIFFERS {r['case_id']}")
            continue
        d.mkdir(parents=True, exist_ok=True)
        (d / 'SAMPLE.json').write_text(json.dumps(dict(schema='OPL_EXPANSION_SAMPLE_V1', gp=dict(gamma=GN.GAMMA))))
        (d / 'FRESH_CONTEXT.json').write_text(ctx)
    plan = dict(schema='OPL_SEALED_PLAN_V1', seed=SEED, generator_sha256=src_sha, families=a.families,
                cells=[r['case_id'] for r in rows], neighbours=want,
                cases={r['case_id']: dict(kind=r['kind'], volume=r.get('retained_macro_volume_target'), field_kind=r.get('field_kind'))
                       for r in rows},
                counts=dict(cells=len(rows), full=sum(r['kind'] == 'FULL' for r in rows), cut=sum(r['kind'] == 'CUT' for r in rows),
                            neighbours=len(nbs), neighbour_fallback=sum(n['neighbour_draw'] != 'random' for n in nbs)))
    (Path(a.out) / 'PLAN_SEALED.json').write_text(json.dumps(plan, indent=1))
    print(json.dumps(plan['counts']), flush=True)


if __name__ == '__main__':
    main()
