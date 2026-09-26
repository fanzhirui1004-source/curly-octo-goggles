"""Geometry expansion: packet contexts for new cells (runs in the frozen CUTFEM environment: run_frozen_python.sh).

New families come from the registered family generator (fresh_gp.families.generate, read-only import of the frozen source)
with a NEW seed (never the registered seed 2026092101, whose sealed_test stream stays untouched); its sealed_test output for
the new seed is discarded unused. The generator's per-split random streams are prefix-stable: more families later with the
same seed keep the earlier ones identical.
  new train families   fresh_train_<offset+i>   (generator 'train' stream, first --train families), split 'train'
  new val families     fresh_val_<offset+i>     (generator 'development' stream, first --val families), split 'validation'
  each family          <fid>_full + the generator's six cuts <fid>_d{direction}_v{depth} (vertical plane, canonical theta in
                       [0, pi/4] stratified in two halves, retained macro box volume stratified in thirds)
Extra cuts on the existing training families (--extra, from --split's train cases): per family, the point of the heavy
band theta in [0, pi/4] x retained box volume in [--vmin, --vmax] farthest (max-min, theta/(pi/4) and volume) from the
family's existing cuts; one per family, then second points for the families with the largest remaining gap.
  <fid>_x<k>
Diagnostic subset (PLAN.json 'diagnostic'): the first --diag-val new val families (FULL + d0_v0, d1_v1, d0_v2) and the extra
cuts of --diag-extra training families chosen by a seeded shuffle. Everything else is 'production'.
Writes <out>/packets/<case>/FRESH_CONTEXT.json + SAMPLE.json (idempotent: an existing identical context is kept, a different
one is an error) and <out>/PLAN.json. Collision checks: new mother orbits vs the existing families, new cut orbits vs the
family's existing cases, case ids vs the existing packets.

--mode indep (production, 2026-09-26): every cell has its OWN thickness field (no field shared inside a family). Generator
families of the new seed, per block of 8 fields: positions 0, 1 -> FULL; positions 2..7 -> one cut each, the six
(direction half, depth third) strata of the generator design once per block. Train cells from the generator 'train' stream
(fresh_train_<offset+i>_<kind>), validation cells from its 'development' stream (fresh_val_<offset+i>_<kind>).
Neighbours (continuous thickness), only where they are used:
  glued class  --glued-per-cell (1) tag(s) per cell, drawn (seeded by seed, case) among the glue faces px, py, pz, mz whose
               retained area (half-space n.x <= offset of the canonical vertical cut) is >= 2% of the face (prep_geo2 needs > 8 material port nodes) (all four for FULL)
  gate         --gate-tags (mx, my: the intact side of the canonical cut) for the cells of --gate-splits (validation)
<case>_nb<tag> is a FULL cell glued across that face: its four corners on the shared face equal the cell's, its four far corners are drawn uniformly in [TAU_LOWER,
TAU_UPPER] (seeded by seed, case, tag) and accepted inside the generator contract (bounds, span, gradient <= 0.47);
after 5000 rejections the far corners copy the shared ones (zero normal gradient, always admissible; recorded).
Usage: gen_new.py <out_root> [--mode family|indep] [--seed 2026092601] [--train 30] [--val 6] [--offset 1000] [--extra 39] [--diag-val 3]
                  [--diag-extra 12] [--vmin 0.08] [--vmax 0.45] [--split /root/autodl-tmp/OPL/S2/SPLIT.json] [--dry]"""
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np

SRC = Path('/root/autodl-tmp/CUTFEM_DEPENDENCIES_20260924/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/'
           'cut_cover80_source_20260922_03/research')
PACKETS = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/packets')
REGISTERED_SEED = 2026092101
GAMMA = 0.0001
TAU_LO = 0.1755   # frozen geometry contract (stage_cutfem_graded.thickness.LOWER) > generator TAU_LOWER 0.1752016


def context(row, seed):
    return dict(schema='OPL_EXPANSION_CONTEXT_V1', case=row, n=32, material=dict(E=1.0, nu=0.3),
                gp=dict(gamma=GAMMA, orders=[1, 2], scope='module local'),
                provenance=dict(generator='fresh_gp.families.generate (frozen source, read-only)', seed=seed,
                                writer='OPL gen_new.py'))


def ceil8(x):
    return max(8, 8 * math.ceil(x / 8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--seed', type=int, default=2026092601)
    ap.add_argument('--train', type=int, default=30)
    ap.add_argument('--val', type=int, default=6)
    ap.add_argument('--offset', type=int, default=1000)
    ap.add_argument('--extra', type=int, default=39)
    ap.add_argument('--diag-val', type=int, default=3)
    ap.add_argument('--diag-extra', type=int, default=12)
    ap.add_argument('--vmin', type=float, default=0.08)
    ap.add_argument('--vmax', type=float, default=0.45)
    ap.add_argument('--split', default='/root/autodl-tmp/OPL/S2/SPLIT.json')
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--mode', default='family', choices=('family', 'indep'))
    ap.add_argument('--glued-per-cell', type=int, default=1)
    ap.add_argument('--fix-neighbour', default='', help='rewrite <out>/packets/<name> with far corners = the shared face '
                    '(zero normal gradient), e.g. after its topology could not be certified; then exit')
    ap.add_argument('--make-neighbour', default='', help='CASE: choose its glue face from the built body (--body): the '
                    'planned face if it has > 8 box-port nodes, else a seeded draw among the faces that do (all six), '
                    'write <out>/packets/<CASE>_nb<tag> if missing, print the tag (NONE if no face qualifies); then exit')
    ap.add_argument('--body', default='/root/autodl-tmp/OPL/S3/body')
    ap.add_argument('--gate-tags', default='mx,my')
    ap.add_argument('--gate-splits', default='validation')
    a = ap.parse_args()
    if a.seed == REGISTERED_SEED:
        raise SystemExit('REGISTERED_SEED_REFUSED: the registered seed also draws the sealed test families')
    sys.path.insert(0, str(SRC))
    from fresh_gp import families as F
    src_sha = hashlib.sha256((SRC / 'fresh_gp' / 'families.py').read_bytes()).hexdigest()
    if a.fix_neighbour:
        return fix_neighbour(a, F)
    if a.make_neighbour:
        return make_neighbour(a, F)
    if a.mode == 'indep':
        return indep(a, F, src_sha)

    # existing packets: families, their cuts (theta, volume), orbits
    existing, fam_cuts, fam_ctx, mother = set(), {}, {}, {}
    for d in sorted(PACKETS.iterdir()):
        existing.add(d.name)
        if '_rot' in d.name:
            continue
        c = json.loads((d / 'FRESH_CONTEXT.json').read_text())['case']
        fam_ctx.setdefault(c['family_id'], c)
        if c.get('kind') == 'FULL':
            mother[c['family_id']] = c['geometry_orbit_sha256']
        else:
            n = [float(x) for x in c['normal']]
            th = math.atan2(abs(n[1]), abs(n[0]))
            fam_cuts.setdefault(c['family_id'], []).append((min(th, math.pi / 2 - th) / (math.pi / 4),
                                                            float(c['retained_macro_volume_target']), c['geometry_orbit_sha256']))

    # new families (generator; sealed_test stream of the NEW seed discarded)
    fams, cases = F.generate(a.seed, ceil8(a.train), ceil8(a.val), 8)
    keep_ids, rename = {}, {}
    for f in fams:
        split, idx = f['split'], f['family_index']
        if split == 'train' and idx < a.train:
            rename[f['family_id']] = (f'fresh_train_{a.offset + idx:04d}', 'train')
        elif split == 'development' and idx < a.val:
            rename[f['family_id']] = (f'fresh_val_{a.offset + idx:04d}', 'validation')
    new_rows, new_fams = [], []
    for f in fams:
        if f['family_id'] not in rename:
            continue
        if f['mother_orbit_sha256'] in set(mother.values()):
            raise SystemExit(f"MOTHER_ORBIT_COLLISION {f['family_id']}")
        nid, split = rename[f['family_id']]
        new_fams.append(dict(family_id=nid, split=split, generator_family_id=f['family_id'], field_kind=f['field_kind'],
                             tau_corners=f['tau_corners'], mother_orbit_sha256=f['mother_orbit_sha256']))
    for c in cases:
        if c['family_id'] not in rename:
            continue
        nid, split = rename[c['family_id']]
        row = dict(c, case_id=nid + c['case_id'][len(c['family_id']):], family_id=nid, split=split,
                   generator_case_id=c['case_id'], generator_family_id=c['family_id'])
        new_rows.append(row)

    # extra cuts on the existing training families
    sp = json.loads(Path(a.split).read_text())
    train_fams = sorted({'_'.join(x.split('_')[:3]) for x in sp['train']})
    TH = np.linspace(0, 1, 91); V = np.linspace(a.vmin, a.vmax, 75)
    grid = np.stack(np.meshgrid(TH, V, indexing='ij'), -1).reshape(-1, 2)
    pts = {f: [(t, v) for t, v, _ in fam_cuts.get(f, [])] for f in train_fams}

    def best(f):
        P = np.asarray(pts[f]) if pts[f] else np.zeros((0, 2))
        dmin = np.full(len(grid), 9.) if not len(P) else np.linalg.norm(grid[:, None] - P[None], axis=2).min(1)
        i = int(np.argmax(dmin))
        return float(dmin[i]), grid[i]

    extra_rows, per = [], {f: 0 for f in train_fams}
    order = list(train_fams)
    while len(extra_rows) < a.extra:
        if all(per[f] >= 1 for f in train_fams):                      # second round: largest remaining gaps first
            order = sorted(train_fams, key=lambda f: (-best(f)[0], f))
        placed = False
        for f in order:
            if len(extra_rows) >= a.extra:
                break
            if per[f] > min(per.values()):
                continue
            gap, (t, v) = best(f)
            theta = t * math.pi / 4
            n, b = F.plane_from_volume(theta, float(v))
            c0 = fam_ctx[f]
            orbit = F.orbit_key(np.asarray([float(x) for x in c0['tau_corners']]), n, b)
            if orbit in {o for _, _, o in fam_cuts.get(f, [])}:
                raise SystemExit(f'CUT_ORBIT_COLLISION {f}')
            row = dict(case_id=f'{f}_x{per[f]}', family_id=f, family_index=c0.get('family_index'), split='train',
                       field_kind=c0.get('field_kind'), nested_train_level=c0.get('nested_train_level'), kind='CUT',
                       tau_corners=c0['tau_corners'], normal=[format(x, '.16g') for x in n], offset=format(b, '.16g'),
                       theta_radians=theta, retained_macro_volume_target=float(v),
                       retained_macro_volume_check=F.retained_box_volume(n, b), geometry_orbit_sha256=orbit,
                       design='heavy-band max-min gap to the family cuts', design_gap=gap)
            extra_rows.append(row); pts[f].append((t, float(v))); per[f] += 1; placed = True
        if not placed:
            break

    # diagnostic subset
    rng = np.random.default_rng(a.seed)
    diag_f = set(rng.permutation(train_fams)[:a.diag_extra].tolist())
    val_ids = sorted({r['family_id'] for r in new_rows if r['split'] == 'validation'})[:a.diag_val]
    diag = [r['case_id'] for r in new_rows if r['family_id'] in val_ids and
            r['case_id'].endswith(('_full', '_d0_v0', '_d1_v1', '_d0_v2'))]
    diag += [r['case_id'] for r in extra_rows if r['family_id'] in diag_f and r['case_id'].endswith('_x0')]
    rows = new_rows + extra_rows
    ids = [r['case_id'] for r in rows]
    if len(set(ids)) != len(ids) or set(ids) & existing:
        raise SystemExit('CASE_ID_COLLISION')
    plan = dict(schema='OPL_EXPANSION_PLAN_V1', seed=a.seed, generator_sha256=src_sha, args=vars(a),
                counts=dict(new_train_families=a.train, new_val_families=a.val,
                            new_family_cells=len(new_rows), extra_cuts=len(extra_rows), total=len(rows), diagnostic=len(diag)),
                families=new_fams, diagnostic=diag, production=[i for i in ids if i not in set(diag)],
                cases={r['case_id']: dict(split=r['split'], family_id=r['family_id'], kind=r['kind'],
                                          volume=r.get('retained_macro_volume_target')) for r in rows})
    print(json.dumps(plan['counts']), flush=True)
    if a.dry:
        return
    out = Path(a.out) / 'packets'
    for r in rows:
        d = out / r['case_id']
        ctx = json.dumps(context(r, a.seed), indent=1)
        if (d / 'FRESH_CONTEXT.json').exists():
            if (d / 'FRESH_CONTEXT.json').read_text() != ctx:
                raise SystemExit(f"CONTEXT_DIFFERS {r['case_id']}")
            continue
        d.mkdir(parents=True, exist_ok=True)
        (d / 'SAMPLE.json').write_text(json.dumps(dict(schema='OPL_EXPANSION_SAMPLE_V1', gp=dict(gamma=GAMMA))))
        (d / 'FRESH_CONTEXT.json').write_text(ctx)
    (Path(a.out) / 'PLAN.json').write_text(json.dumps(plan, indent=1))
    print('WROTE', len(rows), 'packets to', out, flush=True)


TAGS = {'px': (0, 1), 'mx': (0, -1), 'py': (1, 1), 'my': (1, -1), 'pz': (2, 1), 'mz': (2, -1)}
COMBOS = [(0, 0), (1, 1), (0, 2), (1, 0), (0, 1), (1, 2)]                 # (direction half, depth third) per block position 2..7


FACE = {'px': (0, 1), 'mx': (0, 0), 'py': (1, 1), 'my': (1, 0), 'pz': (2, 1), 'mz': (2, 0)}


def make_neighbour(a, F, min_nodes=9):
    """Material-based glue face (the Schwarz-P surface decides which faces carry material: a heavy vertical cut keeps the
    stub of the strut through x = 0 only). Face counts as in prep_geo2 (box-port nodes on the face > 8)."""
    c = a.make_neighbour
    pk = Path(a.out) / 'packets'
    row = json.loads((pk / c / 'FRESH_CONTEXT.json').read_text())['case']
    n = 32; M = 2 * n + 1
    B = np.load(Path(a.body) / c / 'BOX_NODES.npy')
    g = np.stack(np.unravel_index(B, (M,) * 3), 1)
    cnt = {t: int((g[:, ax] == (2 * n if side else 0)).sum()) for t, (ax, side) in FACE.items()}
    ok = sorted(t for t, k in cnt.items() if k >= min_nodes)
    planned = sorted(d.name.split('_nb')[-1] for d in pk.glob(f'{c}_nb*') if d.name.split('_nb')[-1] in ('px', 'py', 'pz', 'mz'))
    keep = [t for t in planned if t in ok]
    if keep:
        tag, how = keep[0], 'planned'
    elif ok:
        h = hashlib.sha256(f"{a.seed}:{c}:glued-auto".encode()).digest()
        tag, how = ok[int(np.random.default_rng(int.from_bytes(h[:8], 'little')).integers(len(ok)))], 'material'
    else:
        print('NONE', flush=True)
        return
    d = pk / f'{c}_nb{tag}'
    if not (d / 'FRESH_CONTEXT.json').exists():
        nb = neighbour(F, row, tag, a.seed)
        nb['glue_face_choice'] = dict(how=how, face_port_nodes=cnt)
        d.mkdir(parents=True, exist_ok=True)
        (d / 'SAMPLE.json').write_text(json.dumps(dict(schema='OPL_EXPANSION_SAMPLE_V1', gp=dict(gamma=GAMMA))))
        (d / 'FRESH_CONTEXT.json').write_text(json.dumps(context(nb, a.seed), indent=1))
    print(tag, flush=True)


def fix_neighbour(a, F):
    d = Path(a.out) / 'packets' / a.fix_neighbour
    ctx = json.loads((d / 'FRESH_CONTEXT.json').read_text())
    nb = ctx['case']
    row = json.loads((Path(a.out) / 'packets' / nb['neighbour_of'] / 'FRESH_CONTEXT.json').read_text())['case']
    new = neighbour(F, row, nb['neighbour_tag'], a.seed, force_copy=True)
    new['replaced_draw'] = dict(tau_corners=nb['tau_corners'], reason='topology not certified (fast_prep4)')
    ctx['case'] = new
    (d / 'FRESH_CONTEXT.json').write_text(json.dumps(ctx, indent=1))
    print(json.dumps(dict(event='NEIGHBOUR_FIXED', case=a.fix_neighbour, tau_corners=new['tau_corners'])), flush=True)


def neighbour(F, row, tag, seed, force_copy=False):
    """FULL neighbour across face `tag` with continuous thickness (see the module docstring)."""
    axis, sgn = TAGS[tag]
    tau = np.asarray([float(x) for x in row['tau_corners']])
    idx = np.arange(8); bits = np.stack([(idx >> 2) & 1, (idx >> 1) & 1, idx & 1], 1)   # corner index = 4x + 2y + z
    c_t, c_n = (1, 0) if sgn > 0 else (0, 1)                                # shared face: test side / neighbour side
    near = bits[:, axis] == c_n
    src = bits.copy(); src[:, axis] = c_t
    shared = tau[src @ np.array([4, 2, 1])]                                 # neighbour corner value on the shared face
    h = hashlib.sha256(f"{seed}:{row['case_id']}:{tag}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], 'little'))
    how = 'fallback_copy' if force_copy else 'fallback'
    for _ in range(0 if force_copy else 5000):
        c = shared.copy(); c[~near] = rng.uniform(TAU_LO, F.TAU_UPPER, int((~near).sum()))
        c = np.round(c, 12)
        if (c.min() > TAU_LO and c.max() < F.TAU_UPPER and np.ptp(c) <= F.SPAN_MAX and F.gradient_max(c) <= F.GRADIENT_MAX):
            how = 'random'
            break
    else:
        opp = bits.copy(); opp[:, axis] = c_n
        c = shared[opp @ np.array([4, 2, 1])]
    name = f"{row['case_id']}_nb{tag}"
    return dict(case_id=name, family_id=row['family_id'], split=row['split'], kind='FULL', role='neighbour',
                neighbour_of=row['case_id'], neighbour_tag=tag, neighbour_draw=how, field_kind='NEIGHBOUR',
                tau_corners=[format(x, '.12f') for x in c], normal=['1', '0', '0'], offset='1',
                retained_macro_volume_target=1.0, geometry_orbit_sha256=F.orbit_key(c))


def admissible_glue(row, min_frac=0.02):
    """Glue faces (px, py, pz, mz) with material on >= min_frac of the face: FULL all four; a cut keeps n.x <= offset."""
    tags = ['px', 'py', 'pz', 'mz']
    if row['kind'] == 'FULL':
        return tags
    nrm = np.asarray([float(x) for x in row['normal']]); b = float(row['offset'])
    s_ = (np.arange(40) + .5) / 40
    U, V = np.meshgrid(s_, s_, indexing='ij'); U, V = U.ravel(), V.ravel()
    out = []
    for t in tags:
        axis, sgn = TAGS[t]
        P = np.zeros((len(U), 3)); others = [k for k in range(3) if k != axis]
        P[:, others[0]], P[:, others[1]] = U, V; P[:, axis] = 1.0 if sgn > 0 else 0.0
        if float((P @ nrm <= b).mean()) >= min_frac:
            out.append(t)
    return out


def indep(a, F, src_sha):
    """The generator nests training families in levels up to 512 per call: more training cells use consecutive seeds
    (seed, seed + 1, ...; 512 train fields each, prefix-stable); validation fields come from the first seed only."""
    rows, k, left = [], 0, a.train
    while left > 0 or k == 0:
        t, v = min(512, left), (a.val if k == 0 else 0)
        seed_k = a.seed + k
        if seed_k in (REGISTERED_SEED, 2026092601):
            raise SystemExit(f'SEED_RESERVED {seed_k}')
        fams, cases = F.generate(seed_k, ceil8(t) if t else 8, ceil8(v) if v else 8, 8)
        byid = {c['case_id']: c for c in cases}
        for f in fams:
            split, i = f['split'], f['family_index']
            if not ((split == 'train' and i < t) or (split == 'development' and i < v)):
                continue
            g = 512 * k + i if split == 'train' else i                      # global index: blocks of 8 stay intact
            j = g % 8
            suffix = '_full' if j < 2 else '_d{}_v{}'.format(*COMBOS[j - 2])
            c = byid[f['family_id'] + suffix]
            nid = f"fresh_{'train' if split == 'train' else 'val'}_{a.offset + g:04d}"
            rows.append(dict(c, case_id=nid + suffix, family_id=nid, split='train' if split == 'train' else 'validation',
                             generator_seed=seed_k, generator_case_id=c['case_id'], generator_family_id=f['family_id'],
                             block=g // 8, block_position=j, design='independent field per cell'))
        left -= t; k += 1
    if len({r['geometry_orbit_sha256'] for r in rows}) != len(rows):
        raise SystemExit('DUPLICATE_ORBIT_IN_PLAN')
    gate_tags = [t for t in a.gate_tags.split(',') if t]
    gate_splits = set(x for x in a.gate_splits.split(',') if x)
    want = {}
    for r in rows:
        adm = admissible_glue(r)
        h = hashlib.sha256(f"{a.seed}:{r['case_id']}:glued".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], 'little'))
        g = sorted(rng.choice(adm, size=min(a.glued_per_cell, len(adm)), replace=False).tolist()) if adm else []
        want[r['case_id']] = dict(glued=g, gate=gate_tags if r['split'] in gate_splits else [])
    nbs = [neighbour(F, r, t, a.seed) for r in rows for t in want[r['case_id']]['glued'] + want[r['case_id']]['gate']]
    ids = [r['case_id'] for r in rows + nbs]
    existing = set(x.name for x in PACKETS.iterdir())
    if len(set(ids)) != len(ids) or set(ids) & existing:
        raise SystemExit('CASE_ID_COLLISION')
    mothers = {json.loads((d / 'FRESH_CONTEXT.json').read_text())['case']['geometry_orbit_sha256'] for d in PACKETS.iterdir()
               if d.name.endswith('_full')}
    if {r['geometry_orbit_sha256'] for r in rows} & mothers:
        raise SystemExit('ORBIT_COLLISION_WITH_EXISTING')
    fb = sum(n['neighbour_draw'] == 'fallback' for n in nbs)
    plan = dict(schema='OPL_EXPANSION_PLAN_V2', mode='indep', seed=a.seed, generator_sha256=src_sha, args=vars(a),
                counts=dict(train=sum(r['split'] == 'train' for r in rows), validation=sum(r['split'] == 'validation' for r in rows),
                            full=sum(r['kind'] == 'FULL' for r in rows), cut=sum(r['kind'] == 'CUT' for r in rows),
                            neighbours=len(nbs), neighbour_fallback=fb),
                train=[r['case_id'] for r in rows if r['split'] == 'train'],
                validation=[r['case_id'] for r in rows if r['split'] == 'validation'],
                neighbours={r['case_id']: want[r['case_id']] for r in rows},
                no_glue_face=[r['case_id'] for r in rows if not want[r['case_id']]['glued']],
                cases={r['case_id']: dict(split=r['split'], kind=r['kind'], block=r['block'], position=r['block_position'],
                                          volume=r.get('retained_macro_volume_target'), field_kind=r.get('field_kind'))
                       for r in rows})
    print(json.dumps(plan['counts']), flush=True)
    if a.dry:
        return
    out = Path(a.out) / 'packets'
    for r in rows + nbs:
        d = out / r['case_id']
        ctx = json.dumps(context(r, a.seed), indent=1)
        if (d / 'FRESH_CONTEXT.json').exists():
            if (d / 'FRESH_CONTEXT.json').read_text() != ctx:
                raise SystemExit(f"CONTEXT_DIFFERS {r['case_id']}")
            continue
        d.mkdir(parents=True, exist_ok=True)
        (d / 'SAMPLE.json').write_text(json.dumps(dict(schema='OPL_EXPANSION_SAMPLE_V1', gp=dict(gamma=GAMMA))))
        (d / 'FRESH_CONTEXT.json').write_text(ctx)
    (Path(a.out) / 'PLAN_INDEP.json').write_text(json.dumps(plan, indent=1))
    print('WROTE', len(rows), 'cells and', len(nbs), 'neighbour packets to', out, flush=True)


if __name__ == '__main__':
    main()
