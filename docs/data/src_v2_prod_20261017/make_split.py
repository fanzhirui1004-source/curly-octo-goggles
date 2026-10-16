"""Step-2 split by family (a FULL parent and its cut variants never cross sets).
test  = the 'development' split of the packets; val = VAL_FAMILIES random families of the 'train' split;
train = the rest, ordered by a fixed random family permutation so that nested learning-curve subsets are prefixes.
Only geometries with DONE.json are used (NO_INTERIOR slivers and failures are listed separately).
Usage: make_split.py <data_dir> <out.json> [val_families=4] [seed=0]"""
import sys, json, random
from pathlib import Path
from collections import defaultdict

P = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/packets')
data, out = Path(sys.argv[1]), sys.argv[2]
nval = int(sys.argv[3]) if len(sys.argv) > 3 else 4
rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 0)
fam, split, excluded = defaultdict(list), {}, {}
for d in sorted(P.iterdir()):
    c = json.loads((d / 'FRESH_CONTEXT.json').read_text())['case']
    if not (data / d.name / 'DONE.json').exists():
        f = data / d.name / 'FAILED.json'
        excluded[d.name] = json.loads(f.read_text()).get('error', '')[:80] if f.exists() else 'missing'
        continue
    fam[c['family_id']].append(d.name); split[c['family_id']] = c['split']
test_f = sorted(f for f in fam if split[f] == 'development')
train_f = sorted(f for f in fam if split[f] != 'development')
rng.shuffle(train_f)
val_f, train_f = train_f[:nval], train_f[nval:]
order = [g for f in train_f for g in sorted(fam[f])]
curve = {}
for n in (50, 100, 150):
    k, acc = 0, []
    for f in train_f:                                 # whole families until at least n geometries
        if len(acc) >= n:
            break
        acc += sorted(fam[f]); k += 1
    curve[str(n)] = dict(families=train_f[:k], geometries=acc)
rec = dict(test=[g for f in test_f for g in sorted(fam[f])], val=[g for f in val_f for g in sorted(fam[f])],
           train=order, train_families=train_f, val_families=val_f, test_families=test_f, curve=curve, excluded=excluded)
Path(out).write_text(json.dumps(rec, indent=1))
print(json.dumps({k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in rec.items()}),
      {n: len(v['geometries']) for n, v in curve.items()})
