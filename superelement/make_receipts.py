"""Select FULL labels and write input receipts (SHA verification against each packet MANIFEST). CPU only, read-only on packets."""
import json, hashlib, glob, os, sys, time
from pathlib import Path
root = Path('/root/autodl-tmp/CUTFEM_INGEST_R38/dataset_independent_20260910')
out = Path('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS'); (out / 'receipts').mkdir(parents=True, exist_ok=True)
qmax = int(sys.argv[1]) if len(sys.argv) > 1 else 26000
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''): h.update(c)
    return h.hexdigest()
rows = []
for sj in sorted(glob.glob(str(root / '*/payload/packet/SAMPLE.json'))):
    d = json.load(open(sj)); ctx = d['dataset_context']
    if d.get('additional_scalar_cut_coordinates', 0) > 0: continue
    rows.append(dict(seat=int(Path(sj).parents[2].name.split('_')[-1]), split=ctx['split'], thickness=ctx['thickness_class'], q=d['full_trace_dimension'], packet=str(Path(sj).parent)))
rows = [r for r in rows if r['q'] <= qmax]
train = [r for r in rows if r['split'] == 'train']; val = [r for r in rows if r['split'] == 'validation']
print('FULL labels with q<=%d: train %d validation %d test %d' % (qmax, len(train), len(val), sum(1 for r in rows if r['split'] == 'test')))
import random; random.Random(2026091207).shuffle(train); random.Random(2026091208).shuffle(val)
chosen = train[:32] + val[:6]
t0 = time.time(); records = []
for r in chosen:
    packet = Path(r['packet']); manifest = json.load(open(packet / 'MANIFEST.json'))
    files = {}; ok = True
    for name, digest in manifest.items():
        st = (packet / name).stat(); h = sha(packet / name); passed = (h == digest)
        files[name] = {"bytes": st.st_size, "sha256": h, "pass": passed, "mtime_ns": st.st_mtime_ns}; ok &= passed
    rec = dict(packet=str(packet), seat=r['seat'], all_files_verified=ok, seconds=time.time() - t0, files=files)
    rp = out / 'receipts' / f"{r['seat']}.json"; rp.write_text(json.dumps(rec, indent=2))
    records.append(dict(seat=r['seat'], split=r['split'], thickness=r['thickness'], q=r['q'], packet=str(packet), receipt=str(rp), verified=ok))
    print(r['seat'], r['split'], r['thickness'], r['q'], 'verified' if ok else 'FAILED', flush=True)
(out / 'LABELS.json').write_text(json.dumps(records, indent=2))
print('done', round(time.time() - t0, 1), 's')
