"""Build teacher factors (dense_reference) for the selected labels and write the V1 label list. GPU; run after V0F."""
import json, subprocess, sys, time
from pathlib import Path
base = Path('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS'); labels = json.load(open(base / 'LABELS.json'))
src = '/root/cutfem_neural_a_20260910/source_14301bc56'; py = '/root/cutfem_neural_a_20260910/env/bin/python'
out = []
for r in labels:
    if not r['verified']: continue
    ref = base / f"REFERENCE_{r['seat']:04d}"
    if not (ref / 'RESULT.json').exists():
        if ref.exists(): subprocess.run(['rm', '-rf', str(ref)])
        t = time.time()
        p = subprocess.run([py, '-m', 'stage_cutfem_neural_a.dense_reference', '--packet', r['packet'], '--receipt', r['receipt'], '--output', str(ref)],
                           cwd=src, capture_output=True, text=True, env=dict(__import__('os').environ, PYTHONPATH=src, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'))
        ok = (ref / 'RESULT.json').exists()
        print(r['seat'], r['split'], r['q'], 'ok' if ok else 'FAILED', round(time.time() - t, 1), 's', '' if ok else p.stderr[-400:], flush=True)
        if not ok: continue
    out.append(dict(seat=r['seat'], split=r['split'], thickness=r['thickness'], q=r['q'], packet=r['packet'], receipt=r['receipt'], reference=str(ref), trace_cache=str(ref / 'input' / 'TRACE_CACHE.npz')))
# include the existing 0415 reference as a training label and 0401 as validation
existing = json.load(open('/root/autodl-tmp/CUTFEM_NEURAL_DENSE_20260911_P01/INPUT_MANIFEST.json'))['samples']
for s in existing:
    if int(s['seat']) in (415, 401) and int(s['seat']) not in {o['seat'] for o in out}:
        out.append(dict(seat=int(s['seat']), split='train' if int(s['seat']) == 415 else 'validation', thickness='AFFINE', q=20700 if int(s['seat']) == 415 else 17586,
                        packet=s['packet'], receipt=s['receipt'], reference=s['reference'], trace_cache=s['trace_cache']))
json.dump(out, open(base / 'V1_LABELS.json', 'w'), indent=2)
print('labels ready:', len(out), 'train', sum(1 for o in out if o['split'] == 'train'), 'validation', sum(1 for o in out if o['split'] != 'train'))
