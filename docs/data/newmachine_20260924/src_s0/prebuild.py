"""Build and store the training slots (slimmed trainlib.Geo on the host) of a split, so train2 starts without rebuilding.
Usage: prebuild.py <config.json>   (uses split, body, data, slot_cache)"""
import sys, json, time
from pathlib import Path
import torch
import train2 as T2

cfg = json.loads(Path(sys.argv[1]).read_text())
split = json.loads(Path(cfg['split']).read_text())
out = Path(cfg['slot_cache']); out.mkdir(parents=True, exist_ok=True)
cases = split['train'] + split['val']
t0 = time.perf_counter()
for i, c in enumerate(cases):
    f = out / f'{c}.pt'
    if f.exists():
        continue
    g = T2.build_geo(c, cfg)
    T2.move(g, 'cpu'); torch.save(g, f)
    del g; torch.cuda.empty_cache()
    print(json.dumps(dict(i=i, case=c, s=time.perf_counter() - t0)), flush=True)
print('PREBUILD_DONE', flush=True)
