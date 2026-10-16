"""T64 health on the host (memory-mapped, row blocks): non-finite count, max asymmetry, diagonal range and the number of
port DOFs with diagonal below 1e-9 x max."""
import sys, json
import numpy as np
body = sys.argv[1]
for c in sys.argv[2:]:
    T = np.load(f'{body}/{c}_portview/T64.npy', mmap_mode='r'); n = T.shape[0]; bad = 0; asym = 0.0
    for i in range(0, n, 2048):
        b = np.asarray(T[i:i + 2048]); bad += int((~np.isfinite(b)).sum())
        if bad == 0:
            asym = max(asym, float(np.abs(b - np.asarray(T[:, i:i + 2048]).T).max()))
    d = np.array([T[k, k] for k in range(n)])
    print(json.dumps(dict(case=c, n=int(n), nonfinite=bad, asym=asym, diag_min=float(d.min()), diag_max=float(d.max()),
                          weak=int((d < 1e-9 * d.max()).sum()))), flush=True)
