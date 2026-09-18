"""Is the residual error a global calibration bias or scatter?

The probe says the first-order compliance error is m - 1 with
m = sum_j wc_j / r_j over sum_j wc_j, so a single global scale alpha = m zeroes it
EXACTLY at first order -- which makes the first-order test vacuous.  The real test is
the full re-solve, which is not linear in alpha.  So scale the factor by sqrt(alpha),
re-run the two-cell assembly, and see how much of the measured error was bias.

alpha comes out near 0.85 for route 5 on all four loads independently (0.852, 0.842,
0.839, 0.860), which is itself evidence of a genuine global bias rather than a
load-specific artefact.
"""
import argparse, numpy as np
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--source', type=Path, required=True)
ap.add_argument('--alpha', type=float, required=True)
ap.add_argument('--output', type=Path, required=True)
a = ap.parse_args()
p = np.load(a.source, mmap_mode='r', allow_pickle=False)
out = np.lib.format.open_memmap(a.output, mode='w+', dtype=np.float64, shape=p.shape)
s = float(np.sqrt(a.alpha))
step = 1 << 23
for lo in range(0, len(p), step):
    out[lo:lo + step] = np.asarray(p[lo:lo + step]) * s
out.flush()
print(f'wrote {a.output} scaled by sqrt({a.alpha}) = {s:.6f}')
