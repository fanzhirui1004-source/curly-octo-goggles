#!/usr/bin/env python3
"""Prune training checkpoints to a geometric ladder, keeping every run's tail intact.

Touches ONLY files matching CHECKPOINT_<digits>.pt.  Labels, spectra, eigenvalues,
predictions, JSON receipts and CHECKPOINT.tmp (a possibly in-flight write) are never
considered.  Keeps, per run directory:
  - the last KEEP_TAIL checkpoints by step (so a running job cannot be left with none),
  - every step that is 64 * 2^k (a geometric ladder for evaluating mid-run convergence).
Dry run by default; --apply deletes.
"""
import argparse, os, re, sys
from pathlib import Path

PATTERN = re.compile(r'^CHECKPOINT_(\d+)\.pt$')
KEEP_TAIL = 3
LADDER = {64 * 2 ** k for k in range(0, 14)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/root/autodl-tmp')
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    runs = {}
    for dirpath, dirnames, filenames in os.walk(a.root):
        hits = [(int(PATTERN.match(f).group(1)), f) for f in filenames if PATTERN.match(f)]
        if hits:
            runs[dirpath] = sorted(hits)
    total_del = total_keep = 0
    plan = []
    for d, hits in sorted(runs.items()):
        steps = [s for s, _ in hits]
        keep = set(steps[-KEEP_TAIL:]) | (set(steps) & LADDER)
        delete = [(s, f) for s, f in hits if s not in keep]
        dbytes = sum((Path(d) / f).stat().st_size for _, f in delete)
        kbytes = sum((Path(d) / f).stat().st_size for s, f in hits if s in keep)
        total_del += dbytes; total_keep += kbytes
        plan.append((d, len(hits), len(delete), dbytes, kbytes, sorted(keep)))
        if a.apply:
            for _, f in delete:
                (Path(d) / f).unlink()
    for d, n, nd, db, kb, keep in plan:
        print(f'{d}\n  {n} checkpoints -> delete {nd} ({db/2**30:.1f} GiB), keep {n-nd} ({kb/2**30:.2f} GiB): {keep}')
    print(f'\nTOTAL {"DELETED" if a.apply else "TO DELETE"}: {total_del/2**30:.1f} GiB   kept: {total_keep/2**30:.1f} GiB')


if __name__ == '__main__':
    main()
