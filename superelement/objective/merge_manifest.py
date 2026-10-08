"""Merge the newly built labels into one manifest the M4 harness can read.

Rows keep the frozen manifest's shape.  The original 33 rows are copied verbatim
so the existing splits, including the 6 holdout seats and the validation seat,
are preserved exactly; new rows are train only, because the producer refuses
anything else.
"""
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frozen', type=Path, required=True)
    ap.add_argument('--new', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()

    frozen = json.loads(args.frozen.read_text())
    new = json.loads(args.new.read_text()) if args.new.exists() else []
    keys = ('seat', 'split', 'thickness', 'q', 'packet', 'receipt', 'reference', 'trace_cache')
    seats = {int(r['seat']) for r in frozen}
    rows = list(frozen)
    added = 0
    for r in new:
        if int(r['seat']) in seats:
            raise ValueError(f"DUPLICATE_SEAT {r['seat']}")
        if r['split'] != 'train':
            raise ValueError('NEW_ROWS_MUST_BE_TRAIN')
        for k in keys:
            if k not in r:
                raise ValueError(f'MISSING_KEY {k}')
        if not (Path(r['reference']) / 'R_UPPER.npy').exists():
            raise ValueError(f"MISSING_FACTOR {r['reference']}")
        seats.add(int(r['seat']))
        rows.append({k: r[k] for k in keys})
        added += 1
    args.output.write_text(json.dumps(rows, indent=1))
    counts = {}
    for r in rows:
        counts[r['split']] = counts.get(r['split'], 0) + 1
    print(json.dumps(dict(output=str(args.output), total=len(rows), added=added, splits=counts)))


if __name__ == '__main__':
    main()
