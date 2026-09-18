"""Build the A^{1/2} label beside every Cholesky label in a manifest.

Each one is an exact symmetric eigendecomposition of A = R^T R followed by the
same packed-upper layout, with the self-tests build_sqrt_label.py already runs:
exact symmetry, M@M against A, a read_blocks round trip that must be exactly 0,
and the whitened spectrum of the exact label.  Nothing is repaired.
"""
import argparse, json, shutil, subprocess, time
from pathlib import Path


def free_gib(path):
    total, used, free = shutil.disk_usage(path)
    return free / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--builder', type=Path, required=True)
    ap.add_argument('--python', default='/root/cutfem_neural_a_20260910/env/bin/python')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--min-free-gib', type=float, default=40.0)
    ap.add_argument('--only-seats', type=int, nargs='*', default=None)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = [r for r in json.loads(args.manifest.read_text()) if r['split'] == 'train']
    if args.only_seats is not None:
        keep = set(args.only_seats)
        rows = [r for r in rows if int(r['seat']) in keep]
    rows.sort(key=lambda r: int(r['q']))

    log = args.output / 'SQRT_LABELS.jsonl'
    built = []
    for index, r in enumerate(rows):
        reference = Path(r['reference'])
        target = reference / 'M_UPPER.npy'
        d = int(r['q']) - 6
        record = dict(index=index, of=len(rows), seat=int(r['seat']), q=int(r['q']), d=d)
        if (reference / 'SQRT_RESULT.json').exists() and target.exists():
            record.update(phase='already_built')
            print(json.dumps(record), flush=True)
            built.append(record)
            continue
        free = free_gib(reference)
        need = (d * (d + 1) // 2) * 8 / 2**30
        if free - need < args.min_free_gib:
            record.update(phase='stopped_low_disk', free_gib=round(free, 1), need_gib=round(need, 2))
            print(json.dumps(record), flush=True)
            with log.open('a') as fh:
                fh.write(json.dumps(record) + '\n')
            break
        tick = time.perf_counter()
        proc = subprocess.run([args.python, '-u', str(args.builder),
                               '--reference', str(reference), '--d', str(d),
                               '--seat', str(int(r['seat']))],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            record.update(phase='failed', stderr=proc.stderr[-600:])
            print(json.dumps(record), flush=True)
            with log.open('a') as fh:
                fh.write(json.dumps(record) + '\n')
            continue
        result = json.loads((reference / 'SQRT_RESULT.json').read_text())
        record.update(phase='built', seconds=round(time.perf_counter() - tick, 2),
                      square_residual=result['square_residual'],
                      sym_residual=result['sym_residual'],
                      read_blocks_roundtrip=result['read_blocks_roundtrip'],
                      label_eps_op=result['label_eps_op'],
                      log_pivot_bounds_ok=result['calibrate_log_bounds_ok'],
                      condition=result['condition'])
        for key, limit in (('square_residual', 1e-10), ('sym_residual', 0.0),
                           ('read_blocks_roundtrip', 0.0), ('label_eps_op', 1e-6)):
            if record[key] > limit:
                record.update(phase='selftest_out_of_tolerance', violated=key, limit=limit)
                break
        if not record['log_pivot_bounds_ok']:
            record.update(phase='selftest_out_of_tolerance', violated='log_pivot_bounds')
        print(json.dumps(record), flush=True)
        with log.open('a') as fh:
            fh.write(json.dumps(record) + '\n')
        built.append(record)

    ok = [r for r in built if r['phase'] in ('built', 'already_built')]
    summary = dict(status='SQRT_LABEL_EXPANSION_COMPLETE', planned=len(rows), built=len(ok),
                   seats=sorted(r['seat'] for r in ok),
                   free_gib_after=round(free_gib(args.output), 1),
                   worst_square_residual=max((r.get('square_residual', 0.) for r in ok), default=0.),
                   worst_label_eps_op=max((r.get('label_eps_op', 0.) for r in ok), default=0.))
    (args.output / 'SQRT_LABELS.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in
                      ('status', 'planned', 'built', 'free_gib_after',
                       'worst_square_residual', 'worst_label_eps_op')}), flush=True)


if __name__ == '__main__':
    main()
