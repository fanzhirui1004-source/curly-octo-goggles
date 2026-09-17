"""Turn already-condensed packets into training labels.

dense_reference.run reproduces seat 0253's frozen R_UPPER.npy bit-for-bit from
its packet alone, so the same call turns any other verified train packet into a
label.  This driver writes the input receipt that verified_packet requires,
runs the producer, and appends a manifest row in the shape the M4 harness reads.

Holdout hygiene is the producer's own: verified_packet refuses any packet whose
split is not train.
"""
import argparse, hashlib, json, re, subprocess, sys, time
from pathlib import Path

SOURCE = Path('/root/cutfem_neural_a_20260910/source_14301bc56')


def sha256(path, chunk=1 << 23):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def write_receipt(packet, path):
    """Verify every manifest digest, then record the stat the producer rechecks."""
    packet = Path(packet)
    manifest = json.loads((packet / 'MANIFEST.json').read_text())
    files = {}
    for name, digest in manifest.items():
        st = (packet / name).stat()
        got = sha256(packet / name)
        files[name] = {'bytes': st.st_size, 'sha256': got, 'pass': got == digest,
                       'mtime_ns': st.st_mtime_ns}
    record = dict(packet=str(packet.resolve()),
                  all_files_verified=all(v['pass'] for v in files.values()), files=files,
                  scope='input verification only; no numerical content is derived here')
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    return record['all_files_verified']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--census', type=Path, required=True)
    ap.add_argument('--existing', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--max-q', type=int, default=18000)
    ap.add_argument('--max-seats', type=int, default=10000)
    ap.add_argument('--python', default='/root/cutfem_neural_a_20260910/env/bin/python')
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'receipts').mkdir(exist_ok=True)
    rows = json.loads(args.census.read_text())
    existing = json.loads(args.existing.read_text())
    have = {int(r['seat']) for r in existing}
    have_packets = {str(Path(r['packet']).resolve()) for r in existing}

    pool = [r for r in rows if r['split'] == 'train' and r['q'] <= args.max_q
            and str(Path(r['packet']).resolve()) not in have_packets]
    pool.sort(key=lambda r: r['q'])
    pool = pool[:args.max_seats]

    used, plan = set(have), []
    for r in pool:
        m = re.fullmatch(r'(\d+)', r['tag'])
        seat = int(m.group(1)) if m else None
        if seat is None or seat in used:
            seat = 100000 + len(plan)
            while seat in used:
                seat += 1
        used.add(seat)
        plan.append(dict(r, seat=seat))
    (args.output / 'PLAN.json').write_text(json.dumps(
        dict(max_q=args.max_q, candidates=len(pool), already_built=len(have),
             estimated_gib=sum(((r['q'] - 6) * (r['q'] - 5) // 2) * 8 for r in plan) / 2**30,
             rows=plan), indent=1))
    print(json.dumps(dict(phase='plan', seats=len(plan),
                          gib=round(sum(((r['q'] - 6) * (r['q'] - 5) // 2) * 8 for r in plan) / 2**30, 1))), flush=True)

    manifest_path = args.output / 'NEW_LABELS.json'
    built = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    done = {int(r['seat']) for r in built}
    for index, r in enumerate(plan):
        if r['seat'] in done:
            continue
        tick = time.perf_counter()
        reference = args.output / f"REFERENCE_{r['seat']:05d}"
        if reference.exists():
            print(json.dumps(dict(phase='skip_existing_directory', seat=r['seat'])), flush=True)
            continue
        receipt = args.output / 'receipts' / f"{r['seat']:05d}.json"
        if not write_receipt(r['packet'], receipt):
            print(json.dumps(dict(phase='receipt_failed', seat=r['seat'], tag=r['tag'])), flush=True)
            continue
        proc = subprocess.run([args.python, '-u', '-m', 'stage_cutfem_neural_a.dense_reference',
                               '--packet', r['packet'], '--receipt', str(receipt),
                               '--output', str(reference)],
                              cwd=SOURCE, capture_output=True, text=True)
        if proc.returncode != 0:
            print(json.dumps(dict(phase='producer_failed', seat=r['seat'], tag=r['tag'],
                                  stderr=proc.stderr[-600:])), flush=True)
            continue
        result = json.loads((reference / 'RESULT.json').read_text())
        row = dict(seat=r['seat'], split='train', thickness=r['thick'], q=r['q'],
                   packet=r['packet'], receipt=str(receipt), reference=str(reference),
                   trace_cache=str(reference / 'input' / 'TRACE_CACHE.npz'),
                   tag=r['tag'], mother_field_id=r['mother'], dimension=result['dimension'],
                   producer_seconds=result['total_seconds'], driver_seconds=time.perf_counter() - tick)
        built.append(row)
        manifest_path.write_text(json.dumps(built, indent=1))
        print(json.dumps(dict(phase='built', index=index, of=len(plan), seat=r['seat'],
                              q=r['q'], seconds=round(row['driver_seconds'], 2))), flush=True)
    print(json.dumps(dict(phase='complete', built=len(built), planned=len(plan))), flush=True)


if __name__ == '__main__':
    main()
