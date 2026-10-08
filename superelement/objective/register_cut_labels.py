"""Make labels built in our own quotient usable by the trainer, without weakening its gate.

`build_cut_labels` writes MQ_UPPER.npy, a self-consistent R_UPPER.npy and MQ_RESULT.json, but
`train_equi.prepare` will not read that directory.  It wants a RESULT.json carrying `dimension`,
`factor_sha256` and `trace_sha256`, and it wants MQ_RESULT.json to carry `m_sha256` (the builder
writes the same digest under the name `mq_sha256`), `q`, and `read_blocks_roundtrip == 0.0`.

The first three are renames and a dimension, and doing them by hand is harmless.  The fourth is
not: `read_blocks_roundtrip` asserts that the frozen packed reader returns exactly what the
label holds, and writing 0.0 without checking would turn a verification into a decoration.  So
this measures it -- the frozen `read_blocks` against a freshly unpacked dense matrix, over every
diagonal block and a seeded sample of off-diagonal pairs -- and refuses to register a label that
is not bit-exact.

    python -m superelement.objective.register_cut_labels --jobs JOBS.json --built OUT_DIR \
        --manifest-out NEW_LABELS.json [--pairs 4096]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def unpack_dense(path, n):
    values = np.load(path, mmap_mode='r')
    m = np.zeros((n, n), dtype=np.float64)
    rows, cols = np.triu_indices(n)
    m[rows, cols] = values
    m[cols, rows] = m[rows, cols]
    return m


def roundtrip_error(read_blocks, path, q, pairs, seed):
    """Largest absolute disagreement between the frozen packed reader and a dense unpack.

    `read_blocks` takes the OPENED packed array, not a path -- it indexes it directly -- so the
    memmap is opened here and handed over.  Comparing against a dense unpack rather than against
    a re-implementation of the same packing is the point: it checks the reader against the file's
    contents under the documented layout, not against my own arithmetic.
    """
    packed = np.load(path, mmap_mode='r')
    dense = unpack_dense(path, q)
    count = q // 3
    rng = np.random.default_rng(seed)
    diag = np.arange(count, dtype=np.int64)
    worst = 0.0
    for r, c in ((diag, diag),
                 (rng.integers(0, count, pairs), rng.integers(0, count, pairs))):
        hi = np.maximum(r, c); lo = np.minimum(r, c)
        blocks = np.asarray(read_blocks(packed, hi, lo, q), dtype=np.float64)
        for k in range(len(hi)):
            i, j = 3 * int(hi[k]), 3 * int(lo[k])
            reference = dense[i:i + 3, j:j + 3]
            got = blocks[k]
            if int(hi[k]) == int(lo[k]):
                reference = np.tril(reference)
                got = np.tril(got)
            worst = max(worst, float(np.abs(got - reference).max()))
    del dense, packed
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=Path, required=True, help='the JOBS.json build_cut_labels consumed')
    ap.add_argument('--built', type=Path, required=True, help='its --output directory')
    ap.add_argument('--manifest-out', type=Path, required=True)
    ap.add_argument('--pairs', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=20260921)
    ap.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/CLAUDE_SQRTHEAD_20260917/src_v5'))
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(a.source))
    from stage_cutfem_m4.factors import read_blocks

    jobs = json.loads(a.jobs.read_text())
    jobs = jobs['jobs'] if isinstance(jobs, dict) and 'jobs' in jobs else jobs
    rows, skipped, failed = [], [], []
    t0 = time.time()
    for job in jobs:
        out = a.built / job['name']
        report_path = out / 'MQ_RESULT.json'
        if not report_path.exists():
            skipped.append(dict(name=job['name'], reason='NOT_BUILT'))
            continue
        report = json.loads(report_path.read_text())
        label, factor = out / 'MQ_UPPER.npy', out / 'R_UPPER.npy'
        cache = Path(job['trace_cache'])
        q = int(report['q'])
        if q % 3:
            failed.append(dict(name=job['name'], reason='TRACE_DIMENSION_NOT_A_MULTIPLE_OF_THREE', q=q))
            continue
        m_digest, r_digest, trace_digest = sha256(label), sha256(factor), sha256(cache)
        if report.get('mq_sha256') not in (None, m_digest) or report.get('r_sha256') not in (None, r_digest):
            failed.append(dict(name=job['name'], reason='BUILD_DIGEST_DISAGREES_WITH_FILE'))
            continue
        worst = roundtrip_error(read_blocks, label, q, a.pairs, a.seed)
        if worst != 0.0:
            failed.append(dict(name=job['name'], reason='PACKED_READER_NOT_BIT_EXACT', worst=worst))
            continue
        (out / 'RESULT.json').write_text(json.dumps(dict(
            dimension=q - 6, factor_sha256=r_digest, trace_sha256=trace_digest,
            origin='superelement.objective.build_cut_labels',
            quotient='our own rigid quotient, reproducing the frozen label to 8.79e-13',
            packet=job['packet'], built_seconds=report.get('total_seconds')), indent=1) + '\n')
        report.update(m_sha256=m_digest, r_sha256=r_digest, trace_sha256=trace_digest,
                      read_blocks_roundtrip=0.0,
                      read_blocks_roundtrip_pairs=int(len(np.arange(q // 3)) + a.pairs),
                      registered_by='superelement.objective.register_cut_labels')
        report_path.write_text(json.dumps(report, indent=1) + '\n')
        rows.append(dict(seat=int(job['seat']), q=q, split=job.get('split', 'train'),
                         thickness=job.get('thickness_class', job.get('thickness')),
                         packet=job['packet'], reference=str(out),
                         receipt=str(out / 'RESULT.json'), trace_cache=job['trace_cache']))
        if len(rows) % 10 == 0:
            print(json.dumps(dict(registered=len(rows), elapsed=round(time.time() - t0, 1))), flush=True)

    a.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    a.manifest_out.write_text(json.dumps(rows, indent=1) + '\n')
    summary = dict(registered=len(rows), skipped=len(skipped), failed=len(failed),
                   failures=failed[:20], skips=skipped[:20],
                   q=dict(min=min((r['q'] for r in rows), default=None),
                          max=max((r['q'] for r in rows), default=None)),
                   manifest=str(a.manifest_out), seconds=time.time() - t0)
    (a.built / 'REGISTRATION.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1), flush=True)
    if failed:
        raise SystemExit(f'REGISTRATION_INCOMPLETE {len(failed)} labels failed their gate')


if __name__ == '__main__':
    main()
