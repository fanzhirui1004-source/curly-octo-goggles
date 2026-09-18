"""Build MQ_UPPER.npy for a list of seats, one subprocess per seat, resumable.

Peak memory is about seven simultaneous q x q float64 buffers, so the device is chosen per
seat: the 32 GB card takes everything that fits with headroom, and the rest goes to CPU
(754 GB of RAM, and the eigendecomposition is the only expensive step).  A seat whose
MQ_RESULT.json already exists is skipped, so the driver can be re-run after an interruption;
a seat that fails its self-test gates leaves MQ_FAILURE.json and is reported, and the
remaining seats still build.

    python -m superelement.equi.expand_mq_labels --manifest V2_LABELS.json --seats 114 181 ... \
        --output <log dir> [--device-budget-gib 26] [--threads 8]
"""
from __future__ import annotations

import argparse, json, subprocess, sys, time
from pathlib import Path


def peak_gib(q):
    return 7 * q * q * 8 / 2 ** 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--seats', type=int, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device-budget-gib', type=float, default=26.,
                    help='a seat whose estimated peak exceeds this goes to CPU instead of cuda:0')
    ap.add_argument('--force-cpu', action='store_true')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/CLAUDE_SQRTHEAD_20260917/src_v5'))
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    rows = {int(r['seat']): r for r in json.loads(a.manifest.read_text())}
    plan = []
    for s in a.seats:
        r = rows.get(s)
        if r is None:
            raise ValueError(f'SEAT_NOT_IN_MANIFEST {s}')
        q = int(r['q'])
        plan.append(dict(seat=s, q=q, reference=r['reference'], trace_cache=r['trace_cache'],
                         peak_gib=round(peak_gib(q), 1),
                         device='cpu' if (a.force_cpu or peak_gib(q) > a.device_budget_gib) else 'cuda:0'))
    plan.sort(key=lambda d: d['q'])
    (a.output / 'PLAN.json').write_text(json.dumps(plan, indent=1))
    print(json.dumps(dict(phase='plan', seats=len(plan), on_cpu=sum(d['device'] == 'cpu' for d in plan),
                          q_min=plan[0]['q'], q_max=plan[-1]['q'],
                          bytes_gib=round(sum(d['q'] * (d['q'] + 1) // 2 * 8 for d in plan) / 2 ** 30, 1))), flush=True)
    done = failed = skipped = 0
    start = time.perf_counter()
    for i, d in enumerate(plan, 1):
        ref = Path(d['reference'])
        if (ref / 'MQ_RESULT.json').exists() and (ref / 'MQ_UPPER.npy').exists():
            skipped += 1
            print(json.dumps(dict(index=i, of=len(plan), seat=d['seat'], phase='skipped')), flush=True)
            continue
        cmd = [a.python, '-u', '-m', 'superelement.equi.build_mq_label', '--reference', str(ref),
               '--seat', str(d['seat']), '--trace-cache', d['trace_cache'], '--device', d['device'],
               '--threads', str(a.threads), '--source', str(a.source)]
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True)
        row = dict(index=i, of=len(plan), seat=d['seat'], q=d['q'], device=d['device'],
                   peak_gib=d['peak_gib'], seconds=round(time.perf_counter() - t0, 1))
        if p.returncode != 0:
            failed += 1
            row.update(phase='failed', returncode=p.returncode, stderr=p.stderr[-400:])
        else:
            done += 1
            rep = json.loads((ref / 'MQ_RESULT.json').read_text())
            row.update(phase='built', label_g=rep['label_g'], rigid_nullspace=rep['rigid_nullspace'],
                       rigid_span_residual=rep['rigid_span_residual'],
                       round_trip=rep['round_trip_to_M'], gates_pass=all(g['pass_'] for g in rep['gates'].values()))
        print(json.dumps(row), flush=True)
        with (a.output / 'BUILD.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
    result = dict(status='MQ_EXPANSION_COMPLETE' if not failed else 'MQ_EXPANSION_WITH_FAILURES',
                  built=done, skipped=skipped, failed=failed, seconds=round(time.perf_counter() - start, 1))
    (a.output / 'RESULT.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
