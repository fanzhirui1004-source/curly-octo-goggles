"""Re-run evaluation for a finished run from its final checkpoint.

The training in that run completed; only the evaluation raised.  Nothing here
retrains or reinitialises: the conditioning comes from the checkpoint, so the
head, the normalisers and the weights are exactly the ones the run ended with.
"""
import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, default=None)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--inference-chunk', type=int, default=8192)
    ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args()

    sys.path.insert(0, str(a.source))
    from stage_cutfem_m4.run import selected_rows, prepare, evaluate, write
    from stage_cutfem_m4.factors import Conditioning
    from stage_cutfem_m4.adapter import model_for

    protocol = json.loads((a.run / 'PROTOCOL.json').read_text())
    checkpoint = a.checkpoint or sorted(a.run.glob('CHECKPOINT_*.pt'))[-1]
    saved = torch.load(checkpoint, map_location='cuda:0', weights_only=False)
    conditioning = Conditioning(**saved['conditioning'])
    conditioning.validate()

    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    args = SimpleNamespace(manifest=Path(protocol['source_files'] and
                                         '/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'),
                           seats=protocol['seats'], known_328=protocol['known_328_diagnostic'],
                           patch_size=protocol['patch_size'] if 'patch_size' in protocol else 32,
                           head=protocol['head'], steps=protocol['steps'],
                           inference_chunk=a.inference_chunk, pair_sampling=protocol['pair_sampling'],
                           pairs_per_bucket=protocol['same_patch_pairs'], seed=protocol['seed'])
    rows = selected_rows(args.manifest, args.seats, args.known_328)
    samples = [prepare(r, args) for r in rows]
    model = model_for(samples[0]['context']).to('cuda:0')
    model.load_state_dict(saved['model'])

    a.output.mkdir(parents=True, exist_ok=True)
    write(a.output / 'REEVAL.json',
          dict(run=str(a.run), checkpoint=str(checkpoint), step=int(saved['step']),
               head=args.head, conditioning=saved['conditioning'],
               scope='evaluation only; the weights and normalisers are the ones the run ended with',
               reason='the original evaluation raised in a diagnostic guard, not in training'))
    results = [evaluate(model, s, conditioning, a.output / f'EVAL_{s["seat"]:04d}', args) for s in samples]
    summary = {str(r['seat']): dict(e_A=r['spectrum']['e_A'], D_per_mode=r['spectrum']['D_per_mode'],
                                    mu_min=r['spectrum']['mu_min'], mu_max=r['spectrum']['mu_max'],
                                    eps_op=max(abs(r['spectrum']['mu_min'] - 1), abs(r['spectrum']['mu_max'] - 1)),
                                    factor_relative=r['spectrum']['factor_relative'],
                                    root_determinant_sign=r['spectrum'].get('root_determinant_sign'),
                                    numerical_gate_pass=r['spectrum']['numerical_gate_pass'])
               for r in results}
    write(a.output / 'RESULT.json', dict(status='REEVALUATION_COMPLETE', step=int(saved['step']),
                                         head=args.head, evaluations=summary))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
