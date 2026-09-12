"""Consistency test of the probe-free objective against the exact whitened spectrum (CPU, float64).
Checks: divergence_terms == sum(mu - log mu - 1) and the log-det gap == sum(log mu) for a V1 checkpoint on one label,
in the original frame and in a cube-symmetry frame; the extreme-mode Ritz values bracket the true extremes; gradients are finite."""
import json, sys, os, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v0_superelement as V0, v1_superelement as V1
CPU = torch.device('cpu'); V0.DEV = CPU; V1.DEV = CPU
from stage_cutfem_neural_a.elimination_reference import quotient_dense
torch.set_num_threads(int(os.environ.get('THREADS', '7')))
run, ckpt, seat = sys.argv[1], sys.argv[2], int(sys.argv[3])
proto = json.load(open(f'{run}/PROTOCOL.json')); off_scale = proto.get('off_scale', 1.0)
record = next(r for r in json.load(open('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json')) if int(r['seat']) == seat)
label = V1.Label(record, proto['r_near'], proto['decay'], off_scale)
sd = torch.load(ckpt, map_location='cpu')['net']; vw = proto.get('volume_width', 0) if any(k.startswith('volume.') for k in sd) else 0
net = V1.GeometryNet(label.images.shape[1], proto['width'], proto['rank'], proto['hidden'], off_scale=off_scale, volume_width=vw); net.load_state_dict(sd)
t0 = time.time(); g = label.to_gpu(need_A=False, need_Z=True, z_dtype=torch.float64); print('prepared', round(time.time() - t0, 1), 's', flush=True)
group = V1.cube_group()
frames = [f if f.startswith('Q') else int(f) for f in os.environ.get('FRAMES', '0,Q7').split(',')]
for frame in frames:
    ga = dict(g)
    if isinstance(frame, str):                                   # frame-0 student conjugated by a cube symmetry: exercises the Pi handling on a well-conditioned operator
        _, Q = V1.transform_grid(label.grid_np, *group[int(frame[1:])], V1.N); ga['Q'] = torch.from_numpy(Q)
    elif frame: ga.update(label.transformed(*group[frame]))
    values, M = net(ga, g['w'].log()); op = V1.Operator(ga, values, M)
    D, trace, gap, residual = V1.divergence_terms(g, op, torch.float64)
    (D / g['d'] + 0.1 * V1.extreme_terms(g, op, 2, 4)[0]).backward(); grads = [p.grad for p in net.parameters() if p.grad is not None]
    finite = all(bool(torch.isfinite(x).all()) for x in grads); net.zero_grad(set_to_none=True)
    with torch.no_grad():
        ge = dict(g); ge.pop('ext', None); _, ritz = V1.extreme_terms(ge, op, 40, 8)
        S = op.materialize(); Ahat, _ = quotient_dense(S, g['data'].quotient); del S
        R = g['Rstar']; Y = torch.linalg.solve_triangular(R.T, Ahat, upper=False); del Ahat
        W = torch.linalg.solve_triangular(R.T, Y.T.contiguous(), upper=False); del Y; mu = torch.linalg.eigvalsh(0.5 * (W + W.T)); del W
    D_exact = float((mu - mu.log() - 1).sum()); gap_exact = float(mu.log().sum())
    print(json.dumps(dict(frame=str(frame), D=float(D), D_exact=D_exact, gap=float(gap), gap_exact=gap_exact, trace_per_mode=float(trace) / g['d'],
                          ritz_max=float(ritz.max()), mu_max=float(mu.max()), ritz_min=float(ritz.min()), mu_min=float(mu.min()), solve_residual=residual, conditioning=V1.factor_conditioning(op), grad_finite=finite, seconds=round(time.time() - t0, 1))), flush=True)
