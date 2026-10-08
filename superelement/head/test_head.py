"""Self-tests for the symmetric square-root head.

Exercises the exact index arithmetic predict_full uses, on a small synthetic
SPD matrix where the truth is known, for both heads.
"""
import json, sys
import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
from stage_cutfem_m4.factors import read_blocks, sample_pairs
from stage_cutfem_m4 import mechanics as F

torch.manual_seed(0)
dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
nodes, patch = 40, 8
d = 3 * nodes
report = {}

G = torch.randn(d, d, dtype=torch.float64, device=dev)
A = G @ G.T + d * torch.eye(d, dtype=torch.float64, device=dev)
A = (A + A.T) * .5
Rstar = torch.linalg.cholesky(A, upper=True)
lam, V = torch.linalg.eigh(A)
Mstar = (V * lam.sqrt()) @ V.T
Mstar = (Mstar + Mstar.T) * .5


def to_packed(dense):
    host = dense.cpu().numpy()
    return np.concatenate([host[row, row:] for row in range(len(host))])


def assemble(packed, head):
    """Byte-for-byte the expressions in run.predict_full."""
    out = torch.zeros((d, d), dtype=torch.float64, device=dev)
    a = torch.arange(3, device=dev)[None, :, None]
    b = torch.arange(3, device=dev)[None, None, :]
    total = nodes * (nodes + 1) // 2
    for lo in range(0, total, 97):
        index = np.arange(lo, min(total, lo + 97), dtype=np.int64)
        rows = np.floor((np.sqrt(8 * index + 1) - 1) / 2).astype(np.int64)
        cols = index - rows * (rows + 1) // 2
        pred = torch.as_tensor(read_blocks(packed, rows, cols, d), dtype=torch.float64, device=dev)
        r = torch.tensor(rows, device=dev); c = torch.tensor(cols, device=dev)
        if head == 'chol':
            out[3 * c[:, None, None] + b, 3 * r[:, None, None] + a] = pred
        else:
            out[3 * r[:, None, None] + a, 3 * c[:, None, None] + b] = pred
    if head == 'sqrt':
        assert int(torch.count_nonzero(torch.triu(out, 1))) == 0, 'wrote above the diagonal'
        out = out + torch.tril(out, -1).T
    F.validate_head(out, head)
    return out


for head, truth in (('chol', Rstar), ('sqrt', Mstar)):
    got = assemble(to_packed(truth), head)
    err = float((got - truth).abs().max())
    report[f'{head}_assembly_max_abs_error'] = err
    assert err == 0.0, f'{head} assembly is not exact: {err}'
    Ahat = got.T @ got
    report[f'{head}_Ahat_relative_error'] = float((Ahat - A).norm() / A.norm())
    div = F.head_divergence(got, Rstar, True, head)
    C = div.pop('C'); W = C.T @ C; W = (W + W.T) * .5
    mu = torch.linalg.eigvalsh(W)
    report[f'{head}_eps_op_of_exact_label'] = float((mu - 1).abs().max())
    report[f'{head}_D_closed'] = float(div['D'])
    report[f'{head}_D_spectral'] = float(F.spectral_phi(mu).sum())
    report[f'{head}_logdet_gap_closed'] = float(div['logdet_gap'])
    report[f'{head}_logdet_gap_spectral'] = float(mu.log().sum())

# the two divergence routes must agree on a triangular argument
perturbed = Rstar + 1e-3 * torch.triu(torch.randn_like(Rstar))
tri = F.divergence(perturbed, Rstar)
gen = F.head_divergence(perturbed, Rstar, False, 'sqrt')   # forces the general route
report['divergence_route_D_relative_difference'] = abs(float(tri['D'] - gen['D'])) / abs(float(tri['D']))
report['divergence_route_logdet_relative_difference'] = abs(float(tri['logdet_gap'] - gen['logdet_gap'])) / max(1e-30, abs(float(tri['logdet_gap'])))
assert report['divergence_route_D_relative_difference'] < 1e-8
assert report['divergence_route_logdet_relative_difference'] < 1e-8

# a symmetric root and a Cholesky factor of the SAME A must give the same spectrum
report['both_heads_give_identical_exact_spectrum'] = bool(
    abs(report['chol_eps_op_of_exact_label'] - report['sqrt_eps_op_of_exact_label']) < 1e-9)
report['status'] = 'HEAD_SELFTEST_PASS'
print(json.dumps(report, indent=1))
