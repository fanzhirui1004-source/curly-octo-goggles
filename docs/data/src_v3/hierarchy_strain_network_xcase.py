"""Fixed-coefficient recursive mechanics feasibility baseline, not trained NN.

The compiler owns every coordinate and both nested prolongations.  All fine
coordinates remain active.  V-cycle residuals use full A and A1, including
off-diagonal coupling.  The only dense inverse factor is the <=800-DOF A2
reference constructed and timed offline by the caller.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import torch
from torch import Tensor, nn

# CROSS-CASE PORT (Claude, 2026-09-23): the only change from Codex source_chebyshev_full_01
# is this size guard, raised from 800 so the dense coarsest factor fits FULL/moderate cells.
COARSEST_LIMIT = 8192


def load_frozen_core(path: str | Path, expected_sha256: str):
    """Import the original complete-interface energy readout by verified file."""
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError("frozen mechanics core SHA mismatch")
    name = "_hierarchy_frozen_mechanics_" + digest
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclass resolves its defining module.
        spec.loader.exec_module(module)
    return sys.modules[name]


def mm(matrix: Tensor, panel: Tensor) -> Tensor:
    return matrix @ panel if matrix.layout == torch.strided else torch.sparse.mm(matrix, panel)


def _matrix(matrix: Tensor) -> Tensor:
    return matrix.detach() if matrix.layout == torch.strided else matrix.detach().to_sparse_coo().coalesce()


class BlockJacobi(nn.Module):
    """Disjoint, complete ragged blocks. -1 padding and factor padding are zero.

    F F.T is the corresponding inverse principal block.  Indices are padded
    on the right; every real coordinate occurs exactly once.  No clipping,
    pseudoinverse or factorization is performed here or in forward.
    """
    def __init__(self, dimension: int, indices: Tensor, factors: Tensor, omega: float):
        super().__init__()
        if indices.ndim != 2 or indices.dtype != torch.long or indices.shape[1] > 12:
            raise ValueError("block indices must be int64 [blocks,r<=12], -1 padded")
        blocks, width = indices.shape
        if factors.shape != (blocks, width, width) or factors.dtype != torch.float64:
            raise ValueError("padded block factors must be FP64 [blocks,r,r]")
        if factors.device != indices.device or not torch.isfinite(factors).all():
            raise ValueError("block indices/factors device or finiteness mismatch")
        valid = indices >= 0
        if torch.any(indices < -1) or not valid.any(dim=1).all():
            raise ValueError("each block needs real coordinates; padding sentinel is -1")
        if torch.any(valid[:, 1:] & ~valid[:, :-1]):
            raise ValueError("ragged block padding must be on the right")
        real = indices[valid]
        if int(real.max()) >= dimension or real.numel() != dimension:
            raise ValueError("blocks must cover the complete coordinate space")
        if not torch.equal(torch.sort(real).values, torch.arange(dimension, device=real.device)):
            raise ValueError("block coordinates must be disjoint and complete")
        allowed = valid[:, :, None] & valid[:, None, :]
        if torch.any(factors[~allowed] != 0):
            raise ValueError("padded factor entries must be zero")
        if torch.any(torch.tril(factors, diagonal=-1) != 0) or torch.any(factors.diagonal(dim1=-2, dim2=-1)[valid] <= 0):
            raise ValueError("reference factors must be upper inverse-Cholesky factors")
        if not 0 < float(omega) < float("inf"):
            raise ValueError("positive finite smoother omega required")
        self.dimension = dimension
        self.register_buffer("indices", indices.detach().clone())
        self.register_buffer("valid", valid)
        self.register_buffer("factors", factors.detach().clone())
        self.register_buffer("omega", factors.new_tensor(float(omega)))

    def forward(self, rhs: Tensor) -> Tensor:
        panel = rhs[self.indices.clamp_min(0)] * self.valid[:, :, None]
        values = self.omega * (self.factors @ (self.factors.transpose(-1, -2) @ panel))
        result = rhs.new_zeros(rhs.shape)
        return result.index_copy(0, self.indices[self.valid], values[self.valid])


class HierarchyCorrection(nn.Module):
    """Symmetric multiplicative V-cycle or additive 1/1/1 component control.

    Required setup certificates: A1=P1.T A P1, A2=P12.T A1 P12, full-column
    rank nested spaces, and omega_j*lambda_max(F_j.T A_j F_j)<=1. With these
    certificates, the symmetric pre/post V-cycle is SPD and B_V<=A^-1.
    Additive uses the same three components on the unchanged residual; its
    outer Richardson scale is 1/3. These are safeguards on iteration, never
    changes to the physical energy operator.
    """
    def __init__(self, A: Tensor, P1: Tensor, A1: Tensor, P12: Tensor, A2: Tensor,
                 fine_indices: Tensor, fine_factors: Tensor,
                 level1_indices: Tensor, level1_factors: Tensor,
                 coarse_factor: Tensor, omega0: float, omega1: float, *, mode: str):
        super().__init__()
        if mode not in ("vcycle", "additive"):
            raise ValueError("mode must be vcycle or additive")
        n0, n1, n2 = A.shape[0], A1.shape[0], A2.shape[0]
        if A.shape != (n0, n0) or A1.shape != (n1, n1) or A2.shape != (n2, n2):
            raise ValueError("energy blocks must be square")
        if P1.shape != (n0, n1) or P12.shape != (n1, n2):
            raise ValueError("nested prolongation dimensions do not match")
        if not 0 < n2 <= COARSEST_LIMIT or coarse_factor.shape != (n2, n2):
            raise ValueError("only explicit <=COARSEST_LIMIT-DOF complete coarsest reference allowed")
        tensors = [A, P1, A1, P12, A2, fine_factors, level1_factors, coarse_factor]
        if any(x.dtype != torch.float64 or x.device != A.device for x in tensors):
            raise ValueError("all hierarchy tensors must share one FP64 device")
        if fine_indices.device != A.device or level1_indices.device != A.device:
            raise ValueError("block indices must share the energy device")
        if not torch.isfinite(coarse_factor).all() or torch.any(torch.tril(coarse_factor, diagonal=-1) != 0) or torch.any(coarse_factor.diag() <= 0):
            raise ValueError("coarsest factor must be unrepaired upper inverse-Cholesky")
        self.mode = mode
        self.dimensions = (n0, n1, n2)
        for name, value in (("A", A), ("P1", P1), ("A1", A1), ("P12", P12), ("A2", A2)):
            self.register_buffer(name, _matrix(value))
        self.register_buffer("P1t", _matrix(P1.T))
        self.register_buffer("P12t", _matrix(P12.T))
        self.register_buffer("coarse_factor", coarse_factor.detach().clone())
        self.fine = BlockJacobi(n0, fine_indices, fine_factors, omega0)
        self.level1 = BlockJacobi(n1, level1_indices, level1_factors, omega1)

    def coarsest(self, residual: Tensor) -> Tensor:
        return self.coarse_factor @ (self.coarse_factor.T @ residual)

    def coarse_cycle(self, residual: Tensor) -> Tensor:
        state = self.level1(residual)
        updated = residual - mm(self.A1, state)
        state = state + mm(self.P12, self.coarsest(mm(self.P12t, updated)))
        return state + self.level1(residual - mm(self.A1, state))

    def forward(self, residual: Tensor) -> Tensor:
        if residual.ndim != 2 or residual.shape[0] != self.dimensions[0]:
            raise ValueError("correction expects a complete internal residual panel")
        if residual.dtype != self.A.dtype or residual.device != self.A.device:
            raise ValueError("residual must share the hierarchy FP64 device")
        if self.mode == "additive":
            coarse_residual = mm(self.P1t, residual)
            coarse = self.level1(coarse_residual)
            coarse = coarse + mm(self.P12, self.coarsest(mm(self.P12t, coarse_residual)))
            return self.fine(residual) + mm(self.P1, coarse)
        state = self.fine(residual)
        updated = residual - mm(self.A, state)
        state = state + mm(self.P1, self.coarse_cycle(mm(self.P1t, updated)))
        return state + self.fine(residual - mm(self.A, state))

    def operation_counts(self) -> dict:
        """One correction call; one matmul counts a complete RHS panel action."""
        v = self.mode == "vcycle"
        return {"A": 2 if v else 0, "A1": 2 if v else 0, "A2": 0,
                "fine_block_inverse": 2 if v else 1,
                "coarse1_block_inverse": 2 if v else 1,
                "coarsest_factor_pair": 1, "P1": 1, "P1t": 1,
                "P12": 1, "P12t": 1,
                "full_internal_solve": 0, "physical_directions_removed": 0}


def boundary_action_counts(mode: str, cycles: int) -> dict:
    """Complete inherited boundary_action, including explicit extension adjoint."""
    if mode not in ("vcycle", "additive") or cycles < 1:
        raise ValueError("invalid mode or cycle count")
    v = mode == "vcycle"
    calls = 2 * cycles  # extension and extension_adjoint
    return {"A": calls * (3 if v else 1) + 1,
            "A1": calls * (2 if v else 0), "A2": 0,
            "C": 2, "Ct": 2, "D": 1,
            "fine_block_inverse": calls * (2 if v else 1),
            "coarse1_block_inverse": calls * (2 if v else 1),
            "coarsest_factor_pair": calls,
            "P1": calls, "P1t": calls, "P12": calls, "P12t": calls,
            "full_internal_solve": 0}


def install_hierarchy(model: nn.Module, correction: HierarchyCorrection, *, cycles: int,
                      step_scale: float | None = None) -> nn.Module:
    if cycles < 1 or correction.A.shape != model.A.shape:
        raise ValueError("positive cycles and matching complete internal dimension required")
    if correction.A.dtype != model.A.dtype or correction.A.device != model.A.device:
        raise ValueError("model and hierarchy dtype/device must agree")
    if step_scale is None:
        step_scale = 1.0 if correction.mode == "vcycle" else 1.0 / 3.0
    if not 0 < float(step_scale) < float("inf"):
        raise ValueError("outer scale must be positive finite")
    model.corrections = nn.ModuleList([correction])
    model.layers, model.share_layers = int(cycles), True
    model.step_scale = model.step_scale.new_full((cycles,), float(step_scale))
    return model


def build_hierarchy_model(*, frozen_core_path: str | Path, frozen_core_sha256: str,
                          A: Tensor, C: Tensor, D: Tensor, Ub: Tensor, Ui: Tensor,
                          P1: Tensor, A1: Tensor, P12: Tensor, A2: Tensor,
                          fine_indices: Tensor, fine_factors: Tensor,
                          level1_indices: Tensor, level1_factors: Tensor,
                          coarse_factor: Tensor, omega0: float, omega1: float,
                          mode: str, cycles: int, step_scale: float | None = None):
    """Build full-trace model; Ub/Ui must already be the exact normalized pair.

    Fine blocks remain physical xyz triplets. Ragged <=12 blocks are latent
    coarse variables, never a reinterpretation of solid node semantics.
    """
    core = load_frozen_core(frozen_core_path, frozen_core_sha256)
    correction = HierarchyCorrection(A, P1, A1, P12, A2,
                                     fine_indices, fine_factors,
                                     level1_indices, level1_factors,
                                     coarse_factor, omega0, omega1, mode=mode)
    model = core.MechanicsNetwork(A, C, D, Ub, Ui, fine_factors, (), layers=1,
                                  step_scale=1.0, learnable=False, share_layers=True)
    return install_hierarchy(model, correction, cycles=cycles, step_scale=step_scale)
