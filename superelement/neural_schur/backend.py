"""Minimal operator backend: A_hat = T^T D T on the rigid quotient.

T acts on the d-dimensional quotient, not the q-dimensional trace space.  That choice is what
makes logdet(A_hat) = logdet(D) exact, with no solve against any auxiliary matrix anywhere; see
docs/ROADMAP_NEURAL_SCHUR_20260916.md section 3.  Locality still means physical locality because
quotient coordinate j is physical dof order[j+6] (measured: B's row j has its largest entry
exactly there for 100% of rows).

T is a product of unit-triangular lifting layers.  A layer picks a bipartition (A, B) of the
coordinates and updates

    x[A] <- x[A] + K x[B],      x[B] unchanged

with K supported on a chosen set of pairs.  Every such layer has determinant exactly 1 and is
invertible for ANY K, so invertibility is structural rather than a constraint to be enforced or
monitored.  Multiscale range comes from the layer schedule: early layers couple near neighbours,
later ones couple across the domain.

D is block diagonal with each block stored as its Cholesky factor, so positive definiteness is
also structural and logdet D is a sum of small block determinants.

Nothing here is claimed to be expressive enough for the real target.  It is the smallest thing
with the right guarantees, built so that E0 can check the guarantees hold numerically.
"""
from __future__ import annotations

import numpy as np
import torch


class LiftingLayer:
    """x[rows] += K x[cols], with K supported on `pairs`.  Determinant 1, always invertible."""

    def __init__(self, d: int, pairs: np.ndarray, device=None, dtype=torch.float64):
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if pairs.size and (pairs[:, 0] == pairs[:, 1]).any():
            raise ValueError('LIFTING_DIAGONAL_PAIR')          # a diagonal entry would change det
        rows, cols = pairs[:, 0], pairs[:, 1]
        if pairs.size and len(set(rows.tolist()) & set(cols.tolist())):
            raise ValueError('LIFTING_PARTITION_OVERLAP')      # updated and source sets must be disjoint
        self.d = int(d)
        self.pairs = torch.as_tensor(pairs, device=device)
        self.n_coeff = len(pairs)
        self.device, self.dtype = device, dtype

    def _sparse(self, k: torch.Tensor, transpose: bool) -> torch.Tensor:
        """(I + K) or its transpose as a sparse matrix.

        index_add on (n_pairs, n_cols) intermediates is fine for a few right-hand sides and
        impossible at d = 12792 with 400k pairs: the intermediate alone would be 41 GB in
        float64.  A sparse product never forms it.
        """
        dev, dt = k.device, k.dtype
        diag = torch.arange(self.d, device=dev)
        rows, cols = self.pairs[:, 0].to(dev), self.pairs[:, 1].to(dev)
        if transpose:
            rows, cols = cols, rows
        idx = torch.stack([torch.cat([diag, rows]), torch.cat([diag, cols])])
        val = torch.cat([torch.ones(self.d, device=dev, dtype=dt), k])
        return torch.sparse_coo_tensor(idx, val, (self.d, self.d)).coalesce()

    def apply(self, k: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """T_layer x."""
        if self.n_coeff == 0:
            return x
        return torch.sparse.mm(self._sparse(k, False), x)

    def apply_transpose(self, k: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """T_layer^T x.  Exactly the transpose of `apply`, not a second learned map."""
        if self.n_coeff == 0:
            return x
        return torch.sparse.mm(self._sparse(k, True), x)

    def dense(self, k: torch.Tensor) -> torch.Tensor:
        """Explicit matrix, for tests only."""
        M = torch.eye(self.d, dtype=k.dtype, device=k.device)
        if self.n_coeff:
            M[self.pairs[:, 0], self.pairs[:, 1]] += k
        return M


class QuotientOperator:
    """A_hat = T^T D T, T a product of lifting layers, D block diagonal."""

    def __init__(self, d: int, layers: list[LiftingLayer], blocks: list[np.ndarray]):
        self.d = int(d)
        self.layers = layers
        self.blocks = [torch.as_tensor(np.asarray(b, dtype=np.int64)) for b in blocks]
        seen = np.concatenate([np.asarray(b) for b in blocks]) if blocks else np.empty(0, np.int64)
        if len(seen) != d or len(np.unique(seen)) != d:
            raise ValueError('BLOCKS_MUST_PARTITION_ALL_COORDINATES')
        self.layer_sizes = [l.n_coeff for l in layers]
        self.block_sizes = [len(b) for b in self.blocks]
        self.n_layer_coeff = int(sum(self.layer_sizes))
        # each block stores an upper-triangular Cholesky factor: n(n+1)/2 entries
        self.n_block_coeff = int(sum(n * (n + 1) // 2 for n in self.block_sizes))

    # ---------------------------------------------------------------- coefficients
    def split(self, coeff: torch.Tensor):
        ks = torch.split(coeff[:self.n_layer_coeff], self.layer_sizes) if self.layer_sizes else []
        rest = coeff[self.n_layer_coeff:]
        sizes = [n * (n + 1) // 2 for n in self.block_sizes]
        cs = torch.split(rest, sizes)
        return list(ks), list(cs)

    def identity_coefficients(self, device=None, dtype=torch.float64) -> torch.Tensor:
        """T = I and D = I.  New layers initialised here leave any previous solution intact."""
        c = torch.zeros(self.n_layer_coeff + self.n_block_coeff, dtype=dtype, device=device)
        off = self.n_layer_coeff
        for n in self.block_sizes:
            tri = torch.zeros(n, n, dtype=dtype, device=device)
            tri[range(n), range(n)] = 1.0
            iu = torch.triu_indices(n, n)
            c[off:off + n * (n + 1) // 2] = tri[iu[0], iu[1]]
            off += n * (n + 1) // 2
        return c

    def _chol_blocks(self, cs):
        """Upper-triangular factors with positive diagonal, so D_b = C_b^T C_b is PD by construction."""
        out = []
        for n, c in zip(self.block_sizes, cs):
            tri = torch.zeros(n, n, dtype=c.dtype, device=c.device)
            iu = torch.triu_indices(n, n, device=c.device)
            tri = tri.index_put((iu[0], iu[1]), c)
            diag = torch.nn.functional.softplus(tri.diagonal()) + 1e-12
            tri = tri - torch.diag(tri.diagonal()) + torch.diag(diag)
            out.append(tri)
        return out

    # ---------------------------------------------------------------- actions
    def apply_T(self, coeff, z):
        ks, _ = self.split(coeff)
        for layer, k in zip(self.layers, ks):
            z = layer.apply(k, z)
        return z

    def apply_T_transpose(self, coeff, z):
        ks, _ = self.split(coeff)
        for layer, k in zip(reversed(self.layers), reversed(ks)):
            z = layer.apply_transpose(k, z)
        return z

    def apply_D(self, coeff, z):
        _, cs = self.split(coeff)
        out = torch.zeros_like(z)
        for idx, tri in zip(self.blocks, self._chol_blocks(cs)):
            idx = idx.to(z.device)
            sub = z[idx]
            out[idx] = tri.T @ (tri @ sub)
        return out

    def apply_A(self, coeff, z):
        """A_hat z = T^T D T z."""
        return self.apply_T_transpose(coeff, self.apply_D(coeff, self.apply_T(coeff, z)))

    def apply_S(self, coeff, u, quotient):
        """S_hat u = B^T A_hat B u on the full trace space."""
        return quotient.lift(self.apply_A(coeff, quotient(u)))

    def energy(self, coeff, u, quotient):
        z = quotient(u)
        return 0.5 * (z * self.apply_A(coeff, z)).sum(0)

    def logdet_A(self, coeff):
        """logdet A_hat = logdet D, because every lifting layer has determinant 1."""
        _, cs = self.split(coeff)
        return 2.0 * sum(tri.diagonal().log().sum() for tri in self._chol_blocks(cs))

    # ---------------------------------------------------------------- tests only
    def dense_T(self, coeff):
        ks, _ = self.split(coeff)
        M = torch.eye(self.d, dtype=coeff.dtype, device=coeff.device)
        for layer, k in zip(self.layers, ks):
            M = layer.dense(k) @ M
        return M

    def dense_D(self, coeff):
        _, cs = self.split(coeff)
        M = torch.zeros(self.d, self.d, dtype=coeff.dtype, device=coeff.device)
        for idx, tri in zip(self.blocks, self._chol_blocks(cs)):
            idx = idx.to(coeff.device)
            M[idx.unsqueeze(1), idx.unsqueeze(0)] = tri.T @ tri
        return M

    def dense_A(self, coeff):
        T = self.dense_T(coeff)
        return T.T @ self.dense_D(coeff) @ T


def multiscale_layers(points: np.ndarray, levels: int, radius0: float, growth: float = 2.0,
                      max_pairs_per_level: int | None = None, device=None, seed: int = 0):
    """A lifting schedule whose coupling range grows with level.

    points: (d, k) coordinates of each quotient coordinate, used only to define neighbourhood.
    Each level bipartitions by a coordinate-parity rule that changes with level, then couples
    pairs within the level's radius.  Early levels are local, later ones reach across the domain.
    """
    d = len(points)
    rng = np.random.default_rng(seed)
    layers, meta = [], []
    for lv in range(levels):
        r = radius0 * (growth ** lv)
        axis = lv % points.shape[1]
        med = np.median(points[:, axis])
        side = points[:, axis] > med if lv % 2 == 0 else points[:, axis] <= med
        A = np.flatnonzero(side); Bs = np.flatnonzero(~side)
        if len(A) == 0 or len(Bs) == 0:
            continue
        from scipy.spatial import cKDTree
        tree = cKDTree(points[Bs])
        nbr = tree.query_ball_point(points[A], r)
        pairs = [(int(A[i]), int(Bs[j])) for i, js in enumerate(nbr) for j in js]
        if max_pairs_per_level and len(pairs) > max_pairs_per_level:
            keep = rng.choice(len(pairs), max_pairs_per_level, replace=False)
            pairs = [pairs[i] for i in keep]
        layers.append(LiftingLayer(d, np.asarray(pairs, dtype=np.int64).reshape(-1, 2), device=device))
        meta.append(dict(level=lv, radius=float(r), axis=int(axis),
                         updated=int(len(A)), pairs=int(len(pairs))))
    return layers, meta


def spatial_blocks(points: np.ndarray, block_size: int):
    """Partition coordinates into spatially coherent blocks by recursive bisection."""
    d = len(points)
    out = []

    def rec(idx):
        if len(idx) <= block_size:
            out.append(np.sort(idx)); return
        axis = int(np.argmax(points[idx].max(0) - points[idx].min(0)))
        order = idx[np.argsort(points[idx, axis], kind='stable')]
        h = len(order) // 2
        rec(order[:h]); rec(order[h:])

    rec(np.arange(d, dtype=np.int64))
    return out
