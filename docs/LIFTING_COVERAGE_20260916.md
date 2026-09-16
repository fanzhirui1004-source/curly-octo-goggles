# The multiscale lifting schedule was local to a plane, not local everywhere

Date 2026-09-16.  Seat 0328, `d = 12792`.
Scripts `superelement/factor_fit/slab.py` (coverage), `prof.py` (timing), `e1c.py`, `e1d.py`.

## What was found

`backend.multiscale_layers` documents itself as bipartitioning "by a coordinate-parity rule that
changes with level".  The implementation instead splits on a **median plane** of one axis and
couples pairs across it within radius `r`.  For a small `r` only a slab of thickness `~2r` around
that plane contains any pair at all, so an early layer is local *to a plane*, not local
everywhere.  Measured, rather than read off the code:

| level | radius | pairs | coordinates appearing in any pair |
|---:|---:|---:|---|
| 0 | 0.035 | 5,418 | 744 (5.8%) |
| 1 | 0.070 | 28,494 | 1,434 (11.2%) |
| 2 | 0.140 | 140,112 | 3,066 (24.0%) |
| 3 | 0.280 | 220,000 | 7,106 (55.6%) |

Union over all four levels: 10,514 of 12,792 (82.2%).  **2,278 coordinates (17.8%) appear in no
pair at any level**, so their rows of `T` are pinned to the identity exactly, whatever the
coefficients do.  394,024 lifting coefficients, concentrated in slabs around four planes.

`backend.checkerboard_layers` colours by the parity of a coarse cell index of side `r`.  Every
coordinate then has opposite-coloured neighbours within `r`:

| level | radius | pairs | coordinates appearing in any pair |
|---:|---:|---:|---|
| 0 | 0.035 | 185,130 | 12,792 (100.0%) |
| 1 | 0.070 | 220,000 | 12,792 (100.0%) |
| 2 | 0.140 | 220,000 | 12,792 (100.0%) |
| 3 | 0.280 | 220,000 | 12,792 (100.0%) |

Rows come from one colour and columns from the other, so the row and column sets stay disjoint and
each layer keeps determinant exactly 1.

This does **not** say the old runs are invalid — they measured what they measured.  It says the
capacity they measured was the capacity of a schedule touching 82% of the domain, which is not the
question anyone meant to ask.  `e1d.py` re-runs both rules at a matched coefficient count under an
identical optimiser, init and step budget so the rule is the only difference.

## Two speed fixes that made the experiment affordable

**The lifting gradient.**  `torch.sparse.mm` is fine forwards, but its gradient with respect to
the sparse operand's values materialises a dense `d x d` product.  Timed at `d = 12792`,
`nnz = 220k`, float64:

```
    sparse.mm  forward only          0.078 s
    sparse.mm  forward + backward    5.419 s      -- a 70x backward
```

The required work is `nnz * d`, so the whole gap is an implementation artefact.
`backend._LiftApply` writes the rule out by hand:

```
    dL/dk_i = sum_j gy[rows[i], j] * x[cols[i], j]
    dL/dx   = gy + K^T gy
```

with column chunking to bound the `(nnz, chunk)` temporaries.  The backward **recomputes** those
temporaries rather than storing them — a plain chunked loop under autograd keeps every chunk's
graph alive and held 29 GB at this scale before running out of memory.

**The block determinants.**  `floor_objective` indexed `G` once per block; each `G[idx]` allocates
a full `zeros_like(G)` for its gradient, so ~200 blocks at `d = 12792` push ~260 GB of allocation
traffic per step.  `fit.block_groups` gathers blocks of equal size together, turning 200 gathers
into one per distinct size, with a batched `bmm` and a batched `slogdet`.

Together: **11.6 s -> 0.53 s per descent step**, a factor of 22.

`floor_objective` now also accepts `logdet_G` as a constant, since `T` is a product of
unit-triangular lifting layers so `det T = 1` exactly and `logdet(T R*^-1) = -logdet R*` whatever
the coefficients are.  Verified on the real operator to `diff 0.00e+00`.

## Tests

`superelement/neural_schur/test_lift.py`, 7 checks, all passing on the GPU box:

- forward and transpose against the explicit dense matrix
- gradients against autograd through the dense matrix, both orientations
- `torch.autograd.gradcheck` on the custom Function
- chunk size invisible to value and gradient, including a chunk that does not divide the width
- the multi-layer product, `det T = 1`, and `apply_T` / `apply_T_transpose` against the dense `T`
- the grouped floor against the per-block floor, in value and gradient
- passing `logdet_G` as a constant changes nothing

A hand-written backward is exactly the kind of thing that is silently wrong, which is why the
reference in every one of these is the dense path rather than a previous version of itself.

## A methodological note on the learning-rate probe in `e1c.py`

The probe ran each candidate under a cosine schedule of the probe's own length, then handed the
winner to a much longer run with a cosine of *that* length.  Those are different schedules: `3e-2`
won a 120-step probe at `2.843e-01`, and then, over 2500 steps, sat at `3.215e-01` at step 300 --
worse than the probe's own result, because the annealing keeps the rate high far longer.  Any
learning rate picked this way is picked for the wrong schedule.  `e1d.py` drops the probe and
fixes one identical schedule across both arms instead.
