# Multi-level elimination tree: measured cost profile

Date: 2026-09-15. Measurement only — no learning, no factorization run.
Script: `scratchpad/tree_profile.py`, executed against five no-GP bodies from
`/autodl-fs/data/COMPOSABILITY_R1`. Leaf size 8 cells, bisection on the
longest axis of the cell-centroid bounding box.

## Purpose

The proposed architecture replaces the single monolithic Schur complement
with a nested-dissection tree: eliminate leaf interiors first, then
separators level by level, so that the learned operator is a composition of
small local eliminations rather than one global one. Before designing that
operator, the question is whether the tree is actually cheaper, and where
the remaining cost sits.

This profile answers the cost question on real cut-cell sparsity. It does
not touch the learning question.

## Method

For each body the incidence matrix (cells x coordinates) is read from the
frozen composability bodies. A cell set is bisected recursively on the axis
of largest centroid spread. At each node:

- `priv` = coordinates touched only by cells inside this node (the node's
  private interior, including its own separator)
- `child_priv` = union of the two children's private sets
- `sep` = `priv` minus `child_priv`, the separator eliminated at this node
- `bnd` = coordinates touched by this node but not private to it, the
  front's boundary

Flops use the dense frontal model `e^3/3 + e^2*b + e*b^2` with `e = |sep|`,
`b = |bnd|`. Front memory is `(e+b)^2 * 8` bytes. The monolithic reference
eliminates all interior coordinates against the full external trace in one
front, same model.

## Results

| body | cells | coords | trace | trace% | tree flops | monolithic | ratio | peak front |
|---|---|---|---|---|---|---|---|---|
| input_0693 | 14 | 693 | 552 | 79.7% | 1.774e7 | 5.487e7 | 3.1x | 0.002 GiB |
| input_0448 | 19 | 891 | 711 | 79.8% | 1.917e7 | 1.160e8 | 6.0x | 0.004 GiB |
| input_0269 | 68 | 2757 | 2088 | 75.7% | 4.058e8 | 3.951e9 | 9.7x | 0.034 GiB |
| input_0010 | 250 | 8991 | 5433 | 60.4% | 1.477e10 | 1.888e11 | 12.8x | 0.241 GiB |
| input_0245 | 391 | 13287 | 7080 | 53.3% | 3.489e10 | 6.636e11 | 19.0x | 0.420 GiB |

Per-level flop share:

```
input_0693  d0 15.7%  d1 84.3%
input_0448  d0 32.2%  d1 20.3%  d2 47.5%
input_0269  d0 36.0%  d1 36.4%  d2 11.5%  d3 13.7%  d4  2.4%
input_0010  d0 52.7%  d1 26.0%  d2 13.4%  d3  3.9%  d4  1.4%  d5  2.5%
input_0245  d0 65.4%  d1 13.4%  d2 13.7%  d3  3.9%  d4  1.9%  d5  0.6%  d6  1.1%
```

Root separator sizes are small relative to the trace they face:
`e = 9, 12, 33, 252, 429` against `b = 552, 711, 2088, 5433, 7080`.

## Reading

**The tree is cheaper, and increasingly so with size.** 3.1x at 14 cells
rising to 19.0x at 391 cells. The saving comes from never forming a front
that couples all interior coordinates at once.

**Cost concentrates at the root.** Root share climbs 15.7% -> 32.2% ->
36.0% -> 52.7% -> 65.4%; leaf share collapses 84.3% -> 1.1%. By 391 cells
the top two levels carry 79% of the work and the bottom three carry 3.6%.
A tree operator that spends its parameters uniformly per level would be
spending them where the flops are not.

**The root front is the memory-binding object for a dense delivery.** It is
`(e+b)^2` with `b` the full external trace. That bound applies to writing
out every entry of `S`; it does not apply to an implicit operator that
answers `u ↦ Su` while keeping all `q` interface degrees of freedom. Any
single column is still available as `S e_j`. Keeping the whole interface
and declining to expand the matrix is not port reduction. An implicit tree
is not free either — internal factors, tree construction and each
application have to be timed — but the cost of dense delivery cannot by
itself rule the architecture out.

## Caveat that limits these numbers

The profiled bodies are trace-dominated: 53%-80% of their coordinates are
external trace. Production bodies are roughly 13% (about 12800 trace
coordinates out of about 100000). Since root flops are `e^3/3 + e^2*b +
e*b^2` with `b` equal to the whole trace, a body whose trace is most of its
coordinates puts an unusually large share at the root.

**So the 52.7% and 65.4% root shares are upper bounds, not production
estimates.** At production trace fraction the interior is far larger
relative to the trace, so lower levels carry proportionally more and the
root carries less. The direction is certain; the magnitude is not measured.

Two things are not sensitive to this caveat:

1. The tree-vs-monolithic ratio grows with interior size, so at production
   trace fraction it can only be larger than 19x, not smaller.
2. The root front at production scale is computable exactly, because it
   depends only on `e + q`. With `q = 12800` and a root separator of a few
   hundred to about 1500, the front is 1.3 to 1.6 GiB. The delivered
   Schur complement is 1.22 GiB dense, 0.61 GiB packed upper. The root
   front and the deliverable are the same order.

## What this means for the architecture question

Absolute elimination cost at these scales is not the obstacle. The largest
body profiled needs 3.49e10 flops for the whole tree, which is well under a
second of dense GPU throughput. Even the monolithic 6.64e11 is seconds.

If the deliverable is written out entry by entry, `O(q^2)` is unavoidable
and no tree removes it. If the deliverable is an operator, it is not.

The measurement that matters for the leaf-versus-root question is not here.
On the largest body profiled the leaves carry 1.1% of the work, so a
network replacing leaf construction entirely would save 1.1%. But these
bodies are trace-dominated and small, so that number does not transfer to
production, and it is equally not a proven bound in the other direction.
**A layered cost profile on a real n32 GP teacher is what would decide
which level a network should replace, and it has not been run.** These
no-GP bodies give a hint and nothing more.

## Note on GP labels

The GP label packets contain `S_UPPER.npy`, `TRACE.npz` and `PROBES.npz`
only — no body or cell factors. Profiling a GP-teacher tree would require
recompiling the body from the frozen source. The question profiled here is
geometric, so the no-GP bodies answer it. GP does add face-coupling edges
between neighbouring cut cells, which increases fill, so a GP tree would be
somewhat more expensive than these numbers at the same cell count.
