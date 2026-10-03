# Hierarchical low rank on the real teacher: a much better spend, still short of the gate

Date 2026-09-16.  Seat 0328, `d = 12792`, dense `S` = 8.182e7 independent entries.
Script `superelement/factor_fit/hmat.py`; raw `docs/data/HMAT_ECONOMY3.json`.

Each row is an **achieved** operator carrying a real full-spectrum audit, so a pass would be a
pass and a failure rejects **this construction at that tolerance** — not the hierarchical class.
See `docs/CLAIM_SCOPE_20260916.md`.

## Construction

Bisection cluster tree on the quotient coordinates (511 nodes, leaf 64).  Standard admissibility
`min(diam(s), diam(t)) <= 2.0 * dist(s, t)` gives **2965 admissible + 1344 near-field** blocks.
Near-field blocks are kept dense and exact; admissible blocks are replaced by a truncated SVD at a
blockwise relative-Frobenius tolerance.  No fitting of any kind — this is direct approximation.

**0 of 2965 admissible blocks hit the rank cap of 320.**  The far field of this operator really is
numerically low rank, which is the structural claim the construction was built to test.

### Self-test, which the first two versions of this script would have failed

```
EXACT fill (every block taken straight from A):
    0 entries uncovered,  0 entries written twice,  eps_op = 1.807e-13   -> OK
```

This asserts partition coverage and block orientation before any row is read.  It exists because
v1 of the script filled one orientation per block and then called `triu`; the block index sets
come from a spatial bisection and are scattered, not contiguous, so `triu` silently zeroed about
half of every off-diagonal block.  That produced `eps_op ~ 778` flat across four decades of
tolerance, which is what gave it away.  v2 fixed the fill but double-counted parameters, because
the recursion reaches each off-diagonal node pair from both orderings; deduplicating halved the
block counts (5930 + 2432 -> 2965 + 1344).  Both runs are void.

## Measured

| tol | params | % of dense | `eps_op` | `D/d` | verdict |
|---:|---:|---:|---:|---:|---|
| 3e-1 | 4,851,102 | 5.93% | 6.914 | inf | 4 non-positive `mu` |
| 1e-1 | 5,723,470 | 6.99% | 3.415 | inf | 6 non-positive `mu` |
| 3e-2 | 7,021,512 | 8.58% | 1.555 | inf | 3 non-positive `mu` |
| 1e-2 | 8,257,822 | 10.09% | 1.129 | inf | 3 non-positive `mu` |
| 3e-3 | 9,748,477 | 11.91% | 6.987e-01 | 1.2796e-04 | fails the gate |
| 1e-3 | 11,282,414 | 13.79% | 2.289e-01 | 1.2738e-05 | fails the gate |

`eps_op` now falls monotonically with tolerance, 6.91 -> 0.23, as a truncation error should.
Nothing reaches +-10% (`eps_op <= 0.10`); the best row misses it by 2.3x at 13.8% of dense.

## The finding that matters most here

At `tol = 1e-3` the divergence is `D/d = 1.2738e-05`, which clears the **+-3% necessary
condition** (`4.5921e-04`) by a factor of **36** — while the actual gate `eps_op = 0.2289` misses
**+-10%** by a factor of 2.3.

That is the necessary-vs-sufficient gap, measured on the real teacher rather than argued.  A mean
over `d = 12792` modes is tiny when a handful are badly wrong, and `eps_op` is exactly the
statistic that refuses to average them away.  Two consequences:

- Any run reporting only `D/d` against a necessary line can be off by orders of magnitude on the
  real acceptance criterion.  Every result in this line of work that quotes a floor has to be read
  that way, including the banded sweep.
- It is direct support for the extreme-mode supervision question staying open: the failure is
  concentrated, so a statistic that averages cannot see it.  (How concentrated is being measured
  now at tighter tolerances.)

## Against the banded spend, with the scope attached

The banded sweep's best point was `floor(T) = 2.952e-02` at 6.03% of dense, where `floor` is
already minimised over all block-diagonal `D`.  The hierarchical construction at 11.91% of dense
reaches `D/d = 1.280e-04` with **no fitting at all** — 230x better in the same statistic, at 2x
the parameters.

These are two different ways to spend parameters on the same `A`, not two members of one
parameterised family, and one is optimised over `D` while the other is not.  What the comparison
supports: the hierarchical prior is a far better structural match to this operator than a band,
which is what the Dirichlet-to-Neumann argument predicts.  What it does not support: any statement
about which family a *trained* model should use, or about a class bound for either.
