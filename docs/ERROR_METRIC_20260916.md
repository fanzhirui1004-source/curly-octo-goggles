# The gate is a relative error on the factor, not an accuracy on `S`

Date 2026-09-16.  Seat 0328 of the production GP teacher, `d = 12792`.
Script `superelement/factor_fit/amp.py`; raw `docs/data/AMPLIFICATION.json`.

This is an exact statement about two error metrics on one measured `A`.  It says nothing about
whether any network can hit either budget — see `docs/CLAIM_SCOPE_20260916.md`.

## Measured

```
    lam_min(A) = 4.8777e-07     lam_max(A) = 3.5244e-02     cond(A) = 7.2254e+04
```

Acceptance is `eps_op = ||R*^-T Ahat R*^-1 - I||_2 <= tau`.  Write a candidate two ways.

**Additively**, `Ahat = A + E`.  Then `eps_op = ||A^-1/2 E A^-1/2||` and in the worst alignment
`eps_op = ||E||_2 / lam_min(A)`, so passing needs

| gate | `||E||_2` allowed | relative to `lam_max(A)` |
|---|---:|---:|
| +-3%  | 1.463e-08 | 4.15e-07 |
| +-10% | 4.878e-08 | 1.38e-06 |

**Multiplicatively**, `Rhat = (I + Delta) R*`.  Then `H = (I+Delta)^T (I+Delta)` exactly, with no
reference to `A` at all, so there is no conditioning factor.  Measured with random `Delta` at fixed
spectral norm:

| `||Delta||_2` | `eps_op` | bound `2d + d^2` |
|---:|---:|---:|
| 1e-3 | 1.4135e-03 | 2.0010e-03 |
| 1e-2 | 1.4187e-02 | 2.0100e-02 |
| 3e-2 | 4.2987e-02 | 6.0900e-02 |

So `eps_op ~ 1.42 ||Delta||`, and +-3% needs `||Delta|| ~ 2.1e-2`.

## The consequence

```
    to pass +-3% :   2.1e-2  relative error on the FACTOR
                     4.2e-7  relative error on the ENTRIES of A
                     ------  a factor of 5.1e4 in the error budget
```

Every compression whose tolerance is entry-relative — SVD block truncation, banding, thresholding,
and any loss of the form `||Ahat - A||` — is spending its accuracy in the metric that costs
`cond(A)` more.  Any candidate built or trained multiplicatively on a factor is spending it in the
metric that costs nothing.

This is a reason to prefer the factored form `Ahat = T^T D T`, and equally a warning that fitting
it by matching entries of `A` throws the advantage away.  It also reframes "can a network predict
an operator this large": the target accuracy on a *factor* is 2%, not 4e-7.

## What this does **not** say

It does not say a factored parameterisation has the capacity to reach `||Delta|| = 2e-2` on this
teacher, and it does not say a network can learn one from geometry.  Both are open.  It also does
not license reading a conditioning argument into any *specific* failed run: the first hierarchical
attempt reported `eps_op = 779` at blockwise tolerance `1e-3`, where this amplification predicts
only `72`.  The 10.8x gap was **not** conditioning; it was an assembly bug in that script (block
index sets are scattered, so filling only one orientation and then calling `triu` zeroed roughly
half of every off-diagonal block).  That run is void and is being redone with a `tol = 0`
self-test row that must return `eps_op ~ 1e-10` before any other row is believed.
