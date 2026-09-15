# The current backend cannot pass the work gate, for any M, at any checkpoint measured

Date: 2026-09-16. Seat 0328.
Script: `superelement/factor_fit/reachability.py`.

> **Scope correction (2026-09-16).** "The low-rank term can only soften" is true **relative to
> `H_0`**, and that is the only way it is used below — both necessary conditions are stated against
> `H_0` and remain valid.  It was separately over-read elsewhere as "the parameterisation's sign is
> backwards, so the model can never stiffen", which is **false**: reducing an existing `M` stiffens
> the current operator (`L=2, M^2=7 -> K0=0.5`; `M^2=3 -> K0=1.0`), and `L` is trainable as well.
> This document rules out the **three specific `L` measured**, not the joint `(L, M)` class.  See
> `docs/CLAIM_SCOPE_20260916.md`.

## The criterion

The backend is `A_hat = W^T (I + M M^T)^-1 W` with `W = L pi(B^T)` (q x d). Since
`(I + M M^T)^-1 = I - M (I + M^T M)^-1 M^T`, in teacher-whitened coordinates

```
H = H_0 - P,    H_0 = R*^-T W^T W R*^-1,    P >= 0,   rank(P) <= k
```

with `k` the number of columns of `M` (1500 here). **The low-rank term can only soften.** Two
necessary conditions follow, both exact:

1. `H <= H_0` gives `lam_min(H) <= lam_min(H_0)`, so passing needs `lam_min(H_0) >= 1 - tau`.
2. If `H_0` has `m` eigenvalues above `1 + tau`, the m-dimensional top eigenspace meets `ker(P)`
   (dimension at least `d - k`) in dimension at least `m - k`, and on that intersection
   `x^T H x = x^T H_0 x > (1+tau)|x|^2`. So passing needs `m <= k`.

`m > k` or `lam_min(H_0) < 1 - tau` therefore **proves** that no `M` whatsoever lets this `L`
pass. The converse would additionally need the softening to be realisable as
`M (I + M^T M)^-1 M^T`, whose eigenvalues are strictly below 1; that step is not checked here,
and is not needed for the discriminating direction.

Cost: one `d x d` symmetric eigendecomposition, 22 s at `d = 12792`.

## Measured

| state | `lam_min(H_0)` | `lam_max(H_0)` | m above 1.1 | m above 1.03 | realised mu range | D/d |
|---|---:|---:|---:|---:|---|---:|
| initial, untrained | 0.0172 | 275.5 | 1809 | 1940 | [0.0172, 275.1] | 1.1194 |
| R7 step 1000 | 0.0695 | 142.1 | 1540 | 2637 | [0.0695, 1.2045] | 0.0350 |
| R8 step 1000, gauge pinned | 0.1301 | 371.5 | 2095 | 3050 | [0.00142, 107.4] | 0.0962 |

`k = 1500` throughout. **Every state fails both conditions at both gates.**

## Reading

**The binding constraint is the soft side, not the rank side.** At R7 step 1000 the work gate
needs `lam_min(H_0) >= 0.9` and it is 0.0695, a factor of 13 short. The rank condition is nearly
satisfied there (1540 against 1500). At the initial state 10629 of 12792 directions of `H_0` sit
below 0.9; at R7 step 1000, 4536 do. The low-rank term cannot help any of them.

**`L` is learning the right thing, slowly.** `lam_min(H_0)` moves 0.0172 → 0.0695 → 0.1301 across
the three states. So training does stiffen the softest direction. Whether it can reach 0.9 is an
open question about `L` alone, and it is now a cheap scalar to track per checkpoint rather than a
6000-step run to discover.

**A concrete mechanism for why the gauge pin hurt.** R8's `L` is *better* than R7's on the soft
side, `lam_min(H_0)` 0.1301 against 0.0695. But R8's realised `mu_min` is 0.00142, ninety times
below its own baseline, while R7's realised `mu_min` equals its baseline exactly. So under the pin
the low-rank term oversoftened the very directions that were already too soft. That is more
specific than "the pin restricts reachability", and it is checkable on any future run.

## What this does and does not establish

It establishes that, at these three checkpoints, this `L` cannot pass either gate for any `M`.
It does not establish that no `L` in this class can, because `L` is still training and its soft
side is improving.

It also gives a design consequence with evidence behind it rather than preference: a backend of
the form `A_hat = T^T D T` with `D` free positive definite carries no one-sided limitation, since
`D` can stiffen as well as soften. The current backend's inability to stiffen is measured, not
assumed.
