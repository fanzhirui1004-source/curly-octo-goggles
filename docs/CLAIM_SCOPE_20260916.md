# What each number is allowed to support

Written after an outside audit caught a recurring failure in how results from this line of work
were being reported: a quantity computed with most variables **held fixed** was repeatedly
described as if it bounded the **whole trainable class**.  Those are opposite directions.  This
file fixes the vocabulary, records the withdrawals, and restates what survives with its scope
attached.

## 0. The four kinds of number, and what each can do

| kind | what it can support | what it cannot |
|---|---|---|
| proved lower bound over an entire class | above the gate ⇒ **the class is excluded** | — |
| conditional optimum: best `D` for a **fixed** `T` (closed form) | excludes every `D` **at that `T`** | says nothing about other `T` |
| best value found after optimising `T, D` | shows a level that **was achieved** | a failure proves nothing about reachability |
| independent full-spectrum audit of one concrete operator | proves **that operator** passes or fails | does not generalise to the class |

Only the first kind can retire an architecture.  We have produced **none** of the first kind.

The reason is one line of algebra.  With

```
    F(T) = min_{D block}  divergence(T^T D T, A) / d ,     F_class = inf_{T in calT} F(T)
```

we always have `F_class <= F(T)`.  So a **high** `F(T)` at some particular `T` is an upper bound
that has come out large — it excludes nothing.  Only a **low** `F(T)` is informative, because it
is achieved.

The audit's two-by-two example makes the gap concrete, and it reproduces exactly:

```
    A = [[1,1],[1,2]]
    T = I,  best diagonal D = diag(1/2, 1)   ->   mu = (0.29289, 1.70711),  D/d = 0.346574 = log(2)/2
    T = [[1,1],[0,1]],  D = I                ->   T^T D T = A exactly,      D/d = 0
```

One trainable lifting coefficient moves the same problem from a bad conditional floor to an exact
fit.  Every "the oracle floor is high, therefore ..." sentence has to survive this example first.

*(A correctness note worth keeping: computing that number by `eigvalsh(A^-1 D)` returns
`(0.5, 1.5)` and `D/d = 0.1438`, because `A^-1 D` is not symmetric and `eigvalsh` silently reads
one triangle.  The whitened form `H = R^-T Ahat R^-1` is symmetric, which is why
`evaluator.whitened_spectrum` is written that way.  A throwaway check of mine hit this; the
shipped path does not.)*

## 1. Withdrawn

**W1 — "sweep block size, layer count and radius; a few seconds a point, and you know how many
parameters the whole structure needs to pass the gate."**
Withdrawn.  Every point on such a sweep is `F(T)` at one explicit `T`, i.e. the second or third
row of the table above.  The curve reports **what one construction achieves at a parameter
count**, and nothing about what the class requires.  Consequently the parameter-economy figure is
titled *conditional optimal divergence under a fixed transform* and never *minimum parameter
requirement*, and any extrapolation off it is a property of that construction's trend only.

**W2 — "the old backend's parameterisation has its sign backwards."**
Withdrawn.  `K0 = L^T (I + M M^T)^-1 L <= L^T L` says the model can only soften **relative to
`L^T L`** — not that it cannot stiffen relative to **where it currently sits**, and `L` is
trainable too.  Scalar counterexample, verified:

```
    L = 2, M^2 = 7  ->  K0 = 0.500          reducing the existing softening
    L = 2, M^2 = 3  ->  K0 = 1.000          stiffens the current operator 2x
```

So "4537 directions need to stiffen" does not imply the sign is wrong.  What survives is W2', below.

**W3 — "divergence barely penalises softening, so the mean loss cannot fix the soft modes."**
Withdrawn, and it is backwards.  With `f(t) = t - log t - 1`:

```
    f(0.0695)  = 1.735929      f'(0.0695)  = -13.3885      (gradient pushes this eigenvalue UP, hard)
    f(1.2045)  = 0.018435      f'(1.2045)  =  +0.1698
```

Divergence punishes that soft mode **94x harder** than the comparable stiff one.  The real case
for an extreme-mode term is **dilution** — one bad mode is averaged over `d = 12792` — not any
asymmetry of `f`.  The paired extreme-term on/off experiment is therefore **not** superseded by
any of this and stays on the list.

**W4 — "the structure's only job is to reduce the output dimension", and "1e7 parameters means
the route is dead, 1e6 means it lives."**
Withdrawn.  Structure also sets whether the coefficients are predictable from geometry at all,
the conditioning, the storage, the apply cost and the assembly cost.  And `1e7` fp32 coefficients
is 40 MB; a shared query-style decoder emits many numbers without carrying that many output
weights, so a parameter count alone cannot retire a route.  What must be compared jointly is
**accuracy + geometric generalisation + per-geometry generation and caching cost + assembly and
solve cost**.

**W5 — "dense apply is ~1 ms against the structured 137 ms, so the structure costs 136x."**
Withdrawn as a measurement.  The 1 ms was a bandwidth estimate; 137 ms was an implementation
timing.  Not like-for-like.  Needs same device, same precision, synchronised timing before any
ratio is quoted.

**W6 — "the 0.01% thickness / 59.76% stiffness sensitivity is fixed."**
Narrowed.  The record supports: one controlled stabilisation-energy rewrite took **that one
adjacent pair's** max stiffness ratio from ~1.598 to ~1.016.  It does not establish that the
production labels all carry that change, nor that every nearby geometry behaves that way, nor
anything about behaviour across active-set changes.

**W7 — "the 12-seat spectral-ratio probe measures the local Lipschitz constant of the map."**
Narrowed.  Twelve geometries that are not close to each other cannot measure a *local* constant.
Large geometry separation producing large operator separation does not make the map unlearnable;
it raises the data and model requirement.  Keep it as a controlled sensitivity diagnostic; it is
not a reason to pause network experiments.

**W8 — "the failing directions are the ones worst for assembly."**
Narrowed.  Those are generalised **error** eigendirections between prediction and teacher, not
the teacher's own stiffness modes, and the coefficient norm used still has to be reconciled with
trace semantics.  Real assembly impact depends on how global displacement and load project onto
them.  What the table does support: the error is **not** confined to the compliant tail, which is
enough to retire "mask the soft tail and the problem goes away" — and masking non-rigid
directions would change the task anyway, not satisfy it.

## 2. What survives, with scope attached

**S1 — pure block-diagonal is excluded, at `T = I`, for the blockings tested.**
The closed form `D*_b = ([T A^-1 T^T]_bb)^-1` is the exact conditional optimum, so at `T = I` the
measured 63x-204x margin over the +-10% necessary condition excludes **every** `D` for those
coordinates and those partitions on the seats tested.  Scope: `T = I` only.  It says nothing
about a trainable `T`, per the two-by-two example above.

**W2' — some specific `L` are unreachable, and that is a real negative.**
At the three checkpoints measured, `H0 = R*^-T B L^T L B^T R*^-1` has `lam_min(H0) = 0.13 < 0.9`.
Since `K0 <= L^T L` for every `M`, **no `M` whatsoever** reaches +-10% at those `L`.  Scope:
those three `L`.  Changing `L` changes `H0`, so the joint `(L, M)` class is **not** closed out —
and scaling `L` up is not a free repair either, since raising the floor can push the count of
over-stiff directions past the low-rank correction's capacity.  Both sides have to be checked
together.

**S2 — the E0 / E0-B structural checks pass** (determinant-1 lifting, PD-by-construction `D`,
exact `logdet`, the closed-form `D` elimination improving the fit 99x).  These are implementation
and optimisation results.  They are not a capacity proof for the real operator class.

**S3 — the best scalar rescale is real but small.**  For a fixed operator with extremes
`a = 0.0695, b = 1.2045`, `s* = 2/(a+b) = 1.5699` and `eps_op` improves `0.9305 -> 0.8909`.  It
is never worse than leaving it alone, and it is nowhere near a repair.

**S4 — the lifting inverse's reach, stated precisely.**  `LiftingLayer` rejects overlapping row
and column sets (`LIFTING_PARTITION_OVERLAP`), so within a layer `K^2 = 0` **exactly** and
`(I+K)^-1 = I - K` on the same support — the audit's caveat is satisfied by construction here,
but only per layer.  For the product `T = T_L...T_1`, `T^-1 = (I-K_1)...(I-K_L)`, whose reach is
`sum_l r_l` — **not** `L*r`, since the layer radii differ by design in a multiscale schedule.
Then `Ahat^-1 = T^-1 D^-1 T^-T` has reach `<= 2 sum_l r_l + max block diameter`, since `D^-1` is
block diagonal.  **But** the delivered object is `Shat^+ = B^T Ahat^-1 B`, and `B` carries six
Householder reflectors, so quotient-space bandedness gives *banded plus a dense rank-<=12
correction* in physical coordinates, not locality.  The correction is bounded and the rigid
directions are genuinely global, so this is a quantifiable caveat rather than a refutation — and
it has not been measured.  **Status: an open structural question, not a decided constraint.**

## 3. Current status of each object

| object | most defensible statement today |
|---|---|
| pure block diagonal | excluded at `T = I` for the tested coordinates and partitions |
| trainable multiscale `(T, D)` | **not** shown to pass, and **not** excluded by the block-diagonal result |
| old `(L, M)` | three specific `L` are unreachable; the joint class and the extreme-supervision question are both open |
| E0 / E0-B | implementation and optimisation progress; not a capacity proof |
| geometry-conditioned learning | no new shared-network generalisation result in this round |
| parameter-vs-error sweeps | legitimate, provided conditional optimum / achieved fit / class lower bound are never merged |

## 4. The three converging actions

1. **Every oracle number ships with its scope**: which variables were fixed, which were minimised,
   closed form or numerical, and whether any all-`T` bound exists.  Absent an all-`T` proof, the
   figure is captioned *conditional optimal divergence under a fixed / fitted transform*.
2. **Pick one concrete multiscale configuration and actually test whether `T` can be learned.**
   The block-diagonal result already says "tuning `D` alone is not enough", so the next change is
   to let a legal `T` move, keeping the closed-form `D*(T)` as a diagnostic and optimisation aid.
   Final acceptance stays with an independent full-spectrum audit of the resulting operator: the
   divergence-optimal `D` is not the worst-direction-optimal `D`.
3. **Keep the paired extreme-supervision experiment on the old backend.**  It is still unanswered,
   the fixed-`L` diagnostic explains rather than replaces it, and the new architecture does not
   have to wait on the old one being closed out.
