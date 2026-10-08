# How accurate must a direct per-entry predictor be?

Date 2026-09-16.  Seat 0328, `d = 12792`.
Scripts `superelement/factor_fit/perentry.py`, `coher.py`;
raw `docs/data/PER_ENTRY.json`, `docs/data/COHERENCE.json`.

## The question

The most intuitive architecture emits `S` (or its factor) **entry by entry** — a shared decoder
queried once per `(i, j)`, which costs no more weights than the decoder itself and, at ~8e7 small
forward passes, runs in well under a second against the 31 s it replaces.  So neither network size
nor speed decides it.  What decides it is accuracy, and **per-entry relative accuracy is a
different norm from the acceptance criterion**: independent errors add incoherently, so the
conversion carries a factor that has to be measured.

`ERROR_METRIC_20260916` gave the **worst-case** conversion: `eps_op <= ||E||_2 / lam_min(A)` puts
+-3% at `4.15e-07` relative to `lam_max`.  That bound assumes `E` aligns with the softest mode.
Predictor error does not; it is spread over all directions.  So the bound is correct and, for this
question, very pessimistic.  Measure instead.

## Independent per-entry error

`Shat = A + E` with `E_ij = eps |A_ij| n_ij` (symmetrised), and `Rhat = R* + E` with
`E_ij = eps |R*_ij| n_ij`, `n` standard normal.  `eps_op` on the real whitened spectrum:

| per-entry `eps` | `eps_op` predicting `A` | `eps_op` predicting the factor `R*` |
|---:|---:|---:|
| 1e-7 | 7.84e-05 | 1.11e-05 |
| 1e-6 | 7.57e-04 | 1.13e-04 |
| 1e-5 | 6.87e-03 | 1.12e-03 |
| 1e-4 | 5.43e-02 | **1.11e-02**  (passes +-3%) |
| 1e-3 | 6.34e-01 | 1.20e-01 |
| 1e-2 | non-PD | 1.83e+00 |

Both arms are almost exactly linear in `eps`, so reading off +-3%:

```
    predicting A entry-wise          about 5.5e-05 per entry     (~4.3 significant digits)
    predicting the factor R*         about 2.7e-04 per entry     (~3.6 significant digits)
```

**Not `4e-07`.**  The worst-case bound overstates the requirement for spread-out error by roughly
three orders of magnitude, and the factor arm is a further 5x more forgiving than the `A` arm — the
same direction `ERROR_METRIC` predicted, at a much smaller magnitude than the `cond` ratio suggests.

## Correlated error, which is how a network actually fails

A network's error field is smooth in the geometry: neighbouring entries are wrong in the same
direction, and coherent errors add linearly rather than as a square root.  Model the coherence
length directly — partition the coordinates into spatial blocks of side `B` and give every entry in
one block pair the same relative offset, drawn once.  `B = 1` is the independent case.

| per-entry `eps` | `B = 1` (independent) | `B = 64` (256 groups) | `B = 512` (32 groups) |
|---:|---:|---:|---:|
| 1e-5 | 1.13e-03 | 2.13e-03 | 2.22e-03 |
| 1e-4 | **1.12e-02** | **1.99e-02** | **2.43e-02** |
| 1e-3 | 1.19e-01 | 2.27e-01 | 3.76e-01 |
| 1e-2 | 1.91e+00 | 5.23e+00 | 1.10e+01 |

**Coherence costs a factor of 2, not `sqrt(d) = 113`.**  At `1e-4` per entry, all three coherence
lengths still pass +-3%.  Reading off the gate at the most correlated setting gives about
**1.2e-04 per entry**.

## What this licenses

A direct per-entry predictor of the Cholesky factor needs roughly **1e-4 relative accuracy per
entry — about four significant digits — and that requirement is only mildly sensitive to how
correlated its errors are.**  That is demanding for a regression target but it is an ordinary
number, not an impossible one, and float32 carries seven digits.

So the dense per-entry architecture is **not excluded**, and the earlier reading of `ERROR_METRIC`
as ruling it out was wrong: that bound is worst-case-aligned and does not apply to predictor error.

## What it does not license

It says nothing about whether a network can **reach** 1e-4 relative on 8.2e7 targets from geometry
alone.  That is the learnability question, and it remains completely untested.

It also does not make the per-entry route better than the hierarchical one.  Both now have a
measured accuracy requirement; the hierarchical representation needs **12.2e6** numbers rather than
**81.8e6**, a 6.7x smaller output, at a comparable per-number tolerance.  Fewer targets at the same
accuracy is the thing to want.  The two are alive on the same axis, and which is easier to learn
is — again — untested.

Finally, this is noise injected around the true value, so it models an **unbiased** predictor whose
error has the stated size.  A predictor with a systematic bias not captured by either the
independent or the block-constant model could behave differently.
