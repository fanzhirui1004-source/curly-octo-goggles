# Codex M4 multi-geometry run: the decoupling is now proved algebraically

Read 2026-09-17 21:40 CST from `/root/autodl-tmp/CUTFEM_M4_MULTI_20260917`
(host `a841046-9982-eae5cdfb.bjb2`). Supersedes the scope of
`docs/CODEX_M4_ASSESSMENT_20260917.md`, which covered only the 01:41-02:28
single-seat baseline. Nothing was running at read time (GPU 0%); the run is
between phases.

## 1. What happened in 19 hours

- Training scaled from 1 seat to **26 geometries** (`RUN_TRAIN26`, seats 120-995),
  then to **32** (`RUN_TRAIN32_*`, + `CUT_ADDITIONS_V1`), batch 4, uniform exposure
  (4615-4616 presentations per seat over 30,000 updates).
- A **three-way feature ablation** - `geometry` / `stiffness` / `multiscale` -
  trained to 30,000 steps each (29,290 s, peak 16.4 GiB).
- A **bucket-substitution attribution** at 30,000 steps (`DIAG_BUCKET_030000`).
- A **paired A/B on the loss**: `baseline` (factor MSE) vs `relative` (per-block
  relative weighting), both to 32,048 then 40,000 steps, 8 eval seats.
- An **error attribution** by load and by geometry, and a **root-cause review**
  (`root_cause_review_76446f3e5.tar.gz`) auditing the loss identity, input
  pullback, optimizer log and output pivots.
- Acceleration work: `preproject` fast inference, dense response, shared backward.

## 2. Feature ablation at 30,000 steps (all seats `split: train`)

| variant | seat | e_A | D/d | mu_min | mu_max | eps_op |
|---|---|---|---|---|---|---|
| geometry | 0253 | 0.0926 | 0.314 | 2.51e-3 | 218.6 | 217.6 |
| geometry | 0403 | 0.0788 | 0.319 | 1.17e-3 | 126.0 | **125.0** |
| multiscale | 0253 | 0.0628 | 0.373 | 6.15e-4 | 249.3 | 248.3 |
| multiscale | 0403 | 0.0547 | 0.371 | 1.05e-3 | 158.1 | 157.1 |
| stiffness | 0253 | **0.0596** | 0.342 | 2.05e-4 | 235.3 | 234.3 |
| stiffness | 0403 | **0.0503** | 0.302 | 7.87e-4 | 145.2 | 144.2 |

The physics features (`stiffness`, `multiscale`) give the **best `e_A`** and the
**worst `eps_op`**. Better features improve the quantity the loss measures and
degrade the quantity the gate measures. First clean instance of the decoupling
inside their own ablation.

Their own milestone notice: *"All six final spectra numerical qualification pass;
all six working and target gates fail. Worst 6D compliance errors 81.6-89.3
percent."*

## 3. Bucket substitution: no single bucket is the problem

Replace one bucket of the predicted factor with the teacher's exact values, keep
the rest predicted. `diag` = diagonal blocks, `same` = same-patch (near),
`cross` = cross-patch (far).

| case | 0253 eps_op | 0403 eps_op | 0403 e_A |
|---|---|---|---|
| unchanged (all predicted) | 234.3 | 144.2 | 0.0503 |
| diag -> teacher | 224.0 | 138.1 | 0.0432 |
| same -> teacher | 162.7 | 107.9 | 0.0435 |
| **cross -> teacher** | **54.4** | **37.6** | 0.0428 |
| diag+same -> teacher | 151.6 | 102.0 | 0.0319 |
| same+cross -> teacher | 48.1 | 27.7 | 0.0309 |
| all teacher | **0** | **0** | **0** |

The `all_teacher` row returns eps_op exactly 0 and compliance error 1.3e-15 /
1.8e-15 - the substitution harness is exact, so the other rows are trustworthy.

Two readings:

1. **The far bucket carries most of the gate error and almost none of the loss.**
   Handing over `cross` alone buys 74% of the eps_op reduction (144.2 -> 37.6).
   From the baseline sampling diagnostic, near blocks are 0.444% of sampled
   cross-patch pairs and **97.09%** of sampled factor squared norm - so the bucket
   that decides three quarters of the gate receives a few tenths of a percent of
   the gradient.
2. **The error is not localised, it is incoherent.** Fixing `diag` alone buys 6
   units out of 144. But after `same`+`cross` are exact, the residual 27.7 is
   *entirely* diag. Bucket contributions are strongly non-additive: they partially
   cancel. There is no bucket whose repair rescues the gate; they must be *jointly*
   coherent. Even with two of three buckets exact, eps_op is still 28-48, i.e.
   ~10^3 x the target gate.

## 4. Paired loss A/B at 40,000 steps (8 seats)

| seat | arm | e_A | D/d | mu_min | eps_op | 6-D compl. err |
|---|---|---|---|---|---|---|
| 0253 | baseline | 0.0863 | 0.524 | 1.10e-3 | 273.9 | 0.915 |
| 0253 | relative | 0.0831 | **0.325** | 9.65e-3 | 264.0 | 0.911 |
| 0403 | baseline | 0.0740 | 0.589 | 2.48e-3 | 174.8 | 0.876 |
| 0403 | relative | 0.0760 | **0.258** | 2.35e-2 | 172.6 | 0.874 |
| 0186 | baseline | 0.523 | 5.07 | 1.66e-3 | 2775 | 0.980 |
| 0186 | relative | 0.618 | **1.70** | **-1.35e-16** | 2210 | 0.964 |
| 0206 | baseline | 0.462 | 11.6 | 3.96e-4 | 8592 | 0.986 |
| 0206 | relative | 0.534 | **2.26** | 2.19e-12 | 3045 | 0.970 |
| 0210 | baseline | 0.486 | 4.93 | 1.28e-4 | 8236 | 0.947 |
| 0210 | relative | 0.588 | **2.40** | 3.58e-15 | 7356 | 0.901 |
| 0212 | baseline | 0.523 | 5.34 | 1.95e-4 | 1927 | 0.983 |
| 0212 | relative | 0.640 | **2.40** | 4.19e-13 | 1820 | 0.955 |
| 0353 | baseline | 0.579 | 4.98 | 2.49e-4 | 5781 | 0.970 |
| 0353 | relative | 0.653 | **1.76** | 1.18e-15 | 2197 | 0.948 |
| 0814 | baseline | 0.503 | 4.67 | 1.22e-3 | 1513 | 0.966 |
| 0814 | relative | 0.585 | **1.65** | 4.17e-13 | 1657 | 0.931 |

Three results, all of them informative:

- **Relative weighting works on D/d and is not enough for the gate.** D/d improves
  in 8/8, by up to 5.1x (0206: 11.6 -> 2.26). `eps_op` improves 1.1-2.6x and gets
  *worse* on 0814. Reweighting fixes the bulk of the spectrum, not the extremes.
- **`e_A` moves the other way in 6 of 8 seats.** The two objectives are in direct
  tension, not merely different.
- **Relative weighting drives `mu_min` toward zero and through it.** Baseline
  `mu_min` ~1e-3..1e-4; relative gives 1e-12, 1e-15, and **-1.35e-16** on 0186.
  `numerical_gate_pass` flips to false on most relative rows. A one-sided relative
  loss buys the soft end by making the operator nearly singular there - exactly the
  failure our own E1-F note predicted for anything without a two-sided
  `mu - log mu - 1` penalty.

Codex's own decision: `CONTINUE_BOTH_TO_40000_NO_CLEAR_WINNER` - *"All eight D/d
and compliance metrics improve with relative loss, but e_A worsens in eight and
max displacement worsens in seven... Both remain outside gates."* It also ran a QR
probe (`qr_all_pass: true`) to rule out the response comparison being a solver
artifact. That is correct practice.

Capacity note: 0253 (q=12,828) and 0403 (q=14,568) are the two smallest cells and
the only ones under e_A 0.1. The six larger seats (q=14,388-22,830) sit at
e_A 0.46-0.65. The complete-factor output grows as q^2/2, so the output budget per
cell grows quadratically while the network does not.

## 5. What the errors actually are, physically

`ERROR_ATTRIBUTION_40000_LOADS.json`, baseline arm, per seat: the worst of the 18
probe loads and the work under it.

| seat | worst load | teacher work | predicted work | predicted too stiff by |
|---|---|---|---|---|
| 0403 | Fz | 1.056e-6 | 1.388e-7 | 7.6x |
| 0253 | Fz | 1.702e-6 | 1.575e-7 | 10.8x |
| 0210 | Mz | 4.794e-5 | 3.407e-6 | 14.1x |
| 0212 | Mz | 2.260e-4 | 6.426e-6 | 35.2x |
| 0186 | Fz | 4.334e-6 | 1.190e-7 | 36.4x |
| 0206 | Fz | 9.720e-6 | 1.360e-7 | 71.5x |

`c18_signed_error` is **negative on all eight seats** (-0.87 to -0.99), and
`c6_mu_max < 1` on all eight (0.30-0.61). So this is not scatter. It is a single-
signed, systematic bias: **the predicted unit cell is 8-70x too stiff under every
global load direction**, worst under pure axial Fz and torsion Mz.

Plain reading: the network has learned the near field - which is essentially local
element stiffness - and discarded the cell's long-range compliance. What it emits
behaves like a nearly rigid block. That is precisely the failure mode a
norm-weighted factor loss should be expected to produce, because long-range
compliance lives in the modes with the least factor norm.

## 6. Codex's own root-cause review, and the algebraic witness

`LOSS_IDENTITY.json` (status `SMALL_ALGEBRAIC_WITNESS_PASS`, verified at d = 6, 12,
24 to 1e-16) contains the scalar table that settles the argument:

| mu | relative factor MSE | D | d(MSE)/d(log factor) | d(D)/d(log factor) |
|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 0 |
| 1e-2 | 0.810 | 3.615 | -1.80e-1 | -1.98 |
| 1e-6 | 0.998 | 12.816 | -2.00e-3 | -2.00 |
| 1e-12 | 1.000 | 26.631 | -2.00e-6 | -2.00 |

As a mode softens, **the factor-MSE loss saturates at 1 and its gradient decays to
zero, while the divergence grows logarithmically and its gradient stays pinned at
-2.** On a mode that is 10^6 x too soft, the MSE objective pushes with gradient
2e-6 and the log-det objective pushes with gradient 2 - a factor of 10^6
difference in optimizer pressure on the modes that are most wrong.

This is the amplification-ladder argument of `docs/FIRST_PRINCIPLES_PATH_20260917.md`
restated exactly, derived independently, and proved algebraically rather than
inferred. It is the strongest single piece of evidence produced in this project so
far that the objective, not the budget or the architecture, is what is failing.

The review also **rules things out**, correctly and with evidence:

- **Not gradient clipping.** Trigger rate 0.93% (baseline) / 1.19% (relative) over
  7,952 records each; 1.57% / 2.02% on CUT updates.
- **Not output saturation.** Predicted log-pivots sit inside the [-20, 5] bounds;
  `within_half_log_unit_of_bound_fraction = 0.0` on every arm.
- **Not blindness to the cut.** No CUT cell's feature pullback is identically zero;
  L1 retention median 0.59-0.65. Codex is careful to add that L1 retention is not an
  information-fidelity or energy ratio.
- **Honestly flagged confound**: the 6 new CUT geometries entered at step 30,001,
  when LR had already decayed from a 2e-4 peak to 3.65e-5, with no warm restart and
  no adaptation schedule. Codex states this is not demonstrated to be the root cause.
- Gradient cosine between the MSE and relative terms: 0.021 / 0.033 on the two FULL
  seats, 0.332 on 0206 - near-orthogonal. Codex explicitly declines to call them
  anti-correlated.

Its proposed next step is a 2x2: {current front-end, + CUT support-point features}
x {current loss, factor regression + a **D-aligned work / exact log-det term**},
same decoder head, same frozen start, zero-gated residual so the new arm starts
identical to the old. That is the right experiment.

## 7. Assessment

The harness is now genuinely strong: 32 geometries, exact substitution attribution,
paired loss A/B with a QR control, load-level physical attribution, an algebraic
witness for the loss identity, ruled-out alternatives with numbers attached, SHAs
throughout, and an explicit refusal to claim generalisation from training-seat
errors. Very little of this needs redoing.

The result is equally clear, and it is negative in a useful way. Across 5 step
budgets (256 -> 40,000), 3 feature sets, 2 loss functions and 8 geometries, the
6-D compliance error has never fallen below 0.83, `eps_op` has never come within
three orders of magnitude of the target gate, and the predicted cell is uniformly
8-70x too stiff. `e_A` and `D/d` both improve steadily; neither drags the gate with
it. The bucket substitution shows why no reweighting can fix it, and the loss
identity shows why no amount of training can.

## 8. Recommended next steps

1. **Run Codex's 2x2, with arm B/D's objective specified two-sided.** The D-aligned
   term must be `f(mu) = mu - log mu - 1` on *both* spectral ends, with `mu_min` from
   power iteration on `H^-1 = M^-1 M^-T` (not on `cI - H`, which does not converge
   and returns plausible garbage silently), and *not* `relu(mu_max - 1.03)^2`, which
   with `mu_max ~ 1e3` dominates the gradient direction while Adam normalises only
   magnitude. Section 4 of this document is direct evidence for the two-sidedness:
   the one-sided relative loss already pushed `mu_min` to -1.35e-16.
2. **Give the new geometries a warm restart** when they enter, or re-run the CUT
   addition from a scheduled LR. The confound Codex flagged is cheap to remove and
   currently contaminates every conclusion about the 6 new seats.
3. **Run the two falsifiers before another long training budget.** (A) gamma sweep at
   1e-3 / 1e-5 on seat 0328: if the labels move when the ghost-penalty parameter
   moves 100x, no objective fixes that. (B) near/far coherence oracle on the
   H-matrix that already passes +-3%: the bucket substitution here gives the
   *achieved* incoherence; the oracle gives the *required* coherence, and the two
   together decide whether any separate-prediction architecture can reach the gate.
4. **Open the local-multiplicative-field arm in parallel**, not instead. Section 5
   says the network is emitting a near-rigid block; section 6 says the objective
   cannot punish that. Predicting per-element physical fields and condensing exactly
   makes the amplification 1 instead of 7.2254e4 and shrinks the output from
   O(q^2) to O(elements) - which also removes the capacity wall in section 4, where
   the only two seats under e_A 0.1 are the two smallest cells. Their geometry
   adapter, frozen quotient, spectrum and response harness all carry over; only the
   head changes. Measured support: encoder time is 0.031 s of a 2.07 s inference on
   seat 253 - essentially all of the cost is emitting 9.14e6 factor blocks
   (5.67e13 trunk FLOPs), not understanding the geometry.
