# Routes 3 and 4, measured - and 882 unused labels

2026-09-17. Scripts `superelement/objective/{sqrtA,precond}.py`.

## Route 3: `A^{1/2}` is a viable head

Same seat (0328), same tree, same admissibility, same per-block truncation as the
Cholesky result. Self-tests: `||A^{1/2}A^{1/2} - A||/||A||` = **1.360e-14**; exact fill
`eps_op` = **7.92e-12**.

### Compressibility

| tolerance | `A^{1/2}` numbers | % | `A^{1/2}` `eps_op` | Cholesky numbers | % | Cholesky `eps_op` |
|---|---|---|---|---|---|---|
| 3e-2 | 8,144,216 | 9.95% | 0.2010 | 8,581,236 | 10.49% | 0.152 |
| 1e-2 | 11,032,008 | 13.48% | 0.1191 | 10,221,964 | 12.49% | 0.081 |
| 3e-3 | 14,859,451 | 18.16% | 0.0380 | 12,229,217 | 14.95% | 0.026 |
| 1e-3 | 18,703,203 | 22.86% | 0.0075 | 14,083,953 | 17.21% | 0.011 |
| 3e-4 | 23,383,353 | 28.58% | 0.0015 | - | - | - |

`A^{1/2}` costs ~10% more numbers at the operative 0.15 gate and ~45% more at +-3%.

### Per-entry amplification

| entry relative noise | `eps_op` | amplification | Cholesky |
|---|---|---|---|
| 1e-6 | 1.115e-4 | **111.5** | 104 |
| 1e-5 | 1.113e-3 | **111.3** | 104 |
| 1e-4 | 1.124e-2 | **112.4** | 104 |
| 1e-3 | 1.167e-1 | 116.7 | 104 |
| 1e-2 | 1.864 | 186.4 | 104 |

**112 against 104 - essentially identical.** There is no accuracy penalty for choosing
`A^{1/2}`; the cost is ~10% more numbers.

### Verdict

Route 3 stands. Its advantages are entirely on the learnability side and are real:
unique and continuous gauge, **permutation equivariant** (so the 48-element cubic
symmetry group can be made exact by frame averaging, turning 33 labels into ~1,600),
and outside the scope of *"Message-Passing GNNs Fail to Approximate **Sparse
Triangular** Factorizations"* (TMLR 2025), which covers Cholesky and not a symmetric
square root. The objection that was supposed to kill it -
`FIRST_PRINCIPLES` §2.2's near/far coherence requirement of 3e-6..3e-5 - was measured
today at **2.5x** and is false.

## Route 4: the recorded column scale has its direction inverted

`REVIEW_20260916` states the Newton run's key as the column scale `1/sqrt((A^-1)_jj)`.
Measured on seat 0328 against three alternatives, at equal **loss-measured** error
(an absolute perturbation in units of each coordinate's RMS - what a bucket-normalised
squared loss actually controls):

| scaling | dynamic range (99%/1%) | `diag((Dg A Dg)^-1) - 1` | `eps_op` at delta=1e-3 |
|---|---|---|---|
| raw | 2.53e5 | - | 0.1906 |
| `1/sqrt((A^-1)_jj)` *(as recorded)* | 2.16e6 | 2.18e12 | **37.85** |
| **`sqrt((A^-1)_jj)`** | **1.26e5** | **2.22e-16** | **0.1124** |
| `1/sqrt(A_jj)` | 1.17e5 | 24.9 | 0.1216 |

The scaling that makes `diag((Dg A Dg)^-1) = 1` is `sqrt((A^-1)_jj)`, verified to
2.2e-16. The figure in `REVIEW_20260916` is its reciprocal and is **46x to 1200x
worse** at every perturbation level. That is a documentation error, and it matters
because it is the stated reason the Newton run succeeded.

Sustained across the sweep, the correct scaling buys **1.6-2.5x** and halves the
target's dynamic range:

| delta | raw | `sqrt((A^-1)_jj)` | gain |
|---|---|---|---|
| 1e-4 | 0.01762 | 0.01068 | 1.65x |
| 1e-3 | 0.19058 | 0.11245 | 1.69x |
| 1e-2 | 3.7320 | 1.7455 | 2.14x |
| 3e-2 | 27.521 | 10.918 | 2.52x |

### Verdict

Real but modest. **1.65x does not close a 1e4 gap.** Note the scope: this measures
**error propagation**, not **optimisation**. The 9/15 Newton result was about the
optimisation channel (NaN -> convergence), which is a different mechanism and may
still matter a great deal for a network. Route 4's memorisation test is still worth
running; it just should not be sold as "preconditioning fixes the accuracy budget".

## The dataset is 24x bigger than anyone has been using

Checked while looking for floor-scan data:

```
packet directories                     920
packets with S_UPPER.npy already built 920   (1,216 GB on disk)
turned into usable labels (REFERENCE_*)  38
```

**920 unit cells have already been through the expensive CutFEM condensation. 882 of
them have never been used.** Converting one packet into a label costs **4.36 s**
(`REFERENCE_0328/RESULT.json`: `total_seconds` 4.363, `load_seconds` 0.244) - it is a
Cholesky plus validation, not a condensation.

This is item 0f of `FIRST_PRINCIPLES_PATH_20260917.md` - *"把 1024 个 packet 全部做
Cholesky，标签数 33 -> 1024"* - listed and never done.

### Is fp32 storage good enough? Yes, and today's measurement says so quantitatively

`S_UPPER.json` records `schema: CUTFEM_PACKED_UPPER_F32_V1`, `source_dtype: float64`,
`storage_dtype: float32`. GPT Pro flags precisely this ([E05]): *"FP32标签提升到FP64
则不能恢复原先已丢失的信息."*

fp32 gives independent per-entry relative rounding of `2^-24` = 6.0e-8. Today's oracle
measurement gives `eps_op = 104 * eps` for exactly that error model. So the storage
contributes

        eps_op ~ 104 * 6.0e-8 = 6.2e-6

against a target gate of 0.03 and an operative gate of 0.15 - **5,000x below the
tightest gate.** The concern is real in principle and negligible in this case, and it
is now settled with a number rather than an argument.

### What it takes

`stage_cutfem_neural_a/elimination_reference.py` has a CLI
(`--input-run --previous-diagnostic --output`) but reads a **prior run directory**
containing `RUN.json` and `input/TRACE_CACHE.npz`, so an upstream trace-compile stage
must be identified first. That is a much shallower job than the CutFEM pipeline - no
geometry, no quadrature, just data reshaping plus a Cholesky - and it does not need
the frozen FE code.

Disk: each `R_UPPER.npy` is 654 MB; 725 GB free. **+100 seats = 65 GB and ~7 minutes.**
All 882 would be 577 GB and ~64 minutes, which fits but leaves little headroom - build
in batches.

## Why this matters more than it looks

Three separate standing objections are attacked by the same action:

1. **Sample complexity.** `FIRST_PRINCIPLES` §2.5: 33 geometries over 9-12 geometric
   axes is ~3 points per axis. With 920 it is ~1.8 per axis before symmetry and ~2.4
   after the 48-fold group - still thin, but 24x more data for an hour of compute.
2. **No holdout has ever existed.** Only 7 of the 33 manifest seats were never trained
   on by Codex (196, 328, 401, 434, 575, 810, 825), and 328 is the anchor for nearly
   every oracle result, so a clean holdout today would have **6 seats spanning only
   `tau` 0.277-0.484** against a manifest range of 0.194-0.578 - the thinnest and
   thickest cells are already contaminated. With 882 fresh cells this stops being a
   constraint.
3. **Generalisation has never been tested at all.** Every seat in every run is
   `split: train`.
