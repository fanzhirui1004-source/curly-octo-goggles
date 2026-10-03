# Three measurements on Codex's saved factors: where the error is, how accurate it
# must be, and whether coherence is the problem

All three run on the 40000-step predicted factors Codex saves
(`RUN_TRAIN32_ACCEL_V1/{baseline,relative}/EVAL_040000_{seat}/R_PRED_UPPER.npy`)
against the teacher `REFERENCE_{seat}/R_UPPER.npy`. Nothing trained, nothing of
Codex's modified. Scripts: `superelement/objective/{diagsplit,band,oracle,oracle2}.py`.
Data: `docs/data/codex_probe_20260917/`. Every run's self-test (teacher everywhere)
returns `eps_op = 0` exactly, and `as_predicted` reproduces Codex's published
`eps_op` and `D` to all printed digits, so the harness is sound.

## 1. The gate error is in the off-diagonals, not the pivots

Split the factor by **role** - the diagonal (the pivots, which alone set `logdet H`
and which the network emits as an explicit bounded `log_pivot`) versus everything
above it. Codex's own bucket study split by *distance*; this is the orthogonal cut.

| case | baseline/0253 | relative/0253 | baseline/0403 | relative/0403 |
|---|---|---|---|---|
| as predicted | 273.9 | 264.0 | 174.8 | 172.6 |
| teacher **diagonal** | **275.2** | 236.0 | 173.9 | 149.2 |
| teacher **off-diagonal** | **51.4** | 43.0 | 65.7 | 20.4 |
| teacher everything | 0 | 0 | 0 | 0 |

Handing the network every pivot exactly changes `eps_op` by nothing - it gets
slightly *worse* on baseline/0253. Handing it the off-diagonals cuts `eps_op` 5.3x.

Consequence: any objective term that acts only on the pivots (a log-det term, a
log-pivot regression) improves `D` and `logdet` and does not move the gate.

## 2. No sub-block of the off-diagonal carries it either

Same substitution, by Codex's own patch structure (`patch_size = 32`, so same-patch
<=> `r//32 == c//32`) and by node-index distance inside the cross bucket.

baseline/0253, `as_predicted` = 273.9:

| replaced with teacher | blocks | eps_op |
|---|---|---|
| same-patch off-diagonal | 607,911 | 191.0 |
| all cross-patch | 81,587,520 | 73.3 |
| cross, node distance <= 96 | 3,055,743 (3.7%) | 175.9 |
| cross, node distance > 96 | 78,531,777 (96.3%) | 142.8 |
| cross, node distance <= 384 | 13,510,575 (16.6%) | 123.6 |
| cross, node distance > 384 | 68,076,945 (83.4%) | 203.0 |
| cross, node distance <= 1536 | 47,864,943 (58.7%) | 92.5 |
| cross, node distance > 1536 | 33,722,577 (41.3%) | 247.9 |

Replacing **99.3% of the factor** (every cross-patch block) still leaves
`eps_op = 73.3`, 2400x the target gate, with the residual sitting in the 0.7% that
was harmless a moment ago. The pieces are strongly non-additive; they must be
*jointly* coherent. Same conclusion as Codex's distance-bucket study, now with the
role axis added and at 40000 steps.

## 3. How accurate does a per-entry predictor have to be? Measured: 3e-4

Perturb the **teacher's** factor with independent multiplicative per-entry relative
noise `eps` and measure the gate. Seat 0253, d = 12822:

| per-entry relative noise | eps_op |
|---|---|
| 1e-6 | 1.02e-4 |
| 1e-5 | 1.02e-3 |
| 1e-4 | 1.04e-2 |
| 1e-3 | 1.07e-1 |
| 1e-2 | 1.696 |
| 1e-1 | 107.7 |

Linear at **`eps_op = 104 * eps`** over four decades. So the +-10% work gate needs
~1.0e-3 per entry and the +-3% target gate needs **~2.9e-4** per entry.

Two independent confirmations: the project's own `PER_ENTRY_ACCURACY_20260916.md`
measured 2.7e-4 (independent) / 1.2e-4 (most correlated) on a different seat with a
different code path; and `FIRST_PRINCIPLES_PATH_20260917.md` predicted
`eps_op ≈ 2 eps sqrt(rho_J) ≈ 111 eps` from the near/far cancellation ratio. The
measured 104 matches the predicted 111. The `4e-7` figure quoted elsewhere is the
worst-case-aligned bound and overstates the requirement by ~3 orders.

## 4. How accurate is it actually? Measured: 3.2 to 5.7 - four orders short

Per-entry relative error of the prediction, upper triangle, seat 0253:

| arm | median | q90 | q99 | \|\|E\|\|/\|\|R*\|\| | eps_op |
|---|---|---|---|---|---|
| baseline | **5.725** | 102.3 | 1264 | 0.0971 | 273.9 |
| relative | **3.211** | 55.16 | 686 | 0.0970 | 264.0 |

The **median entry is wrong by 320-570%** where it must be right to 0.03%. The
Frobenius error is only 9.7% because the large near-field entries are predicted
well; the typical entry - and there are 8.2e7 of them - is essentially unpredicted.
The shortfall is a factor of ~1.1e4 (relative arm) to ~2.0e4 (baseline).

## 5. Is it a coherence problem? Measured: no. Coherence costs 2.5x

Take the **actual** error `E = R_hat - R_star`, randomly re-sign every entry keeping
every magnitude, and re-measure. Independent error would be unchanged; systematic
error would collapse.

| arm | as predicted | re-signed (3 trials) | coherence factor |
|---|---|---|---|
| baseline | 273.9 | 101.0, 106.0, 103.1 | **2.65** |
| relative | 264.0 | 109.5, 108.9, 105.5 | **2.45** |

Destroying all correlation in the error buys only 2.5x. This matches the repo's
earlier independent measurement that coherence costs 2.17x, not `sqrt(d) = 113`.
**Coherence is not the binding constraint; per-entry magnitude is.** Even with
perfectly independent errors at the achieved magnitude, `eps_op` would be ~105,
still 3500x the target gate.

This also corrects the pessimistic reading of the amplification ladder: the
`3e-6..3e-5` coherence requirement in `FIRST_PRINCIPLES_PATH_20260917.md` is a
worst-case systematic-bias figure, and the measured error is not that kind of error.

## 6. What the three measurements jointly say

The complete-dense-factor representation fails on **magnitude**, by four orders of
magnitude, distributed over essentially every off-diagonal entry, with no
concentrated sub-block and no coherence pathology to blame. That is not a shape-of-
loss problem and not a training-budget problem.

**On cost the picture is different, and an earlier draft of this document
overstated it.** The bound is FLOP-limited, not implementation-limited. Codex's own
speed audit gives 5.67e13 trunk FLOP per cell (6.21 MFLOP per emitted 3x3 block):

| | FLOP/s | ms per cell |
|---|---|---|
| measured, Codex (TF32 off) | 2.74e13 | 2070 |
| RTX 5090 fp32 peak | 1.05e14 | 540 |
| RTX 5090 TF32 tensor peak | 4.20e14 | **135** |

So the complete-factor decode can in principle reach ~135 ms, which is a **236x
speedup over the 31.85 s teacher** - real, and not to be dismissed. It is 135x from
a literal 1 ms, not the ~1900x an implementation-bound reading suggests. Whether
that matters is a question about the target, not a disqualification.

What is a disqualification is section 4: the accuracy shortfall is ~1e4 and it is
not a cost, budget, or coherence problem.

For reference, the same FLOP bound applied to the other representations
(seat 0253, at TF32 peak):

| representation | numbers | ms at peak | gate reached |
|---|---|---|---|
| complete factor | 8.22e7 | 135 | 264 (network), 0 (teacher) |
| H-matrix at 15% of dense | 1.22e7 | 20 | **2.6e-2** (compression of the teacher) |
| collar core material | 2.3e4 | 0.04 | **0.15** (fit to one seat) |

The last row's 0.04 ms is only the cost of emitting the parameters. The collar route
still has to **condense** at inference: measured at 0.27x the teacher's dofs, 0.076x
its factorisation and 0.105x one right-hand-side application. Per cell that is
seconds, not milliseconds. Its cost advantage is architectural - assemble the
reduced models of all cells into one global reduced system and never form a per-cell
S at all - which is a different contract from the one this project has been
measuring against, and one of the open user decisions.

## 7. How few factor entries could represent S at all? Not few enough

Keep the k largest-magnitude entries of the **teacher's** factor (pivots always
kept) and measure the gate. The network is removed entirely; this is pure
representability. Seat 0253, dense = 82,208,253 entries.

| kept | % of dense | time to emit at 3.97e7/s | eps_op |
|---|---|---|---|
| 12,823 | 0.016% | 0.32 ms | 8031 |
| 30,001 | 0.036% | 0.75 ms | 2978 |
| 100,000 | 0.122% | 2.5 ms | 733 |
| 300,000 | 0.365% | 7.6 ms | 305 |
| 1,000,000 | 1.22% | 25 ms | 93.9 |
| 3,000,002 | 3.65% | 75 ms | 20.2 |
| 10,000,001 | 12.2% | 252 ms | 1.15 |
| 30,000,001 | 36.5% | 755 ms | **0.192** |

At 36.5% of dense - 755 ms just to write the numbers out - naive magnitude
truncation still fails the +-10% work gate.

Magnitude truncation is a weak compressor and this is an upper bound on `eps_op`,
not the optimum: the project's hierarchical low-rank compression reaches
`eps_op = 2.6e-2` at 12.23e6 numbers (14.95% of dense), where magnitude truncation
at 12.2% gives 1.15 - **structure is worth ~44x at equal budget**. So the right
statement is not "sparsity fails" but:

> The best factor-shaped representation anyone here has produced needs 1.2e7 numbers
> and 309 ms to emit. A millisecond budget allows ~4e4 numbers. No factor-shaped
> output closes that gap, and the gap is ~300x against a compressor that already
> exploits the physics (low rank of far-field interaction).

Combined with sections 3-5, the complete-dense-factor route is disqualified twice
over and independently: on per-entry accuracy (4 orders short, not a coherence
problem) and on emission cost at perfect accuracy (~1900x). Neither is a training
problem.
