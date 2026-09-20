# Handover, 2026-09-21 — for Codex

Tonight was a dataset-repair session, by the user's explicit choice: *"今晚不跑实验，纯属修数据集."* No
training arm ran. Nothing is running now.

Read section 3 before proposing anything. Five of my own conclusions died tonight, one of them
twice, and two of them are still written down in older docs in this repo.

The user's last instruction was **stop and settle the route and the sample design** — *"请你先不要
着急往下推进了，先确定路线能跑通，以及我们的具体样本设计即可"*. So no teacher production has been
launched and none should be until section 7's decisions come back.

---

## 0. Standing constraints — carry these forward verbatim

A dropped constraint is the most expensive kind of handover defect, so these come first.

- **Branch**: develop and push only to `claude/wizardly-euler-3m9cwx`.
- **Two frozen trees, bound by sha256, never modify**:
  `/root/autodl-tmp/CUTFEM_INGEST_R38/source_independent_6624dc8_20260910` (HEAD
  `6624dc86706bbd8e3cc16b4b3adbfb043eff08ec`) and
  `/root/autodl-tmp/CLAUDE_SQRTHEAD_20260917/src_v5`.
- **`/root/autodl-tmp/CUTFEM_INGEST_R38/dataset_independent_20260910` (1.3 TB) must not be deleted.**
- **Never nest heredocs and never use an unquoted heredoc to generate a remote script.** Write the
  script locally, base64-ship it. A nested heredoc silently produced an empty file once and cost an
  evening.
- **`pkill -f "<the script's own text>"` kills the shell running your script.** The jrun kernel's own
  subprocess command line contains the literal string you are matching. Learned the hard way.
- **Our code on the box is a shipped copy, not a checkout.** Tonight's modules live in `/root/_cb/`,
  which is a bare staging tree (`superelement/__init__.py`, `superelement/objective/*.py`) created by
  base64-shipping. It is not a git checkout and has no history. Re-ship after every edit.
- **Do not widen the acceptance gate.** No jitter, no pseudo-inverse, no arbitrary spectral clipping,
  no deleting directions to hide a failure, and never rewrite a failed run as a pass by raising a
  tolerance.
- **The contract is the whole point**: the assembled lattice's compliance *and* its optimisation
  sensitivities must be right to **3 %**, with τ as the design variable.
- **All reporting to the user is in Chinese, plain undergraduate-level language.** The user is not a
  native English reader and asked for this explicitly.
- Commit messages end with the two trailers (`Co-Authored-By: Claude Opus 5 …` /
  `Claude-Session: …`).

Remote access: `scratchpad/jrun3.py` with `c3.sh` (new kernel, deletes it afterwards) or `c3r.sh`
(reuses an existing kernel — use this when an arm holds tens of GB, because forking a new kernel can
fail with HTTP 500 under the cgroup ceiling). Host and token are baked into both.
**One shared Jupyter kernel — do not run two of these concurrently, and do not let subagents touch
the box.**

---

## 1. State of the box

Nothing running. GPU idle (0 MiB of 32607 MiB used). cgroup 90 GiB, ~69 GiB of it reclaimable page
cache — release it with `/root/cutfem_neural_a_20260910/env/bin/python /root/_fadv.py <dir>` before
starting anything large, or the next job OOMs on `PREALLOCATION_CGROUP_BUDGET`.

| filesystem | size | free | note |
| --- | --- | --- | --- |
| `/root/autodl-tmp` | 5.7 T | **3.4 T** | the GPU box's data disk, where labels live |
| `/autodl-fs/data` | 200 G | **158 G** | **the shared volume — same AutoFS mount on the GPU box and on all four new CPU instances** |
| `/autodl-pub` | 14 T | 2.7 T | AutoDL's public dataset mount. **Not ours.** I previously mistook this for our shared disk; the shared buffer is 158 G, not 2.8 T. |
| each new CPU box `/root/autodl-tmp` | 50 G | — | ≈70 packets of local queue at ~0.65 GB per `S_UPPER` |

The 50 G local disk is why the ACK-gated cleanup in the user's architecture is mandatory rather than
prudent: the local disk fills before the shared buffer does.

Tonight's directories:

| path | size | what |
| --- | --- | --- |
| `/root/autodl-tmp/CLAUDE_CUTLABELS_20260921/` | 120 G | 44 rebuilt `M_q` labels (`out/CUT_*/{MQ_UPPER,R_UPPER}.npy`, `MQ_RESULT.json`), `JOBS.json`, `build.log` |
| `/root/autodl-tmp/CLAUDE_FILL_20260921/` | 2.6 M | every fill design iteration; `DESIGN_300/` is the current one, the rest are superseded probes |
| `/root/autodl-tmp/CLAUDE_BUDGET_20260921/all/` | 268 K | the 920-packet calibration of the trace screen |
| `/root/autodl-tmp/CLAUDE_STORAGE_20260921/` | 48 K | the deletion manifest for the 219 GB freed tonight |
| `/root/autodl-tmp/CLAUDE_TAUQ_20260921.json` | — | (τ corners, plane, q) for all 920 packets. Cheap and reusable; most of tonight's analysis reads it. |

Four new AutoDL CPU instances exist (32 cores, 64 GB each) — **four, not three**. Nothing has been
installed on them. No source tree, no environment, no config.

---

## 2. What tonight established

### 2.1 The 44 labels are built and self-consistent, but NOT yet trainable

44/44 via `superelement/objective/build_cut_labels.py`. Filed per-packet at
`docs/data/cut_labels_20260921/LABEL_BUILD_44.json`; extrema over all 44:

| | min | median | max |
| --- | --- | --- | --- |
| q | 18012 | 19203 | 19956 |
| seconds | 25.96 | **31.32** | 35.11 (1362.7 s total) |
| peak GiB | 19.34 | — | **23.739** |

Self-tests, worst case over all 44: rigid nullspace residual **5.30e-17**, `‖M(MA) − I‖/√(q−6)`
**2.60e-10**, round trip to `A^{-1/2}` **6.48e-16**. Zero failures.

Do not confuse these with the figures in commit `da3f6a7` (27 s, 1.8e-17, 1.6e-10, 19.3 GiB): that
was the **first** packet, read while the run was still in flight, at the smallest q. These are the
extrema, and they are larger because q is.

**They cannot be trained on yet.** `train_equi.prepare` refuses the directory: it wants a
`RESULT.json` with `dimension`/`factor_sha256`/`trace_sha256`, and it wants `MQ_RESULT.json` to carry
`m_sha256` (the builder writes the same digest under the name `mq_sha256`), `q`, and
`read_blocks_roundtrip == 0.0`. `superelement/objective/register_cut_labels.py` does exactly that —
and measures the round trip rather than asserting it, because writing `0.0` unchecked would turn a
verification into a decoration.

**It has never completed a run.** Its first attempt died on
`TypeError: 'PosixPath' object is not subscriptable` — the frozen `read_blocks(packed, rows, cols, d)`
takes the *opened* packed array, not a path. Note the history, because `git log` will mislead you:
the broken version existed only as the shipped copy on the box and was **never committed**, so
`41a1a10` *introduces* this file already carrying the fix. There is no "before" commit to diff
against. On disk right now: `0` `RESULT.json` under `CLAUDE_CUTLABELS_20260921/out/`, and no
`NEW_LABELS.json`. **This is task 1 in section 4.**

### 2.2 The label ceiling is a measured property of this card, not a law

The user asked why `q ≤ 23000`. Across all 44 builds the peak is exactly

```
peak = 5.961e-8 · q²  GiB   =   8.00 float64 q×q buffers
```

(19.34 GiB at q=18012, 22.11 at 19260, 23.74 at 19956 — the coefficient is constant to three
figures). So a 32 GB card with ~30 GiB usable stops at **q = 22433**, and the 90 GiB cgroup on the
CPU path would reach **q = 38421**. `eigh` costs 18.6 s at q=18012 and 24.6 s at q=19956 on the GPU.
Filed at `docs/data/cut_fill_20260921/LABEL_MEMORY_LAW.json`.

Three ways to raise it, none tried: the CPU path (q≈38000, at maybe 15–25× the `eigh` time); a leaner
builder (8 buffers is not minimal, 3–4 is reachable in-place, giving q≈32000 on the card); or
Newton–Schulz for `A^{-1/2}`, which needs only matmuls and stays in float64. **But raising it to cover
ρ > 0.4 is not worth it** — that band has q median 32484, max 47760, ≈4.6 GB per label and ~40 min of
`eigh` each.

### 2.3 The trace dimension can be aimed at before the teacher runs

New module `superelement/objective/cut_budget.py`.

- **For an UNCUT cell it is an identity**: `q = 3 × |Q2 nodes on the box-face patches|` that the
  stage-1 topology emits. Verified on packets 0257 (q=19038) and 0258 (q=27744) — ratio **1.0000**
  both. `compile_topology` costs 22–31 s at 4 workers and no assembly, so this is free: it is already
  stage 1.
- **For a genuine cut cell no node count can work.** `compile_trace` adds one exact functional per
  sextic triangle point on every `MACRO_CUT_FACE` polygon and keeps a *maximal independent subset*, so
  the cut contribution is a **rank**, and it needs the assembled body's node list. Packet 0267 proves
  it: its trace, 28308, *exceeds* the 24732 the whole box boundary carries. An exact `q` before
  assembly is not available at any price.
- **What is available is a calibrated screen on the two counts that drive it** — box-face nodes
  carrying material on the retained side, and material cells the plane passes through. Fitted on all
  920 packets at **82 ms** a shot: cut cells median |error| **1.49 %**, p95 6.23 %; uncut 0.54 %, p95
  2.11 %; worst 15.85 %; 672/920 within 2 %, 871/920 within 5 %. Coefficients: box **0.9789** (the
  sampled cell set is a ~2 % superset of the certified one), cut **8.4439** independent functionals
  per cut cell, against the 9 a Q2 face carries. Filed at `docs/data/cut_budget_20260921/`.

### 2.4 The τ generator is in the frozen tree, and its bounds are a volume-fraction range

`stage_cutfem_graded/sampling.py` + `thickness.py`. Contract enforced by `Thickness.__post_init__`:
corners ∈ `[0.1755, 0.8775]`, span ≤ `0.47`, max |∇τ| ≤ `0.47`. The eight corners come from one
centred polynomial in `x,y,z,xy,xz,yz,xyz` scaled *as a whole* — individual corners are deliberately
never clipped, because that would turn an AFFINE field into a MIXED one. Cut plane is
`(1, tanθ, 0)` with θ uniform on `[0°,45°)`; the offset is solved from a **uniform retained box
volume** in 10 bins. `POLICY` seed 20260909, 256 targets, and a `replacement_policy` of "new case ID
in same slot and strata".

**Measured on a 400³ grid of the Schwarz-P sheet `|φ| ≤ τ`:**

| τ | ρ (volume fraction) |
| --- | --- |
| 0.1755 (frozen LOWER) | **0.1001** |
| 0.3502 | 0.200 |
| 0.5250 | 0.300 |
| 0.699267 | **0.400** |
| 0.8775 (frozen UPPER) | **0.5026** |

and `UPPER/LOWER = 5.0000` exactly. So **the frozen corner window *is* ρ ∈ [0.10, 0.50]**, and ρ is
very nearly affine in τ across it. `fill_cut_family.volume_fraction_curve` re-derives this at run time
and **raises unless both bounds reproduce to 2e-3**, so a bound stated in ρ cannot silently come to
mean something else.

Print the τ for ρ = 0.40 as **0.699267**, which is what the run-time inverse resolves to at its 256³
resolution and what `SUMMARY.json` records. My earlier `0.6994` came from a one-off 400³ probe; the
two are the same quantity at two resolutions, and quoting the coarser one means the document will not
match what the next run prints.

`c = 0` is not merely the sampler's habit: `GradedContract.__post_init__` raises
`GRADED_SINGLE_ZERO_COMPONENT_PLANE_FAMILY` unless exactly one normal component is zero. **A generic
normal is rejected by the frozen teacher itself.** The user's scoping decision is baked into the
contract.

### 2.5 Capping at ρ = 0.40 is right, and the argument is waste, not storage

Of the 237 packets (26 % of 920) above ρ = 0.40, only **18** are under the label ceiling — **92.4 % of
that band is teacher time that cannot be used**, and for uncut cells it is **0 of 77**. The band is
**43 % of the `S_UPPER` bytes** — 493 GB of 1133 GB, where 1133 GB is the packed upper-triangle size
computed from q, not an on-disk total. (`docs/ROUTES_3_4_RESULT_20260917.md:90` records 1,216 GB on
disk for the same 920 packets; the difference is everything in a packet that is not `S_UPPER.npy`.)
For the thick cells we *can* label, the cap is
q × 0.853 and bytes × 0.728. The cap is strictly *inside* the frozen contract, so no frozen file
changes.

### 2.6 The coverage problem is our budget, not the generator

All 607 cut packets are near-uniform in retained volume — the ten deciles hold 52 to 66 each. The
**in-domain subset is not**: 52 in the bottom decile, 17 in the top. The mechanism, measured: at fixed
τ a deep cut carries **1.61×–1.76×** the dofs of a shallow one, consistently across five τ bands. A
steeper plane also cuts a larger polygon and so carries more cut coordinates. **The q ceiling is a
filter on depth and density**, and the surviving set leans shallow and thin.

`q ~ τ_mean^p` fits **p = 0.704, R² = 0.9917** on the 313 uncut packets and **p = 0.667, R² = 0.301**
on the 607 cut ones — the cut scatter is depth, not noise. Filed at
`docs/data/cut_labels_20260921/TAU_Q_LAW.json`. **This law is descriptive only; it is not the
predictor.** The predictor is the calibrated screen of section 2.3, which works on different inputs.
Do not let the 0.99 tempt you into using τ as a stand-in for q.

### 2.7 The dataset was never the bottleneck; label production was

| | count | note |
| --- | --- | --- |
| V2_LABELS rows | 273 | |
| …with `MQ_UPPER`, usable today | **197** | 140 cut + 57 box, all `train` |
| …lacking `MQ_UPPER` | **76** | 69 train / 6 holdout / 1 validation; 45 cut + 31 box |
| ……of which built tonight but unregistered | 44 | section 2.1 |
| ……still to build | **32** | includes seats 0257, 0328, 0401, 0415 |
| packets outside the manifest | 647 | 422 of them cut |
| …cut and in-domain (`q ≤ 23000`) | **111** | need the ~25 s trace stage first, then `M_q` |

**Ceiling with zero new geometry: 296 usable cut cells, up from 140.** That is the cheapest available
test of "is the dataset too small", and it costs no teacher time.

It will **not** fix coverage: those 111 were selected by the same q ceiling, so they lean the same way.
Adding the 101 under the older τ rule left the largest hole unchanged.

**Two corrections to previously filed counts**, both resolved and filed at
`docs/data/cut_labels_20260921/INVENTORY_RECONCILED.json`:

- `docs/data/unused_packets_20260921/FINDINGS.json` records **603 cut / 317 box**. That is wrong by
  four packets. Three independent tests — offset < 2, the plane actually meeting the box
  (0 < d < 1+b), and retained volume strictly below 1 — all give **607 / 313** with **zero**
  packets on which they disagree. The consequent corrections: V2_LABELS is 185 cut / 88 box (filed:
  184 / 89), and 422 cut packets sit outside the manifest (filed: 419). The script behind the 603 was
  never committed, only its output, so the cause is unrecoverable.
- **"In domain" has meant two different things** and I am the one who overloaded it. The earlier
  sessions meant **τ_mean ≤ 0.45** — that is the filed `cut_in_domain_tau_le_0.45: 239` and the "100
  unused in-domain cut geometries" of commit `b5e4bcb`, which re-measure to **241** and **101** under
  the corrected split. Tonight meant **q ≤ 23000**, which gives **296** and **111**. The two rules
  overlap on 202 cut packets. `q ≤ 23000` is the one that matters, because q is what the label builder
  is limited by. **Never write "in domain" again without the predicate beside it.**

### 2.8 Conditioning does not depend on τ

Over all 44 of tonight's builds, τ_mean spanning 0.197–0.788: the fit is `κ(A) ~ τ^-0.167` with
**R² = 0.0074** — i.e. no relationship at all, and the sign is not even positive. Thin-half median
**3.50e6**, thick-half **3.41e6**. **Do not use ill-conditioning as an argument about thin walls.** I
intended to and the measurement refused.

Two cautions, both of which I got wrong first. My earlier figures (R² = 0.030, 3.47e6 vs 4.76e6) were
the 19-packet subset available mid-build; the 44-packet version above supersedes them. And this
`condition_of_A` is **not** the same quantity as the `κ(A*)` recorded over 47 cut seats in
`docs/THE_CUT_FAMILY_IS_TWO_PARAMETERS_20260920.md` (median 7.75e5, Spearman +0.26 in the
previous handover's ruled-out list). Do not compare them. Filed at
`docs/data/cut_labels_20260921/KAPPA_VS_TAU.json`.

### 2.9 219 GB freed, and one thing that looked deletable is not

Deleted **tonight**: 370 network prediction dumps (`MQ_PRED_UPPER.npy`, `A_PRED_UPPER.npy`),
**219.0 GB** — the class the user named. Re-derivable from any `CHECKPOINT_*.pt` in one decode per
seat, and every summary number is already filed under `docs/data/`.

**This is separate from, and additional to, the 230.9 GB deleted on 2026-09-20** (itemised at
`docs/HANDOVER_20260920_2330.md:178-183`, of which 67.1 GB was prediction dumps from superseded
arms). Do not try to reconcile the two totals — they are different days and different files. Summary
filed at `docs/data/cut_labels_20260921/DELETION_20260921.json`; the full per-file manifest with paths,
bytes and reason stays on the box at
`/root/autodl-tmp/CLAUDE_STORAGE_20260921/DELETED_PREDICTION_DUMPS.json`.

**`R_UPPER` (213 GB) is NOT deletable.** I expected it to be a provenance token; it is loaded and used
as the whitener at `superelement/equi/train_equi.py:725`.

Remaining candidates, not touched: `G_UPPER` 20.9 GB (the superseded route-5 inverse-factor label);
`dataset_n32_20260909` 70 GB and `dataset_n32_actual_20260909` 71 GB (predecessors of the frozen
dataset — 8 and 6 text references exist outside the ingest tree, so check before deleting).

### 2.10 Literature: the central claim survives, three of my framings do not

See `docs/data/prior_art_20260920/` plus tonight's follow-up sweep.

- **Nobody has learned a cut cell's condensed operator.** The claim stands.
- But **"nobody has put a network on a cut cell" is false four times over**: Lee et al. (Eng. with
  Computers 40:105, doi 10.1007/s00366-023-01785-z, Jan 2023) learn moment-fitting *quadrature*;
  Daviet et al. (ACM TOG 44(2), arXiv:2410.09417) learn cut-cell *quadrature*; Saberi, Zhao & Vogel
  (arXiv:2403.11632v2) learn the Nitsche *stabilisation scalar* on 197,796 cut cells of a 988,837-cell
  mesh at <5 % error; Mika et al. (CMAME 452:118700, Jan 2026) learn cut-cell quadrature via splines.
  None learns an operator, reports a trace dimension, or verifies a derivative. We kept missing them
  because we search for a learned *operator* and they learn *quadrature* — the queries never intersect.
- **The literature's largest learned trace is 392** (Du & Stechmann, "element learning",
  arXiv:2308.02467, J. Comput. Math. doi 10.4208/jcm.2407-m2024-0047 — an MLP emitting a 392×392
  Dirichlet-to-Neumann map). Our widest is q = 12798 (seat 0328), so we are **≈33×**, not the 500× or the
  "4,100–342,000× more entries" figure I gave the user. That figure was against two specific papers
  and must be stated that way. Scope any record claim by physics: learned dense BEM/MoM impedance
  matrices are a literature we have never entered.
- **Nearest neighbour, and it is close**: arXiv:2608.02036v1 (Aug 2026) learns
  `M_g = sqrtm(S_g − λ(I − P_c))` — a matrix square root of a regularised Schur complement with the
  rigid nullspace projected out, hypernetwork, 38 M parameters, **400 training geometries**, 386×386
  for *scalar heat*, 128×128 for 2-D elasticity, PSD with exactly three zero modes verified to 1e-14.
  We take `sqrt(A^{-1})`, they take `sqrt(S)`; no cut cells and no derivative on their side. **Read the
  primary source before writing any novelty sentence about the label construction.**
- **The strongest wedge is the derivative.** Nobody has verified a derivative of a learned operator
  with respect to a design or geometry variable. That survived a determined search.
- Chen & Li (CAD 193:104038) has **zero citations**; the EML 63:102041 §5(3) proposal to learn cut
  element stiffness is three years old, unexecuted, and that group's four 2026 papers all went to the
  conforming shape-function route.

---

## 3. What died tonight — do not re-propose these

Five, four of them mine.

1. **"A thickness range of [0.1, 0.4] is unreachable because the wall would be 0.59 voxels."**
   Wrong question. `0.1–0.4` is **volume fraction**, whose image is τ ∈ [0.1755, 0.6994]; the floor
   never moves and the one-voxel argument lived entirely at the floor. The user corrected me. (The
   voxel arithmetic itself is right — τ = 0.1755 is a 1.03-voxel wall at n=32 and τ = 0.1 would be
   0.59 — it is simply about something nobody proposed.)
2. **"κ(A) gets worse toward thin walls."** R² = 0.030. No relationship. I was about to use this as
   my second argument and the measurement killed it.
3. **"The box-node identity is verified on one cut and one uncut case."** Both were uncut. An uncut
   cell still carries a `cut_plane` field, written `(1,0,0,2)`, and I keyed on its presence instead of
   its value.
4. **Depth fraction `s = d/(1+b)` as the coverage coordinate.** I had argued for it on the grounds
   that every point in `s` is physically reachable. That is true and irrelevant: equal steps in `s`
   are wildly unequal in retained material once the plane tilts — at b = 1, `s = 0.033` is a corner
   wedge of volume 0.0005; at b = 0 it is a slab of volume 0.033. A maximin fill in `(θ, s)` spent
   **274 of 574 attempts** on a band holding no material at all, every failure at `s ≤ 0.122`.
   **Retained volume is the coverage coordinate.** Older docs in this repo still say `(b, s)`.
5. **"The 44 rebuilt labels are dataset-designated test/validation packets never presented to
   anything."** They are 44 of the 76 V2_LABELS rows lacking `MQ_UPPER`, and that pool is 69 train /
   6 holdout / 1 validation.

Also retired, from earlier sessions but still worth not re-proposing: "2 s per cell" was never a
measurement, it was a budget; the GPU numbers are 5.372 s to materialise the dense factor at q=10812
and 9.363 s for the full chain.

---

## 4. What is still open, ranked

**1. Register the 44 labels.** `register_cut_labels.py` is fixed but unrun. ~40 min, pure
verification, no new compute. Until it passes, 120 GB of correct labels are files the trainer refuses.
This gates everything downstream.

**2. Build the remaining 32 `M_q` labels.** Use the existing, proven
`superelement/equi/expand_mq_labels.py --manifest V2_LABELS.json --seats …` — **not**
`build_cut_labels`, because these rows already carry a frozen `R_UPPER` bound by sha256 and a fresh
one in our own basis would break the binding. Note the peak estimate in that module is 12 buffers,
not the 8 the new builder actually uses, so `--device-budget-gib 26` routes most of these to CPU;
either raise the budget or accept CPU. Seat list:
`122 181 196 220 226 240 257 328 366 399 401 415 434 483 575 636 713 725 810 825 849 866 895 995
100191 100194 100199 100200 100204 100206 100213 100232`.

**3. Add the 111 out-of-manifest in-domain cut packets.** Each needs the ~25 s trace stage before
`M_q`. Takes usable cut cells to **296**.

**4. Retrain on 296 and see whether 0.4092 moves.** This is the cheap, decisive answer to "is the
dataset too small", and it costs no teacher time. Held-out median for cut cells is **0.4092** against
a 3 % contract — 14× off. Causes already ruled out: family dimension, n (8 = 28 = 59), τ sampling
density, κ(A*), teacher non-equivariance, teacher τ-non-differentiability, coordinate degeneracy.

**5. Validate the q screen on a geometry we designed.** It is calibrated only against packets the
teacher already made, so its 1.5 % median error is an in-sample-domain figure. Take 3–5 rows from
`DESIGN_300/CASES.json`, run `compile_topology`, and compare. Budget ~22–32 s each at 4 workers on a
busy 16-core box (the filed 15.53 s at
`docs/data/pipeline_20260917/TRACE_REPLAY_0253.json` was 8 workers on an idle one; the two agree).
This is also the first exercise of the generation path at all — and note that for a **cut** cell the
topology alone does not give you q, per section 2.3; you need the trace stage after it.

**6. Only then, production.** See section 6.

**Still open from the previous handover**: the teacher symmetry witness on axis-aligned AFFINE seats
(task 25) was never run.

---

## 5. Code — what is new and what to know about it

Three new modules, all under `superelement/objective/`. Commits `742054d`, `41a1a10`, `a1d3467`.

| module | what | status |
| --- | --- | --- |
| `cut_budget.py` | exact trace identity for uncut cells; calibrated 82 ms screen for cut cells | calibrated on 920 packets, working |
| `fill_cut_family.py` | the sample design: 3-D maximin fill under a trace budget | working; see section 6 |
| `register_cut_labels.py` | makes a label built in our own quotient acceptable to the trainer, with a measured round trip | **fixed, never run** |

Things that will bite:

- `compile_topology` uses a process pool, so **any script calling it needs
  `if __name__ == '__main__':`**. Without it you get `BrokenProcessPool`, which looks exactly like an
  OOM and is not — I blamed the cgroup ceiling for two runs before the pool's stderr showed the
  standard `freeze_support() … Safe importing of main module` text. Filed with the timings at
  `docs/data/cut_labels_20260921/TOPOLOGY_COST.json`.
- `fill_cut_family.py` defines a module-level `offset_for_volume` that raises `NotImplementedError`
  and replaces it via `globals()` from the frozen sampler inside `main()`. **That means importing this
  module as a library and calling `offset_for_volume` fails.** It is deliberate — the frozen inverse
  is the contract and I did not want a second implementation — but it is a wart; if you need the
  library path, import the frozen function directly.
- The private helpers `_shape_values`, `_shape_gradient_squared`, `_direction` are imported *from the
  frozen sampler* on purpose, so the τ construction cannot drift from the one that made the existing
  dataset. Keep it that way.
- `build_cut_labels.py` writes the label digest as `mq_sha256`; the trainer's gate reads `m_sha256`.
  `register_cut_labels` bridges it.
- All 920 packets are **float32** `S_UPPER`, including the 273 in use. That is not an inconsistency:
  the production recipe (`final_packet_recipe_v1`) computed in FP64 and stored the final Schur upper
  triangle in FP32 by design.

---

## 6. The sample design, concretely

`superelement/objective/fill_cut_family.py`. Current output: `/root/autodl-tmp/CLAUDE_FILL_20260921/DESIGN_300/`
(`CASES.json`, `AFFORDABLE_MAP.json`, `UNREACHABLE.json`, `SEARCH_TRACES.json`, `SUMMARY.json`).
Everything else in `CLAUDE_FILL_20260921/` is a superseded probe; ignore it.

**Coverage space is three-dimensional**: `(θ/45°, retained volume, volume fraction)`.

Density has to be its own axis. An earlier version made it a consequence — take the thickest field
whose predicted trace still fits — and that collapsed the axis: of 300 designed cells, **over half
came out pinned at the ceiling and the whole design spanned ρ ∈ [0.214, 0.397] with nothing thin**.
Since τ is the design variable the deployed network must differentiate, a fill that holds τ almost
constant is close to worthless however well it covers angle and depth.

**The budget is a constraint on the box, not a knob inside it**, and the box is not uniformly
affordable: depth and density both push the trace up, so the deep-and-dense corner is unreachable
however the grading is chosen. So affordability is **mapped first**, on a coarse grid with a nearly
ungraded field, and both the fill and the hole metric are confined to that region. Measuring over the
whole box reports a hole no sampling can close — the same mistake as measuring the 2-D coverage over
the degenerate edges.

Mapped at resolution 11 (1089 probes, ~90 s):

| axis | affordable fraction |
| --- | --- |
| angle | 73–75 %, flat — **angle is essentially free** |
| retained volume | 82 % up to v = 0.6, then 73 % / 55 % / 45 % at v = 0.7 / 0.8 / 0.9 |
| density | 100 % up to ρ = 0.25, then 89 % / 78 % / 75 % / 70 % |

That map also caught a defect at **both** ends of the density axis, which had read 0/99. Neither was
the budget: at either end one frozen headroom term collapses, no nonzero scale survives the contract,
and `corners_at_mean` returned nothing — 107 of 522 targets had been failing for that. The cell can
only be **ungraded** there, which is a class the frozen dataset already recognises (`field_class`
calls it `UNIFORM_ANCHOR`), so that is now what is emitted.

Rules, all of them checked by frozen code:

- plane `(1, tanθ, 0)`, θ ∈ [0°, 45°) — the only family `GradedContract` accepts;
- retained volume v ∈ [0.02, 0.98], offset from the frozen `offset_for_volume`; both tails excluded
  because v = 0 is an empty cell and v = 1 an uncut one;
- volume fraction ρ ∈ [0.10, 0.40], τ from the run-time-verified ρ(τ) inverse;
- eight corners by the frozen centred-polynomial construction, validated by the frozen `Thickness`
  (corners in range, span ≤ 0.47, |∇τ| ≤ 0.47) and classified by the frozen `field_class`;
- every row through the frozen `validate_case`;
- **trace budget 19500 predicted**, with deliberate headroom: the screen's p95 error is 6.2 % on cut
  cells, so a target screened at 19500 lands under 20700 against the measured card limit of 22433;
- a target the budget cannot afford is **replaced, not dropped** — maximin picks edges first, edges
  are where targets fail, so dropping them throws away the picks covering the biggest holes (measured:
  27 drops left the hole at 0.1195 when the insertion order had earned 0.0373). `reject` blanks the
  neighbourhood, which is the frozen sampler's own replacement policy applied to a region rather than
  a bin;
- splits assigned to mother fields **before any geometry is generated**, so the held-out set cannot be
  chosen to flatter a result.

### What `DESIGN_300` actually is

300 cases, 84 targets replaced along the way (35 "retains almost no material" at the shallow edge,
49 "trace exceeds budget at this density" in the deep-dense corner), 127 s wall clock including the
affordability map. Affordable fraction of the box after the ungraded-cell fix: **88.15 %** (960 of
1089 probes), up from 73.7 % before it.

| quantity | min | median | max |
| --- | --- | --- | --- |
| angle | 0.000° | 23.625° | 45.000° |
| retained volume | 0.025 | 0.525 | 0.975 |
| volume fraction ρ | **0.100** | **0.258** | **0.400** |
| τ mean | 0.176 | 0.451 | 0.699 |
| τ span (grading) | 0.000 | 0.061 | 0.305 |
| predicted q | 2016 | 12072 | 19488 |

Splits 237 train / 63 holdout. Classes 144 AFFINE / 133 MIXED / 23 UNIFORM_ANCHOR (the 23 are the
density-floor cells, where the frozen contract admits no grading at all).

Coverage over the affordable region, 3-D: largest hole **0.5888 → 0.1436**, a 4.1× improvement. Note
the 3-D metric is not comparable to the 2-D numbers quoted elsewhere — a unit cube with 485 points
cannot do better than ≈0.06 — so read it only against its own baseline.

Occupancy, rows = retained-volume quartile, columns = ρ quartile:

```
 15   8  12  14
 22  11  20  20
 11  22  31  23
 26  26  15   1
```

The lone `1` in the bottom-right is the unaffordable deep-and-dense corner, and it agrees with the
map rather than being a sampling accident.

**Cost if this is executed**, from the predicted q: **87 GB** of `S_UPPER`, **173 GB** of `M_q`, and
another 173 GB if `R_UPPER` is kept beside it — call it **434 GB** landed on the 3.4 TB data disk.
Teacher time is the unknown; nothing in this design has been run. Median q is 12072 against 18033 for
the earlier thickness-pinned design, so this is also ~40 % cheaper per label, purely as a side effect
of covering density properly instead of pinning it high.



---

## 7. Stale assertions still in the repo — fix or ignore, but do not trust

An audit pass over the repo found sixteen places that still assert something tonight overturned.
**These are unverified leads** — the adversarial verification pass that was supposed to confirm them
died on a session limit, so treat them as "check this before believing it", not as findings. I
verified the inventory ones myself (section 2.7); the rest I have not.

| file | what it still says |
| --- | --- |
| `docs/data/cut_fill_20260921/COVERAGE.json:5,15-24,26-43` | the whole coverage study and the +50/+300/+800 fill ladder, in the retired `(b, s)` coordinate |
| `docs/data/cut_fill_20260921/cov2.py`, `fill.py` | same, in their docstrings |
| `docs/data/augcost_20260920/COST_PER_CELL.json:22,25,28,29` | "15–350× faster per entry / 4,100–342,000× more entries" as a headline |
| `docs/data/prior_art_20260920/PRIMARY_SOURCE_VERDICT.md:72,93` | ~830× / ~120× trace-width multipliers, load-bearing for the paper framing at :93 |
| `docs/data/prior_art_20260920/OVERLAP_AUDIT.md` | "2 s/cell" treated as a measurement |
| `docs/data/harvest_20260920/SHORTLIST.md:77` | the literature trace list gives 386 and has no Du & Stechmann (392) entry |
| `docs/data/unused_packets_20260921/FINDINGS.json:6` | 603 cut / 317 box — **confirmed wrong**, see 2.7 |

One provenance gap I should name rather than leave: **the 392 / Du & Stechmann figure is not in this
repo.** It came from tonight's literature sweep, whose output lives only in a task file, and
`SHORTLIST.md` still shows 386 as the maximum. Before that 33× goes into a paper, file the sweep and
read the primary source.

---

## 8. Decisions waiting on the user — do not proceed past these

1. **Is 300 the right count, and is ρ ≤ 0.40 accepted?** The ρ cap costs some coverage (in the 2-D
   study, hole 0.0411 uncapped vs 0.0572 capped) and buys the removal of a band that is 92 % wasted.
   I recommend accepting it. The user proposed it; it has not been formally confirmed against these
   numbers.
2. **Free data before new data.** The user's own concern was *"现在裁切单胞还需要学规则"* — the network
   still has not learned the cut rule. Tasks 1–4 of section 4 double the cut-cell count for zero
   teacher time and answer the dataset-size question directly. Generating 300 new geometries before
   that answer risks designing for the wrong problem.
3. **The production plan for the four CPU boxes** is reusable but not written. The user supplied the
   architecture from the original run and it should be followed rather than reinvented:
   single controller → independent slots on each CPU node → compute on local disk → background
   `rsync -aH` at `nice 19` to the shared buffer → receiver on the GPU box (`.partial`, checksum,
   atomic rename, ACK) → **cleanup of the shared copy and the local original only after the ACK bound
   to that sample**. Slots never wait for delivery. Code to reuse, all in the frozen tree's
   `stage_cutfem_cluster/`: `independent_controller.py`, `independent_worker.py`, `memory_requeue.py`,
   `shared_worker.py`, `remote_destination.py`, `final_retention.py`. Directory convention from the
   original run:
   ```
   CPU source        /root/autodl-tmp/CUTFEM_INGEST_R38/source_independent_6624dc8_20260910
   CPU local active  /root/autodl-tmp/CUTFEM_INGEST_R38/active_independent_20260910
   shared buffer     /autodl-fs/data/CUTFEM_SHARED_R38/buffer_independent_20260910
   GPU final         /root/autodl-tmp/CUTFEM_INGEST_R38/dataset_independent_20260910
   ```
   Measured slot layouts (sample 0328, the no-GP FP64 route): **10 cores × 3 slots = 15.13 cells/hr**
   at 21.38 GiB peak, beating 16×2 (10.15) and a genuine 8×4 (13.76) — fuller CPU is not higher
   throughput. Heavy FP64 factor jobs used an exclusive 32×1 instead. Receiver: 104.62 MB/s single
   stream, 162.92 MB/s with two concurrent transactions at a 0.5 CPU quota, hence one receiver service
   with two in-flight deliveries. The GPU box only receives files here — **no-GPU mode suffices**.
   What must be updated for the new machines: the actual CPU sets, memory, shared mount and target
   directories. Ports, pause state and numeric entry points cannot be copied.
4. **Storage**: `G_UPPER` (20.9 GB) and the two predecessor datasets (141 GB) are plausible deletions
   but have live text references; they need a human yes.
