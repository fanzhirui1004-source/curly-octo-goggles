# Stage 2b verification: Theme T1 (cost and baselines)

**Scope:** I-01, I-02, I-06, I-07, I-11, I-12, I-13, I-27, I-36, I-41, I-51, I-52. I-22 is already decided (removed) and is not covered here.
**Role:** adversarial verifier. For each issue I first tried to refute the reviewers' concern from the manuscript, the evidence files and the code mirror, and marked it CONFIRMED only when that failed.
**Constraints:** Nothing in the manuscript, evidence or code was edited. There was no server access. All compute estimates assume one RTX 5090 and the 16-core host container, which has a 90 GiB memory limit.
**Decided context, taken as given:**
- The cost baseline is host MKL PARDISO, not a GPU exact solver.
- The same-GPU exact-factor study has been deleted.
- Table 6 has whole-lattice host PARDISO results for two four-cell lattices. For the eight-cell lattices it gives only the predicted memory.

Line numbers refer to `docs/paper_p1/MANUSCRIPT_EN.md` unless stated otherwise. The same sentences appear in `latex/main.tex`: Abstract l. 43, §6.10 l. 730/732/790, §7.4 l. 828, Conclusions l. 838. Both files must be changed together.

---

## 0. Findings that change the picture (new evidence found during verification)

**F1. The single-vector host solve times in Table 5 include a large Python-wrapper overhead, not only PARDISO work.**
- `bench_cpu.py` applies the interior factor through `pypardiso.PyPardisoSolver.solve(A, b)` (`src_v2_wip/bench_cpu.py`, class `_S.solve`).
- I inspected pypardiso 0.4.7 (downloaded wheel, `pardiso_wrapper.py`). On every call, `solve()`:
  1. checks whether `A` is the factorised matrix. For `A.nnz > 5e7` it computes a SHA-1 hash of `indices`, `indptr` and `data`; otherwise it runs a full `array_equal` against a stored copy.
  2. converts `indptr` and `indices` to new 1-based int32 arrays (`astype(np.int32) + 1`, two full copies).
- Per call, this touches roughly 12–24 bytes per stored nonzero on a single core.
- From `bench_A3_cells.jsonl`, `K_nnz_upper` is 46.8 M for G1 and 78.0 M for G3. With `mtype=11` the full K_II is passed, which is about 70 M nonzeros for G1 and 140–160 M for G3 and G4. That is 0.9–1.9 GB hashed per call.
- At typical SHA-1 rates (about 1–2 GB/s), this plausibly accounts for about 0.4–1 s of the 0.83 s (G1) to 2.26 s (G4) single-vector host times. It is only a few percent of the 64-vector times.
- The lattice script `lat_direct_cpu.py` calls `_call_pardiso` with `phase=33` directly and does not pay the hash.
- Consequence: the "7.4 to 23 times faster for a single vector" ratio, and the sentence "the host solves are limited by reading the factor from memory" (l. 505), are not established. This strengthens I-36 beyond what R3 wrote.
- Also, pypardiso leaves `iparm(1)=0`, so all MKL defaults apply, including the solve-phase parallelism (iparm(25)). The implied factor-read rate of the Table 6 solves (26–51 GiB factors read in 20–42 s, i.e. about 1.8–3.5 GB/s) is consistent with a sequential forward/backward solve.

**F2. The explicit-S baseline is already blocked. The caption describes it less favourably than it is.**
- `bench_cpu.py` forms S in column blocks of 256 retained unit vectors per PARDISO call (docstring lines 9–10; `blk = 256`).
- The Table 5 caption (l. 496) says "one host solve per retained coordinate". That is inaccurate. R3-02(3)'s "straw man" charge is therefore partly refuted on the blocking.
- The PARDISO Schur-complement feature (iparm(36)) is not used. With it, forming S would very probably take far less than the measured 4–59 min, and the favourable sentence "4 to 59 minutes ... against an estimated 2 to 12 minutes for Ŝ" (l. 505) could flip.

**F3. Unreported A3 runs beyond eight cells, and warm-start data, exist in the repository.**
- `docs/data/newmachine_20260924/p2/lat_scale_554_r2.json` and `lat_scale_554.json` hold a **92-cell** A3 lattice (layout hlat554) with the same BNN:K_PP:q1r preconditioner. Per design iteration:
  - 1,232,085 free retained DOFs, of which 245,265 are private cut-band DOFs;
  - 188 PCG iterations to 1e-6 from a cold start, and 135 with a warm start;
  - 2,115 s (cold) and 1,727 s (warm);
  - peak GPU memory 18.3 GB, host RSS 78.3 GB, 73.6 GB of operator state streamed from the host.
- `lat_scale_554_20*.json` is a 20-cell uncut subset: 108 iterations cold, 81 warm.
- With the preconditioner reused across a design change (`_20re`), iterations rise to 136 and then 201.
- `lat_scale_hlat222_reg.json` (2×2×2, tolerance 1e-4): 95 iterations cold, 56 warm.
- None of these runs has an exact reference, so accuracy at 92 cells is unverified. They bear directly on I-13 and I-41.

**F4. Offline cost can largely be reconstructed from repository records** (details under I-02):
- `evidence/meta_p1.json` training curves;
- `docs/PROGRESS_20260924_CN.md:265,316`;
- `docs/V2_ARMS_SPEED_DEPLOY_20260925_CN.md:130`.

**F5. The fixed-weight evidence already contains larger smoothing budgets and a sixth cell.**
- `evidence/p1_checks.json`, `p1_checks_cpu.json` and `p1_checks_u2.json` contain zero, harmonic and B starts with 8, 16, 32 and 64 steps per stage around Q1(17), for M1, M2, H1, H2, U2 and U1. U1 appears in neither Table 3 nor ST18.
- This supports a matched-work statement for I-07 and I-12 without new runs (table under I-07).

**F6. Existing profiling contradicts the §7.4 attribution of application cost.**
- `p2/ks_benchB.json` (cell hlat222_000, 329k DOFs) gives an fp32 stiffness action of 0.42–0.78 ms at batch 1 and 9.7 ms at batch 64.
- 34 such actions are about 14–27 ms of a roughly 100 ms application (batch 1), and about 330 ms of roughly 1.8 s (batch 64). That is about 15–25%.
- The fp64 action (7.5 ms at batch 1) cannot be the deployed path, because 34 × 7.5 ms exceeds the whole measured application.
- "The learned applications are dominated by the 32 stiffness actions of the correction, not by the network" (l. 558) is therefore probably wrong (I-51).

**F7. Unconditional cost claims that remain in the text:**
- l. 505, §6.10: "Since both preparation and every application are cheaper, the learned route is cheaper for any number of queries per design iteration."
- l. 507: "Since every application is cheaper, this count does not change the comparison with the conventional route."
- l. 558, §7.4: "... so neither preparation can be reused across iterations ..." and "... so this count does not change the comparison."
- l. 558: "A direct solution of the whole lattice ... already takes more time and memory than the learned route at four cells" (no fine-scale iterative competitor).
- l. 568, Conclusions: "... so it is cheaper for any number of queries." This sentence also drops the GPU qualifier that the Abstract has.
- l. 5, Abstract: the cost sentence is hardware-qualified ("on the host cores", "on one GPU"). It still states LU-based ratios and gives no offline cost.
- `RESULTS_A3_CN.md:39` records the authors' own removed same-GPU result: learned preparation 3.9–8.3× faster, applications 1.35–16× slower, break-even about 30–340 retained vectors per cell per design iteration. The lattices need 183–234 applications per cell. R3 and R7 have seen these data. **Any "any number of queries" wording is therefore indefensible in the response letter and must go.**

---

## 1. Issue-by-issue verification

### I-01: GPU-vs-host cost comparison and headline cost claims

**Verdict: CONFIRMED** for the wording. The baseline choice itself is DECIDED and not relitigated.

**Refutation attempt:**
- The Abstract does qualify the hardware (l. 5).
- §6.10 names 16 threads and one RTX 5090 (l. 494).
- Under that hardware pairing, the statement "preparation and every application are cheaper" is literally true of the measured numbers.

**Why the refutation fails:**
1. "Cheaper for any number of queries" is still unconditional in §6.10 (l. 505), §7.4 (l. 558) and the Conclusions (l. 568). The Conclusions sentence also omits the GPU.
2. The claim excludes offline cost (I-02).
3. The host numbers are LU (not Cholesky) and include a per-call wrapper overhead (F1). The 1-vector ratio is the most inflated.
4. The per-cell "stores 4 to 14 times less" compares the learned state *without* K with the host LU factor *without* PARDISO's permanent memory:
   - `bench_A3_cells.jsonl`: G1 `learned_state_GB` 0.74 vs `K_GB` 1.39;
   - `bench_cpu_quiet.json`: G1 `permanent_kB` 1.42 GB not counted.
   With K included and Cholesky, the per-cell memory ratio is plausibly only about 1.5–4×.
5. The lattice-level memory advantage is robust:
   - learned: at most 9.9 GB GPU + 4.9 GB host;
   - Cholesky: 29–32 GiB (four cells) and 66.6–80.7 GiB predicted (eight cells).

**Action class:** (a) text now, plus (c) through the I-36 re-run to get correct numbers.

**Priority:** must.

**Proposed replacement wording:**

*§6.10, l. 505, last sentence.* Replace "Since both preparation and every application are cheaper, the learned route is cheaper for any number of queries per design iteration." with:

> "On this hardware pairing, both the preparation and every application are cheaper on the learned route, so the per-iteration comparison with the host route does not depend on the number of queries. The comparison excludes the one-off cost of generating training data and training the network (Section 6.1), and it is specific to the host direct solver used here."

*§6.10, l. 507, last sentence.* Replace "Since every application is cheaper, this count does not change the comparison with the conventional route." with:

> "Against the host route of Table 5, this count therefore does not change the comparison; against a baseline with cheaper applications it would determine the break-even point."

*§7.4, l. 558.* Replace "Against the conventional host route, the learned route is cheaper in preparation, in memory and in every application (Table 5), so this count does not change the comparison." with:

> "Against the conventional host route of Table 5, the learned route is cheaper in preparation, in memory and in every application, so on that hardware this count does not change the comparison; the offline cost is amortised separately (Section 6.1)."

*Conclusions, l. 568, last sentence.* Two variants.

(i) Without the I-36 re-run:

> "On one GPU, against conventional condensation with a sparse direct solver on 16 host cores, the learned operator prepares a cell 9 to 29 times faster and applies it 2 to 23 times faster; for the eight-cell lattices, a design iteration takes 81 to 110 s and at most 15 GB of device and host memory, whereas the whole-lattice direct factor alone would require 67 to 81 GiB. These figures exclude the one-off offline cost of data generation and training."

(ii) After the re-run: the same sentence with the recomputed ranges. Lead with the lattice memory figure.

*Abstract, l. 5, cost sentence.* Recommended:

> "On one GPU, against sparse direct condensation on 16 host cores, the learned operator prepares a cell [X–Y] times faster, and a design iteration of an eight-cell lattice takes 81–110 s with at most 15 GB of memory, where the whole-lattice direct factor requires 67–81 GiB."

Drop the per-application ratio from the Abstract. It is the number most affected by F1 and I-36.

*§6.10, l. 494.* Add one sentence justifying the baseline, per the DECIDED position:

> "Host sparse direct condensation is the conventional way to condense a cell and is the reference against which learned substructure methods are commonly assessed; GPU sparse direct solvers are not included in this comparison."

**Response-letter paragraph (for R2, R3, R5, R7):**

> We compare against host sparse direct condensation because it is the standard practice that a learned condensed operator replaces, and because learned substructure methods are commonly assessed against full-scale finite element solution on conventional hardware. We agree that a comparison across devices cannot separate algorithm from hardware. We have therefore (i) qualified every cost statement with its hardware, (ii) removed "cheaper for any number of queries", (iii) reconfigured the host baseline (Cholesky, direct solve phase, parallel solve, Schur-complement feature; see I-36) and recomputed all ratios, and (iv) given the offline cost and the break-even counts. The claim we retain as most robust is memory at the lattice level: a design iteration of an eight-cell lattice needs at most 15 GB, whereas the whole-lattice direct factor needs 67–81 GiB.

---

### I-02: Offline cost and break-even not reported

**Verdict: CONFIRMED.** It is partly resolvable from existing records.

**Refutation attempt:** Could "cheaper for any number of queries" hold as a *per-iteration* statement that deliberately excludes one-off costs? Only if it is stated so. It is not, and R2, R3, R6 and R7 all read it as a total-cost claim.

**Reconstructable offline cost** (repository records; the authors should confirm the gaps from the server logs):

| Item | Record | Value |
|---|---|---|
| Direction banks, first 206 geometries (148 train + 20 val + 38 test) | `docs/PROGRESS_20260924_CN.md:219,265` | About 5 h on one RTX 5090 while shared with training (estimated 3.5–4 h exclusive); median 70 s per geometry; 58 GB |
| Direction banks, augmentation cells (443 train_s3 + 80 val_s3 in `meta_p1.json` splits) | `docs/V2_ARMS_SPEED_DEPLOY_20260925_CN.md:130` | About 3 min and about 1 GB per cell on four rented RTX 5090s, i.e. about 26 GPU-h and about 0.5 TB for 523 cells (estimate) |
| Bank content per geometry | `meta_p1.json` → `bank_columns_summary` | 5 classes × (512 train + 64 val + 64 test) = 3,200 labelled directions, with exact sensitivities |
| Step-2 base network (init of the chain) | `PROGRESS_20260924_CN.md:316` | 60,000 steps, 4.6 h |
| P0 (c_oh), v2s | `meta_c_oh.json`; v2s not in the repository | Time not recorded in the repository (P0: 15,000 steps) |
| B (v2L1) | `meta_p1.json` runs.v2L1.log.curve | 40,000 steps, 16,433 s = 4.56 h, peak 28.1 GB |
| A3 (A3_2grid) | runs.A3_2grid.log | Curve starts at step 4,050 after a restart; 9,807 s for 10,950 steps (0.89 s/step), so about 3.7 h for 15,000 steps (extrapolated). "2.7 h" covers only the final 10,950 steps (R6-15 confirmed) |
| C, S8 (ablations, not needed for deployment) | runs.A0_ctrl, A2_tail8 | 5,799 s and 5,472 s (A2b not in the repository) |

**Provisional A3 lineage cost:**
- data about 30 GPU-h (3.5–5 + about 26);
- training about 13 GPU-h (4.6 + 4.6 + 3.7, excluding P0, v2s and development runs);
- total **about 43 GPU-h plus the unrecorded v2s/P0 time**, and roughly 0.5–0.6 TB of banks.
- CPU-h for geometry and CutFEM preprocessing is not recorded. The pipeline ran on 16 host processes in parallel with the GPU stage.

**Provisional break-even** (wall-clock seconds of one GPU against 16 host cores; state explicitly that the resources differ):
- **Per cell, against Table 5 readiness:** 9.9–61 s minus 3.2–7.6 s gives 6.7–54 s saved per cell preparation. 43 GPU-h ≈ 155,000 s, so about **2,900–23,000 cell preparations**. With the I-06 same-device front end (below) the saving shrinks to about 1.2–30 s, and break-even rises to about 5,000–130,000 cell preparations.
- **Per eight-cell design iteration, against the whole-lattice direct Cholesky:**
  - measured lower bound: > 342 s vs 81 s, and > 380 s vs 110 s, so ≥ 261–270 s saved per iteration;
  - estimated full direct time about 555–710 s (§I-36), so about 475–600 s saved;
  - break-even about **260–590 design iterations of an eight-cell lattice**, i.e. roughly 1–6 optimisation runs of 100–300 iterations.
- The offline cost recurs for any change of TPMS family, n, ν or γ (R3-08, R4-08). The reuse domain must be stated next to the claim.

**Action class:** (b) re-analysis of existing logs, plus text.

**Spec:**
1. From the server, collect wall time for every training stage in A3's lineage from `train.log`. This means the step-2 base network, P0/c_oh if on the path, v2s, B and A3 including the 4,050 steps before the restart. Also collect the production-queue logs of `prep_geo.py`/`prod_worker.sh` (per-geometry GPU and CPU seconds, bytes).
2. Count exact solves per geometry by class: interior factorisations, Neumann/pinned and spring solves, sensitivity labels, and adversarial S⁻¹ applications.
3. Produce a supplementary table: stage, device, wall time, GPU-h, CPU-h, storage.

**Cost:** 2–4 h of author time; no compute.

**Priority:** must (consensus 6).

**Proposed wording, §6.1, l. 355.** Replace "A3's continuation took 2.7 h on one NVIDIA GeForce RTX 5090 (peak device memory 29.6 GB)." with:

> "B's 40,000 steps took 4.6 h and A3's 15,000-step continuation about 3.7 h on one NVIDIA GeForce RTX 5090 (2.7 h measured for the final 10,950 steps after a restart; peak device memory 29.6 GB). Generating the direction banks of the 591 training and 100 validation geometries took about [N] GPU-hours and [M] CPU-hours and [S] TB of storage, so that the offline cost of A3, including all training stages of its lineage, was about [T] GPU-hours (Table ST_offline). This cost recurs for another TPMS family, background resolution or material; within the trained family it is amortised after about [K] eight-cell design iterations against the whole-lattice direct solution of Table 6."

---

### I-06: Asymmetric timing conditions (front end, shared host, stopping criteria)

**Verdict: CONFIRMED on four points, PARTLY REFUTED on one.**

**Front-end asymmetry: confirmed.**
- The same assembly code runs on the GPU for the learned route (G1: setup 1.64 + moments 0.87 + assembly 0.79 + topology 2.02 = 5.3 s, `bench_A3_cells.jsonl`) and on the host for the conventional route (18.7 s).
- A GPU front end followed by a host factorisation is entirely feasible. Copying K_II over PCIe costs about 0.1 s per GB.
- Re-analysis from existing data, readiness with the GPU front end for both routes (learned = GPU front end + condensation):

| Cell | Learned (s) | GPU front end + host LU (s) | Ratio | Estimated ratio with Cholesky |
|---|---|---|---|---|
| G1 | 6.0 | 13.0 | 2.2 | about 1.5 |
| G2 | 3.2 | 4.4 | 1.4 | about 1.2 |
| G3 | 6.9 | 25.3 | 3.7 | about 2.3 |
| G4 | 7.6 | 37.8 | 5.0 | about 2.9 |

- The claimed "3.1–8.1" readiness ratio (l. 505) is therefore mostly hardware.
- In Table 6 the host front end is 140–152 s of 242–256 s. With a GPU front end (about 5–11 s) the direct Cholesky total would be about 108 s at four cells, against 38 s learned (about 2.8×).

**Shared host: confirmed.**
- Direct runs: load average 11–37 (Note S7; `lat_direct_*.json` → `loadavg_start` 11.1–37.5).
- Quiet Table 5 run: load 10.3–18.2. First run: 9.6–16.3.
- Partial refutation of R6-02: `RESULTS_A3_CN.md:40` notes that the load average is host-wide, across 128 logical CPUs and other tenants, and says nothing about the 16-core container. The "quiet" run means "no other jobs of ours in the container". It does not mean an idle host.
- **But the caption's "agreed within 10% for G1–G3" is false as written.** G1 64-vector: 4.38 vs 5.81 s (−25%). G3 explicit S: 1,873 vs 1,609 s (+16%). G4 was worse still (factor 47.3 vs 31.3 s, +51%) and is carved out of the claim.

**Stopping criteria: confirmed, but the effect on conclusions is small.**
- The direct solve's cost barely depends on its accuracy.
- The three extra random loads share one factor. Their marginal cost is below half of the 20.6–30.5 s solve phase.
- The learned compliance error (0.010–0.022%) is set by the operator, not by the 1e-6 residual.
- The learned route's recomputed residual is unreported (I-04, T4).

**Eight-cell Cholesky "not run": confirmed as a script choice, not a physical limit.**
- `lat_direct_cpu.py --mem-gb` defaulted to 70. Note S7 says 60 GB was used.
- 2×2×2: 66.6 GiB + about 5–6 GiB overhead ≈ 72.6 GiB, which fits the 90 GiB container.
- 3×3×1: 80.7 + about 6 ≈ 86.7 GiB, which is marginal.

**Sensitivities excluded from direct times: confirmed.** The learned side needs 1.9–4.1 s. A direct-route sensitivity costs one element-moment derivative pass per cell with u known, of similar order. The effect is small.

**Action class:** (a) caption and text, plus (b) re-analysis above, plus (c) re-run (shared with I-36).

**Spec (joint with I-36):**
- Re-run the Table 6 direct Cholesky and LU for both four-cell lattices, three repetitions in an otherwise empty container.
- Run the 2×2×2 Cholesky factorisation (`--mem-gb 85 --mtypes 2`). Attempt the 3×3×1 Cholesky with release of the assembly (`rss_after_release` is 5.6 GB); report predicted if it fails.
- Time a GPU front end + host factorisation variant: assemble with the deployed GPU code, copy K to the host, run PARDISO.

**Cost:**
- 4-cell runs: about 6 min each × 2 lattices × 2 mtypes × 3 repetitions ≈ 1.2 h.
- 8-cell Cholesky: about 10–12 min each (my estimate, below) × 2 ≈ 0.5 h, plus repetitions about 1 h.
- GPU-front-end variant: about 15 min.
- Total about 2.5–3 h on the host, a few minutes on the GPU, and 0.5 day of scripting.

**Priority:** must for the caption and text; should for the re-runs (strongly recommended, piggybacking on I-36).

**Proposed wording, Table 5 caption (l. 496).** Replace "Host timings on an otherwise idle job; a first run with other jobs sharing the host cores agreed within 10% for G1–G3." with:

> "Host timings are single runs in a 16-core container without other jobs of ours; other tenants of the 128-core host were not controlled. An earlier run agreed within 10% for G1–G3 except for G1's 64-vector application (−25%) and G3's explicit S (+16%)."

(Use "median of three runs (range)" once the re-runs exist.)

**§6.10, l. 505.** Replace "Including geometry preprocessing and assembly, a cell is ready for queries after 3.2 to 7.6 s on the learned route and 9.9 to 61 s on the conventional route, a factor of 3.1 to 8.1 that grows with cell size." with:

> "Including geometry preprocessing and assembly, a cell is ready for queries after 3.2 to 7.6 s on the learned route and 9.9 to 61 s on the conventional route. Most of this difference comes from running the method-independent front end on different devices: with the GPU front end for both routes, the conventional route is ready after 4.4 to 38 s, a factor of 1.4 to 5.0."

**§6.10, l. 520.** Make the solver-phase comparison primary:

> "The analysis, factorisation and solution take 91 s with Cholesky, against 30 to 31 s for the learned preconditioner setup and solve; cell setup and assembly, which run on the host for the direct route and on the GPU for the learned route, add 140 to 152 s and 5 s, respectively."

**Response-letter note:** "The load average reported in the benchmark files is that of the whole 128-CPU host, which is shared with other tenants. It does not measure contention inside our 16-core container. We have nevertheless repeated the host runs three times and report medians and ranges."

---

### I-07: Weak graph-harmonic starting-field baseline; scope of "7 to 290 times"

**Verdict: CONFIRMED on scope and weakness. The *direction* of the claim is strongly supported by unused existing data.**

**Refutation attempt using F5.** From the `p1_checks*.json` means of energy excess (%), for B's learned field at 8/Q1(17)/8 against the harmonic start at larger budgets:

| Cell | Class | B 8/Q1/8 | Harmonic 16 | Harmonic 32 | Harmonic 64 | Harmonic 64 / B 8 |
|---|---|---|---|---|---|---|
| M1 | force_c | 0.187 | 8.14 | 4.11 | 1.43 | 7.7 |
| M2 | force_c | 0.0273 | 3.53 | 1.52 | 0.484 | 18 |
| H1 | force_c | 0.0131 | 0.546 | 0.198 | 0.0342 | 2.6 |
| H2 | force_c | 0.0117 | **0.00482** | 0.000614 | 1.2e-5 | 0.001 |
| U1 | force_c | 0.0267 | 2.65 | 1.87 | 1.33 | 50 |
| U2 | force_c | 0.00424 | 0.693 | 0.380 | 0.194 | 46 |
| M1 / M2 / H1 / U1 / U2 | force | — | — | — | — | 5.8 / 3.7 / 2.1 / 11 / 17 |
| H2 | force | 0.0249 | 0.181 | **0.0246** | 5.2e-4 | 0.021 |

- In five of six cells, eight times the smoothing work (128 vs 16 smoothing K-actions per extension) does not let the harmonic start reach B's accuracy.
- In the heavily cut H2 (2.7% retained volume) it does so at 16–32 steps per stage.
- This is a stronger and more honest statement than "quadrupling does not close the gap", but it covers smoothing work only. Network cost in stiffness-action equivalents is not measured (I-51).

**Why the concern stands:**
- The Abstract's "7 to 290" rests on B (not A3), five development cells, one loading class and 32 directions. With A3 (Table 3) the range is 5.4–265.
- Under nodal forces the harmonic start ends *worse than a zero interior* at 32 and 64 steps for U2 and H2 (ST18; `force` rows: U2 1.25 vs 1.19; H2 0.0246 vs 0.0056 at 32; H2 5.2e-4 vs 1.1e-4 at 64). That confirms R1-06: it is a weak comparator. It is scalar and volume-weighted, with no elastic coupling.
- No elastic (vector) harmonic, coarse-first or multilevel non-learned start was tested.

**Action class:**
- (a) must: scope the claim;
- (b) should: add the 16/64 and U1 data (existing) to ST18 and one sentence to §6.5;
- (c) optional: cheap stronger starts.

**Spec for (c):**
- Extend `p1_checks.py` with two starts under the same rigid split and correction:
  1. coarse-first: Q1(17) Galerkin solve before any smoothing;
  2. vector elastic extension: 10–20 AMG-PCG iterations on A = K_II with rigid near-nullspace, e.g. pyamg smoothed aggregation, on the host.
- Optionally add an n = 16 exact extension interpolated to n = 32; it needs a coarse CutFEM build per cell, which is more work.
- Run on the six cells, both classes. Optionally run zero/harmonic/elastic starts on all 80 validation geometries.
- Code: 0.5–1 day. Compute:
  - six cells: < 1 h GPU plus about 1 h CPU for AMG setup (M1 CPU p1_checks took 2,098 s; GPU 164 s);
  - 80 geometries: about 3–4 h GPU.

**Priority:** must for (a); should for (b); optional for (c). (c) becomes should if the authors keep "7 to 290" in the Abstract.

**Proposed wording, Abstract, l. 5.** Replace "Under the same correction budget, the learned starting field is 7 to 290 times more accurate than a harmonic extension." with:

> "On five detailed cells under consistent tractions, the learned starting field leaves 7 to 290 times less energy error than a graph-harmonic extension under the same correction, and a harmonic start given eight times the smoothing work still does not reach it in five of six cells."

Shorter alternative: "Under the same correction, a learned starting field leaves 7 to 290 times less energy error than a graph-harmonic extension on five detailed cells."

**§6.5, l. 419.** After "... still above B's field at the eight-step budget in four of the five cells.", add:

> "With 64 steps per stage, eight times the smoothing work, the harmonic start remains 2.6 to 50 times above B's eight-step result in five of six cells (M1, M2, H1, U1, U2) and reaches it only in the heavily cut H2; under nodal forces the scalar harmonic extension ends above the zero interior at 32 and 64 steps in U2 and H2, so it is a weak but inexpensive comparator (Table ST18)."

Also state "a zero interior leaves 2,400 to 12,000 times" in §6.5, as R6-27 asks.

---

### I-11: No whole-lattice iterative or domain-decomposition baseline

**Verdict: CONFIRMED.** It can be argued partly away, but not safely.

**Refutation attempt:**
- The manuscript frames the cost comparisons as context ("with population, retained-space and workload comparisons providing context", l. 27). The contributions (i)–(iv) contain no speed claim.
- The DECIDED baseline class is host sparse direct.
- The method's object is a reusable condensed operator with an analysable, controllable error. A fine-scale iterative solve delivers no condensed operator.
- A text-only answer is therefore defensible *if* §6.10/§7.4/§8 stop implying superiority over whole-lattice solution in general.

**Why the concern stands:**
1. Table 6 and §7.4 ("A direct solution of the whole lattice ... already takes more time and memory than the learned route", l. 558) enter the whole-lattice comparison. A direct solver is the weakest fine-scale competitor at 1–2 M DOFs.
2. On the manuscript's own logic (§7.4: "the total work required to attain the prescribed response accuracy"), the correction is itself a two-grid method. NICE's lattice solve is an inexact matrix-free substructuring solver: BNN with K_PP⁻¹ and a q1r coarse space (Note S1.1).
3. Four reviewers (R1, R3, R4, R7) ask for it, and a CMAME solver referee will repeat the request.

**Risk estimate (my own, to size the decision):**
- The 2×2×2 full matrix has about 516 M stored nonzeros (upper 258 M): about 6 GB, plus an AMG hierarchy of about 1–2×.
- On 16 host cores a smoothed-aggregation AMG-PCG iteration costs about 1–2 s. Setup is about 30–90 s.
- If AMG needs 50–200 iterations on thin-walled Q2 CutFEM with ghost penalty, the host AMG route is about 100–400 s per load block. That is comparable to or slower than NICE's 81–110 s.
- A GPU AMG (AmgX) would be about 10–20× faster per iteration and could beat NICE. That falls outside the DECIDED host baseline class, but R4 and R7 name GPU GMG/AMG explicitly.
- If AMG struggles with the thin walls and small cuts, which is plausible, that is itself a useful result (R3-04).

**Action class:** Author decision between two options.
- **Option A (recommended): (c)**, a host AMG-PCG baseline, consistent with the DECIDED host policy.
- **Option B: (d)**, disagree, with scoped text.

**Spec (Option A):**
- Reuse `lat_direct_cpu.global_matrix()` for the full CutFEM lattice matrix, including the same scaling, clamp and loads.
- Build the rigid-body near-nullspace (6 vectors) from nodal coordinates of every active DOF. The per-cell node coordinates are available in `teacher.Cell`.
- Solve the three consistent loads with PCG preconditioned by smoothed-aggregation AMG, to the compliance accuracy that NICE attains (relative compliance change < 1e-4, i.e. about 0.01%), and also to 1e-8.
- Candidates: AMGCL (`pyamgcl`, OpenMP, block size 3, near-nullspace), PETSc GAMG via petsc4py (16 MPI ranks), or pyamg (single-threaded; slower, but always installable).
- Report setup, iterations, solve time, peak memory and compliance error for the 2 four-cell and 2 eight-cell lattices. Optionally report the 20-cell uncut subset of hlat554.
- Cost: 1–2 days of implementation. About 1–2 h of host compute for the four lattices; memory ≤ 20 GB each.

**Priority:** should. Treat it as must if Table 6 and the §7.4 whole-lattice sentence stay as they are.

**Option B wording, §7.4, l. 558.** Replace "A direct solution of the whole lattice without condensation already takes more time and memory than the learned route at four cells, and its factor memory grows slightly faster than the number of degrees of freedom (Table 6)." with:

> "A sparse direct solution of the whole lattice without condensation already needs more time and, with a Cholesky factor, about 3.5 times more memory than the learned route at four cells (about 5.5 times at eight cells, predicted), and its factor memory grows slightly faster than the number of degrees of freedom (Table 6). Fine-scale iterative solvers of the whole lattice, such as algebraically preconditioned conjugate gradients or substructuring methods of the BDDC and FETI-DP type, were not compared; the learned route is itself an inexact substructuring solver, and its distinguishing output is a condensed operator per cell with a quantified and reducible error, rather than a faster fine-scale solve."

**Response paragraph (Option B):**

> We agree that the lattice solve of NICE is an inexact substructuring iteration and that fine-scale iterative solvers are its natural competitors for a single analysis. The purpose of the paper is a learned condensed operator whose error can be controlled and followed into compliance and sensitivity; its cost is reported as context against the conventional condensation workflow it replaces. We have removed wording that implied superiority over whole-lattice solution in general, stated explicitly that algebraic multigrid and BDDC/FETI-DP solvers were not compared, and positioned the method relative to inexact-subdomain domain decomposition in §7.

This is risky with R3: expect a second-round request. Option A is cheap enough to prefer.

---

### I-12: No matched-accuracy (Pareto) comparison against non-learned corrections

**Verdict: PARTLY CONFIRMED.** Part is answerable from existing data.

**Refutation attempt:**
- F5 gives, on six cells and two classes, the energy excess of zero, harmonic and B starts at 8, 16, 32 and 64 steps per stage.
- In smoothing-step units this is already an accuracy-vs-work front for the non-network part. The learned start at 8/Q1/8 is non-dominated against the harmonic and zero starts up to 64/Q1/64 in 5 of 6 cells.

**Remaining gaps:**
1. The network's cost in K-action equivalents is not measured (I-51).
2. A3 is evaluated only at its training budget. B is not the deployed predictor.
3. Multiple two-grid cycles (8/Q1/8 repeated) are never shown.
4. There is no interior AMG-PCG comparator.
5. Wall-clock times per budget are missing.
- R7-10's point stands: M1 improves only 2.6× from 8 to 32 steps; H2's harmonic start reaches 0.0006% at 32/Q1/32.

**Action class:** (b) + small (c).

**Spec:**
1. Re-analysis (b): plot energy excess vs smoothing K-actions for zero, harmonic, B and A3 from the `p1_checks*.json` files (six cells, both classes); zero compute.
2. Small (c): extend `p1_checks.py` with:
   - A3 at k = 2, 4, 16, 32;
   - B and A3 with 2 and 3 repeated cycles;
   - an interior PCG with SA-AMG (pyamg; rigid near-nullspace) iterated to the energy excess of A3.
3. Time each budget on G1 or M1 on the GPU with `prof_apply3.py`-style timing. Report the network forward+transpose in K-action equivalents.

**Cost:** code 1 day; GPU about 1 h; CPU AMG about 1–2 h for six cells.

**Priority:** should. Four reviewers ask for it, and R5 flags the risk that "the network only replaces a stronger preconditioner".

**Proposed text (with (b) only), after Table 3 in §6.5:**

> "Measured in smoothing steps, the learned start at eight steps per stage is not matched by the harmonic or zero start at up to 64 steps per stage in five of the six cells examined (Figure S_x); the network's own cost corresponds to [k] stiffness actions (Section 7.4)."

---

### I-13: Scalability beyond eight cells; assembled solver under-specified

**Verdict: CONFIRMED for the manuscript, but evidence to answer it largely exists (F3).**

**Refutation attempt:** Could the paper argue it only claims tens of cells? §6.11's placeholder plans 8 or 27 cells, and R7-07 notes that NICE sits at "fine-scale accurate analysis of tens of cells". This is a positioning answer, but the preconditioner is still unnamed in the main text. The only main-text hint is "preconditioner setup" in Table 6 and "preconditioned iteration" at l. 507.

**Existing evidence (A3, BNN:K_PP:q1r, tolerance 1e-6, three consistent loads, one RTX 5090):**

| Lattice | Cells | Free retained DOFs (private cut-band) | PCG iterations: cold / warm | s per design iteration: cold / warm | Peak GPU / host RSS (GB) | Source |
|---|---|---|---|---|---|---|
| 2×2×1 | 4 | 71–77 k | 114–119 / – | 38 / – | 6.8 / 2.5 | `evidence/learned_hlat221*.json` |
| 2×2×2 | 8 | 139,002 (69,156) | 129 / – | 81 / – | 9.1 / 4.1 | `evidence/d5_off.json` |
| 3×3×1 | 8 | 143,685 (53,085) | 165 / – | 110 / – | 9.9 / 4.9 | `evidence/learned_hlat331.json` |
| hlat554 subset, uncut | 20 | 266,478 (0) | 108 / 81 | 366–434 / 319–367 | 6.2–10.5 / 20–31 | `p2/lat_scale_554_20*.json` |
| hlat554 | 92 | 1,232,085 (245,265) | 188 / 135 | 2,115 / 1,727 | 18.3 / 78.3 (73.6 GB streamed) | `p2/lat_scale_554_r2.json` |

What these runs show:
- Iterations grow mildly: 114–165 at 4–8 cells, 188 at 92 cells.
- Time per cell per iteration roughly doubles from 8 to 92 cells (10–14 s → 23 s), because of host streaming.
- The host-memory ceiling (90 GiB) is reached at about 100 cells: 78 GB RSS at 92 cells.
- K_PP factorisation: 1.2 s at 8 cells, 6.2 s at 92 cells. Coarse AZ: 6.1 s and 85 s.
- Not recorded: accuracy (no reference), the recomputed residual (floor), and a CG-Lanczos condition estimate.

**Action class:**
- (a) must: describe the preconditioner in the main text;
- (b) should: include the 20/92-cell scaling rows from existing JSONs, with explicit "accuracy not verified at this scale";
- (c) optional: a posteriori accuracy check at 92 cells.

**Spec for (c):**
- For the stored learned retained solution of the 92-cell run, compute each cell's exact interior extension at the learned trace. That is one host PARDISO (Cholesky) solve per cell with three RHS, factor freed after each: about 20–40 s per cell, so about 0.5–1 h in total, with < 10 GB peak.
- This gives each cell's energy error δ at the learned trace and a participation-weighted estimate of the compliance error (Eq. 12, as in §6.9).
- Also record the recomputed residual ‖f − K̂U‖/‖f‖ and the CG-Lanczos extreme Ritz values from the stored α, β.
- Requires re-running `lat_scale.py` with a solution dump: about 35 min GPU plus 1 h host.

**Priority:** must for (a); should for (b); optional for (c). (c) is needed only if the 92-cell run is presented as more than a cost and scaling demonstration.

**Proposed wording.** §6.9 (or §6.10 before Table 6), new sentence:

> "The learned lattices are solved by preconditioned conjugate gradients on the free retained coordinates with a balanced two-level preconditioner: the fine action is the inverse of the assembled retained stiffness block K_PP, factorised once per design iteration, and the coarse space consists of trilinear macro-vertex functions multiplied by the six rigid-body modes (Supplementary Note S1.1)."

§6.10, after the Table 6 paragraph:

> "With the same preconditioner, a 92-cell lattice (1.23 million free retained coordinates) needs 188 iterations to a relative residual of 10⁻⁶, against 114 to 165 for four to eight cells; one design iteration takes 35 min on one GPU, 18 GB of device memory and 78 GB of host memory, from which 74 GB of operator state are streamed. Host memory, not iteration growth, limits one GPU to about a hundred cells of this size; accuracy at this scale was not verified against a reference."

---

### I-27: Benchmark reporting incomplete; table layout

**Verdict: CONFIRMED** (minor).

Checks:
- No CPU model, MKL, CUDA, PyTorch or pypardiso version, GPU clock or power appears anywhere in the manuscript, appendices or supplement (grep: none).
- Partly available in the repository: `docs/data/newmachine_20260924/README_CN.md` gives torch 2.8.0+cu128, scipy 1.18.1, cuDSS 0.8, RTX 5090 32 GB, a 16-core container, and a 128-CPU host (`bench_cpu*.json` → `cpu: 128`). The CPU model and MKL version must be read on the server (`lscpu`, `mkl_get_version_string`).
- Host timings are single runs, and the "within 10%" remark is false (I-06). The learned side uses 3 repetitions after 1 warm-up (`bench_cpu.py timed()` for the host; `bench_A3` for the GPU; confirm).
- Table 5 column order switches: "Front end: host / GPU" vs "Condensation: learned / host" vs "Memory: learned / host". Confirmed.
- ST20c phase sums: 5.47 + 0.07 + 3.53 + 27.41 + 1.88 = 38.36 vs 38.65; 3×3×1: 109.70 vs 110.12. Confirmed. The difference is `lattice_setup`/`kpp_assemble` rounding plus unlisted synchronisation (`learned_hlat331.json` phases: `lattice_setup` 0.065 s is not a column).
- Table 5 host memory is the factor only (`factor_kB`), while Table 6 sums permanent + factor. The two tables use different memory definitions (this also belongs to I-28).
- Energy normalisation: not reported. A rough upper bound gives GPU 575 W × 38.6 s ≈ 22 kJ vs 16 host cores at about 100–150 W × 256 s ≈ 26–38 kJ per four-cell iteration. On energy the advantage is at most about 1–2×. Recommend logging `nvidia-smi --query-gpu=power.draw` and RAPL during the re-runs if feasible.

**Action class:** (a), with the re-runs of I-06/I-36 supplying repetitions.

**Spec:** a supplementary benchmark-environment table covering:
- CPU model, sockets and cores allotted, memory, NUMA;
- GPU model, driver, clocks and power limit;
- OS, Python, PyTorch/CUDA, MKL, pypardiso and SciPy versions;
- PARDISO iparm settings;
- threads, repetitions and the warm-up policy.

Also: fix Table 5's column order to "learned / host" throughout and add the ST20c caption note. Cost: 1–2 h plus server `lscpu`.

**Priority:** should.

**Proposed ST20c caption addition:** "Phases do not add up to the total by 0.3–0.4 s, which is lattice setup, retained-block assembly bookkeeping and device synchronisation not listed separately."

---

### I-36: The host sparse-direct baseline is configured unfavourably

**Verdict: CONFIRMED.** The finding is stronger than stated (F1). Point 3 is PARTLY REFUTED (F2); point 4 is PARTLY REFUTED.

**Point by point:**

1. **LU on an SPD matrix: confirmed.**
   - `bench_cpu.py` uses the pypardiso default `mtype=11`. `lat_direct_cpu.py` docstring: "11 = real nonsymmetric LU (pypardiso default, the setting of bench_cpu.py / Table 5)".
   - Table 6's own data give LU/Cholesky ratios of 1.98 × memory (63.6/32.1; 57.4/29.0) and 2.0–2.2 × analysis + factorisation time (139.7/70.6; 130.2/60.3).
   - Expected Table 5 ratios with Cholesky:
     - condensation: 8.5–29 → about 4–14;
     - memory: 4.4–13.5 → about 2.2–6.8 (factor only; lower still if the learned side includes K, I-01 and R3-10);
     - application: see point 2.
2. **Low bandwidth: confirmed, with a second cause.** Besides a possibly sequential solve (iparm(25) under MKL defaults, since pypardiso leaves iparm(1)=0), each single-vector solve pays pypardiso's per-call SHA-1 hash or matrix comparison plus two index copies (F1).
   - My estimate: 30–60% of the 0.18–2.26 s single-vector host times are wrapper overhead.
   - The remainder would roughly halve with Cholesky and shrink further with a parallel solve.
   - The 1-vector ratio (7.4–23×) could plausibly fall to about 2–8×.
   - The 64-vector ratios (2.3–7.6×) are less affected (overhead about 2–10%) but still roughly halve with Cholesky, to about 1.2–4×.
   - "The host solves are limited by reading the factor from memory" (l. 505) is unsupported as written.
3. **Explicit S column by column: partly refuted.** The code already uses 256-column blocks (F2), so the manuscript misdescribes its own baseline. PARDISO's Schur-complement option (iparm(36)) is not used. With it, forming S would likely drop from 4–59 min to about 0.5–5 min (my estimate: one partial factorisation plus a dense update of order n_P²). That could reverse the sentence comparing it with the 2–12 min estimated for Ŝ.
4. **16 of 128 cores: partly refuted.**
   - The container was allotted 16 cores (`README_CN.md`: "16 核"). `os.cpu_count()` = 128 reports the shared host (`RESULTS_A3_CN.md:40`), whose other tenants cannot be controlled. A full-node run is impossible on this machine.
   - One RTX 5090 against 16 cores of a server CPU is a plausible single-workstation pairing.
   - The price asymmetry should be acknowledged (R3-14).
   - On whether more threads would change the conclusions, see below.
5. **No settings reported: confirmed.** There are no iparm settings or MKL version anywhere.

**Would Cholesky with more threads change the conclusions?** The estimate below uses the `lat_direct_*.json` data; the scaling assumptions are mine.
- **Four cells, measured 16-thread Cholesky:**
  - cells 140–152 s + global assembly 11–12 s + analysis 13 s + factorisation 48–57 s + 6-load solve 21–31 s = 242–256 s;
  - learned 38 s: 6.3–6.6×.
- **Eight cells, estimated 16-thread Cholesky:**
  - factor size ×2.07 (2×2×2) and ×2.51 (3×3×1) relative to four cells. For nested dissection in 3D, flops scale roughly as (factor size)^1.5.
  - So numerical factorisation ≈ 57 s × 3.0–3.7 ≈ 170–210 s (2×2×2) and ≈ 57 s × 4.0–4.9 ≈ 230–280 s (3×3×1). The solve is about 43–52 s (sequential).
  - Totals ≈ 555–595 s and 650–710 s, against 81 and 110 s learned: about 6–7×. Memory 66.6/80.7 GiB against 13–15 GB learned (GPU + host).
- **More threads / parallel solve / parallel METIS:**
  - The factorisation typically scales about 1.6× from 16 to 32 threads and about 2–2.5× to 64.
  - The solve could become about 5–10× faster with a parallel solve phase, and analysis about 2–3× faster with iparm(2)=3.
  - The four-cell PARDISO phases (91 s) could plausibly fall to about 30–40 s. That is **parity with the learned solver phases (30–31 s)**.
  - End-to-end the learned route stays about 4–5× faster only because of the host front end (140–152 s), which is itself the I-06 asymmetry.
  - With a GPU front end for both routes and an optimised PARDISO, the four-cell advantage shrinks to about 1.3–2× and the eight-cell advantage to about 2–4×.
- **Memory is unaffected by thread count.** It remains the robust conclusion: 3–6× at the lattice level.
- **Conclusion:** time claims at the lattice level must be phrased as solver-phase and memory comparisons with explicit configuration. The 16-core container makes "more threads" moot on this machine, but the parallel-solve and wrapper effects are configuration issues that must be fixed.

**Action class:** (c) re-run, cheap; plus (a) text.

**Spec:**
- Add a flag to `bench_cpu.py`. Do not change the default, so that the old records remain reproducible.
- Under that flag:
  1. factorise K_II with `mtype=2` on the upper triangle (as `lat_direct_cpu.py` already does);
  2. set `iparm(1)=1` with explicit settings: iparm(2)=3 (parallel METIS) or 2, iparm(25)=1 (parallel solve; check the MKL documentation for the installed version), iparm(8)=0, and no matching/scaling for SPD;
  3. solve through `set_phase(33); _call_pardiso(A, b)` with pre-built 1-based int32 `ia`/`ja` arrays cached once. That bypasses pypardiso's hash, compare and copies; it needs a small subclass because `_call_pardiso` rebuilds `ia`/`ja` on each call;
  4. form S with iparm(36)=1 (Schur complement via `perm` marking the retained DOFs); keep the 256-block path as a fallback;
  5. record the iparm array, the MKL version (`mkl_get_version_string` via ctypes) and three repetitions.
- Keep the LU run once for continuity, and time "hash on / hash off" once on G1 to document F1.
- Run on G1–G4. Recompute Table 5, the Abstract and Conclusions ratios, and the I-02 break-even.

**Cost:** 0.5 day scripting. About 1.5–2 h host compute (G1–G4 × 3 repetitions; Schur formation is the unknown and could take up to about 10 min per cell). No GPU.

**Priority:** **must.** The DECIDED baseline is only credible if it is configured well; R3 will check.

**Proposed wording (after the re-run), §6.10, l. 494:**

> "The conventional route factorises the symmetric positive definite interior block by MKL PARDISO (Cholesky, nested-dissection ordering, parallel solve phase; settings in Table ST_env) on the 16 host cores available to the job, which run on a shared 128-core host, and forms the dense condensed matrix with PARDISO's Schur-complement option; the learned route runs on one RTX 5090."

Remove "the host solves are limited by reading the factor from memory" unless the re-run demonstrates it with a bandwidth measurement.

**Table 5 caption (l. 496):** "Explicit S: dense condensed matrix by one host solve per retained coordinate" → "Explicit S: dense condensed matrix from PARDISO's Schur-complement option", or, if not re-run, "by blocked host solves of 256 retained unit vectors each".

---

### I-41: Reuse and warm starts across design iterations dismissed

**Verdict: CONFIRMED.** Easily fixed with existing data.

**Refutation attempt:**
- The cell preparations genuinely cannot be reused once the thickness changes. The network encoding, correction setup and interior factor all depend on the geometry.
- The existing data also show that reusing the *global preconditioner* across a design step is counterproductive: `lat_scale_554_20re.json` goes 108 → 136 → 201 iterations with reuse, against 81 when rebuilt with a warm start.

**Why the concern stands:**
- The sentence "neither preparation can be reused across iterations" (l. 558) overgeneralises. The assembled solve warm-starts well: 188 → 135 at 92 cells, 108 → 81 at 20 cells, and 95 → 56 at 2×2×2 to 1e-4 (F3). This is already implemented (`lat_scale.py --iters`).
- The conventional route can reuse its symbolic analysis while the active-element pattern is unchanged (13–19 s of the 4-cell direct time), and it can use the previous factor as a preconditioner (approximate reanalysis).

**Action class:** (a) text, plus (b) citing the existing warm-start runs. Optionally add one small run on 2×2×2 at tolerance 1e-6 with `--iters 3` (about 5 min GPU) so that the numbers come from a published configuration.

**Priority:** should.

**Proposed wording, §7.4, l. 558.** Replace "In design optimisation every iteration changes the thickness parameters, and with them the geometry of every cell, so neither preparation can be reused across iterations;" with:

> "In design optimisation every iteration changes the thickness parameters, and with them the geometry of every cell, so each cell's preparation, whether interior factorisation or network encoding and correction setup, is repeated in every iteration. The assembled solve can nevertheless start from the previous iteration's solution, which reduced the conjugate-gradient iterations by 25 to 41% in our lattice runs, whereas reusing the previous preconditioner increased them; the conventional route can likewise reuse its symbolic factorisation while the active-element pattern is unchanged, or precondition with the previous factor."

Then continue: "the relevant comparison is the cost per cell and design iteration ...". §6.11 must state which of these the reported timings use.

---

### I-51: The learned application barely benefits from batching

**Verdict: PARTLY CONFIRMED.**

**Refutation attempt:**
- Per-vector cost does fall with batch width: G1 54 → 33 → 14.4 ms per vector for batches 1/16/64, a 3.7× amortisation (`bench_A3_cells.jsonl`). "Almost linear" overstates it between 16 and 64.
- The roofline argument (about 25–30 ms for 34 K-actions at 1.8 TB/s) assumes a CSR SpMM. The deployed action is an element-wise dense action (`fastidx`, fp32 K_e), whose gather/scatter pattern is not bandwidth-optimal.

**Why the concern stands (F6):**
- Existing profiling (`p2/ks_benchB.json`) shows the fp32 K-action at 0.42 ms (batch 1) to 9.7 ms (batch 64) for a 329k-DOF cell. The 34 K-actions of an application account for only about 15–25% of its time.
- So "dominated by the 32 stiffness actions of the correction, not by the network" (§7.4, l. 558) is probably wrong. The network forward/transpose, gathers and the coarse solves (1.2–4.8 ms each, `coarse_chol`) take the rest.
- The 16-vector time (531 ms, 33 ms per vector) is anomalously close to linear, which suggests a per-column loop somewhere, e.g. the network transpose.

**Action class:** (b) profiling with the existing script, plus (a) text.

**Spec:** run `src_v2_wip/prof_apply3.py` on G1 and G4 with `--B 1`, `6`, `16`, `64` and `--rep 5`. Report per-segment ms for:
- network forward and transpose;
- 2 × 8 smoothing K-actions and their vector operations;
- 2 coarse restrict/solve/prolong;
- the final Fᵀ K F.

Compare with a bandwidth estimate (bytes of K_e, network weights and activations).

**Cost:** about 30 min GPU plus 1 h analysis.

**Priority:** should if §7.4 keeps any attribution sentence; otherwise optional (then delete the sentence).

**Proposed wording, §7.4, l. 558.** Replace "The learned applications are dominated by the 32 stiffness actions of the correction, not by the network; fewer smoothing steps, a cheaper coarse space or fusing the stiffness actions would reduce this cost, at an accuracy cost that Section 6.5 quantifies for the smoothing budget." with:

> "In the present implementation an application takes 25 to 99 ms for one vector and 0.39 to 1.83 s for 64 vectors (Table 5), of which the network accounts for [x]% and the stiffness actions of the correction for [y]%; fusing these operations or reducing the smoothing budget would lower this cost, at an accuracy cost that Section 6.5 quantifies for the smoothing budget."

Without profiling, drop the attribution clause and end at "(Table 5)".

---

### I-52: Retaining the cut band when cut surfaces are traction-free

**Verdict: PARTLY CONFIRMED.** The size concern is real; the cost impact is smaller than implied.

**Evidence:** private (cut-band) DOFs as a share of the free retained system:
- 69,156 / 139,002 = **50%** (2×2×2, `lat_scale_hlat222_reg.json`);
- 53,085 / 143,685 = 37% (3×3×1, `learned_hlat331.json`);
- 245,265 / 1,232,085 = 20% (92 cells).
In the §6.9 lattices the cut surfaces are traction-free: clamp on y = min, loads on y = max.

**Refutation:**
1. Cut-surface loads and supports are part of the stated scope: §2.2, and ST06 reports cut-surface tractions. The specimen boundary is where an implant or a test fixture typically acts, so the retained band keeps the operator valid for such loads without retraining.
2. The dominant PCG cost is the cell application, which scales with the cell's full DOF count, not with the global retained dimension. Condensing the cut band would shorten global vectors and reduce the K_PP factorisation (1.2 s at 8 cells) and the coarse AZ product (6.1 s). It would not remove the per-cell work.
3. Exact elimination of the band at the retained level would need Ŝ_cc⁻¹ per cell: dense, 15k² per cut cell, and expensive. Moving the band into the interior changes the network's input space and requires retraining.
4. What is not known is the effect on PCG iteration counts. The private DOFs are coupled only through their own cell, so they are well handled by the K_PP fine action.

**Action class:** (a) text, with an optional note in §7.

**Priority:** optional (minor, single reviewer).

**Proposed wording**, §7.1 or the new Limitations paragraph (I-05):

> "The retained cut band keeps cut-surface loads and supports available without retraining; in the lattices examined, where cut surfaces are traction-free, it accounts for 37 to 50% of the free retained coordinates. Its effect on cost is limited because each cell application acts on the complete cell regardless of how its retained coordinates are shared; a traction-free variant that treats the band as interior is possible but would change the network's input space."

**Response paragraph:**

> We agree that for traction-free cut surfaces the cut-band coordinates could be eliminated. We retain them so that cut-surface loads and supports, which arise at specimen boundaries, remain available without retraining (Table ST06). In the lattices they are 37–50% of the global retained coordinates, but the cost of the iteration is dominated by the cell applications, which act on complete cells irrespective of the partition of their retained coordinates; the global retained dimension enters mainly the retained-block factorisation and coarse space, which take 1.2 s and 6.1 s of a 110 s iteration for eight cells. We now state this in §7.

---

## 2. One benchmark campaign that covers most of T1

A single host/GPU campaign serves I-01, I-02, I-06, I-11, I-27, I-36 and I-41.

| Step | What | Script | Device | Estimated cost |
|---|---|---|---|---|
| 1 | Environment capture: `lscpu`, MKL version, `nvidia-smi -q`, library versions | shell | – | 15 min |
| 2 | Table 5 host re-run: Cholesky, explicit iparm, direct phase 33, Schur iparm(36), 3 repetitions; one LU and hash-on control on G1 | `bench_cpu.py` + new flag | host | 0.5 d code + 2 h |
| 3 | Table 6 re-run: 4-cell Cholesky and LU × 3 repetitions; 8-cell Cholesky (`--mem-gb 85`); GPU-front-end variant | `lat_direct_cpu.py` | host (+GPU 15 min) | 3 h |
| 4 | Host AMG-PCG whole-lattice baseline, 4 and 8 cells, matched compliance | new, reusing `global_matrix` | host | 1–2 d code + 2 h |
| 5 | Warm-start / 3-iteration run on 2×2×2 to 1e-6 | `lat_scale.py --iters 3` | GPU | 15 min |
| 6 | Application profiling at B = 1/6/16/64 | `prof_apply3.py` | GPU | 30 min |
| 7 | Matched-work fixed-weight study: A3 at other budgets, repeated cycles, coarse-first and elastic starts | `p1_checks.py` extension | GPU + host | 1 d code + 2–3 h |
| 8 | Offline-cost table from server logs | – | – | 2–4 h author time |
| 9 (opt.) | 92-cell a posteriori check (solution dump + per-cell exact extension) | `lat_scale.py` + small script | GPU 35 min + host 1 h | 0.5 d |

**Total:** about 3–4 person-days, about 10–12 h host and about 2–4 h GPU. It fits in one week on the existing machine and needs neither fp16/bf16 nor any exotic hardware.

---

## 3. Summary table

| Issue | Verdict | Action class | Priority | Cost (RTX 5090 + 16-core host) |
|---|---|---|---|---|
| I-01 GPU-vs-host claims | CONFIRMED (wording) | (a) + numbers from I-36 | must | text 2 h |
| I-02 Offline cost, break-even | CONFIRMED | (b) logs + text | must | 2–4 h author, no compute |
| I-06 Timing asymmetry | CONFIRMED (load-average point partly refuted) | (a) + (b) + (c) with I-36 | must (text), should (re-run) | about 3 h host + 0.5 d |
| I-07 Harmonic baseline, "7–290" | CONFIRMED (scope); direction supported by unused data | (a) must, (b) should, (c) optional | must / should | (b) 0; (c) 1 d + 1–4 h GPU |
| I-11 No whole-lattice iterative baseline | CONFIRMED | (c) host AMG-PCG, or (d) with scoped text | should (must if Table 6 claims kept) | 1–2 d + about 2 h host |
| I-12 Pareto vs non-learned | PARTLY CONFIRMED | (b) + small (c) | should | 1 d + about 1 h GPU + 2 h host |
| I-13 Scaling > 8 cells; preconditioner | CONFIRMED in the paper; evidence exists (92 cells) | (a) must, (b) should, (c) optional | must / should | (b) 0; (c) about 2 h |
| I-27 Reporting and layout | CONFIRMED | (a) + environment table | should | 2 h + lscpu |
| I-36 Host PARDISO configuration | CONFIRMED and stronger (wrapper overhead); explicit-S point partly refuted | (c) re-run + (a) | must | 0.5 d + about 2 h host |
| I-41 Reuse and warm starts | CONFIRMED | (a) + (b), optional 15-min run | should | about 0 |
| I-51 Batching | PARTLY CONFIRMED; the §7.4 attribution is probably wrong | (b) profiling + (a) | should (or delete the sentence) | 30 min GPU |
| I-52 Retained cut band | PARTLY CONFIRMED (cost impact small) | (a) / (d) | optional | 0 |

## 4. Top three decisions for the authors

1. **Re-run the DECIDED host baseline properly before any cost number is final (I-36, I-06).**
   - Table 5 uses LU on an SPD matrix.
   - Its single-vector host times include pypardiso's per-call SHA-1 hashing and index copies, estimated at 30–60% of those times.
   - MKL defaults were left in place, the solve phase was probably sequential, and the Schur-complement option was unused. The caption also misdescribes the blocked explicit-S baseline.
   - Expected result: preparation and memory ratios roughly halve, and the single-vector ratio drops the most (plausibly to about 2–8×).
   - The four-cell solver-phase advantage (91 s vs 31 s) could approach parity with an optimised PARDISO.
   - Decide to lead the Abstract and Conclusions with the lattice-level memory advantage and a qualified per-cell preparation ratio, and to delete "cheaper for any number of queries" everywhere (l. 505, 507, 558, 568). The authors' own removed same-GPU data (break-even at 30–340 applications) make that phrase indefensible in the response letter.

2. **Whole-lattice iterative competitor (I-11): run it or scope it out.**
   - A host AMG-PCG with rigid near-nullspace on the full CutFEM lattice is consistent with the DECIDED host policy and costs about 2 days.
   - Its outcome is uncertain in both directions: thin walls and small cuts may defeat AMG, or AMG may match NICE's 81–110 s at eight cells.
   - If the authors decline, §7.4 and Table 6 must stop implying superiority over whole-lattice solution in general, and the paper must position NICE as an inexact substructuring solver whose value is the controllable condensed operator.

3. **Report the offline cost and the scale evidence already in the repository (I-02, I-13, I-41).**
   - About 43 GPU-h plus unrecorded stages for A3's lineage, about 0.5–0.6 TB of banks, and a break-even of about 260–590 eight-cell design iterations against the whole-lattice direct solve.
   - The "2.7 h" in §6.1 must be corrected: it covers the final 10,950 steps only; about 3.7 h in total.
   - The unreported 92-cell A3 run (`p2/lat_scale_554_r2.json`: 188 iterations, 35 min per iteration, 78 GB host) and the warm-start data answer the scaling and reuse questions directly. Decide whether to publish them, with "accuracy not verified at this scale" or with the optional a posteriori check.
   - The preconditioner must be named in the main text in any case.

---
**Coordinator notes:** (1) The 92-cell hlat554 runs were produced in the P2 (speed/scale) line; whether any of them enter P1 is an author decision. Check which run's flags each JSON records before quoting: OPL_COARSE_INV is known to break the lattice PCG convergence and must be off for any quoted number. (2) The F1 pypardiso-overhead share is an estimate; measure before rewording Table 5 around it.
