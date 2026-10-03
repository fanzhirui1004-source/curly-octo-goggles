# FACTS_R1: fact checks for revision round 1 (connectivity, lattice cells, offline cost, container limits)

Date: 2026-09-28. Server: the RTX 5090 container `autodl-container-473f4db77f-254db1e6`, accessed read-only through `rr.sh`. The only files written were scratch scripts and outputs under `/root/_ks/`: `conn_check.py`, `q1.py`–`q3.py`, `tim.py`, `agg*.py`, `du.txt` and `du2.txt`. Local copies of the scripts are in the session scratchpad under `fk/`. No GPU was used and no process was signalled. All CPU work ran under `nice 19`.

Abbreviations: B = v2L1 (base network), C = A0_ctrl (Uncorrected), S8 = A2_tail8, A2b = A2b_tail8 (Smoothing-trained), A3 = A3_2grid (NICE), P0 = c_oh.

---

## 1. Face-connectivity rejection (manuscript §2.2)

### Where the check lives

The check is not in a geometry generator. It sits in the body builder `fast_prep4.py`, which is mirrored at `docs/data/newmachine_20260924/src_v2_wip/fast_prep4.py:56-69` and is identical in `src_s0/`. The builder takes the active background cells and forms the graph of 6-neighbour face adjacencies. It then calls `scipy.sparse.csgraph.connected_components` and raises `ValueError('ACTIVE_CELLS_NOT_FACE_CONNECTED:<ncomp>')` unless `ncomp == 1`. A cell without `PREP.json` has no body and cannot enter a bank.

The builder also raises `BOX_NODE_NOT_IN_BODY`. The "carrying retained box-face coordinates" part is not a separate test: `box_nodes` may be non-empty on only some faces (see H2 below).

### Generator and builder used for each family

| Family | Geometry generator (packet writer) | Seed | Body builder | Evidence |
|---|---|---|---|---|
| Legacy `fresh_train_00xx`, `fresh_development_00xx`: 148 train (incl. 15 `_rot*` views of 0013/0021 cells), 20 legacy val, 38 test | frozen registered generator `fresh_gp.families.generate` (packets in `CUTFEM_FRESH_GP_20260921/packets`) | 2026092101 (registered) | `fast_prep4.py` (Step 0) | every `S0/<case>/PREP.json` has `"active_cell_components": 1` |
| New `fresh_train_2000–2451` (443 in the pool), `fresh_val_2000–2079` (80) | `gen_new.py --mode indep` (same frozen generator, new seed) | 2026092602 (`provenance` in `S3/packets/<case>/FRESH_CONTEXT.json`) | `fast_prep4.py` via `prod_worker.sh`, on 4 rented RTX 5090 machines | `S3/logs/<case>.body.log`; `S0/<case>` is a symlink to `S3/body/<case>` |
| Neighbours `<case>_nb{px,py,pz,mx,my,mz}` (incl. the 16 pair neighbours `fresh_val_20{00..06,10}_nbm{x,y}`) | `gen_new.py --make-neighbour`: continuous thickness, shared-face corners copied; if the body fails, `--fix-neighbour` sets the far corners to the shared ones and rebuilds | 2026092602 | `fast_prep4.py` | 21 `NEIGHBOUR_FIXED` events in `/autodl-fs/data/OPL_QUEUE/events.log`; 18 ok, 3 still failing, and those 3 cells were BODY_FAIL and are not used |
| Lattices `hlat222_*`, `hlat331_*` | `gen_hlat.py` | 0 (vertex jitter) | `fast_prep4.py` (`S1/V2/hlat.status`, 2026-09-27 02:26–02:31) | `S4/body/<case>/PREP.json` has `active_cell_components: 1` |

### The check was active in production

It rejected 10 of the 1,109 produced cells:

- 5 with `ACTIVE_CELLS_NOT_FACE_CONNECTED:2`: `fresh_train_2317_d1_v0`, `2498_d0_v0`, `2597_d1_v0`, `2909_d1_v0`, `2965_d1_v0`.
- 5 with `ACTIVE_CELLS_NOT_FACE_CONNECTED:0` (no active cell): `2213_d1_v0`, `2357_d1_v0`, `2453_d1_v0`, `2525_d1_v0`, `2677_d1_v0`.

Source: `S3/logs/*.body.log`. A further 3 cells failed with `GRADED_THICKNESS_RANGE` and 3 with `PREP_FAIL`. None of the 16 failed cells is in any split.

### Independent re-check on the stored active-cell sets

I ran `/root/_ks/conn_check.py` on every stored body. It reads each `CELL_INDICES.npy` in `S0/`, `S3/body/` and `S4/body/`, builds the face-adjacency graph and counts components.

| Set | n | 1 component | Fail | Active cells (min–max) |
|---|---:|---:|---:|---|
| Training pool (SPLIT_ARMS `train`) | 591 | 591 | 0 | 150–16,256 |
| Validation `val_s3` (fresh_val_2000–2079) | 80 | 80 | 0 | 68–16,016 |
| Legacy validation (selection only) | 20 | 20 | 0 | 894–14,560 |
| Legacy test (development families, not used in paper) | 38 | 38 | 0 | 604–15,835 |
| All neighbour bodies on disk | 1,434 | 1,434 | 0 | 7,138–15,996 |
| Pair neighbours (8 targets × `nbmx`, `nbmy`) | 16 | 16 | 0 | 9,717–14,520 |
| All lattice bodies in `S4/body` (incl. the 16 used) | 116 | 116 | 0 | 2,818–12,913 |

- The `S0` and `S3/body` copies are identical: they are symlinks.
- Every `PREP.json` also records `active_cell_components = 1`. No older generator without the check was used for any family in the paper.

### H2 (fresh_val_2010_d0_v0)

**The cell itself passes.** It has 308 active cells in one face-connected component (`PREP.json`: `active_cell_components 1`, 3,858 nodes, 1,008 box nodes, 3,483 port nodes). Its cut has retained macro volume 0.0267.

**The ill-posed H2/y pair is explained by the box faces, not by cell connectivity.** H2's 1,008 retained box nodes all lie on x = 0. It has **0 box nodes and 0 active cells touching y = 0** (and none on x = 1, y = 1, z = 0 or z = 1).

- **Configuration y:** the neighbour `fresh_val_2010_d0_v0_nbmy` is translated by (0, −1, 0). It has 1,136 box nodes on its face y = 1, but the target shares **0** of them. The target is therefore not attached to the clamped neighbour, which explains the negative energy shares and the PCG runs stopping at 400 iterations in `gate_*_fresh_val_2010_d0_v0_y.json`.
- **Configuration x:** 1,008 of 1,008 nodes are shared, so the pair is well posed. No run of it is archived.

### Near-mechanisms and the pool size

A face-connected cell can still be a near-mechanism. `fresh_train_2013_d1_v0` has 198 active cells in 1 component, so it passes the check. `ingest_s3.py` nevertheless excluded it (`excluded_near_mechanism`), because the median Rayleigh quotient of its `force_c` bank lies below `--mech-floor`, about 1000× softer than typical heavy cuts ("hinge-like sliver").

This cell was in B's pool: `v2L1.log` shows `VIEW` records for it. B's 305 therefore consist of:

- the 148 legacy geometries;
- 157 new cells, including `2013_d1_v0`.

B visited all 305. The 591-pool contains **304** of them plus **287** later cells (`2159_d1_v2` is in the 591 list but was not in B's pool). The manuscript sentence "the base network's 305 and 286 produced later" is off by one.

### Suggested wording

**§2.2.** "Geometry preparation enforces this: a cell is accepted only if its active background elements form a single face-connected set (10 of the 1,109 cells generated for the expansion were rejected by this test), and one further cell that passed it but behaved as a near-mechanism (hinge-like sliver; median Rayleigh quotient of its consistent-traction bank about three orders below typical heavily cut cells) was removed from the training pool. Face-connectivity does not imply material on every box face: H2, for example, carries retained coordinates only on x = 0."

**§6.6 / ST09 (H2/y).** "A fifteenth configuration, H2/y, is ill-posed: H2 has no material on its face y = 0, so the target is not attached to the neighbour; it is excluded for all predictors. H2/x is well posed but was not run."

**§6.1.** "The continuations draw from a pool of 591 geometries: 304 of the base network's 305 (one near-mechanism cell was removed) and 287 produced later."

---

## 2. Lattice cells (§6.9)

**Generator.** `gen_hlat.py`: `S4/packets/hlat*/FRESH_CONTEXT.json` carries `provenance.writer = 'OPL gen_hlat.py'` and the full args. It uses one continuous vertex field, and each cell takes its 8 corner values, so neighbours share face corners exactly:

τ(X) = t0 + gx(X/nx − 1/2) + gy·sin(πY/ny) + gz·cos(πZ/nz) + jitter·U(−1, 1)

- Parameters: t0 = 0.38, gx = 0.12, gy = 0.06, gz = 0.05, jitter 0.02, `default_rng(0)`.
- Cut: one vertical plane with normal (cos 20°, sin 20°, 0), global offset b_global. Cells with local offset ≤ 0, or with retained volume < `min_vol` 0.15, are dropped.
- Contract: corners must lie in (0.1755, 0.6983), with per-cell span ≤ 0.3.
- The template context, for n, material and gp, is `S3/packets/fresh_val_2003_d1_v1`. Only the case fields are replaced.

**Layouts** (`S4/hlat222.json`, `S4/hlat331.json`):

| Lattice | args | b_global | τ range (all cell corners) | Cells (position: kind, retained volume, τ min–max) |
|---|---|---|---|---|
| hlat222 (2×2×2) | b0 0.75, θ 20°, seed 0 | 1.6897 | **0.2516–0.5350** | 000 FULL 0.311–0.504; 001 FULL 0.252–0.421; 010 FULL 0.329–0.504; 011 FULL 0.272–0.421; 100 CUT 0.616, 0.393–0.535; 101 CUT 0.616, 0.310–0.507; 110 CUT 0.252, 0.395–0.535; 111 CUT 0.252, 0.332–0.507 |
| hlat331 (3×3×1) | b0 0.70, θ 20°, seed 0 | 2.5794 | **0.2608–0.5618** | 000 FULL 0.261–0.475; 010 FULL 0.303–0.476; 020 FULL 0.279–0.476; 100 FULL 0.327–0.494; 110 FULL 0.342–0.494; 120 CUT 0.835, 0.297–0.483; 200 CUT 0.563, 0.352–0.562; 210 CUT 0.199, 0.387–0.562; **(2,2,0) empty** (cut removes it) |

Position (i, j, k) is the cell origin in units of the cell size. The cut side is +x/+y.

**Checks against the manuscript:**

- "Four of them cut with retained volumes of 62% and 25%": confirmed (0.616 × 2, 0.252 × 2).
- "Corner parameters of the block range from 0.25 to 0.54": confirmed.
- Layer: 8 cells, 3 cut (84%, 56%, 20%), one corner position (2,2,0) empty: confirmed.

**Novelty.**

- No `hlat*` id is in any split: SPLIT_ARMS and SPLIT_V3 `train`, `val`, `val_s3` and `test`.
- No `hlat*` bank exists in `S2/data*` or `S3/data*`.
- The lattice packets were written 2026-09-27 02:25:42 and the bodies 02:26–02:31. That is after A3's `best.pt` (2026-09-27 00:32:38), B's `best.pt` (09-26 09:53) and the final `SPLIT_ARMS.json` (09-26 12:51).
- Geometric distinctness: I compared each lattice cell's corner vector with all 6,493 generated packets (legacy plus expansion, incl. neighbours) under all 48 cube symmetries. The smallest max-abs corner difference is **0.019** (`hlat331_110` vs `fresh_val_2049_full`). For the block it is ≥ 0.026. No lattice cell reproduces any training, validation or neighbour thickness field.

**Suggested wording (§6.9).** "The sixteen cells were generated for these examples after all networks had been trained, from one vertex field τ(X) = 0.38 + 0.12(X/n_x − 1/2) + 0.06 sin(πY/n_y) + 0.05 cos(πZ/n_z) with a seeded ±0.02 vertex perturbation and one vertical cut at ϑ = 20° (Supplementary Note …). Their corner parameters range from 0.25 to 0.54 in the block and from 0.26 to 0.56 in the layer, and differ from those of every generated training, validation and neighbour cell, in any cube orientation, by at least 0.019 in some corner; they entered neither training nor weight selection."

---

## 3. Offline cost (§6.1 "[PENDING: offline-cost table]")

### Sources

- **Per-geometry bank timings:**
  - legacy: `S2/data/<case>/DONE.json:total_seconds` (prep_geo) and `S2/data_v2/<case>/DONE2.json:total_seconds2` (prep_geo2);
  - new cells: `S3/logs/<case>.status.json` (`body_s`, `prep_geo_s`, `prep_geo2_s`; one cell at a time per machine, max concurrency 1 from `events.log`), plus the `REPAIR_END` lines in `/autodl-fs/data/OPL_QUEUE/events.log`.
- **Training:** each run's own `S1/V2/<run>/train.log`, which records STEP `s`/`step_s`/`swap_s`, EVAL `eval_s` (cumulative) and DONE `seconds`.
- **Storage:** `du -s --block-size=1` of the used case directories, without following symlinks (`/root/_ks/du2.txt`).

### A3 (NICE) restart

The top-level `S1/V2/A3_2grid.log` was overwritten by the resume, but **the run's own `S1/V2/A3_2grid/train.log` holds both segments.** Their times are measured, so no extrapolation is needed.

- **Timeline:**
  - Launch at 18:59 crashed at start (coarse Cholesky not PD; about 1 min).
  - Relaunch ran to step 5,600 and hit an OOM in `coarse_setup`, logged "A3_2grid OOM at step 5600 … resuming A3 from ckpt" (`chain_arms.status`).
  - Resume from `ckpt.pt` at step 4,000 (`RESUME` event: `step 4000, rng_restored true`) ran to 15,000, with `TRAINED` at 00:32:39.
- **Segment 1:** s = 2,687 s at step 4,000 and s = 3,642 s at step 5,600. So 955 s of steps 4,000–5,600 were redone.
- **Segment 2:** s = 9,807 s at step 15,000. The 7,500 selection eval took 1,908 s and the final eval 4,871 s, so the segment ended with DONE = 14,678 s.
- **Reconstructed uninterrupted run** = 2,687 + 14,678 = 17,365 s (4.82 h), including both selection evals.
  - To the last update (the same metric as B's "4.6 h") it is 2,687 + 9,807 = 12,494 s (**3.47 h**).
  - Excluding all evals it is 2,687 + 7,899 = 10,586 s (2.94 h).
- **GPU time actually consumed:** 3,642 + 14,678 = 18,320 s (5.09 h).
- The "2.7 h measured after a restart" is the 9,807 s of segment 2. That covers 11,000 updates (4,000 → 15,000), not 10,950, and includes one selection eval.
- The earlier estimates of about 3.7 h (VERIFY_T1: 9,807 × 15,000 / 10,950) and 3.8 h (VERIFY_T6) were extrapolations. The measured figure on B's metric is **3.5 h**.

### Offline cost table

Hardware:

- **"main":** the container's RTX 5090 32 GB with 16-CPU quota. It was shared at times with other jobs (the legacy bank generation overlapped training), so its times are wall-clock rather than exclusive.
- **"rented":** 4 rented RTX 5090 machines, one cell at a time each.

| Item | GPU-h | CPU stage | Hardware | Source | Status |
|---|---:|---|---|---|---|
| Banks, 5 classes × (512/64/64) directions with exact sensitivities, 168 legacy train+val geometries (prep_geo) | 3.05 | incl. | main (shared) | Σ DONE.json `total_seconds` | measured |
| Extra classes force_c/face_c/support_k/glued, same 168 (prep_geo2) | 6.81 | incl. | main | Σ DONE2.json `total_seconds2` | measured |
| New 523 cells (443 train + 80 val): body build (fast_prep4, 8 procs, incl. neighbours) | – | 2.65 h wall (≤ 21 CPU-h) | rented | Σ status.json `body_s` | measured |
| New 523 cells: prep_geo | 7.89 | | rented | Σ `prep_geo_s` | measured |
| New 523 cells: prep_geo2 | 21.70 | | rented | Σ `prep_geo2_s` | measured |
| Glued-class repair of 44 of the 523 | 2.42 | | rented | `REPAIR_END` lines, events.log | measured |
| **Data for the 691 train+val geometries** | **41.9** | + 2.65 h | | | measured (≈ 3.6 min per geometry) |
| (Legacy test 38, not used in paper: 0.74 + 1.48) | (2.2) | | main | | measured |
| (Whole expansion run incl. unused cells: 1,109 cells) | (68.1 machine-h; about 19.5 h wall on 4 machines) | | rented | status.json, events.log | measured |
| Step-2 base network `s2_full`, 60k updates (init of P0) | 4.59 | | main | train.log DONE 16,537 s | measured |
| P0 (`c_oh`), 15k | 1.54 | | main | DONE 5,528 s | measured |
| `v2s`, 5k (init of B) | 1.39 | | main | DONE 5,004 s | measured |
| **B (`v2L1`), 40k** | **4.96** (4.57 to last update; 3.81 excl. evals) | | main | DONE 17,842 s; eval_s 4,124 s | measured; peak 28.1 GiB |
| **NICE (`A3_2grid`), 15k** | **4.82** reconstructed (3.47 to last update; 2.94 excl. evals); 5.09 consumed | | main | both segments of `A3_2grid/train.log` | reconstructed from measured segments; peak 29.6 GiB |
| Uncorrected (`A0_ctrl`), 15k | 2.01 (1.61; 1.35) | | main | DONE 7,234 s | measured |
| S8 (`A2_tail8`), 15k | 2.06 (1.52; 1.23) | | main | DONE 7,398 s | measured |
| Smoothing-trained (`A2b_tail8`), 15k | 3.28 (2.59; 2.19) | | main | DONE 11,801 s | measured |
| **NICE lineage** (s2_full + P0 + v2s + B + A3 consumed) | **17.6** | | | | measured sum |
| **Total, NICE incl. its data** | **≈ 59.5** | + about 3 h CPU-stage wall | | | |

Not included:

- development runs (`mgno*`, `v2L2`, `A1_sens3`, `A3G_ft`);
- the time to form the dense exact condensed matrices of the pair and lattice cells. That time is not logged separately. The exact lattice solves themselves take 60 s (block) and 86 s (layer) (`lat_hetero{222,331}_A3.json`).

Evaluation-time references are part of the banks (the val/test split of each bank).

### Storage

Measured with `du`, in decimal GB:

| Item | Size |
|---|---|
| Banks of the 691 train+val geometries | **1.05 TB** (0.96 TiB): legacy 168 = 58.1 GB (`S2/data`) + 196.4 GB (`S2/data_v2`); new 523 = 178.5 GB (`S3/data`) + 619.0 GB (`S3/data_v2`). About 1.5 GB per geometry |
| All bank directories on disk (incl. test and unused expansion cells) | 1.98 TB |
| Regenerable training slot cache `S2/slots` | 1.33 TB |
| Bodies: `S0` (legacy bodies + ghost-penalty caches) and `S3/body` | 284 GB and 90 GB |

### Suggested wording (§6.1)

"The base network's 40,000 updates took 4.6 h and NICE's 15,000 updates 3.5 h on one NVIDIA GeForce RTX 5090, measured to the last update including the intermediate selection evaluations (5.0 h and 4.8 h including all selection evaluations; NICE's run was interrupted at update 5,600 and resumed from its checkpoint at update 4,000, and the time of both segments is logged; peak device memory 29.6 GiB). Generating the direction banks with exact sensitivities for the 691 training and validation geometries took about 42 GPU-hours, about 3.6 min per geometry, and the banks occupy 1.05 TB; with the three earlier training stages of its lineage, NICE required about 60 GPU-hours in total (Table ST01). This cost recurs for another cell family, background resolution or material."

Table ST01 = the table above. Mark rows "measured" or "reconstructed" and state that main-GPU times are wall-clock on a GPU not guaranteed exclusive.

---

## 4. Container limits (the "90 GB")

These were read on 2026-09-28 in the same container that ran `bench_cpu*.json` and `lat_direct_*` (`S1/V2`). It is cgroup v2 (`0::/`).

| Item | Value |
|---|---|
| `/sys/fs/cgroup/memory.max` | 96,636,764,160 B = **exactly 90 GiB** (96.6 GB) |
| `memory.high` | 94,489,280,512 B = 88 GiB (reclaim/throttling starts here) |
| `memory.swap.max` | max (host has no swap: `Swap: 0`) |
| `cpu.max` | `1600000 100000` → CPU-time quota of **16 CPUs** |
| `cpuset.cpus.effective` | 0–127 (the quota is not pinned to 16 cores) |
| `nproc` / `getconf _NPROCESSORS_ONLN` | 16 / 128 |
| Host | Intel Xeon Gold 6459C, 128 logical CPUs, 754 GiB RAM; `nvidia-smi` RTX 5090, 32,607 MiB |

The "90 GB" is therefore **90 GiB** (cgroup hard limit), with throttling from 88 GiB. For Table 6 this means:

- **2×2×2 Cholesky:** 66.6 GiB predicted factor + 5.2–6.4 GiB process overhead (as in the four-cell runs) ≈ 72–73 GiB. That should fit.
- **3×3×1 Cholesky:** 80.7 + about 6 ≈ 87 GiB. This is at the 88 GiB throttle point.
- **LU (132–160 GiB):** exceeds the limit.

"16 host cores" is a 16-CPU CFS quota on a shared 128-thread host, not 16 dedicated cores.

### Suggested wording

**§6.10.** "The conventional route runs MKL PARDISO within a quota of 16 CPUs on a shared 128-thread host (Intel Xeon Gold 6459C)"

**Table 6 / text.** "the eight-cell LU factorisations exceed the 90 GiB memory limit of the container (cgroup hard limit; reclaim starts at 88 GiB)"

If the eight-cell Cholesky is not re-run, say that the 2×2×2 Cholesky (≈ 73 GiB predicted peak) would fit and was not run for time or threshold reasons (the `--mem-gb 60` threshold), not because of the limit.

---

## Side findings

- **Parameter count.** Every training log's `MODEL` event (`v2L1.log`, `A0_ctrl.log`, `A3_2grid.log`) reports `"params": 603409`. `meta_p1.json` and the manuscript give 603,464. The difference of 55 should be explained or corrected, for example by checking how `meta_p1.py` counts (buffers or bounds tensors).
- **Training-time wording.** "2.7 h for the final 10,950 updates" should read "11,000 updates": resume at 4,000, end at 15,000.
