# Stage 2b verification: T9 (fact-check corrections) and T8 (structure, length, presentation)

**Scope:** I-28, I-29, I-46, I-47, I-57, I-58, I-59 (T9); I-21, I-37, I-45, I-53, I-54, I-55, I-56 (T8).
**Basis:** repository state at 693de8e (after the D-study removal in b7dc534). Every T9 number was recomputed from `docs/paper_p1/evidence/` with read-only scratch scripts. Memory units were traced to the producing code in `docs/data/newmachine_20260924/src_v2_wip/` (`bench_cpu.py`, `bench_deploy.py`, `lat_direct_cpu.py`, `lat_scale.py`, `train3.py`).
**Constraints:** No manuscript, evidence or code file was changed, and nothing was committed.

---

## 0. Summary

| Issue | Verdict | Correct value / edit | Priority |
|---|---|---|---|
| I-28 | **ERROR CONFIRMED**, and wider than R6 reported: Tables 5 and 6 use **four** conventions | Use GiB everywhere. Host factors become 5.30 / 1.05 / 11.8 / 17.7 GiB, the ratio 4.2–12.9 and the Abstract "4 to 13". The content definition is also open (K, PARDISO permanent storage); see §1.1 | **Must fix (Abstract)** |
| I-29 | Numbers CONFIRMED; the display-rounding convention is DECIDED | Caption note, plus "about 9 to 29" in the Abstract and §8 | Must fix (wording) |
| I-46 | **ERROR CONFIRMED** | "nine configurations common to C, S8, A2b, B+W and A3 (B was evaluated on seven of them)" | Minor |
| I-47 | **CONFIRMED.** §6.9 is correct as an operator-accuracy value. Table 6 violates the Eq. (12) bound, so it contains algebraic error | Relabel the Table 6 values as timed-run values at 10⁻⁶. Residual "at most 1.3×10⁻¹⁰". "about 4–5%". Qualify §8. **New:** the "0.03–0.12%" and "0.03–0.08%" cell ranges are random-load ranges, not consistent-load ranges | Should fix |
| I-57 | **ERROR CONFIRMED** | "9×10⁻⁹" (symmetry) and "5×10⁻⁹" (work–energy) | Minor |
| I-58 | All five items **CONFIRMED** | Five sentence replacements (§3.5) | Minor |
| I-59 | **CONFIRMED UNSUPPORTED** (partial reconstruction only) | Archive the layer mask; "maxima over the six loads" | Minor |
| I-21 | Confirmed; S8 is the orange square in Fig. 10 (checked on the image) | Label map (§4.2); replace S8 with A2b in Table 4 and Fig. 10; add a nomenclature table | Major (presentation) |
| I-37 | Confirmed: main text 12,900 words without captions, 14,450 with captions | Per-section cut plan (§4.3) | Major |
| I-45 | Needs author decision | Three options (§4.4) | Minor |
| I-53 | Confirmed; the data for NICE and NICE-post exist (96 + 96 points) | Re-plot Fig. 10; drop Fig. 9(c,d); fold Table 4 | Minor |
| I-54 | Confirmed: App. C, I, J.1, J.2, J.8 and J.9 are uncited, and there are about 30 ST citations | Inline J.4/J.5; move the B.8 sentence; move the uncited appendices | Minor |
| I-55 | Confirmed: 369 words (359 before the placeholder) | Draft Abstract of 240 words + placeholder (§4.6) | Major (desk) |
| I-56 | Confirmed | Move §3.3 ¶3 to a new §5.4 (§4.7) | Minor |
| **N-1 (new)** | Error | §6.9 cell energy-error ranges refer to the random loads | Should fix |
| **N-2 (new)** | Error | Table 5 explicit-S memory for G2: 2.7 → **2.6** (GiB) | Minor |
| **N-3 (new)** | Undisclosed exclusion | The H2/y two-cell records exist for C, S8, A2b and A3 but are degenerate (NaN, PCG at maxit) and are never mentioned | Minor |
| **N-4 (new)** | Unsupported | The "90 GB available to this job" has no archived source, and its unit is unknown | Minor |

**Word counts.** The main text of Sections 1–8 has **12,897 words**, excluding the title, Abstract, tables, display equations, figure/table captions and references. Captions add **1,549**, so Sections 1–8 with captions come to **14,446** (R5 said about 14,650). The whole file has 16,777 words. The **Abstract has 369 words** (359 before the placeholder). The draft Abstract in §4.6 has **240 words plus a 9-word placeholder**.

**Deleted-ST08a facts (b7dc534 check):**
- **K storage.** The retained text never mentions it, but the retained evidence supports it: `bench_A3_cells.jsonl` → `K_GB` = 1.394 / 0.505 / 2.324 / 2.707 GiB (G1–G4). The formula is 32 B per stored upper-triangle nonzero, in the GPU storage format (`bench_deploy.py` l.113).
- **Fused/sparse difference 2.5×10⁻⁵.** It is not in the retained text and **not supported** by retained evidence. The same file records `fused_vs_csr_rel` = 3.5–4.6×10⁻⁸. Do not quote 2.5×10⁻⁵ in the response letter (I-04).
- **fp64 factor sizes.** They are not in the retained text. `factor_fp64_GB` exists (cuDSS, GPU) but belongs to the removed comparison and should not be revived.

---

## 1. I-28: memory units and content

### 1.1 Unit of every memory number, from the producing code

| Quantity (where shown) | Producing code | Unit as computed | Label in the paper |
|---|---|---|---|
| Table 5, learned state (0.74 / 0.25 / 1.20 / 1.37) | `bench_deploy.py` l.53–55, 138: `learned_state_GB = Δ torch.cuda.mem_get_info()/2**30` | **GiB** | "GB" |
| Table 5, host factor (5.56 / 1.10 / 12.3 / 18.5) | `bench_cpu.py` l.43: PARDISO `iparm(17)` in kB. The paper value is `factor_kB/1e6` (hand-computed; no generator script exists) | **Neither**: kB/10⁶ is 1.024×10⁹ B if kB = 1024 B | "GB" |
| Table 5, explicit S (4.5 / 2.7 / 2.3 / 5.0) | `bench_cpu.py` l.102: `dense_GB = 8·np²/2**30` | **GiB** | "GB" |
| Table 6 / ST20b, PARDISO memory (32.1 … 160.5) | `lat_direct_cpu.py` l.87: `(permanent+factor_solve) kB / 2**20` | **GiB** (kB taken as 1024 B) | "GB" (S7 says 2³⁰ B) |
| ST20b, peak process memory; §6.10 "5.2 to 6.4 GB" | `lat_direct_cpu.py` l.35: `ru_maxrss/2**20` (ru_maxrss is in KiB on Linux) | **GiB** | "GB" |
| Table 6 / ST20c, learned GPU peak (6.8 / 6.5 / 9.1 / 9.9) | `lat_scale.py` l.250: `max_memory_allocated()/1e9` | **GB (10⁹ B)** | "GB" (S7 says 10⁹ B) |
| Table 6 / ST20c, learned host (2.5 / 2.5 / 4.1 / 4.9) | `lat_scale.py` l.32–35: `VmRSS[kB]/1e6` (VmRSS is in KiB) | **Neither** (1.024×10⁹ B) | "GB" |
| ST20c, streamed operator state (1.37 / 2.24) | `lat_scale.py` l.168: bytes/1e9 | GB | "GB" |
| §6.1, training peak 29.6 GB | `train3.py` l.106: `max_memory_allocated()/2**30` (→ `meta_p1.json gpu_GB_max` 29.63) | **GiB** | "GB" |

MKL documents `iparm(15–17)` in "kilobytes". The paper already treats these as 1024 B (S7, `lat_direct_cpu.py`), and this report keeps that reading.

**Verdict: ERROR CONFIRMED.** Table 5 compares a GiB value with a kB/10⁶ value in the same cell. Table 6 compares GiB (direct) with GB (learned GPU). Everything is labelled "GB".

**Content is also inconsistent:**
- Table 5's host figure is `iparm(17)` only (factorisation storage).
- Table 6's direct figure is `iparm(16)+iparm(17)` (permanent + factorisation). MKL gives this sum as the memory needed in phases 2–3.
- Neither route's figure includes the cell stiffness K. Both routes must hold K to apply the operator (the learned route needs K for FᵀKF; the host route needs K_PI, K_IP, K_PP and a scaled copy of K_II).
- The learned state is the free-device-memory change during model caching and freezing (`bench_deploy.py` l.133–138), taken after K is already resident. It therefore excludes K (R3-10's question: **answered, K is not included**).

### 1.2 Recomputed values (G1 / G2 / G3 / G4, all GiB)

| Quantity | G1 | G2 | G3 | G4 | Source |
|---|---|---|---|---|---|
| Learned state | 0.736 | 0.252 | 1.199 | 1.365 | `bench_A3_cells.jsonl learned_state_GB` |
| Host factor only, `iparm(17)` | 5.302 | 1.051 | 11.759 | 17.672 | `bench_cpu_quiet.json factor_kB/2**20` |
| Host permanent + factor, `iparm(16+17)` | 6.654 | 1.436 | 14.254 | 20.769 | idem |
| Cell stiffness K (both routes; GPU format) | 1.394 | 0.505 | 2.324 | 2.707 | `bench_A3_cells.jsonl K_GB` |
| Explicit dense S | 4.52 | **2.65** | 2.28 | 5.01 | `dense_GB` |

Ratios, host/learned (ranges formed from the displayed values, per the I-29 decision):

| Convention | Displayed host values | Ratio range | Abstract wording |
|---|---|---|---|
| (A) Unit fix only: factor vs learned state | 5.30 / 1.05 / 11.8 / 17.7 | **4.2–12.9** (unrounded 4.17–12.94) | "4 to 13 times less operator state" |
| (B) PARDISO permanent + factor (as Table 6) vs learned state | 6.65 / 1.44 / 14.3 / 20.8 | 5.8–15.2 (unrounded 5.70–15.21) | "6 to 15 times" |
| (C1) Totals with K on both sides, factor only | 6.69 / 1.56 / 14.1 / 20.4 vs 2.13 / 0.76 / 3.52 / 4.08 | 2.1–5.0 | "2 to 5 times less memory" |
| (C2) Totals with K on both sides, permanent + factor | 8.04 / 1.95 / 16.6 / 23.5 vs 2.13 / 0.76 / 3.52 / 4.08 | 2.6–5.8 | "about 3 to 6 times less memory" |

The host K is counted at the learned route's GPU-format size, so (C1) and (C2) are *lower* bounds for the host total, because the scaled copy of K_II handed to PARDISO is not counted.

### 1.3 Proposed convention

Use **GiB (2³⁰ B) throughout**, labelled "GiB". Take PARDISO kilobytes as 1024 B, and say so once. Report K separately in Table 5, so that each reader can form either ratio. My recommendation for the text is (C1) as the headline total with (A) as the operator-state ratio:
- (C1) answers R3-10 directly.
- It does not depend on the PARDISO-permanent question, which the Cholesky rerun of I-36 would change anyway.

Keeping (A) alone fixes the unit error but leaves R3-10 unanswered. **Author decision.**

### 1.4 Exact replacements

**Table 5 header.** `Memory (GB): learned / host` → `Memory (GiB): learned state / host factor`.

**Table 5 values:**
- G1 `0.74 / 5.56` → `0.74 / 5.30`
- G2 `0.25 / 1.10` → `0.25 / 1.05`
- G3 `1.20 / 12.3` → `1.20 / 11.8`
- G4 `1.37 / 18.5` → `1.37 / 17.7`

**Explicit S column.** `(s / GB)` → `(s / GiB)`; G2 `247 / 2.7` → `247 / 2.6` (N-2).

**Table 5 caption.**
- Old: "Memory: interior factor, or the learned operator's stored state."
- New: "Memory, in GiB (2³⁰ bytes): PARDISO's factorisation storage (iparm(17), kilobytes taken as 1024 bytes), or the device memory held by the learned operator's network state and correction. Neither includes the cell stiffness *K*, which both routes hold (1.39, 0.51, 2.32 and 2.71 GiB for G1–G4 in the GPU storage format)."

**§6.10, ¶ after Table 5.**
- Old: "Against the conventional route, the learned operator condenses a cell 8.8 to 29 times faster and stores 4.4 to 13.5 times less."
- New: "Against the conventional route, the learned operator condenses a cell 8.8 to 29 times faster; its stored state is 4.2 to 12.9 times smaller than the interior factor, and including the stiffness matrix that both routes hold, the memory per cell is 2.1 to 5.0 times smaller."

**Abstract.**
- Old: "…condenses a cell 9 to 29 times faster, stores 4 to 14 times less, and applies it 2 to 23 times faster."
- New, convention (A): "…condenses a cell about 9 to 29 times faster, stores 4 to 13 times less operator state, and applies it 2 to 23 times faster."
- New, convention (C1): "…stores 2 to 5 times less in total…"
- The draft Abstract in §4.6 omits memory pending this decision.

**§8.**
- Old: "…condenses a cell 9 to 29 times faster, stores 4 to 14 times less and applies it 2 to 23 times faster, so it is cheaper for any number of queries."
- New: "…condenses a cell about 9 to 29 times faster, stores 4 to 13 times less operator state (2 to 5 times less including the stiffness matrix) and applies it 2 to 23 times faster."
- The "any number of queries" clause belongs to I-01 and I-02; it is not re-verified here.

**Table 6.**
- Headers: `memory (GB)` → `memory (GiB)` (twice), and `Learned memory (GB): GPU / host` → `Learned memory (GiB): GPU / host`.
- Direct values are unchanged, since they are already GiB.
- Learned values: `6.8 / 2.5` → `6.3 / 2.4`; `6.5 / 2.5` → `6.1 / 2.4`; `9.1 / 4.1` → `8.5 / 3.9`; `9.9 / 4.9` → `9.2 / 4.7`.

**§6.10, last ¶:**
- "from 29 to 32 GB … 57 to 64 GB … 67 to 81 GB and 132 to 160 GB … a further 5.2 to 6.4 GB" → the same numbers with "GiB".
- "the eight-cell Cholesky factorisations, of 67 and 81 GB" → "GiB".
- "at most 9.9 GB of GPU memory and 4.9 GB of host memory" → "at most 9.2 GiB of GPU memory and 4.7 GiB of host memory".
- "the 90 GB available to this job" needs a source and unit (N-4).

**S7, last sentence of the learned-route paragraph.**
- Old: "GPU memory is the peak memory allocated by the process, in units of 10⁹ bytes; host memory is the resident set size after the front end. PARDISO and peak process memory of the direct solution are given in units of 2³⁰ bytes."
- New: "All memory values are in GiB (2³⁰ bytes): GPU memory is the peak memory allocated by the process, host memory the resident set size after the front end, and PARDISO memory the reported kilobytes taken as 1024 bytes."

**ST20.**
- Headers "(GB)" → "(GiB)".
- ST20c peak GPU: 6.80 / 6.54 / 9.13 / 9.87 → **6.34 / 6.09 / 8.50 / 9.19**.
- ST20c host: 2.54 / 2.51 / 4.07 / 4.92 → **2.42 / 2.40 / 3.88 / 4.69**.
- ST20c streamed: 0.00 / 0.00 / 1.37 / 2.24 → **0.00 / 0.00 / 1.28 / 2.09**.
- ST20b is unchanged apart from the label.

**§6.1.** "peak device memory 29.6 GB" → "29.6 GiB".

---

## 2. I-29: ratios from rounded values (DECIDED)

**Recomputation.** Condensation on the learned route = netdata + geo + model cache + freeze + correction setup (reproduces 0.72 and 0.16 s).

| Quantity | Unrounded | From the displayed values (current text) |
|---|---|---|
| Condensation | 8.53–28.72 | 8.8–29 |
| Memory | 4.37–13.57 (the kB/10⁶ mix) | 4.4–13.5 → becomes 4.2–12.9 in GiB (I-28) |
| Single-vector application | 7.26–22.97 | 7.4–23 |
| Ready time | learned 3.12–7.63 s; host 9.81–61.4 s; ratio 3.15–8.05 | 3.2–7.6 s, 9.9–61 s, 3.1–8.1 |

**Verdict.** R6-03's numbers are CONFIRMED. Under the decision, the text only needs:

1. **Table 5 caption, append:** "Ratios quoted in the text are formed from the displayed values."
2. **Abstract and §8:** "9 to 29 times" → "about 9 to 29 times". (§6.10 keeps "8.8 to 29".)

---

## 3. T9 findings

### 3.1 I-46: "nine configurations common to all predictors": ERROR CONFIRMED

- ST13 rows: B has 7 configurations (U1/x, U1/y, U2/x, M1/x, H1/x, H1/y, M2/x); U2/y and M1/y are missing.
- C has 11, S8 9, A2b 12, and B+W and A3 14 each.
- The nine configurations U1/x,y; U2/x,y; M1/x,y; H1/x,y; M2/x are common to C, S8, A2b, B+W and A3 only.

**§6.6:**
- Old: "The uncorrected continuation C was evaluated on eleven configurations, the nine common to all predictors and L1/x and L1/y;"
- New: "The uncorrected continuation C was evaluated on eleven configurations, the nine common to C, S8, A2b, B+W and A3 (B was evaluated on seven of them) and L1/x and L1/y;"

**ST13 note:**
- Old: "…in the same five of the nine configurations common to all predictors…"
- New: "…in the same five of the nine configurations common to C, S8, A2b, B+W and A3 (B was evaluated on seven of them)…"

### 3.2 I-47: lattice numbers in §6.9, Table 6 and §8

**Sources:**
- §6.9 comes from `lat_hetero222_A3.json` and `lat_hetero331_A3.json`: `A3.compliance_rel_err`, CG to 10⁻¹⁰, evaluation route.
- Table 6 and ST20d come from `d5_off.json` (2×2×2) and `learned_hlat331.json` (3×3×1): the deployed timed route, CG to 10⁻⁶ (reached 8.6×10⁻⁷), with host streaming. Their compliances are compared with the `exact.compliance` of the lat_hetero files.

| Lattice, load | Table 6 (%) | §6.9 (%) | β = Σw_mε_m (%) (`A3.bound`) | Table 6 / β |
|---|---|---|---|---|
| 2×2×2 x | 0.01403 | 0.01371 | 0.01375 | 1.020 |
| 2×2×2 y | 0.01118 | 0.01101 | 0.01102 | 1.014 |
| 2×2×2 z | 0.00974 | 0.00942 | 0.00944 | 1.032 |
| 3×3×1 x | 0.01545 | 0.01466 | 0.01471 | 1.050 |
| 3×3×1 y | 0.01366 | 0.01342 | 0.01345 | 1.016 |
| 3×3×1 z | **0.01514** | **0.01031** | 0.01033 | **1.465** |

**Which is right.** At exact equilibrium of the surrogate, Eq. (12) gives (C−Ĉ)/C ≤ β/(1+β) < β.
- The §6.9 values satisfy the bound (β exceeds them by 0.13–0.33%).
- All six Table 6 values exceed β, by 1.4–47%.
- The Table 6 values therefore contain the signed residual work of the incomplete 10⁻⁶ solve (Eq. (18)). This is consistent with CG underestimating compliance.

§6.9 is the correct **operator-accuracy** value. Table 6 is a correct record of the **timed run** but must not be described as the same quantity. The Abstract's "at most 0.015%" holds for both (0.01545 rounds to 0.015).

**Table 6 caption:**
- Old: "Compliance error of the learned route under the three consistent loads, against the direct solution (four cells) or the exact condensation of Section 6.9 (eight cells); the learned compliance lies below the reference in every case."
- New: "Compliance error of the timed learned run (conjugate gradients to 10⁻⁶) under the three consistent loads, against the direct solution (four cells) or the exact condensation of Section 6.9 (eight cells); the learned compliance lies below the reference in every case. For the eight-cell lattices these values include the algebraic error of the 10⁻⁶ solve (Eq. (18)) and exceed the operator errors of Section 6.9, obtained at 10⁻¹⁰, by up to 0.005 percentage points (3×3×1, z-load: 0.015% against 0.010%)."

**S7:**
- Old: "The compliance errors of the eight-cell lattices differ slightly from those of Section 6.9, which solved six loads to a relative residual of 10⁻¹⁰."
- New: "The compliance errors of the eight-cell lattices exceed those of Section 6.9, which solved six loads to a relative residual of 10⁻¹⁰, by 2–5% of their value and by 47% for the z-load of the 3×3×1 layer (0.0151% against 0.0103%); unlike the Section 6.9 values, they exceed the participation-weighted bound of Eq. (12) and therefore include the algebraic error of the 10⁻⁶ solve."

**§6.9, reference residual (R6-09 CONFIRMED: 9.08×10⁻¹¹ and 1.29×10⁻¹⁰):**
- Old: "…to a recomputed relative residual of \(9\times10^{-11}\)."
- New: "…to a recomputed relative residual of at most \(1.3\times10^{-10}\)."

**§6.9, random loads (R6-26 CONFIRMED: 5.1 / 4.4 / 5.0% and 5.0 / 4.2 / 4.5%):**
- "…and 4–5% above them for the random loads" → "…and about 4–5% above them for the random loads".
- The same edit applies to the 3×3×1 sentence.

**N-1 (new): cell energy-error ranges in §6.9 refer to the random loads.** The `A3.eps` arrays (8 cells × 6 loads) give:
- consistent loads: 2×2×2 **0.006–0.026%**; 3×3×1 **0.005–0.052%**;
- random loads: 0.027–0.123% and 0.023–0.077%.

The text says "The cells' own energy errors at the exact traces range from 0.03% to 0.12%, largest in the cut cells; weighted by the cells' shares …, their sums are 0.014%, 0.011% and 0.0094% for the consistent loads". This reads as a consistent-load statement, and the weighted sums (0.0094–0.014%) are then larger than several of the "0.03%" cell values could produce.

- Old: "The cells' own energy errors at the exact traces range from 0.03% to 0.12%, largest in the cut cells;"
- New: "The cells' own energy errors at the exact traces range from 0.006% to 0.026% under the consistent loads (0.03% to 0.12% under the random loads), largest in the cut cells;"
- Old (3×3×1): "In the \(3\times3\times1\) layer the cells' energy errors range from 0.03% to 0.08%,"
- New: "In the \(3\times3\times1\) layer the cells' energy errors range from 0.005% to 0.052% under the consistent loads (0.02% to 0.08% under the random loads),"

**§8 (R6: consistent loads only; the bound excess is 0.13–0.33% consistent vs 4.2–5.1% random):**
- Old: "…follows the participation-weighted bound of Eq. (12) to within 0.4%…"
- New: "…follows the participation-weighted bound of Eq. (12) to within 0.4% under consistent face loads (about 5% under random loads)…"

**Both 5.83% values CONFIRMED as a coincidence:**
- B, U1/x, N-z: target sensitivity 5.830% (`gate_v2L1_fresh_val_2000_full_x.json`, load 6).
- S8, M1/x, T-y: 5.829% (`gate_A2_tail8_fresh_val_2003_d1_v1_x.json`, load 2).

Also re-verified (no change needed):
- sensitivity maxima 0.144 / 0.454% (2×2×2) and 0.144 / 0.426% (3×3×1);
- solution differences 0.130 / 0.123%;
- random compliance maxima 0.069 / 0.056%;
- learned true residuals 3.6×10⁻⁴ / 3.2×10⁻³.

### 3.3 I-57: operator-consistency numbers in §6.2: ERROR CONFIRMED

Deployed-arithmetic records for the five named cells (`p1_checks_cpu.json` for H2, H1, M2, M1; `p1_checks_u2.json` for U2; both = ST19):
- `sym_rel_max` ≤ **8.73×10⁻⁹** (M2);
- `action_vs_field_energy_rel_max` ≤ **4.49×10⁻⁹** (H2);
- field reproduction ≤ 1.11×10⁻⁷ (U2) ✓;
- rigid energy ≤ 1.72×10⁻¹¹ ✓.

The older `p1_checks.json` gives ≤ 1.15×10⁻⁸ and ≤ 9.99×10⁻⁹. No record gives 3×10⁻⁸ or 2×10⁻⁸.

- Old: "…the bilinear form \(q_i^T\widehat Sq_j\) is symmetric to a relative \(3\times10^{-8}\), the returned work \(q^T\widehat Sq\) agrees with the energy of the recovered field \(q^TF^TKFq\) to \(2\times10^{-8}\),…"
- New: "…the bilinear form \(q_i^T\widehat Sq_j\) is symmetric to a relative \(9\times10^{-9}\), the returned work \(q^T\widehat Sq\) agrees with the energy of the recovered field \(q^TF^TKFq\) to \(5\times10^{-9}\),…"

### 3.4 I-58: descriptive wording (§6.3, §6.4, §6.6, §7.3): all five CONFIRMED

**(a) §6.4 (R6-13).** From `p1_field_M1_small.npz`:
- total B/A3 error-energy ratio 113–138 per direction (log₁₀ 2.05–2.14);
- elementwise log₁₀ ratio at the 10/50/90th percentiles: 1.43 / 1.88 / 2.46 (27× / 76× / 290×);
- only 1.5% of elements reach ≥ 10³.

- Old: "The corrected field reduces the element error energies by two to three orders of magnitude throughout the cell and halves the share of the cut-adjacent layer to 19–22%."
- New: "The corrected field reduces the total error energy by about two orders of magnitude (113–138×) and the element error energies by roughly 25–300× (10th–90th percentiles), and halves the share of the cut-adjacent layer to 19–22%."
- The 19–22% remains unsupported (I-59).

**(b) §6.6 (R6-23).** ST13: on M2, B+W 0.083 / 0.075% vs A3 0.136 / 0.154%; on U2, A3 0.082 / 0.073% vs B+W 0.142 / 0.158%.

- Old: "…while on M2 the untrained combination is marginally more accurate."
- New: "…while on M2 the untrained combination is more accurate (0.075–0.083% against 0.136–0.154%), a difference far below the accuracy reference."
- §7.3, old: "…whereas on cells that the untrained combination already approximates well the two are equivalent."
- §7.3, new: "…whereas on cells that the untrained combination already approximates well both stay below 0.16%, with either one up to twice as accurate as the other."

**(c) §6.6 (R6-24).** ST13 B+W M1/x 0.350, M1/y 0.339.

- Old: "…from 0.35% to 0.16–0.21% on M1…"
- New: "…from 0.34–0.35% to 0.16–0.21% on M1…"

**(d) §6.6 (R6-25).** ST13: C also fails compliance on M1/x (4.119%) and M1/y (3.458%).

- Old: "…it fails the sensitivity requirement in four of them (U1/x, U1/y, M1/x, M1/y, with up to 11.4%),"
- New: "…it fails the sensitivity requirement in four of them (U1/x, U1/y, M1/x, M1/y, with up to 11.4%) and, on M1, also the compliance requirement (up to 4.1%),"

**(e) §6.3 (R6-28).** ST19 ghost share (force_c) covers H2, H1, M2, M1 and U2, with a maximum of 0.048%. There is no U1 record, although §6.4 examines U1.

- Old: "…the ghost-penalty contribution is below 0.05% of the exact field energy in all cells examined in Section 6.4."
- New: "…the ghost-penalty contribution is below 0.05% of the exact field energy in the five cells of Table ST19."

### 3.5 I-59: numbers without an archived source: CONFIRMED UNSUPPORTED

**Figure 7 layer shares.** `p1_field_M1_small.npz` has no plane or layer mask. My own reconstruction took a plane fitted to the 5,057 cut-band retained nodes, element width 1/32, and a layer of |d| ≤ 2h. It gives:
- 12.4% of elements;
- exact energy 3.6–12% (direction-dependent);
- B error 36–40%;
- A3 error 16–19%.

This is qualitatively consistent but does not reproduce 13% / 8% / 39–43% / 19–22%. The quantised element positions make the numbers sensitive to the layer definition. Action: archive the mask (or plane and offset) used for the figure, and state the definition in the caption.

**§6.7 replacement values.** ST05 B M1/x gives full 12.153, field-only 15.602 and solution-only 19.269 as *maxima over six loads*. The full-error maximum is at T-y (`gate_v2L1_…2003_d1_v1_x.json`, 12.15%), but no per-load record of the replacement values is archived.

- Old: "For B on M1/x under the target-face y load, the full sensitivity error is 12.2%. Evaluating the learned extension at the exact retained displacement gives 15.6%, whereas evaluating the exact extension at the learned retained displacement gives 19.3% (Table ST05). The full error is smaller than either replacement error."
- New: "For B on M1/x the full sensitivity error reaches 12.2% (target-face y load). Over the six face loads, evaluating the learned extension at the exact retained displacement gives up to 15.6%, whereas evaluating the exact extension at the learned retained displacement gives up to 19.3% (Table ST05). The largest full error is smaller than the largest error of either replacement."
- Alternatively, archive the per-load values and keep the original wording.

**Other unarchived sources** (ST02, ST03, ST04, per-load ST05, the R2 element-group shares): unchanged; archive them.

### 3.6 Further new observations

**N-2.** Table 5, explicit S, G2: `dense_GB` = 8·18,858²/2³⁰ = 2.6496 GiB, so it should display **2.6**, not 2.7 (probably double rounding). G1, G3 and G4 (4.5 / 2.3 / 5.0) are correct.

**N-3.** Two-cell records `gate_{A0_ctrl,A2_tail8,A2b_tail8,A3_2grid}_fresh_val_2010_d0_v0_y.json` (H2/y) are archived but degenerate:
- neighbour-load compliance errors are NaN;
- target compliance errors are 2×10²–3×10⁴%;
- PCG stopped at maxit 400;
- target energy shares are negative.

This is consistent with a pair that is not mechanically connected (H2 retains 2.7% of the box). The exclusion is legitimate, but the manuscript never mentions it. Add to the ST13 note: "H2/y was not evaluated: the H2 target carries no material on the shared face, so the pair is not connected." (Authors to confirm the cause.)

**N-4.** "The eight-cell LU factorisations therefore exceed the 90 GB available to this job". No evidence file records the job memory limit. The runs used `--mem-gb 60` (S7: "skipped … exceeded 60 GB"). Give the source and unit, or rephrase as "exceed the 60 GiB limit set for these runs".

---

## 4. T8 findings

### 4.1 Word counts (current MANUSCRIPT_EN.md)

Method: lines starting with `|` (tables), `![` (images) and display-math blocks `\[…\]` are excluded; inline math counts as one token; links count as their text. "Captions" are lines starting with `**Figure n.` or `**Table n.`.

| Part | Words (text) | Caption words |
|---|---:|---:|
| Abstract | 369 (359 before the placeholder) | — |
| 1 Introduction | 1,410 | — |
| 2 Substructures | 637 | 85 |
| 3 Neural extension | 1,407 | 281 |
| 4 Error relations | 847 | — |
| 5 Correction | 734 | — |
| 6 Numerical examples | 5,623 | 1,183 |
| 7 Discussion | 1,730 | — |
| 8 Conclusions | 509 | — |
| **Sections 1–8** | **12,897** | **1,549** |
| Sections 1–8 incl. captions | 14,446 | |
| Whole file (`wc -w`, incl. tables and references) | 16,777 | |

§6 subsections, including captions: 6.1 750 · 6.2 576 · 6.3 599 · 6.4 765 · 6.5 810 · 6.6 727 · 6.7 323 · 6.8 371 · 6.9 556 · 6.10 1,122.

### 4.2 I-21: labels, notation, colour

**Label map** (recommended; builds on R5 and uses the DECIDED name). Internal run IDs (v2L1, A0_ctrl, A2_tail8, A2b_tail8, A3_2grid, B2grid) should appear only in a data-availability key in the supplement.

| Current | New label | Where it remains |
|---|---|---|
| A3 | **NICE** | everywhere, including all §6 text, figures and tables |
| B+W | **NICE-post** (base network + correction, not trained through it) | §6.3, §6.5–6.6 |
| A2b | **Smoothing-trained** | §6.3, §6.6, Fig. 9, and Table 4 in place of S8 |
| C | **Uncorrected** (continued network, no correction) | throughout |
| B | **Base network** | §6.4–6.5 (fixed-weight study) |
| S8 | remove from the main text; supplement only, as "S8 (smoothing at evaluation, partial training)" | ST01, ST13, ST15 |
| P0 | remove from the main text and Table 2; supplement ST01/Fig. S01 only | supplement |

**Consequences:**

1. **Table 4.** Replace the S8 column with Smoothing-trained (A2b), which has a documented protocol:

   | Row | A2b compliance / sensitivity |
   |---|---|
   | H1/x T-x | 0.144 / 0.636% |
   | U1/x N-z | 0.00109 / 3.15% |
   | M1/x T-y | 1.14 / 4.43% |

   The U1/x N-z row keeps the separation argument: compliance 0.001%, sensitivity 3.2%, target share 0.13%. Alternatively, fold Table 4 into text (I-53).
2. **§6.6 ¶2 example.** "S8's compliance error is 0.00137% while the target-cell sensitivity error is 3.89%" → "the smoothing-trained predictor's compliance error is 0.00109% while the target-cell sensitivity error is 3.15%" (`gate_A2b_tail8_fresh_val_2000_full_x.json`, load 6).
3. **Table 3.** Rename the column "B (learned)" to "Base network" (it is B's field with the correction applied, i.e. NICE-post) and "A3" to "NICE". Add "(= NICE-post)" to the caption.
4. **Tie W to 𝒲.** Once, in §6.1: "NICE-post applies the correction 𝒲 of Section 5.2 to the base network's unchanged weights."
5. **Cell key.** Add a key next to Table 2 or Fig. 1: U1, U2, L1, M1, M2, H1, H2, H3, G1–G4 with stratum, retained volume and DOFs. It is currently only in Supp. R1.
6. **Colour map.** Fix one colour and marker per predictor across Figs. 5, 8, 9 and 10. At present the orange square is A3 in Figs. 5 and 9 but S8 in Fig. 10 (confirmed on `F10_energy_participation.png`).
7. **Nomenclature table** (half a page, before §2), with these reservations:
   - C for compliance only; Ĉ for the surrogate;
   - A = K_II and D = diag(A) for mechanics only;
   - K_{,c} for the stiffness derivative;
   - H for the extension error;
   - P for the retained index set;
   - ρ for the assembled residual (Eq. 18).

   With the predictor renaming, the collisions C/C, A/A3 and D/D disappear. Remaining overloads to resolve in the table (R1-16): T, M_I, I, γ, b, n.
8. **Support classes (R6-11).** Use "stiffness-scaled spring supports" for `support_k` in §6.1, §6.3, Fig. 5(c) and Fig. S01, and "spring supports" for `support`.
   - §6.3, old: "…and spring-supported faces 0.057% (0.47%)."
   - §6.3, new: "…and stiffness-scaled spring supports 0.057% (0.47%)."

### 4.3 I-37: length plan (concrete)

The target is about 9,500 words of text before §6.11, per R5. The current text is 12,897 words.

| Part | Now | Target | Concrete action |
|---|---:|---:|---|
| Abstract | 369 | ≤250 | §4.6 draft |
| §1 | 1,410 | ~950 | ¶4 (PIML, 246 words): keep the attribution and the one-sentence 78–85% finding, and move the rest to §6.8. Merge ¶6 (137, analysis summary) into the contributions. Cut ¶7 (58, roadmap) to one sentence. |
| §2–5 | 3,625 | ~3,200 | §3.1: move the gather/scatter map detail (eq. block around l.133) to App. G. Move §3.3 ¶3 to §5.4 (I-56). Inline the J.4/J.5 one-liners (I-54). |
| §6 | 5,623 (+1,183 captions) | ~3,800 (+~800) | 6.2: keep one paragraph and move the refinement numbers to Note S5/Fig. S06 (−300). 6.4: keep δ/κ and one spectral sentence (−350). 6.5: cut the H2/M1 narrative already shown in Fig. 8 (−250). 6.6: fold Table 4 into two sentences (I-53) (−200). 6.8: halve (−180). 6.10: absorb §7.4 ¶2 and cut the explicit-S paragraph to one sentence (−250). Captions: trim Fig. 8 and Fig. 9 captions, which repeat the text (−350). |
| §7 | 1,730 | ~850 | Three subsections, as below. |
| §8 | 509 | ~350 | Cut the method recap in ¶1 (158 → 60). |

The §7 restructure:
- **(a) Boundary restriction versus interior approximation.** Keep §7.1 ¶3 minus its repetition of §1 ¶4 and §6.8 (209 → 110).
- **(b) What learning and correction contribute, including NICE-post.** Merge §7.3 ¶1–2 (374 → 220) and §7.2 ¶4 (163 → 120). Delete §7.2 ¶1–3 (331 words, which repeat §4.2–4.3 and §6.7).
- **(c) Limitations and extensions** (new, about 180 words, per I-05). Move the cost content of §7.4 ¶2 (198) into §6.10. Delete §7.4 ¶3 (78, which repeats §5.3/Eq. 18).

The main-text ST citations should also be capped (I-54).

### 4.4 I-45: title options

1. **(R5, recommended once §6.11 exists):** "Neural-initialised static condensation with equilibrium correction for the analysis and thickness design of cut thin-walled TPMS lattices"
2. **(if §6.11 stays small; foregrounds sensitivity):** "Neural-initialised static condensation with equilibrium correction for cut thin-walled TPMS cells: compliance and thickness sensitivity"
3. **(scope-precise; if I-05 asks for P-type in the title):** "Neural-initialised static condensation with equilibrium correction for cut thin-walled Schwarz-P lattices and their thickness sensitivity"

In all three, put NICE in the Abstract and keywords, not in the title.

**Keywords (replacement):** static condensation; neural-initialised condensation; equilibrium correction; cut finite elements; TPMS lattices; thickness sensitivity; thickness optimisation.

### 4.5 I-53: Fig. 10, Fig. 9 and Table 4

**Confirmed.** Fig. 10 shows only B, C and S8: 192 = 54 + 69 + 69 observations, which I recomputed from the `gate_*` files excluding L1 and H2/y.

The retained evidence already contains the per-load `compliance_rel_err`, `sens_vec_rel_err`, `bound` (β) and `energy_share` for NICE (`gate_A3_2grid_*`) and NICE-post (`gate_B2grid_*`): 12 configurations and 96 observations each.

**Edits:**
- Re-plot Fig. 10 with Base network, Uncorrected, Smoothing-trained, NICE-post and NICE. Drop S8, or keep it grey.
- New caption: "Each point is one load for one model–configuration combination: 411 observations from the 52 combinations of Table ST13 other than L1 and S8."
  - These counts are the sum over the five predictors, as observations / combinations: B 54 / 7, C 69 / 9, A2b 96 / 12, B+W 96 / 12, A3 96 / 12 (by files). Keeping S8 adds 69 / 9 (480 / 61).
  - The plotting script should re-derive the counts.
- Remove Fig. 9(c,d); Fig. 10(b) shows the same relation.
- Replace Table 4 by two sentences carrying the U1/x N-z and M1/x T-y numbers, and move Table 4 to ST15.

### 4.6 I-55: revised Abstract (240 words + placeholder; 249 in total)

> Static condensation of cut thin-walled Schwarz-P-type triply periodic minimal surface (TPMS) cells needs one interior solve per retained coordinate, and each cell retains tens of thousands. We approximate the condensed operator on the complete retained space by neural-initialised condensation with equilibrium correction (NICE): a geometry-conditioned network supplies an initial interior field, and a fixed, geometry-specific multilevel correction reduces its equilibrium residual. The resulting matrix-free condensed stiffness is symmetric, positive semidefinite, reproduces rigid motion and is bounded below by the exact Schur complement. Tracing the interior error to assembled compliance through energy participation, and to thickness sensitivity through the stiffness derivative, explains why accurate compliance need not imply accurate local sensitivity. Trained through the correction, NICE attains a mean directional energy error of 0.074% under consistent tractions on 80 validation geometries not used in training (largest geometry mean 0.65%). In two-cell assemblies it keeps compliance errors below 0.06% and eight-corner sensitivity errors below 0.7% in all fourteen configurations, whereas the uncorrected network exceeds 3% sensitivity error in four of eleven. With the same correction, the learned initial field is 7 to 290 times more accurate than a harmonic extension. In two eight-cell heterogeneous lattices with every cell learned, compliance and sensitivity errors stay within 0.015% and 0.15% under consistent face loads. On one GPU, NICE condenses a cell about 9 to 29 times faster than PARDISO on 16 host cores and applies it 7 to 23 times faster for single vectors. [Placeholder: quantitative outcome of the Section 6.11 design example.]

**Notes:**
- Every number is traced (ST01, ST13, Table 3, §6.9, Table 5).
- The draft has no GPU exact-solver comparison.
- The draft avoids "unseen" (I-16) and avoids "at most … for any geometry" (I-18).
- It avoids the memory ratio until I-28 is decided; adding "and stores 4 to 13 (or 2 to 5) times less" costs 8 words.
- Once the real §6.11 sentence (about 25 words) replaces the placeholder, delete the harmonic sentence (19 words, which also depends on I-07) to stay at 250 or fewer.
- Without that sentence and without the placeholder, the draft is 221 words.

**Highlights** (R5 asks for them): write them after the claim decisions.

### 4.7 I-56: training described before the correction

**Edit.** Move §3.3 ¶3 ("The correction of Section 5 can also be placed inside the training loop. … adds no trainable parameters.", 118 words) unchanged into a new **§5.4 "Training through the correction"**, placed after §5.3. Its first sentence becomes: "The correction can also be placed inside the training loop."

In §3.3, replace the moved paragraph by: "Section 5.4 describes how the objective is evaluated on the corrected extension when the network is trained through the equilibrium correction."

In §6.1, change "(Section 3.3)" → "(Section 5.4)". A full reorder (Setting → NICE operator → Training → Error analysis) is not needed.

### 4.8 I-54: appendix reliance

| Location | Current reliance | Edit |
|---|---|---|
| §5.1 (l.283) | "whereas Eq. (B.8) compares the full error with the loaded response" | Delete the clause. The spectral argument does not need it. |
| §6.7 | "Eq. (J.4) relates local relative sensitivity error to participation, derivative coupling and the local reference scale" | State the bound inline in one displayed line, and keep "(Appendix J.4)" as its source. |
| §7.2 | Eq. (J.5) and Eq. (H.5) | Keep the words "quadratic when the extension error is uniformly C¹-small". Drop the H₀ / H_{0,c} detail. |
| Uncited: App. C, I, J.1, J.2, J.8, J.9 | — | Move App. I, J.8 and J.9 to the supplement. Either cite C and J.1–J.2 from §4.2 or merge them into J.3. |

Main-text ST/S citations (about 30) should be capped at about one per subsection. Keep ST01, ST13, ST19 and ST20.

---

## 5. Top three author decisions

1. **Memory convention and the Abstract memory ratio (I-28).**
   - The unit fix alone gives 4.2–12.9 ("4 to 13").
   - Counting the stiffness matrix that both routes hold gives 2.1–5.0, or 2.6–5.8 with PARDISO permanent storage.
   - K is *not* in the learned-state figure (answers R3-10), and the K values are in retained evidence.
   - Decide which ratio headlines, and whether Table 5 adopts Table 6's permanent+factor definition.
2. **Abstract content (I-55, with I-07/I-16/I-18).**
   - Adopt the 240-word draft.
   - Decide whether the harmonic "7 to 290" sentence stays once the §6.11 sentence is written.
   - Decide whether any memory ratio appears.
3. **Labels and figure scope (I-21 and I-53, which constrain I-37).**
   - Adopt NICE / NICE-post / Smoothing-trained / Uncorrected / Base network.
   - Drop S8 and P0 from the main text, replacing S8 by A2b in Table 4 and Fig. 10.
   - Re-plot Fig. 10 with NICE and NICE-post.
   - Approve the per-section length plan.
   - The title follows once the §6.11 scope is known (options in §4.4).
