# The pipeline was never the blocker

2026-09-17, evening. Found while building the line-③ harness. Three documents written
earlier today are wrong and are annotated as such: `THE_BLOCKER_20260917.md`,
`TAU_BLOCKER_20260917.md`, `ROUTE5_COST_20260917.md`.

Every number below is a measurement with a self-test, not an estimate.

## 1. The label producer runs, and reproduces a frozen label bit-for-bit

I had been reading `full_factor_geometry.py`, which is a *diagnostic* that recompiles the
topology from a case. The label producer is
`/root/cutfem_neural_a_20260910/source_14301bc56/stage_cutfem_neural_a/dense_reference.py`,
and its trace-compile step is `data.prepare`, which builds `TRACE_CACHE.npz` **from the
packet's own `TRACE.npz`** in plain numpy: no solver, no topology compile, no cut.

Run on seat 0253's packet, writing to a scratch directory:

```
status         FULL_SAVED_FACTOR_REFERENCE_PASS
dimension      12822
factor_sha256  7ee508ff9c1019e8a59bae8bc159ff9cdc76c635850f912d15f51a506f96a453
trace_sha256   76dfce8fce2d2cea5a4934c3e394d9eb9f19e87a892005a19ce27ccfd875dc66
action_relative        5.05e-15
inverse_relative_max   6.33e-13
total_seconds  4.647
peak_allocated 4.913 GiB
```

Both digests are **identical** to the frozen receipt in
`CUTFEM_SUPERELEMENT_LABELS/REFERENCE_0253/RESULT.json`. This is exactly the self-test
`THE_BLOCKER_20260917.md` demanded before trusting anything else, and it passes.

Separately, and before finding the cheap path, I also reproduced the trace cache the hard
way — `from_case` → `compile_topology` (15.5 s, 8 workers) → `compile_trace` (0.30 s) →
`trace_cache` (0.014 s) — and compared array by array against the frozen file:

```
key_sets_equal  true
all_arrays_equal  background_nodes coefficients cut_plane indices indptr
                  kind order rigid support_centroid tau_corners   -> all true
bit_exact  true
```

So both routes to the trace cache agree with the frozen one exactly. The expensive route
matters only because it is the one that accepts a *perturbed* geometry (section 3).

## 2. What that unblocks: 26 training labels become 266

Census of every packet in `CUTFEM_INGEST_R38/dataset_independent_20260910`:

| | |
|---|---|
| packets | 920 (1025 `S_UPPER.npy` across all dataset directories) |
| splits | train 730, validation 97, test 93 |
| `q` | min 552, median 23 616, max 47 760 |
| distinct mother fields | 920, **all of family size 1** |
| labels built so far | 33 (26 train / 6 holdout / 1 validation) |

The `q` spread is the constraint, because storage is `d(d+1)/2` fp64 and the producer's
peak GPU memory scales as `q²`:

| cap | new train seats | disk | GPU time | train total | max peak |
|---|---|---|---|---|---|
| `q ≤ 16 000` | 127 | +76 GiB | ~10 min | 153 | 7.6 GiB |
| `q ≤ 18 000` | 183 | +137 GiB | ~20 min | 209 | 9.6 GiB |
| **`q ≤ 20 000`** | **240** | **+214 GiB** | **~35 min** | **266** | 11.9 GiB |
| `q ≤ 24 000` | 373 | +438 GiB | ~58 min | 399 | 17.2 GiB |
| all train | 730 | +1807 GiB | — | 756 | 68.1 GiB |

715 GB free, so `q ≤ 20 000` is the pick: **a 10x larger training set for 35 minutes of
GPU**. Holdout hygiene is the producer's own — `verified_packet` raises on any packet
whose split is not `train`, so validation and test cannot be built by accident.

Every `eps_op` this project has quoted — 234, 273.9, 597 — came from at most 26
geometries. With 266 the question stops being memorisation.

## 3. `dS/dtau` is not blocked either, and it found something

`CUTFEM_FULL_FACTOR_LOCAL_GEOMETRY_20260913T2330` contains four **already-built**
perturbed operators for seat 0328, at `tau_corners` scaled by `1+eps`:

| case | `eps` | active cells | `e_A` | `mu_min` | `mu_max` | **`eps_op`** | modes > 1.1 |
|---|---|---|---|---|---|---|---|
| PATH_1 | −1/1000 | 7470 | 1.298e−3 | 0.84728 | 1.00389 | **0.153** | 0 |
| PATH_2 | −1/10000 | 7477 | 1.292e−4 | 0.98429 | 1.00000 | **0.016** | 0 |
| base | 0 | **7478** | — | — | — | — | — |
| PATH_3 | +1/10000 | 7479 | 2.341e−4 | 0.99999982 | 1.59756 | **0.598** | 11 |
| PATH_4 | +1/1000 | 7493 | 1.341e−3 | 0.99938 | 1.67487 | **0.675** | 33 |

All four are `ADMITTED_SAME_TRACE`: same `q` = 12 798, same order, quotient reflectors
agreeing to 2.9e−15. So these five operators live in one coordinate system and can be
differenced directly.

Two things fall out, and they point in opposite directions.

**The operator is smooth in `tau` in Frobenius.** `e_A ≈ 1.3·|eps|` on both sides, four
points, no kink.

**The acceptance metric is not.** Going from `eps` = 0 to `eps` = +1e−4 — a 0.01 %
thickness change — exactly **one cut cell is born** (7478 → 7479), and that single cell
takes `mu_max` from 1 to 1.598. `eps_op` jumps to 0.60, four times our 0.15 gate, on a
perturbation whose Frobenius size is 2.3e−4. Thinning is unremarkable by comparison: one
cell dies and `mu_min` moves 1.6 %. The `+` side is also not linear — 10x the
perturbation only takes `mu_max` from 1.598 to 1.675 — which is the signature of a
threshold crossing, not a derivative.

So a low-rank, high-contrast change: one nearly-null direction of `A(0)` is stiffened by
60 %. In the energy metric that is enormous; in the operator norm it is nothing.

This matters because it is the same subspace our own error lives in. `E3` found that all
20 modes driving `eps_op` sit in the bottom 7 % of true stiffness. Representation error
and design sensitivity are competing on the same directions.

It also puts a measurement where GPT Pro's derivative-oscillation objection currently
sits as a caveat. The open question is no longer rhetorical: **does the assembled
compliance, and its `tau` sensitivity, jump when that cell is born, or is the jump
confined to directions that carry no compliance?** If compliance is smooth through
`eps` = 0 while `eps_op` jumps by 0.6, then `eps_op` over-weights physically irrelevant
directions and the 0.15 gate is if anything conservative. If compliance jumps too, then
`tau` topology optimisation on this teacher has a problem the network cannot cause and
cannot fix. Neither is currently known; `superelement/objective/tau_gate.py` measures it.

## 4. The teacher's cost, decomposed at last

`METADATA_DELIVERY/PATH_*_NUMERIC_R2/RESULT.json` records the stage timings of a full
teacher run from geometry to `S`, at `n = 32`, `q ≈ 12 798`:

| stage | seconds |
|---|---|
| volume assembly (the cut and its quadrature) | 128.4, 128.8, 128.8, 137.3 |
| ghost penalty | 11.0, 11.0, 11.3, 11.0 |
| operator export (the condensation to `S`) | 39.9, 39.6, 40.2, 39.9 |
| **total** | **179.3, 179.3, 180.3, 188.1** |

`ROUTE5_COST_20260917.md` calls the re-cut cost *the decisive unknown* and assumes
0.05–5 s per cell. It is **~140 s** — 28x the top of that band — and the condensation is
40 s, consistent with the corrected teacher figure in `TEACHER_COST_20260917.md`.

Consequences:

- One design iteration of the teacher on a 1000-cell lattice is **~50 hours**, not the
  10.8–26.4 h in `ROUTE5_COST_20260917.md`.
- The network replaces all 180 s, not the 40 s condensation. At the 100 ms/cell target
  that is a **1800x** speedup, and 78 % of what it skips is the geometry pipeline.
- This strengthens the route-5 conclusion rather than weakening it: never forming `S`
  saves the 40 s, but the 140 s cut is unavoidable for any method that starts from
  geometry — except a network, which does not.

## 5. What is still blocked

The *upstream* ingest — geometry → `K` → `S` — has not been run from this workstream.
So these remain blocked, and `THE_BLOCKER_20260917.md` is still right about them:

| measurement | still needs |
|---|---|
| Smetana–Patera floor scan | the **fine** `K_ii`, `K_it`, which is discarded during ingest |
| gamma sweep (1e−3, 1e−5) | new teacher runs; all 920 packets are `gamma = 1e−4` |
| `tau` labels on seats other than 0328 | the 180 s/cell numeric stage, per seat per `eps` |

The third is now merely expensive rather than impossible: `perturbed_case` in
`full_factor_geometry.py` is a registered harness that scales `tau_corners` by `1+eps`,
and section 3's four cases are its output.
