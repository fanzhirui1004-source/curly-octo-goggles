> **Corrected 2026-09-17, later the same day.** The central claim here — *"nobody in this
> workstream can drive the frozen CutFEM pipeline"* — is wrong for the label stage. I was
> reading `full_factor_geometry.py`; the label producer is
> `stage_cutfem_neural_a/dense_reference.py`, and its trace-compile step
> (`data.prepare`) builds `TRACE_CACHE.npz` from the packet's own `TRACE.npz` in plain
> numpy — no solver, no topology compile. Running it on seat 0253's packet reproduced the
> frozen `R_UPPER.npy` **bit-for-bit** (`factor_sha256 7ee508ff…f96a453`) in 4.65 s.
> The doc's own unblocking self-test therefore passes for the label stage. See
> `docs/PIPELINE_UNBLOCKED_20260917.md`. The *upstream* ingest (geometry → `K` → `S`)
> is still unrun here, so the fine-`K` floor scan and the gamma sweep remain blocked.

# Four gating measurements, one blocker

2026-09-17, found while executing the approved plan. This changes the priority order.

## What happened

The plan's next two items were the route-5 geometry timing and the Smetana-Patera
floor scan. Both turned out to need data this project does not retain.

**The fine stiffness matrix `K` is not stored.** `REFERENCE_0328/` holds `RESULT.json`,
`RUN.json`, `R_UPPER.npy` (the fp64 Cholesky factor of `S`, 654 MB) and
`input/{INPUT.json, TRACE_CACHE.npz}`. The packet holds `S_UPPER.npy` (fp32 dense `S`,
327 MB), `TRACE.npz`, `PROBES.npz`, `SAMPLE.json`. **Condensation happens upstream in
the ingest pipeline and `K` is discarded.**

**The geometry/quadrature stage is not separately timed.** `REFERENCE_0328/RESULT.json`
records `total_seconds = 4.363` and `load_seconds = 0.244` - that is the **Cholesky of
`S`**, not the condensation and not the cut. The only decomposition of the teacher's
cost anywhere is `HANDOFF_20260913`'s bundled *"geometry and factor preparation ~29 s"*.

## The four things that are blocked, and what each needs

| measurement | why it matters | what it needs |
|---|---|---|
| **`dS/dtau` gate** | GPT Pro's derivative-oscillation counterexample makes this mandatory, not optional: value accuracy does not imply derivative accuracy, and `tau` is the design variable | two new teacher labels at `tau +- delta`, same seat |
| **geometry re-cut timing** | the decisive unknown in `ROUTE5_COST_20260917.md`; the 0.05-5 s/cell band spans the entire route-5 verdict | time the cut/quadrature stage alone, one cell |
| **Smetana-Patera floor scan** | the highest-leverage experiment on the board - it decides route 4's ceiling AND supplies route 5's coarse space | the **fine** `K_ii`, `K_it` (the collar's `K_COLLAR1.npz` has the core already coarsened, so it cannot answer whether enrichment lowers the 1.111 floor) |
| **gamma sweep** | whether the decisive soft directions are defined by the stabilisation rather than the physics; open since 9/16 | two new teacher labels at `gamma` = 1e-3, 1e-5 |

**Four gating measurements, one blocker: nobody in this workstream can drive the frozen
CutFEM pipeline.**

## What the pipeline looks like

`/root/autodl-tmp/CUTFEM_GP_TASK_V2_20260915_R1/FROZEN_FE_6624dc8/scripts/` - several
hundred scripts; `cutfem_full_interface_stage.py` is only a wrapper taking `--run-id`
and a REMAINDER command. The numerical source is
`/root/autodl-tmp/CUTFEM_INGEST_R38/source_independent_6624dc8_20260910`, commit
`6624dc8`. The runtime is
`/root/autodl-tmp/CUTFEM_GP_TASK_V2_20260915_R1/RUNTIME_RECOVERY/environment/.venv`.

## The unblocking step, and the order it must be done in

**Reproduce one existing label bit-for-bit first.** Take seat 0328, run the pipeline,
and require the resulting `S` (or its Cholesky factor) to match `R_UPPER.npy` to the
recorded `teacher_sha256`. Only after that self-test passes does anything else become
trustworthy - a wrong quadrature policy, ghost-penalty face set or ordering produces a
confident and wrong answer, which is worse than no answer. That failure mode has bitten
this project repeatedly, and twice today in my own work.

Once it passes, all four unlock, and two of them are minutes of compute:

1. `K` retained -> floor scan (no new labels needed, just re-run one cell with `K` kept)
2. stage timing -> route 5 decided
3. `tau +- delta` -> the derivative gate
4. `gamma` 1e-3 / 1e-5 -> the teacher's own well-posedness

## What can proceed without it

| item | status |
|---|---|
| route 3 (`A^{1/2}`) | **done today.** Compressible at a ~10% number penalty at the 0.15 gate; per-entry amplification **112** against Cholesky's 104 - essentially identical. The route stands, and its advantages are on the learnability side (unique, permutation-equivariant, 48-fold symmetry exact, outside the TMLR triangular-factorization negative result) |
| route 4 (preconditioned coordinates) | running; the partial result already shows the scaling quoted in `REVIEW_20260916` has its direction inverted (46x worse), and the correct direction buys 1.6x, not orders |
| route 1 memorisation in corrected coordinates | **can start now** - Codex's harness implements 90% of it |
| route 2 gauge-fixed parameterisation | a design problem, needs no data |
| **freeze a holdout** | **overdue and free.** Every seat in every run is `split: train`; the project has never evaluated anything out of sample |
