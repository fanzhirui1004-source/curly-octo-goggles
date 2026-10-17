# src_v3: working copy for the speed / memory work (not the paper code)

Copy of `docs/data/src_v2_prod_20261017` (= `/root/autodl-tmp/OPL/src_v2` on the RTX 5090 machine, 17 Oct 2026), changed
for speed and memory; the machine copy is `/root/autodl-tmp/OPL/src_v3`. The paper's numbers come from src_v2, which
stays untouched. Plan, scripts and results: `docs/speedup_v3/`.

Changes are listed in `docs/speedup_v3/CHANGES_CN.md` (one entry per change, with its verification) and summarised in
`docs/speedup_v3/REPORT_CN.md`. Every change keeps the numerical method; results agree with src_v2 to rounding (same PCG
iteration counts, compliance within 1e-7, sensitivities within the run-to-run noise of src_v2 itself).

## Switches (environment variables) and defaults

| Variable | Default | What it does (S-number in CHANGES_CN.md) |
| --- | --- | --- |
| OPL_V3_SLIM | 1 | state slimming, shared constants, int32 K_PP triplets, lean / lazy K_PP assembly (S1, S7) |
| OPL_STREAM_MIN | 0 | smallest tensor that is streamed (src_v2: 1 MiB) (S1) |
| OPL_FI_TRITON | 1 | Triton row gather in fastidx, no cached flat indices (S1) |
| OPL_COARSE_PACK | 1 | packed symmetric fp32 coarse inverse + matrix-free Q1 prolongation (S2) |
| OPL_COARSE_INV_TOL | 1e-4 | accuracy check of the packed inverse (fallback: dense fp32 factor) (S2) |
| OPL_COARSE_BLOCKS | 1 | lattice coarse space: per-cell A Z blocks (S3) |
| OPL_V3_SWITCHES | 1 | OPL_KE_SYMV=1, OPL_KE_SYMV64=1 by default (packed element products) (S4) |
| OPL_FULL_KE / OPL_FULL_KE32 | 1 / 0 | full elements share one K_e in the fp64 product only (S4; see the fp32 finding) |
| OPL_V3_SLOTS | store | `recompute`: slot weights recomputed per application (host memory, +30% S_hat) (S5) |
| OPL_KPP_HYBRID / OPL_KPP_MARGIN_GB | auto / 1.5 | cuDSS hybrid-memory fallback for the K_PP factor (S6) |
| OPL_V3_CHEB | 1 | fused Chebyshev sweeps (chebfused.py), bitwise the same iterates (S8) |
| OPL_V3_KE64 | store | keep the fp64 partial-element matrices per design (`form`: rebuild per product) (S9) |
| OPL_V3_HTILE | 1 | tiled hyperedge gather / scatter for B * F <= 128 (S10) |
| OPL_V3_STREAMS / OPL_V3_STREAM_MAXCOLS | 2 / 4 | cell applies of at most 4 columns on 2 CUDA streams (S11) |
| OPL_V3_PIPE_BODIES | 1 | opt_design: body generation overlapped with the analysis front end (S12) |
| OPL_V3_E0FAST | 1 | reduced tetrahedron backward kernel for the volume column of the sensitivities (S13) |
| OPL_V3_GFCAT | 1 | one concatenated ghost-face DOF array: 8 launches per ghost-face product instead of 12 (S20; 51-cell plate solve -7%) |
| OPL_V3_GFBMM | 0 | ghost-face product as a masked gather, two batched GEMMs over the padded templates and a masked scatter: 4 kernels instead of 12 (S20b; same plate time as GFCAT, slower on hlat222) |
| OPL_V3_SLOTFUSE | 1 | recomputed slot weights: the heads' scale and soft clip in one Triton pass, bitwise the same (S19) |
| OPL_V3_BUDGET_ORDER | ke64 | budget priority: float64 matrices first (more time saved per byte; `slots`: S14 order) (S14c) |
| OPL_V3_HOST_TOP / OPL_V3_HOST_TRANSIENT | 0.95 / 4 | host budget target: fraction of the limit, minus a reserve of 4 x the cells' K_PP host triplets for the factorisation peak (S14b); OPL_V3_HOST_FRAC: S14's fixed fraction instead |
| OPL_COARSE_BLK / OPL_COARSE_BLK_MAXFRAC | 1 / 0.5 | block-structured cell coarse inverse: octants + separator Schur complement, 168 -> 29 MiB per full plate cell, more accurate than the packed inverse (S15) |
| (no switch) | | the drivers' per-cell K_PP host triplets are freed once the lattice K_PP is assembled (S16, with OPL_V3_SLIM) |
| OPL_V3_GC | young | per-cell garbage collections of the front end: young generations only, one full collection after it (`full`: as src_v2) (S17) |
| (no switch) | | a streamed cell is copied to the device once for all chunks of a wide apply (S18) |
| OPL_V3_SLOTASSIGN | 0 | recomputed slot weights assigned to the element / ghost-face layers by one Triton kernel per set, bitwise the same (S19b; saves only 0.07 ms per recompute) |
| OPL_V3_KPP_FREEA | 1 | the K_PP CSR operand's device storages released after cuDSS factorises (the solve reads only the factors) (S21) |
| OPL_V3_COARSE_COLS | 1 | the coarse columns kept by coarse_prolong found without forming V (same columns) (S22) |
| --resident-gb auto | (4) | lat_scale / opt_design: the resident-cell device budget from the card's memory and the cell count (hostmem.resident_auto; OPL_V3_RGB_FIXED / _PERCELL / _MARGIN) (S23) |
| OPL_V3_NETFUSE | 1 | fused network paths (netfused.py): port clamp in the forward scatter, adjoint masks in one kernel, U-Net convolution input / output in one kernel each (S24) |
| OPL_V3_SPILL / OPL_V3_SPILL_DIR | 0 / V3W/spill | around the K_PP planning peak, write parked cells' page-locked states to disk and read them back; the budget then needs no reserve (S25) |
| OPL_V3_GRAPH / OPL_V3_GRAPH_MAXB | 0 / 4 | CUDA graph per cell for applies of at most 4 columns (the PCG matvec), replayed on the cell's stream (cellgraph.py) (S26) |
| OPL_V3_KPP_FP32 | 0 | category B (not bitwise): K_PP factor and solves in float32 (S21b) |
