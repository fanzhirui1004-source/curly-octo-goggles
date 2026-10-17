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
| OPL_V3_GRAPH / OPL_V3_GRAPH_MAXB / OPL_V3_GRAPH_STAGES | 1 / 4 / 4 | CUDA graph per cell for applies of at most 4 columns (the PCG matvec), replayed on the cell's stream; streamed cells' states copied ahead into 4 device stage buffers (cellgraph.py; 51-cell plate solve -6%) (S26) |
| OPL_V3_CONVT | 1 | transposed U-Net convolutions as forward convolutions with the flipped, channel-swapped weight: 2.8x faster, deterministic; rounding of each convolution differs (S27) |
| OPL_V3_KPP_FP32 | 0 | category B (not bitwise): K_PP factor and solves in float32 (S21b) |
| OPL_V3_SYMV2 | 1 | float32 element product: one / two columns by a single-read kernel (ke_symv2), wider blocks with retuned tiles: 168.5 -> 120.9 us per product at one column; the summation order differs (P2) |
| OPL_V3_SENS_NOFWD | 1 | cell_sens skips the clipped tetrahedra's forward moments (discarded; the VJP reads only the vertices): bitwise the same, sensitivities -16% (P6a) |
| OPL_V3_FE_THREAD | 1 | opt_design front end: the next cell's constructor, assembly, network files and geometry in a worker thread on the same CUDA stream while this cell's GPU work runs: front end -16% (51 cells) / -18% (110 cells) (P5) |
| OPL_V3_KMAP | 0 | float32 element product: full elements whose packed matrix is bitwise equal to one representative read that row (same values, fewer bytes) (P2b). Finds nothing on plate cells: every full element's float32 moment row is distinct (the float64 rows agree to 1e-12 of the largest entry, which is above float32 resolution for the small entries); kept as a no-op switch |
| OPL_V3_PROF_ITER / _OUT / _TRACE | 0 | diagnostic: device-level profile of PCG iteration k's operator application (gpuprof.py) |
| OPL_V3_CLAMPFREE | 1 | clamped hyperedge layers without the where(port, X0, X) pass: the layer input's port rows already are X0's (checked bitwise at every layer, B = 1, 3, 16), the scatter adds onto a copy of X or onto X itself; S_hat -1% / -4.9% / -5.8% at B = 1 / 3 / 16 (P11) |
| OPL_V3_HFUSE | 0 | not adopted: a hyperedge layer in one kernel (hyperfused; the 4-D register form 4-6x slower, the 2-D dot form no faster than gather + cuBLAS bmm + scatter, S_hat +1% / +4%) (P8) |
| OPL_V3_GFFUSE | 0 | not adopted: 1 = ghost-face product per template in one kernel with atomics (gffused; alone -2% at one column, 1.5-2x slower at 3 / 16; S_hat +11% / +24% / +28% at B = 1 / 3 / 16); 2 = two-phase without atomics (P9, P9b) |
| OPL_V3_CONV3 | 1 | U-Net 3x3x3 convolutions as one Triton implicit-GEMM kernel over the active voxels only (conv3.conv_sparse: reads the sparse layout, gate and residual fused; float32 'ieee' dots, no TF32): level-0 convolution 105 -> 38 us at one column; S_hat -8.0% / -4.9% / -3.8% at B = 1 / 3 / 16; rounding of each convolution's sums differs (P10, P10b) |
| OPL_V3_MSPLIT_IP | 1 | adjoint network: the mask split in place over the port rows only (X0b += port rows of X, which are then zeroed; same values bitwise): 313 -> 15 us at 16 columns; S_hat -0.7% / -1.7% / -4.5% at B = 1 / 3 / 16 (P12) |
| OPL_V3_KMF / _MAXB / _PTS | 1 / 16 / equi | float32 correction products of at most MAXB columns without the element matrices (P13): the moment-fitted Q2 stiffness integrated exactly at 5^3 points with fitted per-element weights (Lame constants folded in; equispaced points: Q2 tables exact in float32), CUDA C++ sum-factorised kernel (kmf_kernel.cu, built with nvcc on first use and cached in ~/.cache/opl_v3_kmf); ghost faces as 45 x 45 Gram matrices per face (b_d (x) I_3). Without nvcc the packed matrices as before. Not bitwise: removes the packed path's systematic float32 error, compliance moves toward the float64-correction reference (hlat222 3.2e-6 -> 2e-8 off it, 24-cell plate 7.0e-6 -> 3e-9); category B, see CHANGES_CN P13 |
| OPL_V3_KMF_GF2 / _EF | auto / 0 | ghost faces: DOF-parallel kernel from two columns on (auto); P13b: elements and ghost faces in one kernel (EF, under test) |
| OPL_V3_KMF64 | 1 | float64 K product without stored element matrices (P14): the same kernel in float64 for the partial elements, fullke's shared-matrix GEMM for the full ones, float64 ghost-face Gram matrices; fp64 rounding level (2.4e-15); 51-cell plate host peak 27.0 -> 19.0 GiB |
| OPL_V3_KC64 | 0 | check only: the float32 correction's K products in float64 (rounded), the reference for P13 |
| OPL_V3_PCG_BLOCK | 0 | category B: block PCG for several right-hand sides (lat_precond.bpcg) (P1) |
| OPL_V3_PCG_RITZ | (off) | category B: "m,k": after m steps the k smallest Ritz vectors join the coarse space, PCG restarts (lat_precond.rpcg) (P1) |
