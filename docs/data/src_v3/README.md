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
| OPL_V3_GFCAT | 0 | one concatenated ghost-face DOF array (experimental) |
