# Revision round 1: execution plan (author approved "按你的建议办", 2026-09-28)

## Decisions (all as recommended in DECISIONS_R1_CN.md)
- Text-only fixes 1–14: apply.
- D1 memory: GiB everywhere; after E1, the Abstract leads with lattice-level memory; the main text reports both conventions (with and without K).
- D2 contributions: inherited results cited + three new contributions. "Accurate compliance ≠ accurate sensitivity" becomes a design criterion, no longer the principal finding.
- D3 §6.8: reframe as a reduced-boundary ablation of our own pipeline, about 200 words in the main text. Drop the 78–85% figure and the PIML attribution from the Introduction. Delete or re-attribute the "Huang 2023 corner-linear 3D" claim. Decline R7's fair-PIML re-implementation, with the rebuttal from VERIFY_T5T7.
- D4 H2/y: one sentence in the supplement ("ill-posed configuration, excluded"); replace "fourteen" with the actual count everywhere.
- D5: the 92-cell hlat554 run stays in P2 and is not used in P1.
- D6 labels: A3 → NICE; B+W → NICE-post; A2b → Smoothing-trained; C → Uncorrected; B → Base network. S8 and P0 appear in the supplement only.
- D7: new Abstract (~240 words). One sentence is added after §6.11.
- D8: title decided after §6.11. Candidate: "Neural-initialised static condensation with equilibrium correction for the analysis and thickness design of cut thin-walled TPMS lattices".
- D9: main text ~9,500 words excluding §6.11; §7 ~850 words.
- D10: availability statement: cleaned, frozen code snapshot plus evaluation data; training data on request.
- E8 (w_s = 0 ablation): NOT run. Remove "with an objective that includes thickness sensitivity" as a claimed benefit. The sensitivity term is only described as part of the objective, with no claim of its effect.
- E9 (AMG whole-lattice baseline): NOT run. Narrow the wording: NICE is positioned as an inexact substructuring solver. No claim of superiority over whole-lattice iterative solvers. AMG/BDDC comparison is listed in Limitations.
- G-cell data stays out of P1.

## Approved experiments
| ID | Content | Owner |
|---|---|---|
| E1 | Host baseline done properly: PARDISO Cholesky (mtype 2), explicit iparm, direct phase-33 calls without the pypardiso hash overhead, parallel solve, Schur option (iparm 36) for the explicit S, 3 repetitions; plus a real 2×2×2 Cholesky whole-lattice solve. Must run on a quiet host (after all GPU jobs). | X1 |
| E7 | CutFEM refinement to n = 56/64 on H1, H2, 2051 and the thinnest-wall cell (CPU only). | X1 |
| E2 | C+W clean control: 80 geometries + all pair configurations. Pre-registered rule: if A3/(C+W) ≤ 1.1 on force_c, contribution (ii) is downgraded. | X2 |
| E5 | Re-evaluate the 80 geometries storing per-direction errors (p95/p99/max) for A3, B+W and C. | X2 |
| E6 | Held-out pair assemblies: about 9 cells not used in selection, including A3's 5 worst geometries. | X2 |
| E3 | Complete surrogate derivative: fixed-q̂ central-difference check of Ĉ,c against s̃_c and the exact s_c, on the pair configurations and both 4-cell lattices. | X3 |
| E4 | Instrumented lattice re-run: ρ, Ūᵀρ, J(Ū), dual-norm residual bound, per-cell vectors, global-gradient metrics. | X3 |
| E14 | Element-level PSD check on the detailed cells. | X3 |

GPU queue order: DF pilot (running) → accuracy jobs (E2, E5, E6, E3, E4; these may share the GPU if memory allows) → E1 timing on a quiet host. E7 may run on the CPU earlier if RAM allows.

## Text streams (stage 3)
- M: MANUSCRIPT_EN.md (all sections) + references_verified.bib. Numbers that depend on experiments get `[PENDING E#]` markers.
- S: APPENDICES_EN.md + SUPPLEMENTARY_EN.md + table/figure generators (labels, units, numbering after the removals).
