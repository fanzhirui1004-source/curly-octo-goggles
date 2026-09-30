<!-- Draft of Supplementary Note S9 for Section 6.11 (review round 1). Tables ST21 to ST25 are the output of
docs/paper_p1/review_r1/results/X6_opt/tables_s9.py, pasted unchanged; every number in the text comes from
FACTS_6_11.json or from that script (internal list of sources in the comment at the end). Visible placeholders:
[TBD-EXACT: ...] for the exact checks of the plate designs; the scale demonstration (S9.6, Table ST26) was filled on
2026-10-01 from results/X6_opt/scale (scale_summary.py/json, scale_dofs.json).
Open items for the coordinator: (1) the reference Svanberg (1987) is not yet in the reference list of the manuscript;
(2) the generator logs of the failing cells (body/<case>.log.attempt* and body/<case>.log.before_eps* of optA,
plateB1, plateB2 and xstartH_z on the GPU host) are not archived in X6_opt, so the stated cause of the generation
failures (Table ST24a note, S9.1) rests on the opt_design.py documentation of the fallback; archive the logs or an
excerpt next to the run records. -->

## Supplementary Note S9. Design optimisation records

This note gives the settings and complete records of the thickness optimisations of Section 6.11: the optimiser and its constraints (S9.1, Table ST21); the \(2\times2\times2\) block of Section 6.9, optimised with NICE and, from the same start, with exact condensation (case A; S9.2, Table ST22); the cut plates B1 and B2, the designs Hom-\(y\) and Hom-\(z\) obtained from a homogenised model, and the NICE optimisations X-\(y\) and X-\(z\) continued from them (S9.3, Table ST23); the geometry-generation fallback and the discrete switches between iterations (S9.4, Table ST24); checks of the analysis route (S9.5, Table ST25); and the scale demonstration and the exact verification of the plate designs (S9.6). All NICE runs use one RTX 5090. Compliance is given in the units of Section 6.1 (\(E_Y=1\), unit cell), memory in GiB (\(2^{30}\) bytes), and times are wall-clock times of complete design iterations, geometry generation included, unless stated otherwise. The geometry of the start design was generated beforehand and is not included at iteration 0 of any run; the same holds at iteration 17 of the exact twin and, in part, at iteration 5 of B1 and B2, where geometry generated earlier was reused.

### S9.1. Optimiser and constraints

The design variables are the corner thickness parameters at the lattice vertices, \(\boldsymbol\tau_g\). Every cell meeting at a vertex takes its value, \(\boldsymbol\tau_m=\boldsymbol\tau_m(\boldsymbol\tau_g)\), so that the thickness field is continuous across shared faces. The objective is the compliance \(\widehat C=f_g^T\bar U\) of the NICE lattice under one consistent face traction of unit resultant. The optimiser receives the field-based estimate of its gradient, \(\sum_m(\partial\boldsymbol\tau_m/\partial\boldsymbol\tau_g)^T\widetilde{\boldsymbol s}_m\), with \(\widetilde s_c\) from Eq. (13) for every corner of every cell; this estimates the exact gradient and is not the complete derivative of the surrogate compliance (Eq. (14), Section 7.3). The vertices in the plane of the loaded face keep their initial values. The thickness field on that face, the material part of the face over which the traction is integrated, and hence the nodal load, then do not depend on the design, and no load-derivative term \(2f_{g,c}^T\widehat U\) arises (Appendix H).

The volume \(V\) is the sum of the zeroth element moments of all cells, that is, the material volume of the discrete model. It is bounded by \(V^*=0.8\,V(\boldsymbol\tau^0)\) or, for the runs continued from the homogenisation designs, by the absolute bound of B1 and B2. The corner span and the gradient norm of every cell's trilinear thickness field are limited to 0.45 and the parameters to [0.18, 0.69], inside the training limits of 0.47 and [0.1752, 0.6993] (Table ST02). The span constraints are linear in the parameters. The gradient norm is evaluated at every corner from the three edge differences; this bounds it over the whole cell, because the largest gradient norm of a trilinear field is attained at a corner. Over all designs analysed, the largest span was 0.4500 and the largest gradient norm 0.4518, and the parameters stayed in [0.1800, 0.6900], so every analysed cell lay within the training limits.

The update is the standard method of moving asymptotes [Svanberg (1987)](https://doi.org/10.1002/nme.1620240207): one convex separable approximation per analysis, solved by a primal–dual interior-point method, with no line search and no conservativeness test (Table ST21). The optimiser thus uses the gradient estimate only to build its approximation and never tests the decrease of \(\widehat C\) along a search direction, where the inconsistency discussed in Section 7.3 would matter. In the runs from an initial design (case A, B1 and B2) the initial volume exceeds the bound by 25%, and the move limit of 0.0255 per iteration brings it to the bound at iteration 4. Meanwhile the compliance rises as material is removed: in case A to 37.86 at iteration 4 (Table ST22a), in B1 to 129.69 at iteration 3 and in B2 to 2,129.57 at iteration 4. In the first three or four iterations the elastic variables of the subproblem are positive (at most 0.19). The optimisation stops when the largest parameter change falls below \(10^{-3}\) or the relative change of the objective stays below \(10^{-4}\) in three consecutive iterations. Neither the gradient norm nor a KKT residual is used for stopping, since the estimate is not the gradient of \(\widehat C\) and \(\widehat C\) changes its discrete model between iterations (S9.4).

Every iteration regenerates the geometry of every cell from its current corner parameters, with the geometry generator used for all cells of this study, and rebuilds every learned operator. The lattice is solved with the arithmetic and the preconditioner of the timed route of Table 5: preconditioned conjugate gradients with the balanced two-level preconditioner of Supplementary Note S6.1 to a recursive relative residual of \(10^{-6}\), with the network and the correction in single precision. The solve starts from the previous iteration's solution, matched coordinate by coordinate by absolute grid position, displacement component and private cut-band flag (unmatched coordinates start at zero) and scaled by the energy-optimal factor \(f_g^TX_0/(X_0^T\widehat{\mathbb K}X_0)\); at iteration 0 of every run, and at iteration 16 of case A and iteration 5 of B1 and B2, it started from zero. In case A all cells stay on the GPU; in the plates, cells stay on the GPU while its allocated memory is below 22 GiB and the others are streamed from host memory, as for the eight-cell lattices of Supplementary Note S5. The sensitivities and the volume gradient are obtained in one reverse-mode pass through the moment integrals at the fixed recovered field; the vertex sensitivities agree with those from the central moment differences of Eq. (H.6) to \(4.6\times10^{-7}\) (Table ST25). The optimisation runs do not use the four implementation options of the timed run of Table 5 (fused per-tetrahedron moment kernels, template-based coarse Galerkin matrices, a sparse coarse space and element gathers; record `evidence/d5_off.json`), and their phase times (Table ST22c) are therefore not comparable with those of Table ST17d.

Geometry generation failed in five design iterations over all runs: the geometry generator could not certify the material patches of an element (Appendix A.1). The fallback of Table ST21 then multiplied the free vertices of the failing cells by \(1+\epsilon\), with \(\epsilon=\pm10^{-4}\), then \(\pm10^{-3}\) and \(\pm3\times10^{-3}\) in turn, regenerated every cell sharing them, and analysed and continued from the perturbed design. It succeeded at its first or third perturbation (\(\epsilon=10^{-4}\) or \(10^{-3}\)) and changed a corner parameter by at most \(5.8\times10^{-4}\) (Table ST24a). At the first failures (iteration 16 of case A, iteration 5 of B1 and B2), perturbations of the failing cell's own corner parameters by up to \(10^{-8}\) had not resolved them (Table ST24a).

### Table ST21. Optimiser, constraints and analysis settings

| Item | Setting |
| --- | --- |
| Design variables | Corner thickness parameters at the lattice vertices \(\boldsymbol\tau_g\), each shared by the cells meeting there; \(\boldsymbol\tau_m=\boldsymbol\tau_m(\boldsymbol\tau_g)\) copies the vertex values to the corners of cell \(m\). Case A: 27 vertices, 18 free; plates: 74 vertices, 64 free |
| Fixed vertices | The vertices in the plane of the loaded face keep their initial values (9 in case A, 10 in the plates), so that the thickness field on that face, the material part of the face and its consistent nodal load do not change with the design (Appendix H: no load-derivative term) |
| Objective | Compliance \(\widehat C=f_g^T\bar U\) under one consistent face traction of unit resultant, divided by its value at the first iteration |
| Gradient | Field-based estimate \(\sum_m(\partial\boldsymbol\tau_m/\partial\boldsymbol\tau_g)^T\widetilde{\boldsymbol s}_m\), \(\widetilde s_c\) from Eq. (13) of every cell; reverse-mode differentiation of the moment integrals at the fixed recovered field. The complete surrogate derivative of Eq. (14) is not used. Exact twin: exact sensitivities from the exact field with the moment derivatives of Eq. (H.6) |
| Volume constraint | \(V/V^*-1\le0\); \(V\) = sum of the zeroth element moments of all cells (material volume of the discrete model), its derivative from the same reverse pass (exact twin: Eq. (H.6)). \(V^*=0.8\,V(\boldsymbol\tau^0)\): 1.03558 (case A), 3.65526 (plates); cross-starts: the same absolute \(V^*=3.65526\) |
| Corner-span constraint | \((\tau_a-\tau_b)/0.45-1\le0\) for every ordered pair of distinct corner vertices of every cell (316 ordered pairs, 276 of them involving a free vertex, in case A; 938 and 896 in the plates); pairs of two fixed vertices are constant and are not passed to the optimiser; training limit 0.47 (Table ST02) |
| Gradient-norm constraint | \(\lvert\nabla\tau\rvert^2/0.45^2-1\le0\) at every corner of every cell, from the three edge differences (64 corner stencils in case A, 192 in the plates, each with a free vertex); for a trilinear field the largest gradient norm over the cell is attained at a corner; training limit 0.47 |
| Bounds | \(0.18\le\tau\le0.69\); training range [0.1752, 0.6993] (Table ST02) |
| Optimiser | Method of moving asymptotes, one update per analysis, no line search and no conservativeness test (no GCMMA inner loop); subproblem solved by a primal–dual interior-point method, the barrier parameter reduced tenfold from 1 to \(10^{-7}\) (at each value, Newton steps until the largest residual is below 0.9 of it, at most 200); \(a_0=1\), \(a_i=0\), \(c_i=1000\), \(d_i=1\) |
| Asymptotes | Initially \(x\pm0.5(x_{\max}-x_{\min})\) (first two iterations), then widened by 1.2 where consecutive steps have the same sign and narrowed by 0.7 where they alternate, kept between 0.01 and 10 variable ranges from \(x\); subproblem bounds at 0.1 of the distance to the asymptotes |
| Move limit | 0.05 of the variable range, 0.0255 per iteration |
| Stopping rule | \(\max\lvert\Delta\tau\rvert<10^{-3}\), or relative objective change below \(10^{-4}\) in three consecutive iterations, or 60 iterations; cross-starts: 12 iterations |
| Geometry | Every iteration regenerates the geometry of every cell from its current corner parameters with the geometry generator used for all cells of this study, in parallel processes on the host (8 for case A, 12 for the plates), and rebuilds every learned operator |
| Lattice solve | Preconditioned conjugate gradients with the balanced two-level preconditioner of Supplementary Note S6.1 to a recursive relative residual of \(10^{-6}\) (at most 3,000 iterations); network and correction in single precision, stiffness actions of the condensed product in double precision: the arithmetic and preconditioner of the timed route of Table 5, but without its implementation options (fused per-tetrahedron moment kernels, template-based coarse Galerkin matrices, sparse coarse space, element gathers; record `evidence/d5_off.json`), so that the phase times are not comparable with Table ST17d. Exact twin: dense exact condensed matrices of every cell, assembled solve to \(10^{-10}\) |
| Warm start | From the previous iteration's solution, matched coordinate by coordinate on absolute grid position, displacement component and private cut-band flag; unmatched coordinates start at zero; the start is scaled by the energy-optimal factor \(f_g^TX_0/(X_0^T\widehat{\mathbb K}X_0)\); the stopping criterion, relative to \(\lVert f_g\rVert\), is unchanged. At iteration 0 of every run, and at iteration 16 of case A and iteration 5 of B1 and B2, the solve started from zero. Exact twin: cold start |
| Operator placement | Case A: all cells on the GPU; plates: cells kept on the GPU while its allocated memory stayed below 22 GiB, the others streamed from host memory as in Supplementary Note S5 |
| Geometry-generation fallback | If generation fails for some cells, the free vertices of those cells are multiplied by \(1+\epsilon\), \(\epsilon=10^{-4},-10^{-4},10^{-3},-10^{-3},3\times10^{-3},-3\times10^{-3}\) in turn (clipped to the bounds), every cell sharing them is regenerated, and the perturbed design is analysed and continued from. In use from the start in the exact twin, X-\(y\) and X-\(z\), and from iteration 16 of case A and iteration 5 of B1 and B2, where an earlier, cell-local scheme had been tried first (Table ST24a) |

Over every design analysed on the fine scale (all iterations of all runs of Tables ST22 and ST23), the largest corner span is 0.4500, the largest gradient norm 0.4518 and the corner parameters lie in [0.1800, 0.6900]. Records in `docs/paper_p1/review_r1/results/X6_opt`: `meta.json` of each run (arguments at the start of the run) and the run logs `optA/optA.log`, `plates/plateB1.log`, `plates/plateB2.log` (arguments from iteration 16 of case A and iteration 5 of B1 and B2 onward); constants of `mma.py`; perturbations of the fallback in `opt_design.py`.

### S9.2. Case A: NICE optimisation, exact twin and exact checks

Case A is the \(2\times2\times2\) block of Section 6.9, with its graded initial corner parameters (0.252 to 0.535) and its four cut cells of 62% and 25% retained volume, clamped on the face \(y=\min\) and loaded by the consistent traction in \(y\) on the face \(y=\max\). Of its 27 vertices, the 9 in the loaded face are fixed and 18 are design variables; \(V^*=0.8\,V(\boldsymbol\tau^0)=1.03558\). The exact twin repeats the optimisation from the same start with the same settings, geometry regeneration and constraints, but with exact condensation: dense exact condensed matrices of every cell, the assembled solve to a recursive relative residual of \(10^{-10}\), exact sensitivities with the moment derivatives of Eq. (H.6), and a cold start. The NICE run was checked with the exact model at iterations 0, 12 and 23. At iteration 0 the design is that of Section 6.9, and the compliances, 24.59081 (NICE) and 24.59356 (exact), are the \(y\)-load values of Table ST17e.

The NICE run stopped after 24 iterations and the twin after 23, both by the objective-change rule (Tables ST22a and ST22c). Both paths first thin the block to the volume bound, the NICE compliance rising from 24.59 to 37.86 at iteration 4, and then redistribute material. Up to iteration 4 the two runs analyse the same design: while the volume bound is violated, every free parameter moves by the move limit or to the lower bound. At these iterations the difference of the two compliances is the surrogate error on the same design, which grows from −0.011% to −0.023% as material is removed. From iteration 5 the paths separate, with corner parameters differing by at most \(8.8\times10^{-4}\), and the NICE compliance lies 0.016–0.030% below the twin's, except at iterations 16 and 19, where the fallback perturbed the NICE design (−0.037% and −0.10%, the latter with the volume 0.036% above the bound). Both final designs reach the lower bound 0.18, have a largest parameter of 0.624, and have the gradient-norm constraint active (0.4500) and the span below its limit (0.444). Their corner parameters differ by at most 0.0051 (root mean square 0.0011), and the exact compliance of the NICE design, 24.110504, is \(1.25\times10^{-5}\) below that of the twin's design, 24.110806. The final compliance is 2.0% below that of the initial design, which holds 25% more material.

In the exact checks (Table ST22b), the surrogate error of the NICE compliance is −0.0112%, −0.0182% and −0.0279% at iterations 0, 12 and 23, and the error of the vertex gradient 0.069%, 0.17% and 0.33%, with cosines of at least 0.9999966 and the signs of all 18 components correct; the component errors are at most \(4.1\times10^{-3}\) of the largest gradient component (95th percentile at most \(2.8\times10^{-3}\)). Both errors increase along the path and remain below 0.03% and 0.33%. Over all iterations the recomputed residual lies between \(5.3\times10^{-5}\) and \(3.2\times10^{-4}\) and the signed residual work \(\lvert\bar U^T\rho\rvert\) is at most \(2.7\times10^{-8}\) of the compliance, as in Section 6.9. Table ST22b also lists a KKT residual computed with the exact gradient, \(\lVert s_g+\lambda\nabla V\rVert/\lVert s_g\rVert\) over the parameters strictly inside the bounds, with the volume multiplier \(\lambda\) fitted by least squares. It omits the multipliers of the span and gradient-norm constraints; since the gradient-norm constraint is active at the final design, its value there, 0.249, overstates the stationarity residual and is not a measure of optimality.

A NICE iteration took 141 s on average (128–171 s), 3,380 s for all 24 iterations together, with peaks of 15.7 GiB of GPU and 3.5 GiB of host memory. The longest iteration and the largest number of conjugate-gradient iterations, 170.7 s and 170 at iteration 16, occur where the solve started from zero and the geometry needed a second generation attempt (Tables ST22a and ST24a). The twin, which forms the dense exact condensed matrices of every cell on the same GPU, took 469 s per iteration and 10,793 s for its 23 iterations, with 17.3 GiB of GPU and 80.0 GiB of host memory. The twin is a verification run; its times are given for completeness, and the cost comparison of the exact and learned routes is that of Section 6.10.

### Table ST22. Case A: NICE optimisation, exact twin and exact checks

#### ST22a. Iteration history

| Iteration | \(\widehat C\) (NICE) | \(V/V^*\) | PCG iterations | Recomputed residual | \(\bar U^T\rho/\widehat C\) | Time (s) | \(C\), exact twin | \(V/V^*\), exact twin | NICE vs twin (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 24.59081 | 1.25000 | 122 | 7.2e-05 | 2.7e-09 | 134.3 | 24.59356 | 1.25000 | −0.011 |
| 1 | 27.14662 | 1.18615 | 102 | 9.9e-05 | 2.3e-09 | 135.2 | 27.14994 | 1.18615 | −0.012 |
| 2 | 30.16595 | 1.12233 | 108 | 1.1e-04 | −1.9e-08 | 137.0 | 30.17025 | 1.12233 | −0.014 |
| 3 | 33.73641 | 1.05885 | 112 | 1.3e-04 | 1.5e-08 | 136.1 | 33.74275 | 1.05885 | −0.019 |
| 4 | 37.85673 | 0.99681 | 117 | 1.5e-04 | 2.6e-09 | 137.0 | 37.86557 | 0.99681 | −0.023 |
| 5 | 35.69282 | 0.99778 | 119 | 1.4e-04 | −1.7e-08 | 137.7 | 35.70004 | 0.99778 | −0.020 |
| 6 | 33.80530 | 0.99727 | 119 | 1.3e-04 | 8.2e-09 | 137.3 | 33.81157 | 0.99727 | −0.019 |
| 7 | 32.05341 | 0.99771 | 121 | 1.2e-04 | −9.4e-09 | 137.8 | 32.05955 | 0.99771 | −0.019 |
| 8 | 30.48399 | 0.99810 | 139 | 9.6e-05 | 9.1e-10 | 145.8 | 30.49000 | 0.99810 | −0.020 |
| 9 | 29.08392 | 0.99844 | 129 | 9.5e-05 | 2.0e-08 | 142.0 | 29.08994 | 0.99844 | −0.021 |
| 10 | 27.85699 | 0.99871 | 145 | 3.2e-04 | 1.5e-08 | 147.0 | 27.86145 | 0.99872 | −0.016 |
| 11 | 26.84866 | 0.99874 | 142 | 8.3e-05 | 1.4e-08 | 145.7 | 26.85332 | 0.99874 | −0.017 |
| 12 | 26.00158 | 0.99867 | 129 | 1.4e-04 | 6.8e-09 | 139.8 | 26.00632 | 0.99867 | −0.018 |
| 13 | 25.28700 | 0.99878 | 133 | 7.0e-05 | 2.0e-09 | 142.6 | 25.29168 | 0.99879 | −0.018 |
| 14 | 24.78587 | 0.99856 | 128 | 6.5e-05 | −1.5e-10 | 140.6 | 24.79256 | 0.99856 | −0.027 |
| 15 | 24.46873 | 0.99892 | 137 | 6.4e-05 | 1.2e-08 | 142.9 | 24.47568 | 0.99892 | −0.028 |
| 16* | 24.24430 | 0.99932 | 170 | 5.3e-05 | −1.5e-08 | 170.7 | 24.25329 | 0.99928 | −0.037 |
| 17 | 24.14971 | 0.99956 | 157 | 6.2e-05 | 1.7e-08 | 151.7 | 24.15703 | 0.99956 | −0.030 |
| 18 | 24.10947 | 0.99993 | 121 | 6.3e-05 | −7.8e-09 | 136.7 | 24.11624 | 0.99992 | −0.028 |
| 19* | 24.08748 | 1.00036 | 104 | 6.2e-05 | 1.1e-08 | 169.0 | 24.11211 | 0.99999 | −0.102 |
| 20 | 24.10488 | 1.00000 | 99 | 6.2e-05 | 1.5e-08 | 127.8 | 24.11155 | 1.00000 | −0.028 |
| 21 | 24.10448 | 1.00000 | 100 | 6.2e-05 | −2.2e-08 | 128.5 | 24.11117 | 1.00000 | −0.028 |
| 22 | 24.10411 | 1.00000 | 100 | 6.3e-05 | −2.7e-08 | 128.1 | 24.11081 | 1.00000 | −0.028 |
| 23 | 24.10378 | 1.00000 | 101 | 6.2e-05 | −1.3e-08 | 128.5 | — | — | — |

\* Design perturbed by the geometry-generation fallback (Table ST24a). NICE vs twin: \((\widehat C-C_{\rm twin})/C_{\rm twin}\) at the same iteration index. The designs coincide (to 4e-14) at iterations 0–4, where every free parameter moves by the move limit or to the lower bound while the volume bound is violated; there the column is the surrogate error of the same design. From iteration 5 the corner parameters differ, by 3.3e-04 at iteration 5 and by at most 8.8e-04 over the remaining iterations. The solve started from zero at iterations 0 and 16. Recomputed residual: \(\lVert f_g-\widehat{\mathbb K}\bar U\rVert/\lVert f_g\rVert\); time: complete design iteration including geometry generation, except at iteration 0, whose geometry was generated beforehand.

#### ST22b. Exact checks of the NICE run

| Iteration | Exact \(C\) | \(\widehat C\) | Surrogate error (%) | Gradient error (%) | Cosine | Component error / \(\max\lvert g\rvert\): median / 95th percentile / max | Sign agreement | KKT residual, volume multiplier only | Exact solve: PCG iterations / recomputed residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 24.59356 | 24.59081 | −0.0112 | 0.069 | 0.9999998 | 2.2e-04 / 5.1e-04 / 6.3e-04 | 1.000 | 0.477 | 178 / 9.1e-11 |
| 12 | 26.00630 | 26.00158 | −0.0182 | 0.170 | 0.9999987 | 4.1e-04 / 1.2e-03 / 1.8e-03 | 1.000 | 0.352 | 244 / 9.0e-11 |
| 23 | 24.11050 | 24.10378 | −0.0279 | 0.329 | 0.9999966 | 3.8e-04 / 2.8e-03 / 4.1e-03 | 1.000 | 0.249 | 262 / 9.9e-11 |

Over the 18 free vertex parameters. Surrogate error \((\widehat C-C)/C\); gradient error \(\lVert\widetilde s_g-s_g\rVert/\lVert s_g\rVert\); component error \(\lvert\widetilde s_{g,i}-s_{g,i}\rvert/\max_j\lvert s_{g,j}\rvert\); KKT residual \(\lVert s_g+\lambda\nabla V\rVert/\lVert s_g\rVert\) over the parameters strictly inside the bounds, with the volume multiplier \(\lambda\) fitted by least squares and no other constraint.

#### ST22c. Final designs and cost

|  | NICE run | Exact twin |
| --- | --- | --- |
| Iterations; stopping rule | 24; objective change \(<10^{-4}\) three times in a row | 23; objective change \(<10^{-4}\) three times in a row |
| Compliance of the run, first → last iteration | 24.59081 → 24.10378 (\(\widehat C\)) | 24.59356 → 24.11081 (\(C\)) |
| Exact compliance of the final design | 24.110504 | 24.110806 |
| Final \(V/V^*\) | 0.999998 | 0.999998 |
| Final \(\tau\) range; largest corner span; largest gradient norm | 0.1800–0.6239; 0.4439; 0.4500 | 0.1800–0.6236; 0.4436; 0.4500 |
| PCG iterations; recomputed residual | 99–170; 5.3e-05–3.2e-04 | 178–262; 8.5e-11–9.9e-11 |
| Time per iteration, mean (range) (s); sum over all iterations (s) | 141 (128–171); 3,380 | 469 (455–534); 10,793 |
| Mean phases (s) | geometry 14.0, front end 20.7, lattice and \(K_{PP}\) assembly 0.1, preconditioner 6.8, PCG 52.7, sensitivities and volume gradient 46.4, MMA 0.005 | geometry 11.4, cell setup with moment derivatives 121.7, dense exact condensation 123.5, PCG 48.6, exact sensitivities 163.2, MMA 0.005 |
| Peak memory, GPU / host (GiB) | 15.7 / 3.5 | 17.3 / 80.0 |

Final designs: exact compliance of the NICE design relative to that of the twin's design −1.25e-05; corner parameters differ by at most 0.0051 (root mean square 0.0011). Mean phases over all iterations; the geometry phase is zero at iteration 0 of the NICE run and at iterations 0 and 17 of the twin, where the geometry had been generated beforehand. The phases of the NICE run are not comparable with Table ST17d, whose timed run used implementation options that these runs did not use (Table ST21).

### S9.3. Plates, homogenisation designs and cross-starts

The plates are a single layer of \(8\times4\) cells, cut by a plane that runs from the bottom-right corner to the top edge at two cells from the left end. Of the 32 cells, 24 remain: 16 uncut and 8 cut, two each with retained volume fractions 0.917, 0.667, 0.333 and 0.083. The layout is stored with its two in-plane axes interchanged, a cube-symmetry image of the plate, so that the cut normal lies at \(\vartheta=33.69^\circ\), within the range \(0<\vartheta<\pi/4\) of the validation geometries (Section 6.1), rather than at its image, 56.31°; the network is not equivariant under the cube symmetries (Section 7.5). The plate is clamped along its long edge of 8 cells and loaded on the opposite face, which after the cut is two cells wide, by a consistent traction of unit resultant: in the plane along the long edge (B1) or normal to the plate, bending the single layer (B2). Of the 74 vertices, the 10 in the plane of the loaded face are fixed and 64 are design variables; the start is uniform, \(\tau=0.40\), and \(V^*=0.8\,V(\boldsymbol\tau^0)=3.65526\).

B1 stopped after 23 iterations by the objective-change rule and B2 after 30 by the parameter-change rule (Table ST23a). With 20% less material than the uniform start, the NICE compliance falls by 7.7% for B1 (94.341 to 87.089) and by 8.3% for B2 (1,467.743 to 1,346.354). Both final designs span the full parameter range [0.180, 0.690], with the span and gradient-norm constraints active. An iteration took 574 s (B1) and 639 s (B2) on average, most of it in the conjugate-gradient solve (294 and 362 s, 138–295 iterations) and the sensitivities (160 and 159 s); the iterations took 3.7 and 5.3 h in total, with at most 24.3 GiB of GPU and 5.0 GiB of host memory. The recomputed residual reached \(3.8\times10^{-3}\) (B1) and \(6.7\times10^{-3}\) (B2), while the signed residual work was at most \(3.1\times10^{-7}\) of the compliance. All plate values are NICE values; their exact verification is pending (S9.6).

For comparison with a homogenisation-based graded design, the effective elasticity tensor \(C^H(\tau)\) of the uncut cell with uniform corner parameter was computed by periodic homogenisation on the same discrete model (\(n=32\), Q2 elements, the stabilised stiffness \(K\) with its ghost penalty). Nodes on opposite faces of the cell are identified, the fluctuation is periodic with one node fixed, and six unit macroscopic strains give \(C^H_{ij}=u_i^TKu_j\) for the unit cell; the material volume fraction \(V^H(\tau)\) is the sum of the zeroth element moments. At twelve thicknesses from 0.18 to 0.70 (Table ST23b), the tensor is cubic to \(3.1\times10^{-13}\) and the periodic fluctuations are in equilibrium to \(2.2\times10^{-13}\); cubic splines in \(\tau\) interpolate \(C^H_{11}\), \(C^H_{12}\), \(C^H_{44}\) and \(V^H\) (Figure S06). The macroscale model meshes the plate with Q1 hexahedra, six per cell and axis (4,464 elements). At every quadrature point it interpolates the local parameter \(\tau(\mathbf x)\) trilinearly from the corners of its cell and evaluates \(C^H(\tau(\mathbf x))\) and \(V^H(\tau(\mathbf x))\); it integrates the cut by a finite-cell rule (\(4^3\) sub-points in intersected elements, the void part carrying \(10^{-6}\,C^H(0.4)\)); and it applies the same clamped face and a uniform traction of unit resultant on the material part of the loaded face. It was optimised with adjoint sensitivities and the same MMA settings, constraints, fixed vertices and volume fraction. At the uniform design its volume agrees with the fine-scale material volume to \(1.2\times10^{-7}\).

At the uniform design the macroscale model underestimates the fine-scale NICE compliance by 26.7% for the in-plane load and by 36.6% for bending (Table ST23c). The plate, one cell thick and four cells wide, offers little separation of scales, most of all through its thickness; the error was not analysed further. The macroscale optimisations stopped after 23 and 27 iterations. Evaluated with NICE on the fine scale, the homogenisation design Hom-\(y\) has a compliance 1.07% above that of B1, with a volume 0.040% above the bound, and Hom-\(z\) one 0.34% below that of B2, with a volume 0.045% below the bound; for these graded designs the macroscale volume model no longer equals the fine-scale material volume exactly. The corner parameters of the B and Hom designs have correlation coefficients of 0.906 (\(y\)) and 0.969 (\(z\)) and differ by 0.074 and 0.045 in root mean square. Continued with NICE for 12 iterations under the absolute volume bound of B1 and B2, the homogenisation designs reach 87.141 (X-\(y\), 0.060% above B1) and 1,334.697 (X-\(z\), 0.87% below B2). Neither continuation met the stopping rule; the last parameter change of X-\(y\) was still close to the move limit (0.0254 of 0.0255). For each load the three final designs thus lie within 1.1% (\(y\)) and 0.9% (\(z\)) of each other in NICE compliance, and for bending the optimisation from the uniform design ended 0.87% above the design reached from the homogenisation start. Differences of a few hundredths of a percent, such as that between X-\(y\) and B1, are of the order of the surrogate errors of case A (Table ST22b); whether the differences between the designs hold in the exact model is part of the pending verification [TBD-EXACT: exact compliances of B1, B2, Hom-\(y\), Hom-\(z\), X-\(y\), X-\(z\) and the exact counterparts of Table ST23c].

### Table ST23. Plates: NICE optimisation, homogenisation designs and cross-starts

#### ST23a. Runs

|  | B1 | B2 | Hom-\(y\) | Hom-\(z\) | X-\(y\) | X-\(z\) |
| --- | --- | --- | --- | --- | --- | --- |
| Start design | uniform 0.40 | uniform 0.40 | uniform 0.40 (macroscale) | uniform 0.40 (macroscale) | Hom-\(y\) design | Hom-\(z\) design |
| Iterations; stopping rule | 23; objective change \(<10^{-4}\) three times in a row | 30; \(\max\lvert\Delta\tau\rvert<10^{-3}\) | 23 (macroscale); objective change \(<10^{-4}\) three times in a row | 27 (macroscale); \(\max\lvert\Delta\tau\rvert<10^{-3}\) | 12; fixed budget of 12 iterations | 12; fixed budget of 12 iterations |
| Compliance, first → last iteration | 94.341 → 87.089 | 1,467.743 → 1,346.354 | 69.177 → 66.566 (macroscale) | 930.693 → 859.925 (macroscale) | 88.019 → 87.141 | 1,341.784 → 1,334.697 |
| Last / first | 0.9231 | 0.9173 | 0.9623 (macroscale) | 0.9240 (macroscale) | 0.9900 | 0.9947 |
| Fine-scale NICE compliance of the final design | 87.089 | 1,346.354 | 88.019 | 1,341.784 | 87.141 | 1,334.697 |
| Final fine-scale \(V/V^*\) | 0.99999 | 1.00000 | 1.00040 | 0.99955 | 0.99998 | 1.00000 |
| Final \(\tau\) range | 0.180–0.690 | 0.180–0.690 | 0.180–0.690 | 0.180–0.690 | 0.180–0.690 | 0.180–0.690 |
| Largest corner span / gradient norm at the end | 0.450 / 0.450 | 0.450 / 0.450 | 0.450 / 0.450 | 0.450 / 0.450 | 0.450 / 0.450 | 0.450 / 0.450 |
| Last \(\max\lvert\Delta\tau\rvert\) | 0.0061 | 0.0005 | 0.0022 (macroscale) | 0.0006 (macroscale) | 0.0254 | 0.0033 |
| Time per iteration, mean (range) (s) | 574 (490–646) | 639 (521–688) | fine-scale evaluation 624 | fine-scale evaluation 674 | 545 (506–599) | 643 (611–666) |
| Sum of iteration times (s) | 13,200 | 19,177 | — | — | 6,542 | 7,715 |
| Mean phases (s): geometry / front end / preconditioner / PCG / sensitivities | 26 / 71 / 23 / 294 / 160 | 26 / 70 / 22 / 362 / 159 | 26 / 71 / 23 / 345 / 160 (evaluation) | 24 / 69 / 21 / 400 / 158 (evaluation) | 23 / 70 / 23 / 269 / 160 | 24 / 70 / 21 / 369 / 158 |
| PCG iterations | 138–248 | 163–295 | 245 (evaluation) | 292 (evaluation) | 160–245 | 242–292 |
| Recomputed residual | 2.6e-04–3.8e-03 | 3.5e-03–6.7e-03 | 2.8e-04 (evaluation) | 3.6e-03 (evaluation) | 2.9e-04–3.5e-04 | 3.5e-03–4.3e-03 |
| \(\max\lvert\bar U^T\rho\rvert/\widehat C\) | 2.5e-07 | 3.1e-07 | 1.3e-08 | 1.3e-07 | 4.4e-08 | 1.7e-07 |
| Peak memory, GPU / host (GiB) | 24.3 / 5.0 | 24.3 / 5.0 | 22.4 / 4.5 (evaluation) | 22.2 / 4.1 (evaluation) | 22.4 / 4.8 | 22.2 / 4.3 |
| Geometry-generation fallback applied | iteration 5 | iteration 5 | none | none | none | iteration 2 |

B1, B2: NICE optimisation from the uniform design, in-plane (B1) and out-of-plane (B2) load. Hom-\(y\), Hom-\(z\): optimisation of the homogenised macroscale model for the same loads (4,464 Q1 elements, 5,719 nodes), final design evaluated once with NICE on the fine scale. X-\(y\), X-\(z\): NICE optimisation continued from Hom-\(y\), Hom-\(z\) under the absolute \(V^*=3.65526\) of B1 and B2. \(V/V^*\) relative to that bound. Mean phases over all iterations; the geometry phase is zero at iteration 0, whose geometry was generated beforehand. Every fine-scale value is a NICE value; the exact verification is pending (S9.6).

#### ST23b. Homogenised law of the uniform-thickness cell

| \(\tau\) | \(V^H\) | \(C^H_{11}\) | \(C^H_{12}\) | \(C^H_{44}\) | DOFs |
| --- | --- | --- | --- | --- | --- |
| 0.18 | 0.1027 | 0.03996 | 0.02806 | 0.01920 | 221,556 |
| 0.22 | 0.1256 | 0.04992 | 0.03432 | 0.02384 | 248,484 |
| 0.27 | 0.1541 | 0.06307 | 0.04212 | 0.02988 | 264,900 |
| 0.32 | 0.1827 | 0.07712 | 0.04989 | 0.03620 | 283,908 |
| 0.37 | 0.2113 | 0.09218 | 0.05765 | 0.04279 | 307,092 |
| 0.42 | 0.2399 | 0.10838 | 0.06541 | 0.04968 | 336,612 |
| 0.47 | 0.2685 | 0.12585 | 0.07323 | 0.05688 | 351,876 |
| 0.52 | 0.2971 | 0.14471 | 0.08117 | 0.06440 | 370,020 |
| 0.57 | 0.3258 | 0.16511 | 0.08929 | 0.07226 | 404,148 |
| 0.62 | 0.3545 | 0.18720 | 0.09768 | 0.08049 | 418,788 |
| 0.67 | 0.3832 | 0.21116 | 0.10644 | 0.08912 | 439,092 |
| 0.70 | 0.4004 | 0.22653 | 0.11193 | 0.09450 | 445,284 |

\(E_Y=1\), \(\nu=0.3\), unit cell of volume 1, Voigt notation with engineering shear strains; \(C^H_{11}\), \(C^H_{12}\), \(C^H_{44}\) are the entry values of the computed tensor (the macroscale model averages the three symmetric entries of each). Largest relative spreads over the twelve thicknesses: \(C^H_{11},C^H_{22},C^H_{33}\) 3.1e-13, \(C^H_{12},C^H_{13},C^H_{23}\) 6.1e-14, \(C^H_{44},C^H_{55},C^H_{66}\) 1.4e-13; largest normal–shear or shear–shear coupling 7.5e-14 of \(C^H_{11}\); largest relative equilibrium residual of the periodic fluctuations 2.2e-13.

#### ST23c. Homogenisation against the fine-scale NICE optimisation

| Load | Initial design: macroscale / NICE compliance | Macroscale prediction error (%) | Final fine-scale NICE compliance: B / Hom / X | Final \(V/V^*\): B / Hom / X | Hom vs B (%) | X vs Hom (%) | X vs B (%) | Corner correlation, B and Hom | Corner RMS difference, B and Hom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \(y\) (B1) | 69.177 / 94.341 | −26.7 | 87.089 / 88.019 / 87.141 | 0.99999 / 1.00040 / 0.99998 | 1.07 | −1.00 | 0.060 | 0.906 | 0.074 |
| \(z\) (B2) | 930.693 / 1,467.743 | −36.6 | 1,346.354 / 1,341.784 / 1,334.697 | 1.00000 / 0.99955 / 1.00000 | −0.34 | −0.53 | −0.866 | 0.969 | 0.045 |

Prediction error: \((C_{\rm macro}-\widehat C)/\widehat C\) at the uniform design \(\tau=0.40\). B: B1 or B2; Hom: Hom-\(y\) or Hom-\(z\); X: X-\(y\) or X-\(z\). Relative differences of the final compliances as stated, e.g. Hom vs B \((\widehat C_{\rm Hom}-\widehat C_B)/\widehat C_B\). Corner correlation and RMS difference over the 24\(\times\)8 cell-corner parameters. The macroscale volume of the uniform design, 4.569075, agrees with the fine-scale material volume 4.569074 to 1.2e-07.

### S9.4. Geometry-generation perturbations and discrete switches

Table ST24a lists the five applications of the geometry-generation fallback. They occurred in the uncut cell 010 of case A at iterations 16 and 19; in both plates at iteration 5, in the cut cell 060 (retained volume 0.917) and, after the perturbation by \(-10^{-4}\), also in the cut cell 160 (0.083); and in the uncut cell 200 of X-\(z\) at iteration 2. The exact twin, whose path differs slightly from the NICE path, needed none. Each perturbed design was generated after one or three perturbations of the fallback, changing a corner parameter by at most \(5.8\times10^{-4}\). At the first failures in case A, B1 and B2, the earlier cell-local scaling of the failing cell by \(1+\epsilon\) (\(\epsilon=10^{-9}\), \(-10^{-9}\), \(10^{-8}\)) had failed.

Table ST24b counts, between consecutive iterations, the cells whose discrete description changed. With the corner parameters moving by up to the move limit, the active elements, ghost faces, retained coordinates and the network's binary node features change in many cells at every step: in the median step all 8 cells of case A, 20 of the 24 cells of B1 and 17 of B2 change at least one of the counts, and never fewer than 3, 16 and 10; in the cross-starts the median is 17 and 9 cells. The coarse-factor shift changed in at most two cells (case A) and four cells (plates) per step. Each iteration therefore evaluates the objective on a different discrete model, and the derivatives of Section 4.3 hold only within each model. Across such switches the discrete reference compliance itself can jump, a non-smoothness that exact condensation meets in the same way, and the surrogate adds at most its own error level (Section 4.3, Supplementary Note S6.3). In case A the NICE and exact compliances differ by 0.011–0.023% on the common design of iterations 0–4 and, after the paths separate, by 0.016–0.030% at the unperturbed iterations (Table ST22a).

### Table ST24. Geometry-generation fallback and discrete switches

#### ST24a. Applied perturbations

| Case | Iteration | Failing cells | Earlier cell-local attempts, all failed | Fallback: \(\epsilon\) tried, in order | Applied \(\epsilon\) | Perturbed vertices | Cells regenerated | Fallback generation attempts; largest \(\lvert\Delta\tau\rvert\) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A, NICE | 16 | 010 (uncut) | +1e-09, −1e-09, +1e-08 (cell 010) | +1e-04 | +1e-04 | 4 (of 18) | 8 | 2; 5.8e-05 |
| A, NICE | 19 | 010 (uncut) | — | +1e-04, −1e-04, +1e-03 | +1e-03 | 4 (of 18) | 8 | 4; 5.8e-04 |
| A, exact twin | — | — | — | — | — | — | — | — |
| B1 | 5 | 060 (cut, 0.917); after −1e-04 also 160 (cut, 0.083) | +1e-09, −1e-09, +1e-08 (cell 060) | +1e-04, −1e-04, +1e-03 | +1e-03 | 12 (of 64) | 5 | 4; 2.7e-04 |
| B2 | 5 | 060 (cut, 0.917); after −1e-04 also 160 (cut, 0.083) | +1e-09, −1e-09, +1e-08 (cell 060) | +1e-04, −1e-04, +1e-03 | +1e-03 | 12 (of 64) | 5 | 4; 2.7e-04 |
| X-\(y\) | — | — | — | — | — | — | — | — |
| X-\(z\) | 2 | 200 (uncut) | — | +1e-04 | +1e-04 | 8 (of 64) | 6 | 2; 5.6e-05 (≤ 6.9e-05 at 2 vertices clipped to 0.69) |

Failing cells by layout index (retained volume fraction of cut cells): the cells whose generation failed for the design returned by MMA and, where stated, in addition after a perturbation. A failure means that the geometry generator could not certify the material patches of an element (Appendix A.1). Earlier cell-local attempts: before the fallback of Table ST21 was in use, the eight corner parameters of the failing cell alone were scaled by \(1+\epsilon\); these attempts are not counted in the last column. Fallback: each tried \(\epsilon\) is applied to the design returned by MMA, not accumulated, to the free vertices of the cells that failed at the preceding attempt; the last one tried is the applied one. Fallback generation attempts: the design returned by MMA and each perturbation. Largest \(\lvert\Delta\tau\rvert\): largest change of a corner parameter by the applied perturbation; at a vertex clipped to a bound the value returned by MMA is not recorded, and only an upper bound of its change is given.

#### ST24b. Cells whose discrete description changed between consecutive iterations

| Case | Active elements | Ghost faces | Retained coordinates | Cut-band nodes | Weak-support nodes | Element fringe hyperedges | Face fringe hyperedges | Coarse-factor shift | Any of these |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A, NICE (8 cells, 23 steps) | 2 / 8 / 8 | 2 / 8 / 8 | 0 / 8 / 8 | 0 / 4 / 4 | 3 / 8 / 8 | 2 / 8 / 8 | 2 / 8 / 8 | 0 / 0 / 2 | 3 / 8 / 8 |
| A, exact twin (8 cells, 22 steps) | 2 / 8 / 8 | not recorded | 0 / 8 / 8 | not recorded | not recorded | not recorded | not recorded | not recorded | 2 / 8 / 8 |
| B1 (24 cells, 22 steps) | 13 / 20 / 24 | 14 / 20 / 24 | 6 / 19.5 / 24 | 0 / 4 / 8 | 15 / 20 / 24 | 12 / 20 / 24 | 15 / 20 / 24 | 0 / 0 / 4 | 16 / 20 / 24 |
| B2 (24 cells, 29 steps) | 9 / 16 / 24 | 9 / 16 / 24 | 4 / 15 / 24 | 0 / 1 / 8 | 10 / 17 / 24 | 9 / 16 / 24 | 10 / 16 / 24 | 0 / 0 / 4 | 10 / 17 / 24 |
| X-\(y\) (24 cells, 11 steps) | 13 / 16 / 18 | 13 / 16 / 18 | 6 / 12 / 15 | 2 / 3 / 4 | 16 / 17 / 18 | 12 / 15 / 18 | 14 / 17 / 18 | 0 / 0 / 1 | 16 / 17 / 18 |
| X-\(z\) (24 cells, 11 steps) | 8 / 9 / 9 | 8 / 9 / 9 | 6 / 8 / 8 | 0 / 0 / 0 | 9 / 9 / 9 | 6 / 8 / 9 | 8 / 9 / 9 | 0 / 0 / 0 | 9 / 9 / 9 |

Minimum / median / maximum over the steps between consecutive iterations of the number of cells whose count changed: active elements, ghost-penalty faces, retained coordinates, nodes flagged by the network's binary node features for cut-band membership and weak support, element and face fringe hyperedges, and the diagonal shift of the coarse factorisation. A change that leaves a count unchanged is not detected, so the numbers are lower bounds. The exact twin recorded active elements and retained coordinates only.

### S9.5. Route checks

Three checks, each over the first one to three iterations of case A, support the analysis route of S9.1 (Table ST25). (i) The implementation of the optimisation runs, with the correction in single precision, gives the compliance of the implementation used for the accuracy results of Sections 6.2–6.9, with the correction in double precision, to \(1.7\times10^{-6}\), with the same conjugate-gradient iteration counts and a vertex gradient within \(1.8\times10^{-4}\). (ii) Reverse-mode differentiation of the moment integrals reproduces the vertex gradient obtained from the central moment differences of Eq. (H.6) to \(4.6\times10^{-7}\), in 47 s instead of 95 s. (iii) The warm start reduces the conjugate-gradient iterations from 127 to 102 and from 132 to 108 at iterations 1 and 2, by 18–20%, and the solve time by 17–18%, with compliances equal to \(2.1\times10^{-8}\); 99.98% and 99.995% of the free retained coordinates were found in the previous solution.

### Table ST25. Route checks on the \(2\times2\times2\) block (case A settings, iterations 0–2)

| Comparison (first / second) | Iteration | Compliance | Relative difference | PCG iterations | Vertex-gradient difference | Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Deployed route vs double-precision correction route | 0 | 24.590814 / 24.590856 | 1.7e-06 | 122 / 122 | 1.8e-04 | iteration 183 / 1,854 |
| Deployed route vs double-precision correction route | 1 | 27.146622 / 27.146669 | 1.7e-06 | 127 / 127 | 1.8e-04 | iteration 194 / 1,871 |
| Deployed route vs double-precision correction route | 2 | 30.165949 / 30.165996 | 1.5e-06 | 132 / 132 | 1.7e-04 | iteration 195 / 1,899 |
| Reverse mode vs central moment differences | 0 | 24.5908146 / 24.5908143 | 1.2e-08 | — | 4.6e-07 | sensitivities 47 / 95 |
| Warm vs cold start | 0 | 24.5908148 / 24.5908143 | 2.1e-08 | 122 / 122 | — | PCG 56 / 56 |
| Warm vs cold start | 1 | 27.1466220 / 27.1466225 | 1.7e-08 | 102 / 127 | — | PCG 47 / 58; matched 0.99982, scale 1.101 |
| Warm vs cold start | 2 | 30.1659498 / 30.1659494 | 1.5e-08 | 108 / 132 | — | PCG 49 / 59; matched 0.99995, scale 1.108 |

Each comparison runs the case A optimisation for up to three iterations with both settings. Deployed route: the implementation of the optimisation runs (Table ST21), network and correction in single precision, cold start, sensitivities from central moment differences (Eq. (H.6)) unless stated; the warm-start run uses reverse-mode sensitivities. Double-precision correction route: the implementation of the accuracy results of Sections 6.2–6.9, with the correction in double precision and operator state held in host memory between applications (peak host memory 30.7 GiB, against 3.4 GiB for the deployed route); the two routes therefore differ in implementation, correction precision and operator placement, and the time of the second is not a cost baseline. Vertex-gradient difference: \(\lVert\widetilde s_{g,1}-\widetilde s_{g,2}\rVert/\lVert\widetilde s_{g,2}\rVert\) over all vertices. Matched: fraction of free retained coordinates found in the previous solution; scale: energy-optimal factor of the warm start. At iteration 0 the warm-start run has no previous solution.

### S9.6. Scale demonstration and exact verification of the plates

**Storage of the streamed cell state.** For lattices whose operator state does not fit on the GPU, the state of the remaining cells is streamed from host memory (Supplementary Note S5). For the scale demonstration it is stored packed without loss: the lower-triangular coarse factors as their triangles and the integer index arrays as 32-bit instead of 64-bit integers. On the \(2\times2\times2\) block with every cell streamed, packing reduces the streamed state from 4.50 to 3.45 GiB (23%) and needs 0.40 GiB more peak GPU memory; the conjugate gradients take 129 iterations in both cases. Packing stores the same values; the compliances under the three face loads differ by at most \(1.1\times10^{-8}\) and the sensitivities by \(5.1\times10^{-7}\), at the level of the nondeterministic summation order of the GPU reductions.

**Scale demonstration.** Plates with the proportions of B1 and B2 (short side : long side 1 : 2, one cell thick), the planar cut of B1 and B2 scaled with the plate, and the clamp and in-plane load of B1 were generated with 24, 51, 88, 110 and 135 cells (Table ST26); all cut cells have the four retained volume fractions of B1 and B2. Each run starts from the uniform \(\tau=0.40\) with \(V^*=0.8\,V(\boldsymbol\tau^0)\) and the settings of Table ST21, and was stopped after four analyses (three MMA updates); the 51-cell run has two analyses. Cells are held on the GPU up to 4 GiB of operator state and streamed from host memory beyond it, with the packed storage above; the assembled retained stiffness \(K_{PP}\) of the preconditioner (Supplementary Note S6.1) is factorised on the GPU with NVIDIA cuDSS. The 24-cell plate is plate B1: its compliances at the first four iterations agree with those of B1 (94.341 at the start and 129.69 at iteration 3). Degrees of freedom are those of the initial design: the cut finite-element model of all cells, with nodes on shared faces counted once, and the retained coordinates of the assembled lattice without the clamped face. GPU memory is the largest memory in use on the device, sampled every 30 s with nvidia-smi while the run was the only process on the GPU; it includes the cuDSS factor, which the allocator statistics of PyTorch (13.3–20.1 GiB for the same runs) do not. Host memory is the peak resident memory of the main process; the geometry generation runs in separate processes. Times include geometry generation for every cell and, as for all optimisation runs, use no fused integration kernels (S9.1). Time per design iteration grows from 534 s at 24 cells to 2,606 s at 110 cells, about in proportion to the number of cells (22.2 s to 24.2 s per cell), with the conjugate gradients taking 46–48% of it; the number of conjugate-gradient iterations stays between 138 and 174. At 110 cells the device memory was fully in use (31.4 of 31.4 GiB). At 135 cells the factorisation of \(K_{PP}\) failed in its analysis phase for lack of device memory; the run stopped before its first solve. At the 23 to 30 iterations taken by B1 and B2, an optimisation of the 110-cell plate would take about 17 to 22 h; this is an extrapolation, not a measured run. The accuracy of NICE at these sizes is not verified against the exact reference.

**Table ST26. Scale demonstration.** Plates with the geometry and load of B1, uniform start, 4 GiB GPU budget for resident cell operators. DOFs: degrees of freedom of the cut finite-element model / free retained coordinates of the assembled lattice (initial design). Time: mean over the analyses (range), with mean phase times, in s. Residual: largest recomputed relative residual / largest \(|\bar U^T\rho|/\widehat C\). Memory: peak GPU memory in use (nvidia-smi) / peak host memory of the main process, GiB. Records: `scale/scale_summary.json`, `scale/scale_dofs.json`.

\(^{a}\) Largest 30-s samples before the failure; host memory from the RSS samples.

| Cells / cut | DOFs: cut model / free retained | Design variables | Analyses | Time per design iteration (s) and phases | PCG iterations | Residual / residual work | Memory, GPU / host (GiB) |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| 24 / 8 | 6.51 M / 0.388 M | 64 | 4 | 534 (505–566): geometry 22, front end 78, preconditioner 24, PCG 247, sensitivities 160 | 138–169 | 3.8e-04 / 2.9e-08 | 18.2 / 14.7 |
| 51 / 12 | 14.57 M / 0.780 M | 128 | 2 | 1,215 (1,177–1,254): geometry 41, front end 182, preconditioner 54, PCG 574, sensitivities 358 | 156–168 | 5.8e-04 / 2.2e-08 | 20.0 / 32.7 |
| 88 / 16 | 25.85 M / 1.305 M | 212 | 4 | 2,126 (2,097–2,178): geometry 52, front end 319, preconditioner 92, PCG 1,022, sensitivities 633 | 158–174 | 9.6e-04 / 5.2e-08 | 28.9 / 62.5 |
| 110 / 18 | 32.70 M / 1.618 M | 262 | 4 | 2,606 (2,524–2,745): geometry 59, front end 407, preconditioner 116, PCG 1,210, sensitivities 800 | 142–172 | 1.2e-03 / 2.2e-08 | 31.4 / 78.4 |
| 135 / 20 | 40.32 M / 1.963 M | 316 | 0 | Failed in the analysis phase of the \(K_{PP}\) factorisation on the GPU (cuDSS: allocation failed) | — | — | 24.1 / 83.8\(^{a}\) |

**Exact verification of the plate designs.** [TBD-EXACT: as Table ST22b for B1 and B2 at the start, an intermediate and the final design (exact compliance, surrogate error, vertex-gradient error, cosine, component-error percentiles, sign agreement, exact solve residual); exact compliance of the final designs of Hom-\(y\), Hom-\(z\), X-\(y\) and X-\(z\), and the exact counterparts of the comparisons of Table ST23c.]

**Table ST27. Exact checks of the plate designs.** [TBD-EXACT: table]

| Design | Iteration | Exact \(C\) | \(\widehat C\) | Surrogate error (%) | Gradient error (%) | Cosine | Component error / \(\max\lvert g\rvert\): median / 95th percentile / max | Sign agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | 0 / [TBD-EXACT] / 22 | [TBD-EXACT] | 94.341 / [TBD-EXACT] / 87.089 | [TBD-EXACT] | [TBD-EXACT] | [TBD-EXACT] | [TBD-EXACT] | [TBD-EXACT] |
| B2 | 0 / [TBD-EXACT] / 29 | [TBD-EXACT] | 1,467.743 / [TBD-EXACT] / 1,346.354 | [TBD-EXACT] | [TBD-EXACT] | [TBD-EXACT] | [TBD-EXACT] | [TBD-EXACT] |
| Hom-\(y\), Hom-\(z\) | final | [TBD-EXACT] | 88.019, 1,341.784 | [TBD-EXACT] | — | — | — | — |
| X-\(y\), X-\(z\) | 11 | [TBD-EXACT] | 87.141, 1,334.697 | [TBD-EXACT] | — | — | — | — |

### Supplementary figure for Note S9

**Figure S06. Homogenised law of the uniform-thickness cell.** (a) Effective elasticity tensor \(C^H_{11}\), \(C^H_{12}\), \(C^H_{44}\) (Voigt notation, engineering shear strains, \(E_Y=1\), \(\nu=0.3\)), cubic to \(3.1\times10^{-13}\); (b) material volume fraction \(V^H\), the material volume of the unit cell from the zeroth element moments. Markers: periodic homogenisation on the discrete model (\(n=32\), Q2 elements, ghost penalty) at 12 thicknesses from 0.18 to 0.70 (Table ST23b); lines: the cubic splines in \(\tau\) used by the macroscale model. Data: `review_r1/results/X6_opt/FACTS_6_11.json` (homog_law); script `figures_src/fig_opt.py`.

<!-- Number sources (internal; not part of the supplement).
All paths relative to docs/paper_p1/review_r1/results/X6_opt; F[...] denotes a key of FACTS_6_11.json (written by
facts_6_11.py from the run records), T the output of tables_s9.py (tables and its derived-number comments). Internal
run keys: XH_y, XH_z = X-y, X-z; H_y, H_z (homog/H_*, plates/hevalH_*) = Hom-y, Hom-z.

* Tables ST21 to ST25: pasted output of tables_s9.py. It reads FACTS_6_11.json; history.jsonl, meta.json and the run
  log of optA/optA, optA/optAx, plates/plateB1, plates/plateB2, plates/xstartH_y, plates/xstartH_z; plates/hevalH_y,
  plates/hevalH_z; homog/H_y, homog/H_z; homog/plate841.json; the pilot histories optA/pilot222, pilot222f, pilot222a,
  pilot222w; the constants of docs/data/newmachine_20260924/src_v2_wip/mma.py, the fallback perturbations of
  opt_design.py and the arguments of docs/paper_p1/evidence/d5_off.json.
* Intro: geometry generated beforehand at iteration 0 of every run and at twin iteration 17 (times.bodies_s < 1 s),
  in part at B1/B2 iteration 5 (resumed there, bodies_new 5 of 24): T (derived comments).
* S9.1: vertex counts, fixed vertices, constraint counts (span pairs with a free vertex computed from meta vid and
  fixed), bounds, move limit, tolerances, GPU memory limit: meta.json of each run (T, ST21); largest span 0.4500,
  gradient norm 0.4518, parameter range [0.1800, 0.6900]: T (note below ST21, from every history.jsonl); initial
  volume 25% above the bound: F['A']['V_rel_trace'][0], F['B1']['V_rel_trace'][0]; bound reached at iteration 4:
  F[...]['V_rel_trace']; compliance peaks 37.86 (A, k=4), 129.69 (B1, k=3), 2,129.57 (B2, k=4): T (derived
  comments, from F[...]['C_trace']); elastic variables positive at iterations 0-3 (A) and 0-2 (B1), at most 0.19: T
  (derived comments, history.jsonl field mma.y_max); cold starts at iteration 0, A 16, B1/B2 5: T (history.jsonl
  entries without times.warm_hit); vertex sensitivities, reverse mode against central differences 4.6e-7:
  F['route_validation']['fd_vs_ad_k0']; the volume gradient was not compared; implementation options of the timed
  run (tet_triton, coarse_tpl, sparse_coarse, fastidx) absent from the optimisation runs: T (checks against
  evidence/d5_off.json args and the env of each meta.json); five fallback events, eps applied, largest parameter
  change 5.8e-4: F[...]['body_perturbations'], T (ST24a); earlier cell-local attempts (1e-9, -1e-9, 1e-8): BODY_RETRY
  events in optA/optA.log, plates/plateB1.log, plates/plateB2.log (T, ST24a); cause of the failures: see the open
  item in the comment at the top.
* S9.2: initial corner range 0.252-0.535: F['A']['tau0_range']; V* = 1.03558: F['A']['Vstar']; compliances 24.59081
  and 24.59356 at iteration 0: F['A']['C0'], F['A_exact_twin']['C0'] (Table ST17e: 24.5908 and 24.5936);
  iterations and stopping rules: F['A']['iterations'], F['A_exact_twin']['iterations'], T (ST22c); peak 37.86 at
  iteration 4: F['A']['C_trace']; common design at iterations 0-4, every free parameter at the move limit or the
  lower bound, same-design error -0.011% to -0.023%, design difference up to 8.8e-4 from iteration 5, NICE against
  twin 0.016-0.030% (unperturbed), -0.037% and -0.10% at iterations 16 and 19: T (ST22a caption and derived
  comments, from the tv of both history.jsonl files and F['A_twin_trace']); volume 0.036% above the bound at
  iteration 19: F['A']['V_rel_trace'][19]; final ranges, span, gradient norm: F['A'], F['A_exact_twin']; design
  differences and exact compliances 24.110504, 24.110806, 1.25e-5: F['A_final_designs']; compliance 2.0% below the
  initial: F['A']['C_ratio']; exact checks: F['A_checks'] (from optA/optA/check_000.json, check_012.json,
  check_023.json); residual range and residual work: F['A']['true_residual_range'], F['A']['Ut_rho_rel_absmax'];
  times (sums over iterations) and memory: F['A'], F['A_exact_twin']; iteration 16 (170.7 s, 170 PCG iterations,
  cold start): T (ST22a); twin phases: T (ST22c, from optA/optAx/history.jsonl).
* S9.3: layout (24 of 32 cells, retained fractions, cut angle 33.69 deg, image 56.31 deg): homog/plate841.json, T
  (derived comments); fixed and free vertices: plates/plateB1/meta.json; V* = 3.65526: F['B1']['Vstar']; B1 and B2
  iterations, compliances, ratios (-7.7%, -8.3%), ranges, times (574 s, 639 s, PCG 294 s and 362 s, sensitivities
  160 s, sums of iteration times 3.7 h and 5.3 h), PCG iterations, memory, residuals: F['B1'], F['B2'], T (ST23a,
  derived comments); homogenised law, symmetry 3.1e-13, equilibrium 2.2e-13: F['homog_law'] (from
  homog/homog_cells.json); macroscale mesh 4,464 elements: homog/H_y/meta.json; volume agreement 1.2e-7: T (ST23c
  note, from homog/H_y/history.jsonl and F['B1']['V0']); prediction errors -26.7% and -36.6%, B/Hom/X compliances and
  volumes, Hom vs B 1.07% and -0.34%, X vs B 0.060% and -0.87%, correlations 0.906 and 0.969, RMS 0.074 and 0.045:
  F['comparison'], T (ST23c); macroscale iterations 23 and 27: F['Hmacro_y'], F['Hmacro_z']; Hom volumes +0.040%
  and -0.045%: T (ST23c, F['comparison'][...]['V_H'] over F['B1']['Vstar']); last parameter change of X-y 0.0254:
  F['XH_y']['dx_final'], move limit 0.0255: T (ST21); spreads 1.1% and 0.9%: T (derived comments). The macroscale
  time per iteration (F['Hmacro_*']['seconds_per_iteration']) is not used: that run has no stated hardware.
* S9.4: failing cells per attempt and their kind: F[...]['body_perturbations'] (field 'failed' of each attempt),
  homog/plate841.json, cut_nodes of optA/optA/history.jsonl (T, ST24a); earlier cell-local attempts: BODY_RETRY
  events of the run logs (T, ST24a); switch counts: F[...]['switches_per_iteration'], the column "Any of these"
  computed by T from the fps records of each history.jsonl; same-design and separated-path differences: as S9.2.
* S9.5: F['route_validation']; warm-start reductions 18-20% (iterations) and 17-18% (solve time): T (derived
  comments); warm-start run with reverse-mode sensitivities: T (checks on the sensitivity phase time and the vertex
  gradient of optA/pilot222w against pilot222a and pilot222f); operator state of the double-precision correction
  route in host memory: peak host memory 30.7 against 3.4 GiB, T (ST25 note, host_peak_gb of optA/pilot222 and
  pilot222f). The number of cells kept resident in that run is not stated, since its argument record is not
  archived here.
* S9.6 (storage): F['storage']['packstore_compare.txt'] (from scale/packstore_compare.txt); GiB conversion (4.50 and
  3.45 GiB), reduction 23%, peak GPU memory +0.40 GiB: T (derived comments).
-->
