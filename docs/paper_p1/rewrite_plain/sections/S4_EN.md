## Supplementary Note S6. Design optimisation records

This note gives the settings (S6.1, Table ST16) and the records of the thickness optimisations of Section 5.10. All NICE runs use one GPU. Compliance is given in the units of Section 5.1 (\(E_Y=1\), unit cell) and memory in GiB (\(2^{30}\) bytes). Times are wall-clock times of complete design iterations and include geometry generation, except at iteration 0, whose geometry was generated beforehand.

### S6.1. Optimiser and constraints

Table ST16 lists the settings. The method of moving asymptotes builds one convex separable approximation per analysis from the estimated gradient. It never tests the decrease of \(\widehat C\) along a search direction, where the inconsistency discussed in Section 4.2 would matter. Neither the gradient norm nor a KKT residual is used for stopping, since the estimate is not the gradient of \(\widehat C\) and \(\widehat C\) changes its discrete model between iterations (S6.4). A design iteration differs from an analysis of Table 5 in that it regenerates every cell's geometry and solves for one load instead of three. For case A, a NICE design iteration takes 74 s on average (Table ST17c), against 79 s for the analysis of the same block in Table 5, which needs no geometry generation but solves for three loads.

### Table ST16. Optimiser, constraints and analysis settings

| Item | Setting |
| --- | --- |
| Design variables | Corner thickness parameters at the lattice vertices \(\boldsymbol\tau_g\); \(\boldsymbol\tau_m=\boldsymbol\tau_m(\boldsymbol\tau_g)\) copies the vertex values to the corners of every cell meeting there. Case A: 27 vertices, 18 free; plate: 74 vertices, 64 free |
| Fixed vertices | The vertices in the plane of the loaded face (9 in case A, 10 in the plate), so that the nodal load obtained by consistent integration of the traction load does not change with the design (Appendix H: no load-derivative term) |
| Objective | Compliance \(\widehat C\) under one face traction load of unit resultant, evaluated as the work \(f_g^T\bar U\) of the final conjugate-gradient iterate and divided by its value at the first iteration |
| Gradient | Sensitivity estimate of Section 5.10, from one reverse-mode pass through the moment integrals at the fixed recovered displacement field; the complete derivative of the approximate compliance, Eq. (18), is not used. Exact twin: exact sensitivities from the exact displacement field with the moment derivatives of Eq. (H.3) |
| Volume constraint | \(V/V^*-1\le0\); \(V\) = sum of the zeroth element moments of all cells (material volume of the discrete model), its derivative from the same reverse pass (exact twin: Eq. (H.3)). \(V^*=0.8\,V(\boldsymbol\tau^0)\): 1.03558 (case A), 3.65526 (plate) |
| Corner-span constraint | \((\tau_a-\tau_b)/0.45-1\le0\) for every ordered pair of distinct corner vertices of every cell (316 ordered pairs, 276 of them involving a free vertex, in case A; 938 and 896 in the plate); pairs of two fixed vertices are constant and are not passed to the optimiser; training limit 0.47 (Table ST02) |
| Gradient-norm constraint | \(\lvert\nabla\tau\rvert^2/0.45^2-1\le0\) at every corner of every cell, from the three edge differences (64 corner stencils in case A, 192 in the plate, each with a free vertex); for a trilinear field the largest gradient norm over the cell is attained at a corner; training limit 0.47 |
| Bounds | \(0.18\le\tau\le0.69\); training range [0.1752, 0.6993] (Table ST02) |
| Optimiser | Method of moving asymptotes [Svanberg (1987)](https://doi.org/10.1002/nme.1620240207), one update per analysis, no line search and without the conservativeness test of its globally convergent variant GCMMA; subproblem solved by the primal–dual interior-point method of Svanberg (2007, Section 5), the barrier parameter reduced tenfold from 1 to \(10^{-7}\) (at each value, Newton steps until the largest residual is below 0.9 of it, at most 200); constants of the standard MMA problem form: \(a_0=1\), \(a_i=0\), \(c_i=1000\), \(d_i=1\) |
| Asymptotes | Initially \(x\pm0.5(x_{\max}-x_{\min})\) (first two iterations), then widened by 1.2 where consecutive steps have the same sign and narrowed by 0.7 where they alternate, kept between 0.01 and 10 variable ranges from \(x\); subproblem bounds at 0.1 of the distance to the asymptotes |
| Move limit | 0.05 of the variable range, 0.0255 per iteration |
| Stopping rule | \(\max\lvert\Delta\tau\rvert<10^{-3}\), or relative objective change below \(10^{-4}\) in three consecutive iterations, or 60 iterations |
| Geometry | Every iteration regenerates the geometry of every cell from its current corner parameters with the geometry generator used for all cells of this study, in parallel processes on the CPU (8 for case A, 12 for the plate), and rebuilds every learned substructure |
| Lattice solve | Preconditioned conjugate gradients with the balanced two-level preconditioner of Supplementary Note S4.1 to a recursive relative residual of \(10^{-6}\) (at most 3,000 iterations), in the arithmetic of the timed route of Table 5, with the coarse-factor pivot rule of Appendix F.1; a run stops if a solve reaches 3,000 iterations or a recomputed relative residual above \(10^{-2}\). Exact twin: dense exact condensed matrices of every cell, assembled solve to \(10^{-10}\) |
| Warm start | From the previous iteration's solution, matched DOF by DOF on absolute grid position, displacement component and the private flag of the cut-plane elements; unmatched DOFs start at zero; the start is scaled by the energy-optimal factor \(f_g^TX_0/(X_0^T\widehat{\mathbb K}X_0)\); the stopping criterion, relative to \(\lVert f_g\rVert\), is unchanged. At iteration 0 of every run the solve started from zero. Exact twin: cold start |
| GPU memory budget for resident learned substructures (Supplementary Note S4.1) | Case A: all cells resident; plate: cells resident while the allocated GPU memory stays below 22 GiB; scale demonstration: 4 GiB (Table ST20) |
| Geometry-generation fallback | If generation fails for some cells, the free vertices of those cells are multiplied by \(1+\epsilon\), \(\epsilon=10^{-4},-10^{-4},10^{-3},-10^{-3},3\times10^{-3},-3\times10^{-3}\) in turn (clipped to the bounds), every cell sharing them is regenerated, and the perturbed design is analysed and continued from. Applied four times over all runs (Table ST19a) |

The designs analysed on the fine scale are all iterations of the runs of Tables ST17 and ST18 and the homogenisation design. Over these designs, the largest corner span is 0.4500, the largest gradient norm 0.4518 and the corner parameters lie in [0.1800, 0.6900], so every analysed cell lay within the training limits.

### S6.2. Case A: NICE optimisation, exact twin and exact checks

Case A is the \(2\times2\times2\) block of Section 5.8, loaded by a traction load in \(y\) on the face \(y=\max\) opposite the clamp. The exact twin runs the same optimisation from the same start, with the same settings, constraints and geometry regeneration, but with exact condensation (Table ST16). The NICE run was checked with the exact model at iterations 0, 12 and 23 (Table ST17b). The twin serves as a verification run and is not used as a cost baseline; the cost comparison of the exact and learned routes is that of Section 5.9.

### Table ST17. Case A: NICE optimisation, exact twin and exact checks

#### ST17a. Iteration history

| Iteration | \(\widehat C\) (NICE) | \(C\), exact twin | NICE vs twin (%) |
| --- | --- | --- | --- |
| 0 | 24.59081 | 24.59356 | −0.011 |
| 1 | 27.14662 | 27.14994 | −0.012 |
| 2 | 30.16595 | 30.17025 | −0.014 |
| 3 | 33.73640 | 33.74275 | −0.019 |
| 4 | 37.85673 | 37.86557 | −0.023 |
| 5 | 35.69282 | 35.70004 | −0.020 |
| 6 | 33.80530 | 33.81157 | −0.019 |
| 7 | 32.05341 | 32.05955 | −0.019 |
| 8 | 30.48399 | 30.49000 | −0.020 |
| 9 | 29.08392 | 29.08994 | −0.021 |
| 10 | 27.85699 | 27.86145 | −0.016 |
| 11 | 26.84866 | 26.85332 | −0.017 |
| 12 | 26.00158 | 26.00632 | −0.018 |
| 13 | 25.28700 | 25.29168 | −0.018 |
| 14 | 24.78587 | 24.79256 | −0.027 |
| 15 | 24.46873 | 24.47568 | −0.028 |
| 16* | 24.24430 | 24.25329 | −0.037 |
| 17 | 24.14971 | 24.15703 | −0.030 |
| 18 | 24.10947 | 24.11624 | −0.028 |
| 19* | 24.08748 | 24.11211 | −0.102 |
| 20 | 24.10488 | 24.11155 | −0.028 |
| 21 | 24.10448 | 24.11117 | −0.028 |
| 22 | 24.10411 | 24.11081 | −0.028 |
| 23 | 24.10377 | — | — |

\* Design perturbed by the geometry-generation fallback (Table ST19a). NICE vs twin: \((\widehat C-C_{\rm twin})/C_{\rm twin}\) at the same iteration index. The designs coincide (to 4e-14) at iterations 0–4, where every free parameter moves by the move limit or to the lower bound while the volume bound is violated; there the column is the approximate compliance error of the same design. From iteration 5 the corner parameters differ, by at most 8.8e-04. Largest \(\lvert\bar U^T\rho\rvert/\widehat C\) over the NICE run: 1.5e-08.

#### ST17b. Exact checks of the NICE run

| Iteration | Exact \(C\) | \(\widehat C\) | Surrogate compliance error (%) | Gradient error (%) | Cosine | Component error / \(\max\lvert g\rvert\): median / 95th percentile / max | Sign agreement | Exact solve: PCG iterations / recomputed residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 24.59356 | 24.59081 | −0.0112 | 0.069 | 0.9999998 | 2.2e-04 / 5.1e-04 / 6.3e-04 | 1.000 | 178 / 9.1e-11 |
| 12 | 26.00630 | 26.00158 | −0.0182 | 0.170 | 0.9999987 | 4.1e-04 / 1.2e-03 / 1.8e-03 | 1.000 | 244 / 9.0e-11 |
| 23 | 24.11050 | 24.10377 | −0.0279 | 0.329 | 0.9999966 | 3.8e-04 / 2.8e-03 / 4.1e-03 | 1.000 | 262 / 9.9e-11 |

Over the 18 free vertex parameters. Surrogate compliance error \((\widehat C-C)/C\); gradient error \(\lVert\widetilde{\boldsymbol s}_g-\boldsymbol s_g\rVert/\lVert\boldsymbol s_g\rVert\); component error \(\lvert\widetilde s_{g,i}-s_{g,i}\rvert/\max_j\lvert s_{g,j}\rvert\).

#### ST17c. Final designs and cost

|  | NICE run | Exact twin |
| --- | --- | --- |
| Exact compliance of the final design | 24.110504 | 24.110806 |
| Final \(\tau\) range; largest corner span; largest gradient norm | 0.1800–0.6239; 0.4439; 0.4500 | 0.1800–0.6236; 0.4436; 0.4500 |
| PCG iterations; recomputed residual | 99–157; 6.2e-05–1.5e-04 | 178–262; 8.5e-11–9.9e-11 |
| Time per iteration, mean (range) (s) | 74 (62–120) | 469 (455–534) |
| Mean phases (s) | geometry 20.8, cell preparation 10.1, lattice and \(\mathbb K_{PP}\) assembly 0.1, preconditioner 5.9, PCG 31.6, sensitivities and volume gradient 5.5, MMA 0.006 | — |

Final designs: exact compliance of the NICE design relative to that of the twin's design −1.25e-05; corner parameters differ by at most 0.0051 (root mean square 0.0011). Iterations and first and last compliances of both runs: Table 6. Mean phases over all iterations; the geometry phase is zero at iteration 0.

### S6.3. Plate supported on its cut and homogenisation design

The plate of Section 5.10 is cut by a plane that runs from the bottom-right corner to the top edge at two cells from the left end. Its eight cut cells have retained volume fractions 0.917, 0.667, 0.333 and 0.083, two each. The layout is stored with its two in-plane axes interchanged, a cube-symmetry image of the plate, so that the cut normal lies at \(\vartheta=33.69^\circ\), within the range \(0<\vartheta<\pi/4\) of the validation geometries (Section 5.1), rather than at its image, 56.31°. The network is not equivariant under the cube symmetries (Section 6.3). Apart from the DOFs of the clamped cut-plane elements, no DOF is fixed.

For comparison with a homogenisation-based graded design, the effective elasticity tensor \(C^H(\tau)\) of the uncut cell with uniform corner parameter was computed by periodic homogenisation on the same discrete model (\(n=32\), Q2 elements, the stabilised stiffness \(K\) with its ghost penalty). Nodes on opposite faces of the cell are identified, and the fluctuation is periodic with one node fixed. Six unit macroscopic strains give \(C^H_{ij}=u_i^TKu_j\) for the unit cell, and the material volume fraction \(V^H(\tau)\) is the sum of the zeroth element moments. Cubic splines in \(\tau\) interpolate \(C^H_{11}\), \(C^H_{12}\), \(C^H_{44}\) and \(V^H\) between the twelve thicknesses of Table ST18b. The macroscale model meshes the plate with Q1 hexahedra, six per cell and axis (4,464 elements, 5,719 nodes). At every quadrature point it interpolates the local parameter \(\tau(x)\) trilinearly from the corners of its cell and evaluates \(C^H(\tau(x))\) and \(V^H(\tau(x))\). It integrates the elements intersected by the cut with \(4^3\) sub-points, the void part carrying \(10^{-6}\,C^H(0.4)\). It applies the clamp on the plane of the cut, \(u=0\) on the section of the plane with the plate, by a penalty \(\alpha\int u\cdot v\,dA\) with \(\alpha=10^6\,C^H_{11}(0.4)/h_M\) and \(h_M=1/6\). The penalty is integrated with a seven-point rule on the triangles of the section polygon of every intersected element. The load is a uniform traction of unit resultant on the loaded face. It was optimised with adjoint sensitivities and the same MMA settings, constraints, fixed vertices and volume fraction.

In iterations 0–3, while the volume bound is violated, the macroscale and NICE runs move every free parameter by the move limit or to the lower bound and analyse the same designs. There the macroscale compliance lies 27.0–31.8% below the NICE compliance, and along the rest of its path it lies 29–33% below it. At the uniform start, refining the macroscale mesh to 12 elements per cell and axis (35,136 elements, 39,949 nodes) raises the macroscale compliance from 55.192 to 56.088 (+1.6%), 25.8% instead of 27.0% below the exact compliance 75.599. Raising the penalty factor from \(10^6\) to \(10^8\) changes it by \(-8.7\times10^{-6}\) (6 elements) and \(-3.7\times10^{-6}\) (12 elements), relative. The optimisation used the six-element mesh. The fine-scale model fixes every DOF of the cut-plane elements of the cut cells (Table ST21). This support is at least as stiff as one acting on the cut surface alone; it lowers the exact compliance and therefore narrows the macroscale underestimation. For the graded homogenisation design, the macroscale volume model no longer equals the fine-scale material volume exactly (Table ST18c).

### Table ST18. Plate supported on its cut: NICE optimisation and homogenisation design

#### ST18a. Runs

|  | NICE run | Homogenised model | Homogenisation design, NICE analysis |
| --- | --- | --- | --- |
| Start design | uniform 0.40 | uniform 0.40 (macroscale) | final macroscale design |
| Largest compliance (iteration) | 110.493 (4) | 74.557 (4) (macroscale) | — |
| Final \(\tau\) range | 0.180–0.690 | 0.180–0.690 | 0.180–0.690 |
| Largest corner span / gradient norm at the end | 0.450 / 0.450 | 0.450 / 0.450 | 0.450 / 0.450 |
| Time per iteration, mean (range) (s) | 200 (155–270) | 4.1 (macroscale, CPU) | — |
| Sum of iteration times (s) | 4,802 | 95 (macroscale, CPU) | — |
| PCG iterations | 73–122 | — | 127 |
| Recomputed residual | 3.5e-04–5.8e-04 | — | 3.9e-04 |
| \(\max\lvert\bar U^T\rho\rvert/\widehat C\) | 5.2e-08 | — | 5.4e-08 |

Every fine-scale value is a NICE value; exact checks in Table ST21. Iterations and first and last compliances: Table 6. Figure 15a,b draws the corner parameters in the layer \(z=0\); those in the layer \(z=1\) differ from them by at most 1.1e-03 (NICE) and 6.3e-13 (homogenisation design).

#### ST18b. Homogenised law of the uniform-thickness cell

| \(\tau\) | \(V^H\) | \(C^H_{11}\) | \(C^H_{12}\) | \(C^H_{44}\) |
| --- | --- | --- | --- | --- |
| 0.18 | 0.1027 | 0.03996 | 0.02806 | 0.01920 |
| 0.22 | 0.1256 | 0.04992 | 0.03432 | 0.02384 |
| 0.27 | 0.1541 | 0.06307 | 0.04212 | 0.02988 |
| 0.32 | 0.1827 | 0.07712 | 0.04989 | 0.03620 |
| 0.37 | 0.2113 | 0.09218 | 0.05765 | 0.04279 |
| 0.42 | 0.2399 | 0.10838 | 0.06541 | 0.04968 |
| 0.47 | 0.2685 | 0.12585 | 0.07323 | 0.05688 |
| 0.52 | 0.2971 | 0.14471 | 0.08117 | 0.06440 |
| 0.57 | 0.3258 | 0.16511 | 0.08929 | 0.07226 |
| 0.62 | 0.3545 | 0.18720 | 0.09768 | 0.08049 |
| 0.67 | 0.3832 | 0.21116 | 0.10644 | 0.08912 |
| 0.70 | 0.4004 | 0.22653 | 0.11193 | 0.09450 |

\(E_Y=1\), \(\nu=0.3\), unit cell of volume 1, Voigt notation with engineering shear strains; \(C^H_{11}\), \(C^H_{12}\), \(C^H_{44}\) are the entry values of the computed tensor (the macroscale model averages the three symmetric entries of each). Largest relative spreads over the twelve thicknesses: \(C^H_{11},C^H_{22},C^H_{33}\) 3.1e-13, \(C^H_{12},C^H_{13},C^H_{23}\) 6.1e-14, \(C^H_{44},C^H_{55},C^H_{66}\) 1.4e-13; largest normal–shear or shear–shear coupling 7.5e-14 of \(C^H_{11}\); largest relative equilibrium residual of the periodic fluctuations 2.2e-13.

#### ST18c. Homogenisation against the fine-scale NICE optimisation

| Quantity | Macroscale model | NICE | Exact condensation |
| --- | --- | --- | --- |
| Compliance of the uniform design | 55.192 | 75.591 | 75.599 |
| Macroscale prediction error at the uniform design (%) | — | −26.99 | −26.99 |
| Compliance of the homogenisation design | 55.271 | 80.172 | 80.198 |
| Homogenisation design vs NICE design (%) | — | +2.44 | +2.44 |
| Fine-scale \(V/V^*\): NICE design / homogenisation design | — | 1.00000 / 0.99821 | — |
| Corner parameters, NICE vs homogenisation design: largest difference / RMS / correlation | — | 0.262 / 0.057 / 0.946 | — |

Prediction error: \((C_{\rm macro}-C)/C\) at the uniform design \(\tau=0.40\). Corner differences and correlation over the \(24\times8\) cell-corner parameters. The macroscale volume of the uniform design, 4.569075, agrees with the fine-scale material volume 4.569074 to 1.2e-07.

### S6.4. Geometry-generation perturbations and discrete switches

Table ST19a lists the four applications of the geometry-generation fallback of Table ST16; a failure means that the geometry generator could not certify the material patches of an element (Appendix A.1). Table ST19b counts the cells whose discrete description changed between consecutive iterations. Each iteration evaluates the objective on a different discrete model, and the derivatives of Section 4.2 hold only within each model.

### Table ST19. Geometry-generation fallback and discrete switches

#### ST19a. Applied perturbations

| Case | Iteration | Failing cells | Applied \(\epsilon\) | Perturbed vertices | Cells regenerated | Largest \(\lvert\Delta\tau\rvert\) |
| --- | --- | --- | --- | --- | --- | --- |
| A, NICE | 16 | 010 (uncut) | +1e-04 | 4 (of 18) | 8 | 5.8e-05 |
| A, NICE | 19 | 010 (uncut) | +1e-03 | 4 (of 18) | 8 | 5.8e-04 |
| Plate | 5 | 060 (cut, 0.917), 160 (cut, 0.083) | +1e-03 | 12 (of 64) | 5 | 2.7e-04 |
| Plate | 14 | 000 (uncut) | +1e-04 | 4 (of 64) | 4 | 3.6e-05 |

Failing cells by layout index (retained volume fraction of cut cells): the cells whose generation failed for the design returned by MMA. The applied \(\epsilon\) acts on the design returned by MMA; largest \(\lvert\Delta\tau\rvert\): largest change of a corner parameter by the applied perturbation; no perturbed vertex was clipped to a bound. The exact twin and the analysis of the homogenisation design needed no fallback.

#### ST19b. Cells whose discrete description changed between consecutive iterations

| Case | Active elements; ghost faces | Retained DOFs | Cut-plane-element nodes | Weakly connected nodes | Weak-region element stencils | Weak-region face stencils | Coarse-factor shift | Any of these |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A, NICE (8 cells, 23 steps) | 2 / 8 / 8 | 0 / 8 / 8 | 0 / 4 / 4 | 3 / 8 / 8 | 2 / 8 / 8 | 2 / 8 / 8 | 0 / 0 / 2 | 3 / 8 / 8 |
| Plate (24 cells, 23 steps) | 8 / 18 / 24 | 3 / 18 / 24 | 0 / 3 / 8 | 11 / 18 / 24 | 9 / 18 / 24 | 11 / 18 / 24 | 0 / 0 / 4 | 11 / 18 / 24 |

Minimum / median / maximum, over the steps between consecutive iterations, of the number of cells whose count changed. Counts were taken of active elements and ghost-penalty faces (identical counts), retained DOFs, nodes flagged by the network's binary node indicators as nodes of the cut-plane elements or as weakly connected nodes, element and face stencils selected for the weak-region layers (Appendix G.1), and the diagonal shift of the coarse factorisation (Appendix F.1). A change that leaves a count unchanged is not detected, so the numbers are lower bounds. The exact twin (8 cells, 22 steps), which recorded active elements and retained DOFs only, gives 2 / 8 / 8 and 0 / 8 / 8.

### S6.5. Scale demonstration and exact verification of the plate designs

**Scale demonstration.** Plates were generated with 24, 51, 88 and 110 cells (Table ST20). They have the proportions of the plate of S6.3 (short side : long side 1 : 2, one cell thick), with its planar cut scaled with the plate, and all cut cells have the four retained volume fractions of S6.3. The plates are clamped on the uncut long side and loaded on the opposite face by a traction load of unit resultant, acting in the plane along the long side. After the cut, this face is two cells wide in the 24-cell plate. The 24-cell plate is the plate of S6.3 with this clamp and load. Each run starts from the uniform \(\tau=0.40\) with \(V^*=0.8\,V(\boldsymbol\tau^0)\) and the settings of Table ST16, and was stopped after four analyses (three MMA updates). Degrees of freedom are those of the initial design: the cut finite-element model of all cells, with nodes on shared faces counted once, and the retained DOFs of the assembled lattice without the clamped face.

**Table ST20. Scale demonstration.** Plates with the proportions and cut of the plate of S6.3, clamped on the uncut long side and loaded in plane on the opposite face, uniform start, four analyses; GPU memory budget for resident learned substructures 4 GiB (Supplementary Note S4.1). DOFs: degrees of freedom of the cut finite-element model / free retained DOFs of the assembled lattice (initial design). Start \(\widehat C\): NICE compliance of the uniform start (first analysis). Time: mean over the analyses (range), with mean phase times, in s, geometry generation included. Residual: largest recomputed relative residual / largest \(|\bar U^T\rho|/\widehat C\). CPU memory: peak resident memory of the main process, which holds the streamed state of the learned cells, per cell (total), GiB; the geometry generation runs in separate processes. GPU memory: largest memory in use on the device, sampled every 30 s with nvidia-smi, GiB; besides the resident learned cells it holds the assembled retained system: \(\mathbb K_{PP}\) and its cuDSS factor in the preconditioner and the vectors of the conjugate gradients.

| Cells / cut | DOFs: cut model / free retained | Design variables | Start \(\widehat C\) | Time per design iteration (s) and phases | PCG iterations | Residual / residual work | CPU memory per cell (total) | GPU memory in use |
| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: |
| 24 / 8 | 6.51 M / 0.388 M | 64 | 94.34 | 242 (229–252): geometry 21, cell preparation 36, preconditioner 21, PCG 144, sensitivities 19 | 138–169 | 3.9e-04 / 2.3e-08 | 0.62 (15.0) | 14.9 |
| 51 / 12 | 14.57 M / 0.780 M | 128 | 89.12 | 547 (539–559): geometry 37, cell preparation 83, preconditioner 45, PCG 334, sensitivities 43 | 156–171 | 6.6e-04 / 3.9e-08 | 0.69 (35.2) | 20.2 |
| 88 / 16 | 25.85 M / 1.305 M | 212 | 86.75 | 962 (955–980): geometry 63, cell preparation 150, preconditioner 79, PCG 586, sensitivities 74 | 158–173 | 9.7e-04 / 6.9e-08 | 0.71 (62.8) | 26.1 |
| 110 / 18 | 32.70 M / 1.618 M | 262 | 85.38 | 1,164 (1,119–1,214): geometry 71, cell preparation 185, preconditioner 99, PCG 698, sensitivities 95 | 142–172 | 1.2e-03 / 7.3e-08 | 0.73 (79.9) | 30.0 |

Exact checks at the first design iteration, with the metrics of Table ST21: 24 cells, \(C=94.353\), \(\widehat C=94.341\), approximate compliance error \(-0.0132\%\), gradient error 0.045%, cosine 0.99999993, component error median / 95th percentile / max 0.011 / 0.048 / 0.059%, exact sign in all 64 components; 51 cells, \(C=89.129\), \(\widehat C=89.119\), \(-0.0113\%\), 0.047%, 0.99999994, 0.009 / 0.034 / 0.063%, all 128 components. The exact reference of the 88- and 110-cell plates was not computed.

**Table ST21. Exact checks of the plate designs.** The uniform start, the final NICE design and the homogenisation design of S6.3, and two uniform designs at the volume bound \(V^*\) for comparison; the intermediate designs were not checked. Exact condensation of the 55 distinct cells with PARDISO's Schur-complement option, verified column-wise against interior solves; assembled solve, with every DOF of the cut-plane elements of the cut cells fixed, to a recursive relative residual of \(10^{-10}\). Surrogate compliance error: \(\widehat C/C-1\). Gradient: vertex gradient with exact sensitivities (central moment differences, Eq. (H.3)) against the NICE sensitivity estimate, over the 64 free vertex parameters. Exact solves: 136 / 9.4e-11, 190 / 8.8e-11, 191 / 9.5e-11 conjugate-gradient iterations / recomputed residual, in the order of the rows.

| Design | Exact \(C\) | \(\widehat C\) | Surrogate compliance error (%) | Gradient error (%) | Cosine | Component error / \(\max\lvert g\rvert\) (%): median / 95th percentile / max | Sign agreement (64 variables) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NICE run, iteration 0 (start) | 75.599 | 75.591 | −0.0105 | 0.029 | 0.99999997 | 0.003 / 0.027 / 0.043 | 1.000 |
| NICE run, iteration 23 (final) | 78.287 | 78.264 | −0.0296 | 0.301 | 0.99999652 | 0.057 / 0.273 / 0.315 | 1.000 |
| Homogenisation design | 80.198 | 80.172 | −0.0317 | 0.320 | 0.99999596 | 0.052 / 0.322 / 0.335 | 1.000 |
| Uniform design at \(V^*\), 64 free parameters (loaded-face parameters 0.40) | 115.357 | 115.335 | −0.0191 | 0.064 | 0.99999995 | — | 1.000 |
| Uniform design at \(V^*\), all 74 parameters | 113.279 | 113.259 | −0.0179 | 0.055 | 0.99999995 | — | 1.000 |

## Supplementary Note S7. Transfer to a second TPMS family: gyroid sheet cells

The construction of Sections 3 and 4 comprises the condensed energy with the transpose of the complete recovery operator, Propositions 1–4 and the correction \(\mathcal W\). It uses the symmetry and positive semidefiniteness of the cell stiffness and the positive definiteness of its interior block, and it does not use the level-set function of Eq. (1). The trained network, in contrast, has seen Schwarz-P cells only (Sections 6.1 and 6.3). This note applies NICE to gyroid sheet cells, first with the network parameters of Table 2 unchanged (zero-shot) and then after a short continuation of training that includes a few gyroid cells. It is not part of the evidence of Section 5.

**Cells.** A gyroid cell replaces the level-set function of Eq. (1) by \(\phi_G(x)=\sin2\pi x_1\cos2\pi x_2+\sin2\pi x_2\cos2\pi x_3+\sin2\pi x_3\cos2\pi x_1\) and keeps the cell box, the trilinear corner parameters, the optional planar cut and the settings of Table 1 (background mesh \(n=32\), material, ghost penalty, retained DOFs and correction 8 / \(Q_1(17)\) / 8). Every gyroid cell is the twin of one P cell and keeps the P cell's position, cut plane and cut-severity group (uncut or cut). Its corner parameters are mapped from the P values so that a uniform-thickness cell keeps its sheet volume fraction (for example 23.3% against 23.2% and 5.8% against 6.2% for two twins). The cells are generated by a version of the geometry stage that takes the level-set function as input; with the P function it reproduces the eight cells of the \(2\times2\times2\) block of Section 5.8 cell by cell. When this version generates a lattice, it keeps the largest face-connected set of active elements of the whole lattice rather than of each cell. It also removes fragments of less than a quarter of an element that have no ghost-penalty support, and it repeats both steps until nothing changes. The retained DOFs of neighbouring cells then coincide on their shared faces except along the cell edges (unmatched DOFs are treated as in Section 2.3). The sixteen single cells are the twins of the validation cells fresh_val_2000 to fresh_val_2015 (geometry identifiers gval_000 to gval_1500 in the data archive). These validation cells include U1, U2, L1, M1, M2 and H1–H3 of R1 and belong to the 20 checkpoint-selection geometries of Table ST01. The lattice is the twin of the \(2\times2\times2\) block (glat222). No gyroid cell entered the training or checkpoint selection of any variant of Table 2 or of the two variants evaluated only in this supplement.

Propositions 1, 3 and 4 and the properties of Section 3.1 hold for these cells as for the P cells. They hold because assumptions (A1)–(A3) concern the stiffness and the assembly, and because the network's displacement recovery reproduces the retained displacements and rigid-body motion by construction. Proposition 2 holds under its spectral condition, which was not checked on the gyroid cells, as for the cells of Sections 5.8–5.10.

**Variants.** *NICE, zero-shot*: NICE with the network parameters of Table 2, unchanged. *NICE, continued* (record identifier A3G_ft): NICE's parameters, trained for a further 8,000 steps. The training set comprises 96 P training cells and 18 gyroid twins of P training cells (cells with at most 420,000 DOFs), and the correction 8 / \(Q_1(17)\) / 8 is inside the training loop as in Section 3.5. Its checkpoint was selected on the P validation list only (selection score 0.00339, against 0.00237 for NICE; Table ST01). The sixteen gyroid twins and the gyroid lattice therefore entered neither its training nor its selection. Both variants apply the correction 8 / \(Q_1(17)\) / 8 at evaluation.

**Single cells.** Each gyroid twin was evaluated in the identity orientation with 64 validation test displacements per load class (Appendix G.2). The energy error of Eq. (7) was averaged over the test displacements of each cell and then over the sixteen cells with equal weights, as in Section 5.1 (Table ST22). Zero-shot, the mean error under traction loads is 4.48% (largest cell 25.6%), 66 times the 0.068% of NICE on the P twins of the same cells. Over the six load and support classes the ratio of the means is 56 to 234; under the imposed polynomial and multiscale displacements it is 14 and 7. The degradation is not confined to a few cells. In every cell, the error under each load and support class is 20 to 1,150 times that of its P twin, and under the imposed displacements 2 to 30 times. Under traction loads, the three gyroid cells with the largest errors are the twins of the three P cells with the largest NICE errors among the sixteen (fresh_val_2007, 2012 and 2015). The correction is the same as on the P cells. The accuracy of the network's interior displacements does not carry over, since it is tied to the family on which the network was trained.

After the continuation, the means under the six load and support classes fall by factors of 5.8 to 10.2. Under traction loads the mean falls from 4.48% to 0.74%, and the largest cell mean from 25.6% to 2.67%. Under the imposed polynomial and multiscale displacements the means fall by factors of 1.6 and 1.2. The means remain 6 to 23 times those of NICE on the P twins (11 times under traction loads). The reduction is uneven across cells: the twin of H2 (fresh_val_2010) now has the largest errors under nodal point loads and traction loads, 4.36% and 2.67% against 4.73% and 3.33% zero-shot. On the 80 P validation cells, the class means of the continued network are 1.1 to 1.7 times those of NICE in the same evaluation (0.026–0.072% against 0.015–0.065%).

**Table ST22. Energy error (%) of the gyroid twins and of the P cells.** Geometry-equal mean, with the largest geometry mean in parentheses, identity orientation. Gyroid twins: the sixteen twins of fresh_val_2000 to fresh_val_2015. P twins: these sixteen P validation cells. P validation: the 80 validation cells of Section 5.1 in one evaluation of both variants.

| Load class | Gyroid twins, NICE, zero-shot | Gyroid twins, NICE, continued | P twins, NICE | P validation, NICE | P validation, NICE, continued |
| --- | ---: | ---: | ---: | ---: | ---: |
| Traction loads (force_c) | 4.48 (25.6) | 0.74 (2.67) | 0.068 (0.28) | 0.065 (0.28)\(^{a}\) | 0.072 (0.30)\(^{a}\) |
| Single-face traction loads (face_c) | 4.30 (27.2) | 0.60 (2.52) | 0.031 (0.17) | 0.029 (0.17)\(^{a}\) | 0.037 (0.21)\(^{a}\) |
| Nodal point loads (force) | 6.10 (14.1) | 0.78 (4.36) | 0.058 (0.18) | 0.058 (0.33) | 0.065 (0.31) |
| Single-face nodal point loads (face) | 7.35 (15.2) | 0.72 (1.59) | 0.031 (0.28) | 0.037 (0.72) | 0.044 (0.47) |
| Spring supports (support) | 5.78 (14.0) | 0.60 (1.54) | 0.053 (0.15) | 0.055 (0.37) | 0.064 (0.28) |
| Stiffness-scaled spring supports (support_k) | 3.01 (17.2) | 0.52 (2.42) | 0.054 (0.22)\(^{b}\) | 0.053 (0.22)\(^{a}\) | 0.059 (0.24)\(^{a}\) |
| Polynomial displacements (macro) | 0.19 (0.89) | 0.12 (0.58) | 0.014 (0.075) | 0.015 (0.12) | 0.026 (0.17) |
| Multiscale displacements (grf) | 0.16 (0.79) | 0.13 (0.68) | 0.024 (0.13) | 0.025 (0.14) | 0.037 (0.21) |

\(^{a}\) Evaluated in this run on the 20 checkpoint-selection geometries only (stiffness-scaled spring supports: 19); Table ST03 gives NICE on all 80 (75) geometries. \(^{b}\) 15 twins; fresh_val_2010 has no stiffness-scaled spring-support test displacements.

**Lattice.** The gyroid twin of the \(2\times2\times2\) block of Section 5.8 has eight cells, four of them cut, and every cell is learned; it has 160,629 free retained DOFs against 139,002 for the P block. It was solved as in Section 5.8, clamped on one face and loaded by the three face loads and three random loads, with the dense exact condensed matrices as reference (Table ST23). Zero-shot, the lattice compliance errors are 0.98–1.53% under the face loads (2.14% under the random loads), and the largest thickness-sensitivity error over the cells is 13.5% (10.6%), against 0.014% (0.069%) and 0.14% (0.45%) for NICE on the P block. With the continued network they are 0.23% (0.35%) and 1.84% (3.57%), and on the P block 0.019% (0.084%) and 0.25% (0.49%). With exact operators, the gyroid lattice needs 414 conjugate-gradient iterations against 183 for the P block; on the gyroid lattice, the learned operators add 20 (zero-shot) and 22 (continued) iterations.

**Table ST23. Lattice compliance and sensitivity errors (%) of the P block and its gyroid twin.** Maxima over the three face loads, and over the three random loads in parentheses; sensitivity error: largest relative error of the eight-component vector over the eight cells. Iterations: preconditioned conjugate gradients (Supplementary Note S4.1) with the learned / the exact condensed operators, to a recursive relative residual of \(10^{-10}\). The exact gyroid solution reaches a recomputed relative residual of \(9.6\times10^{-11}\); for the learned gyroid solves, the recomputed residual stagnates at \(2.7\times10^{-4}\) (zero-shot) and \(2.8\times10^{-4}\) (continued), as for the P block in Section 5.8.

| Lattice | Variant | Compliance error | Sensitivity error | Iterations |
| --- | --- | ---: | ---: | --- |
| P block (\(2\times2\times2\)) | NICE | 0.014 (0.069) | 0.14 (0.45) | 186 / 183 |
| P block (\(2\times2\times2\)) | NICE, continued | 0.019 (0.084) | 0.25 (0.49) | 187 / 183 |
| Gyroid block (\(2\times2\times2\)) | NICE, zero-shot | 1.53 (2.14) | 13.5 (10.6) | 434 / 414 |
| Gyroid block (\(2\times2\times2\)) | NICE, continued | 0.23 (0.35) | 1.84 (3.57) | 436 / 414 |

**Reading.** The construction carried over unchanged: the energy form, the transpose of the complete recovery operator, the correction and the assembly were applied to the gyroid cells without modification. The properties that do not depend on the network parameters hold for them as for the P cells. The trained network is specific to its family: zero-shot, the gyroid errors under loads and supports are 56 to 234 times those on the P twins in the mean, and the unchanged correction does not make up the difference. A continuation of 8,000 training steps with eighteen gyroid cells lowers them six- to tenfold, to 6 to 23 times the P level. It brings the gyroid lattice to 0.23% in compliance and 1.84% in sensitivity under the face loads, while the class means of the continued network on the P validation cells are 1.1 to 1.7 times those of NICE. Whether more gyroid data and training reach the P level was not tested. A new TPMS family therefore needs new training data (Section 6.3); the construction itself needs no change.

**Records.** Record identifiers in the data archive (files in `evidence/gcell/`): gval.json (gyroid twins: P twin, cut-severity group and corner parameters); gval_A3.json and gval_A3G.json (single gyroid cells, NICE zero-shot and continued); newval_A3G.json (80 P validation cells, NICE continued; NICE in the same evaluation: `evidence/newval_A3_2grid.json`); lat_hetero_g222.json and lat_hetero_g222_A3G.json (gyroid block, NICE zero-shot and continued); lat_hetero222_A3G.json (P block, NICE continued); g_vs_p_222.json (P and gyroid blocks, NICE).

## Supplementary figures

![Figure S01](figures/S06_reference_verification.png)

**Figure S01. Verification of the CutFEM reference.** Single cells clamped on one cell face and loaded by unit traction loads on another face in the three Cartesian directions. In (a), (b) and (d), solid lines with filled markers give the compliance and dashed lines with open markers the thickness sensitivity, each as the largest relative change over the three loads. (a) Change against the finest background resolution computed for U1 (\(n=40\)) and for M1 and M2 (\(n=48\)); the reference used throughout has \(n=32\). (b) Change when the ghost-penalty coefficient is varied from the value \(10^{-4}\) used throughout. (c) Filled: largest relative change of the sensitivity when the finite-difference step of the moment derivatives is varied from the value \(h_c=10^{-5}\tau_c\) used throughout, at most \(2.6\times10^{-5}\)% at \(10^{-3}\tau_c\) and a hundredfold smaller at \(10^{-4}\tau_c\); open: largest relative difference between central compliance differences and the sensitivity at the step used. (d) Refinement to \(n=64\) of H1, H2 and the validation cells with the largest NICE error (W1) and the thinnest walls (W3), the data of Table ST10; H2 has no \(n=56\) solution.

![Figure S02](figures/S01_distributions.png)

**Figure S02. Distributions of geometry-level energy errors.** Each point is one validation geometry's mean energy error in the identity orientation; horizontal bars are population medians. The classes with nodal point loads, spring supports, single-face nodal point loads, polynomial displacements, multiscale displacements, traction loads and single-face traction loads contain 80 geometries. The class with stiffness-scaled supports and the class with displacements imposed by a neighbouring cell contain 75, the same populations as Table ST03. The five variants of Table ST03 (Base network, Uncorrected continuation, Smoothing-trained, Base network + correction and NICE) use the markers and colours of the main-text figures; Base network + correction applies the correction to the base network at deployment. Deterministic horizontal offsets separate overlapping observations. All panels share the logarithmic error axis.

![Figure S03](figures/S03_sensitivity_diagnostics.png)

**Figure S03. Sensitivity-error diagnostics.** (a) Paired mean energy and sensitivity errors for responses to traction loads and nodal point loads, using six cells of the base network and five of the Uncorrected continuation. (b) Linear-term norm fraction \(\|D_1\|_F/(\|D_1\|_F+\|D_2\|_F)\) under traction loads, where \(D_1+D_2\) is the sensitivity-error matrix over all eight design components and evaluated test displacements. (c,d) Fractions of absolute elementwise sensitivity-error contributions and of element counts in four mutually exclusive material-volume-fraction groups for the base network under traction loads. Each error group sums absolute contributions over its elements, design components and test displacements before normalisation by the total. Filled markers identify the base network. In (a,b), open markers and hatched bars identify the Uncorrected continuation, which is shown for U1, U2, M1, H1 and M2 (Table ST05a).

![Figure S04](figures/S07_energy_share_variants.png)

**Figure S04. The other four variants follow the relations of Figure 11.** The counterpart of Figure 11 for (a–c) the base network, (d–f) the Uncorrected continuation, (g–i) the Smoothing-trained variant and (j–l) Base network + correction, which applies the correction at deployment. One point per load in the configurations evaluated for each variant (54, 87, 96 and 114 loads; Tables ST08 and ST09); filled markers denote face loads and open markers cut-surface loads. Top row: compliance error against \(\beta\); lines denote equality. The ratio of compliance error to \(\beta\) lies between 0.80 and 0.98, 0.79 and 0.99, 0.92 and 0.99, and 0.96 and 1.006 for the four variants; the six ratios above 1 correspond to at most \(1.3\times10^{-9}\) of the compliance. Middle and bottom rows: compliance error and test-cell sensitivity error against the exact energy fraction \(w\) of the test cell; lines are least-squares fits in logarithmic coordinates, with the slope given in each panel. All panels share the error axis. The circled load in (a–c) is U1/x under the z traction load on the face of the neighbouring cell; its values are given in the note to Table ST08.
