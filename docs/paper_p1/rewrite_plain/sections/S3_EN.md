## Supplementary Note S3. Cost records: whole-lattice direct solution and per-cell condensation

This note defines the three routes of Table 5 and records their phases (Table ST12). It also compares the per-cell cost of conventional condensation and NICE (Table ST13). The four-cell lattices are the two layers \(z=0\) and \(z=1\) of the \(2\times2\times2\) block of Section 5.8, with their corner thickness parameters unchanged. Each layer holds two uncut cells and two cut cells with retained volume fractions 0.616 and 0.252. All lattices are clamped on the face \(y=\min\) and loaded on the face \(y=\max\) by unit traction loads along the three Cartesian axes. Routes (a) and (b) also solve three random loads.

Route (a), the whole-lattice direct solution, assembles the full cut-cell stiffness of every cell, on its retained and interior degrees of freedom, into one global matrix. The retained degrees of freedom are numbered and coupled exactly as in the learned lattice, with the same clamp, free set and load vectors. The interior degrees of freedom of each cell follow the free retained ones. The matrix is scaled symmetrically by its diagonal and factorised by MKL PARDISO 2026.1 on the CPU. The factorisation is a symmetric positive definite Cholesky factorisation of the upper triangle (mtype 2), with nested-dissection ordering (iparm(2) = 3) and 16 threads. The PARDISO phases are called directly with explicitly set parameters, and all six loads are solved at once. Relative residuals \(\|Ku-f\|/\|f\|\) are recomputed with the unscaled matrix.

Route (b), conventional exact condensation, processes the cells one at a time on the CPU with 16 threads. Cell setup and assembly are as in route (a). PARDISO's Cholesky factorisation with the Schur-complement option (iparm(36)) then gives the dense condensed matrix of the cell on its retained degrees of freedom, after which the cell is released. The condensed lattice system is solved exactly by a block Cholesky factorisation in substructuring order. For each cell, its private retained block is factorised and eliminated with dense kernels. The dense interface matrix over the retained DOFs shared by several cells is then factorised and solved, and the private DOFs are recovered by back-substitution. The condensed cell matrices are held until their elimination.

Route (c) runs one lattice analysis with sensitivities in the deployed implementation of NICE on the GPU. Cell preparation comprises cell construction, stiffness and moment assembly, network input and encoding, and a first application that prepares the correction. The lattice and \(\mathbb K_{PP}\) are then assembled, and the preconditioner is set up (the balanced two-level action of Supplementary Note S4.1). Conjugate gradients solve the three traction loads to a recursive relative residual of \(10^{-6}\); the residuals reached are \(9.0\times10^{-7}\) to \(9.9\times10^{-7}\). Finally, the sensitivities \(\widetilde s_c\) of Eq. (17) for the three loads are obtained by reverse-mode differentiation of the moment integrals at the fixed recovered displacement fields. Route (c) uses the arithmetic of the timed route of Table 1. The network and the smoothing and coarse solve of the correction run in single precision, and the stiffness products of the condensed action in double precision (Appendix F.2). The three routes read the same generated cell geometries; geometry generation is not timed.

Whole-lattice iterative solvers (Table ST12e) use the global matrix, load vectors, clamp and numbering of route (a). These are exported once and solved with PETSc 3.26 [Balay et al. (2026)](https://petsc.org/release/manual/) with 16 MPI processes on the CPU used for routes (a) and (b). Conjugate gradients run on the unscaled system and stop on the unpreconditioned relative residual \(\|Ku-f\|/\|f\|\le10^{-9}\). The iterate is recorded where the residual first falls below \(10^{-3}, 10^{-4}, \dots\), together with its compliance and energy-norm difference from the final iterate. GAMG (smoothed aggregation, PETSc defaults) uses block size 3 and the six rigid-body modes of the DOF positions as near-null space. BoomerAMG uses HMIS coarsening, extended+i interpolation, nodal coarsening, strong threshold 0.5 and its default relaxation. The BDDC configuration considered has exact local solvers and one subdomain per cell. The Dirichlet matrix of a cell is its stiffness on the DOFs it shares with no other cell, and its Neumann matrix is the whole cell stiffness on its free DOFs. Each is factorised once, cell after cell, with the PARDISO configuration of route (a) (symmetric Jacobi scaling, real SPD Cholesky, tuned iparm, 16 threads). The singular Neumann matrices of floating cells are shifted by \(10^{-10}\) on the diagonal of the scaled matrix. The sum of these factorisations is a reference for the local factorisation cost of BDDC with exact local solvers in this configuration. It does not include the treatment of the primal constraints and the null space in the local Neumann problems, the coarse problem or the iterations. With MUMPS for the local and coarse problems on eight processes, BDDC with all local factors resident exceeded 90 GiB during its setup.

### Table ST12. Whole-lattice direct solution, conventional exact condensation and learned route: dimensions, phases and memory

Times in s. Memory in GiB (\(2^{30}\) bytes). PARDISO memory is its permanent plus factorisation storage (iparm(16) + iparm(17)) reported by the analysis phase, with kilobytes taken as 1024 bytes. Totals, iteration counts, peak process memory and the memory of route (c) are given in Table 5.

#### ST12a. Lattice dimensions

| Lattice | Cells (cut) | Cut-plane-element DOFs off the cell faces (fraction of free retained DOFs) |
| --- | --- | --- |
| 2×2×1, z=0 | 4 (2) | 35,829 (46%) |
| 2×2×1, z=1 | 4 (2) | 33,327 (47%) |
| 2×2×2 | 8 (4) | 69,156 (50%) |
| 3×3×1 | 8 (3) | 53,085 (37%) |

Total and free retained DOFs: Table 5.

#### ST12b. Route (a), direct solution on the CPU (Cholesky, 16 threads and one thread): phases (s) and memory (GiB)

| Lattice | Threads | Cells: setup + assembly | Global assembly and scaling | Analysis | Factorisation | Solution, 6 loads | Total | PARDISO memory | Max. relative residual |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 2×2×1, z=0 | 16 | 229.9 | 21.9 | 26.0 | 193.1 | 13.5 | 484.4 | 32.0 | 3.3e-10 |
| 2×2×1, z=0 | 1 | 777.3 | 19.8 | 115.8 | 1,500.0 | 12.1 | 2,425.0 | 31.4 | 4.0e-10 |
| 2×2×1, z=1 | 16 | 184.2 | 17.6 | 20.2 | 116.9 | 10.4 | 349.2 | 28.8 | 4.5e-10 |
| 2×2×1, z=1 | 1 | 685.6 | 18.8 | 106.1 | 1,309.4 | 11.3 | 2,131.3 | 28.3 | 5.2e-10 |
| 2×2×2 | 16 | 379.5 | 36.0 | 53.1 | 368.2 | 31.0 | 867.8 | 66.4 | 1.5e-10 |
| 2×2×2 | 1 | 1,402.1 | 39.3 | 254.5 | 4,140.3 | 27.7 | 5,863.9 | 65.5 | 1.8e-10 |
| 3×3×1 | 16 | 431.3 | 42.3 | 50.4 | 481.5 | 36.0 | 1,041.6 | 80.9 | 2.6e-09 |
| 3×3×1 | 1 | 1,669.9 | 45.4 | 324.4 | 6,059.1 | 32.4 | 8,131.0 | 79.8 | 2.8e-09 |

Total: sum of the phases; the 16-thread totals are those of Table 5. With 32 threads, the whole direct solution takes 378.5, 341.1, 823.4 and 955.5 s instead of 484.4, 349.2, 867.8 and 1,041.6 s, because cell setup and assembly do not speed up.

#### ST12c. Route (b), conventional exact condensation on the CPU (16 threads): phases (s) and memory (GiB)

| Lattice | Cell preparation | Cells: Schur complement | Condensed solve | Condensed cell matrices held (GiB) | Compliance vs. (a) |
| --- | --- | --- | --- | --- | --- |
| 2×2×1, z=0 | 210.6 | 274.3 | 86.8 | 17.8 | 5e-10 |
| 2×2×1, z=1 | 263.4 | 279.2 | 71.4 | 14.9 | 4e-10 |
| 2×2×2 | 393.6 | 392.5 | 133.9 | 32.8 | 2e-10 |
| 3×3×1 | 462.0 | 555.0 | 165.7 | 33.4 | 7e-10 |

Condensed solve: private elimination, interface factorisation and solution, and back-substitution for the six loads. The total of Table 5 also includes the per-cell lattice geometry, the matrix scaling and data movement between phases (17 to 35 s). Compliance vs. (a): largest relative difference over the six loads from the whole-lattice direct solution.

#### ST12d. Route (c), learned route: phases of one lattice analysis with sensitivities (s)

| Lattice | Cell preparation | Lattice and \(\mathbb K_{PP}\) assembly | Preconditioner setup | Conjugate-gradient solve, 3 loads | Sensitivities |
| --- | --- | --- | --- | --- | --- |
| 2×2×1, z=0 | 7.50 | 0.08 | 3.78 | 25.63 | 2.12 |
| 2×2×1, z=1 | 6.49 | 0.08 | 3.39 | 25.23 | 1.87 |
| 2×2×2 | 11.82 | 0.17 | 6.57 | 56.37 | 3.80 |
| 3×3×1 | 12.66 | 0.14 | 7.25 | 80.85 | 4.07 |

Total, iterations and memory: Table 5. The phases sum to 0.2–0.5 s less than the totals; the remainder is bookkeeping between phases. GPU memory is the peak allocated by the process; CPU memory is the resident set size after cell preparation. Learned substructures resident on the GPU (Supplementary Note S4.1): all cells of the four-cell lattices and four cells of each eight-cell lattice.

#### ST12e. Whole-lattice iterative solvers on the CPU (16 MPI processes): setup, conjugate gradients and BDDC factorisations (s)

| Lattice | GAMG: setup | GAMG to \(10^{-4}\): iterations / time per load | GAMG to \(10^{-4}\), with setup: 3 loads / 6 loads | GAMG at \(10^{-4}\): max. compliance / energy difference | BoomerAMG: setup / iterations to \(10^{-4}\) / time per load | BDDC factorisations: Dirichlet + Neumann |
| --- | --- | --- | --- | --- | --- | --- |
| 2×2×2 | 35.3 | 90–134 / 52–72 | 221 / 432 | 4.6e-10 / 2.1e-5 | 88.0 / indefinite preconditioner | 207.9 + 241.3 = 449.2 |
| 3×3×1 | 39.2 | 98–123 / 66–84 | 278 / 478 | 1.4e-10 / 1.2e-5 | 96.6 / 250–260 / 405–481 | 245.3 + 277.9 = 523.1 |

Differences are relative to the iterate at a relative residual of \(10^{-9}\), which GAMG reaches in 235–274 (2×2×2) and 204–220 (3×3×1) iterations. Loads: three traction loads, then three random loads, as in route (a). Cell setup and assembly and the global assembly and scaling are taken from route (a) and are common to all fine-scale routes. They are not included (Table ST12b: 416 and 474 s).

### Table ST13. Per-cell cost of conventional condensation and NICE

Four deployment cells, two cut and two uncut (Supplementary Section R1). Cell preparation: geometry preprocessing plus cell setup, moment integration and stiffness assembly, on the GPU for NICE and on the CPU for the conventional route. Condensation: network encoding plus correction setup (smoothing interval and coarse factorisation) for NICE; symbolic analysis and Cholesky factorisation of the interior by MKL PARDISO (16 threads) for the conventional route. Memory, in GiB: GPU memory held by NICE's network state and correction, or PARDISO's factorisation storage (iparm(17)). Neither includes the cell stiffness \(K\), which both routes hold (last column, GPU storage format).

| Cell | DOFs / retained | Cell preparation (s): NICE / conventional | Condensation (s): NICE / conventional | Memory (GiB): NICE state / PARDISO factor | \(K\) (GiB) |
| --- | --- | --- | --- | --- | --- |
| G1 (cut) | 177,507 / 24,636 | 5.3 / 27.3 | 0.72 / 10.5 | 0.74 / 2.89 | 1.39 |
| G2 (cut) | 67,224 / 18,858 | 3.0 / 10.9 | 0.16 / 1.9 | 0.25 / 0.60 | 0.51 |
| G3 (uncut) | 289,494 / 17,508 | 6.1 / 45.0 | 0.78 / 27.3 | 1.20 / 6.18 | 2.32 |
| G4 (uncut) | 404,148 / 25,920 | 6.5 / 47.0 | 1.09 / 48.9 | 1.37 / 9.42 | 2.71 |

## Supplementary Note S4. Global preconditioning, two-cell configurations and residual records

### S4.1. Balanced two-level preconditioner

Let \(\mathbb A\) be the supported assembled operator used in a run, either the exact or learned one. This notation is distinct from the local interior stiffness \(A=K_{II}\). Let
\(\mathbb K_{PP}=\sum_mB_m^TK_{PP,m}B_m\) on the free retained DOFs. The stiffness-block fine action is \(\mathcal B_f=\mathbb K_{PP}^{-1}\).

For a global coarse basis \(Z_g\) on the retained DOFs, the ideal coarse action is
\(Q_g=Z_g(Z_g^T\mathbb A Z_g)^{-1}Z_g^T\), after removal of dependent columns. The balanced action is

\[
\mathcal M^{-1}=Q_g+(I-Q_g\mathbb A)\mathcal B_f(I-\mathbb A Q_g).
\]

The reported coarse basis multiplies the trilinear lattice-vertex functions by three translations and three rotations about each vertex, then restricts the resulting fields to the supported retained DOFs. This global coarse basis differs from the interior basis \(V\) used to correct the network's interior displacements in each cell. The exact and learned lattice solves of Section 5.8, the learned route (c) of Section 5.9 and the learned solves of Section 5.10 use this balanced action with the stiffness-block fine action and this coarse basis.

The coarse matrix \(A_g=Z_g^T\mathbb A Z_g\) is symmetrised and scaled by its diagonal. Eigenvectors whose scaled eigenvalues are positive and exceed \(10^{-10}\) times the largest are kept, and the corresponding normalised columns \(Y_g\) give \(Q_g=Y_gY_g^T\). The stored product \(\mathbb A Y_g\) supplies the two projections.

On the GPU, \(\mathbb K_{PP}\) is factorised once per design iteration by a sparse Cholesky factorisation with NVIDIA cuDSS. The learned substructures are held on the GPU up to a memory budget for resident learned substructures; beyond it, the state of the remaining cells is held in CPU memory and streamed from there. The resident cells or budget of each run are given in the note to Table ST12d, in Table ST16 and in the caption of Table ST20.

### S4.2. Two-cell test configuration

In both configurations of Figure 9, the six face loads are traction loads along the three Cartesian axes on the loaded face of the test cell and the same three loads on the loaded face of the neighbouring cell. The traction loads are obtained by integrating the Q2 surface shape functions. Each nodal load vector is normalised to unit resultant before support elimination, and clamped entries are then eliminated. Three traction loads on the cut surface of the test cell, normalised in the same way, are reported separately when present. The neighbouring cell is uncut. Where the cut plane of the test cell meets the shared face, the pair is therefore a test configuration and not a physically cut specimen.

The pair solves use conjugate gradients preconditioned by the Cholesky factor of the exact assembled reference stiffness, with a relative recursive-residual tolerance of \(10^{-10}\) and at most 400 iterations.

### S4.3. Residual work, gradient checks and difference quotients of the surrogate compliance

The lattices of Section 5.8 and their two \(2\times2\times1\) layers are solved with NICE to a recursive residual of \(10^{-10}\), with the correction in double precision. Three quantities are recorded at the final iterate \(\bar U\). The first is the signed residual work \(\bar U^T\rho\), with \(\rho=f_g-y(\bar U)\) and \(y\) the applied learned action. The second is the action–energy inconsistency \(\omega=\sum_m\omega_m\), \(\omega_m=\bar q_m^Ty_m(\bar q_m)-\bar u_m^TK_m\bar u_m\). Here \(\bar q_m=B_m\bar U\), \(\bar u_m=F_m\bar q_m\) and \(y_m\) is the applied action of cell \(m\), so that \(y(\bar U)=\sum_mB_m^Ty_m(\bar q_m)\) and \(\omega\) is the quantity of Appendix C.2. The third is the dual-norm bound on \(|\bar U^T\rho|\) of Appendix C.2, with \(\rho^T\mathbb K^{-1}\rho\) computed with the exact assembled operator. All quantities are relative to the exact compliance. The recomputed relative residual \(\|f_g-\widehat{\mathbb K}\bar U\|/\|f_g\|\) stagnates at \(3.6\times10^{-4}\) (block) and \(3.2\times10^{-3}\) (layer). Running the correction in single instead of double precision changes this residual of the block by less than 1%. It changes the compliance errors of the block by less than \(3\times10^{-9}\) and those of the layer by less than \(4\times10^{-8}\), and the gradient errors of both by less than \(10^{-6}\).

The gradient metrics compare the sensitivities from NICE with the exact ones after aggregation over the shared lattice vertices, that is, after summing the cells' thickness sensitivity components at each vertex. The aggregated vector is the gradient that an optimiser with shared thickness variables receives; loads are held fixed. For the complete derivative of the surrogate compliance, every cell's condensed stiffness was rebuilt at \(\tau\pm h\tau_ce_c\) for each corner. The rebuilt stiffness was applied to the solved retained displacements at fixed \(\widehat q\), which gives the action-energy difference quotient \(D_{\mathrm{act}}\). A rebuild counts as switched when, relative to the unperturbed build, it changes any of the following: a binary node indicator; the set of weak-region element or face stencils (stencils containing a weakly connected node, Appendix G.1); the active element set; or the diagonal shift of the coarse factorisation (Appendix F.1).

#### Table ST14. Residual work, dual-norm bound, gradient and difference quotients of the surrogate compliance

Face = the three face traction loads; random = the three random loads. Gradient error: \(\|\widetilde{\boldsymbol s}_g-\boldsymbol s_g\|/\|\boldsymbol s_g\|\) over the shared vertex parameters, in percent; cosine over all six loads. \(D_{\mathrm{act}}\): relative error against the exact vertex gradient under the face loads at step \(h\) (in units of \(\tau_c\)), in percent. Switched rebuilds: number of the 49 rebuilds per cell (three steps, both signs and eight corners, plus one unperturbed) that change a discrete choice, range over cells.

| Lattice | \(\max\lvert\bar U^T\rho\rvert/C\) | Dual-norm bound / \(C\) | \(\max\lvert\omega\rvert/C\) | Gradient error, face / random (%) | Smallest cosine | \(D_{\mathrm{act}}\) at \(h=3\times10^{-3}\) / \(10^{-3}\) / \(3\times10^{-4}\) (%) | Switched rebuilds per cell |
| --- | --- | --- | --- | --- | --- | --- | --- |
| \(2\times2\times1\), \(z=0\) | \(2.3\times10^{-8}\) | \(6.8\times10^{-6}\) | \(5.0\times10^{-9}\) | 0.022–0.048 / 0.22–0.27 | 0.9999998 | 0.29 / 0.65 / 1.53 | 31–48 |
| \(2\times2\times1\), \(z=1\) | \(5.3\times10^{-8}\) | \(7.9\times10^{-6}\) | \(9.5\times10^{-9}\) | 0.067–0.130 / 0.29–0.35 | 0.9999993 | 0.21 / 0.27 / 0.35 | 32–47 |
| \(2\times2\times2\) | \(2.9\times10^{-9}\) | \(4.7\times10^{-6}\) | \(2.4\times10^{-9}\) | 0.052–0.069 / 0.24–0.27 | 0.9999996 | 0.18 / 0.29 / 0.59 | 31–48 |
| \(3\times3\times1\) | \(2.5\times10^{-8}\) | \(1.4\times10^{-5}\) | \(7.0\times10^{-9}\) | 0.041–0.088 / 0.22–0.28 | 0.9999996 | 0.19 / 0.23 / 0.35 | 21–48 |

On the lattices, \(D_{\mathrm{act}}\) grows as the step decreases, so the difference quotient does not resolve the complete derivative of the surrogate compliance. The same records were made on the fourteen two-cell configurations of the selection cells (Section 5.6). They give \(|\bar U^T\rho|/C\le1.8\times10^{-8}\), a dual-norm bound of at most \(2.0\times10^{-5}\), \(|\omega|/C\le1.7\times10^{-8}\), thickness-sensitivity errors of 0.07–0.67% with cosines of at least 0.999998, and \(D_{\mathrm{act}}\) errors of 0.05–0.27% at \(h=10^{-3}\) that likewise do not decrease with the step.

The accuracy of the two eight-cell lattices in these runs is reported in Section 5.8. The values below are given for the \(2\times2\times2\) block / the \(3\times3\times1\) layer. The compliance errors under the three face loads are 0.0137, 0.0110 and 0.0094% / 0.0147, 0.0134 and 0.0103%, and at most 0.069% / 0.056% under the random loads. The largest thickness-sensitivity error over all cells is 0.144% / 0.144% under the face loads and 0.454% / 0.426% under the random loads. The relative Euclidean difference between the learned and exact assembled retained solutions over all six loads is 0.130% / 0.123%. The cell energy errors \(\varepsilon_m\) at the exact retained displacements are 0.0061–0.026% / 0.0051–0.052% under the face loads and 0.027–0.123% / 0.023–0.077% under the random loads. The excess of \(\beta\) (Eq. (15)) over the lattice compliance error is 0.13–0.28% / 0.21–0.33% of that error under the face loads and 4.4–5.1% / 4.2–5.0% under the random loads. The exact references, assembled from the dense exact condensed matrices, reach recomputed relative residuals of \(9.1\times10^{-11}\) / \(1.3\times10^{-10}\). The corner parameters of the block range from 0.2516 to 0.5350.

One-corner sweeps test the smoothness of the surrogate compliance across these switches. In the two-cell configurations M1/x and U1/y, one corner parameter of the test cell was varied over up to ±10% of its value; this was the corner with the largest or the median sensitivity. Every design had its own regenerated geometry and learned substructure. The compliance change between neighbouring designs was compared with the change predicted by the trapezoidal rule from the sensitivities at both ends. The same comparison was made for the exact discrete model at every tenth design. Every interval on M1/x and 24 of the 26 intervals on U1/y change at least one discrete choice. Most of these changes concern the active elements, ghost faces and retained DOFs, which belong to the discrete model itself. On these intervals the surrogate compliance departs from the predicted change by up to 0.25% (M1/x), 0.17% and 0.085% (U1/y) of the compliance, and the exact model departs by the same amount. On M1/x the departures are 0.230, 0.252 and 0.228% against 0.228, 0.249 and 0.225%; on U1/y they are identical to three significant digits. On the two U1/y intervals without a switch the departure is \(7\times10^{-8}\). At the reference designs the error of the surrogate compliance stays at its bound \(\beta\) (\(6.0\times10^{-4}\) on M1/x, \(7.2\times10^{-5}\) on U1/y). On the checked intervals the dominant departure of the surrogate objective is therefore also present in the exact discrete model, which an optimiser driven by exact condensation meets in the same way.

## Supplementary Note S5. Network settings and parameter counts

Table ST15 complements Appendix G with the coefficient-bound groups, the initialisation and the parameter count of every block.

### Table ST15. Coefficient bounds, initialisation and parameter counts

| Item | Setting |
| --- | --- |
| Coefficient bounds \(a_{\max}\) | Fixed per group: gather and scatter role of each of the 8 element, 12 face and 4 weak-region layers (48 groups), and restriction and prolongation of each of the 3 levels (6 groups), plus the knee \(k_b\): 55 stored values, part of the stored model state (definition in Appendix G) |
| Initialisation | Channel maps \(W_{\ell h}\): \(0.5\,\mathcal N(0,1)/\sqrt{32}\); \(W_{\rm in}\): \(\mathcal N(0,1)/\sqrt3\); \(W_{\rm out}\): \(0.1\,\mathcal N(0,1)/\sqrt{32}\); convolution kernels: \(0.5\,\mathcal N(0,1)/\sqrt{27\cdot32}\); position-embedding tables: \(0.1\,\mathcal N(0,1)\); \(\sigma_\ell=1\); \(\lambda_{\rm mix}=0\); affine layers: PyTorch default. The base network is initialised from the parameters of three preceding training stages (notes to Table ST01), the first of which uses the random initialisation above; the continuations start from the base network's selected parameters |
| Parameter counts, geometry branch | Element encoder 12,288; node encoder 9,024; message passing 49,664; face encoder 12,608; position-embedding tables 432; element, face and weak-region coefficient heads 12,928, 15,008 and 10,848; restriction/prolongation heads 12,870; convolution-coefficient heads 37,440. Total 173,110 |
| Parameter counts, displacement branch | Element channel maps 32,768; face channel maps 49,152; weak-region element channel maps 16,384; convolution kernels 331,776; \(W_{\rm in}\) and \(W_{\rm out}\) 96 each; skip scalars 3; \(\lambda_{\rm mix}\) 24. Total 430,299 |
| Stored values | 603,409 trainable parameters and 55 fixed bound values: 603,464 in total |
| Training settings | Notes to Supplementary Table ST01; seed 0 for every run |

