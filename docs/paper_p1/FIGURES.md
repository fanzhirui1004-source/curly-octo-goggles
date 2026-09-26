# P1 figures: current integrated version

11 main figures and 5 supplementary figures. PDF and SVG are vector exports.

## Figure 1

**Figure 1. Representative validation geometries.** The same unit-box scale and viewing direction are used for (a) uncut U1, (b) moderately cut M1, and (c,d) heavily cut H1 and H2. Blue denotes the material surface and orange the macro-cut section. Percentages indicate the retained macro-domain volume relative to the unit box, before intersection with the thin-wall material. Surfaces are reconstructed from the trilinear band parameters and cut-plane data in Eq. (1); the visualisation sampling is specified in Supplementary Note S3.

![Figure 1](figures/F08_geometry.png)

[PDF](figures/F08_geometry.pdf) | [SVG](figures/F08_geometry.svg) | [PNG](figures/F08_geometry.png)

## Figure 2

**Figure 2. Learned displacement extension, equilibrium correction and variational assembly.** (a) Rigid motion is separated from the retained displacement before the deformation is extended by the network. The rigid field is reconstructed and the prescribed retained values are restored; correction then reduces internal imbalance at fixed retained displacement. (b) Applying the local stiffness and the complete extension transpose gives the work-conjugate retained force. The assembled solution supplies the inputs for local field recovery. The neural extension is detailed in Figure 3; Section 5 defines the internal correction.

![Figure 2](figures/F01_method_overview.png)

[PDF](figures/F01_method_overview.pdf) | [SVG](figures/F01_method_overview.svg) | [PNG](figures/F01_method_overview.png)

## Figure 3

**Figure 3. Geometry-conditioned neural displacement architecture.** (a) Element and node encoders produce 64-channel geometry embeddings, followed by two rounds of residual exchange. Coefficient heads condition the local interactions, grid transfers and convolutions. (b) The displacement branch propagates the nonrigid retained input through four local interaction pairs, the multilevel block, four further local pairs and four pairs on weakly supported stencils. Linear input and output maps connect the three displacement components to 32 latent channels. Deterministic bypasses reconstruct rigid motion and restore the original retained values. (c) Restriction and prolongation connect grids with 65, 33, 17 and 9 background positions per axis; actual active node counts depend on geometry. Each coarse level has two residual convolutions on each pass, and upward transfers combine with additive skips. (d) A local interaction uses geometry-weighted gathering, four channel-mixing heads and scattering, followed by residual addition and retained-value restoration. E and G denote element and ghost-face interactions. Blue dashed arrows carry geometry-dependent coefficients; solid arrows carry features or displacement states. Channel-mixing matrices and convolution kernels are shared trainable parameters. For fixed geometry, the complete displacement path is linear. Training updates parameters through both branches; inference reuses the geometry coefficients.

![Figure 3](figures/F11_network_architecture.png)

[PDF](figures/F11_network_architecture.pdf) | [SVG](figures/F11_network_architecture.svg) | [PNG](figures/F11_network_architecture.png)

## Figure 4

**Figure 4. Supports and loading of the two-cell examples.** Box envelopes define the coordinate convention. (a) Configuration x: the neighbour is translated by \((-1,0,0)\), the face \(x=-1\) is clamped, and face tractions act at \(y=0\). (b) Configuration y: the translation is \((0,-1,0)\), the face \(y=-1\) is clamped, and tractions act at \(x=0\). Each target (T) and neighbour (N) face carries separate x-, y- and z-directed consistent-traction loads. Coincident box-node coordinates are shared across the interface; non-box cut-band coordinates remain local. Cut targets also receive three macro-cut tractions, analysed separately from the six face loads.

![Figure 4](figures/F09_assembly_loads.png)

[PDF](figures/F09_assembly_loads.pdf) | [SVG](figures/F09_assembly_loads.svg) | [PNG](figures/F09_assembly_loads.png)

## Figure 5

**Figure 5. Geometry and loading dependence of directional energy error.** (a,b) Geometry-weighted means under nodal-force and spring-supported loading, with 20 geometries per cut-severity stratum. (c) Means under consistent tractions, single-face consistent tractions, stiffness-scaled supports and neighbour-induced retained displacements; counts indicate geometries with an available observation. Panels (a–c) use the original orientation. (d) Paired nodal-force results for B on all 80 geometries in the original and transformed orientations. Open circles denote uncut cells and filled triangles cut cells; the dashed line denotes equality. Each observation is a geometry's mean directional energy excess, \(q^T(\widehat S-S)q/(q^TSq)\).

![Figure 5](figures/F02_validation.png)

[PDF](figures/F02_validation.pdf) | [SVG](figures/F02_validation.svg) | [PNG](figures/F02_validation.png)

## Figure 6

**Figure 6. Spectral distribution of the uncorrected extension error.** Panels show U1, M1, M2 and H2 for predictor B. Modes solve \(Av=\lambda Dv\), with \(A=K_{II}\) and \(D=\operatorname{diag}(A)\), and are ordered by increasing eigenvalue. Filled orange markers represent the extension error and open grey markers the exact internal field. Solid circles correspond to consistent tractions and dashed triangles to nodal forces. Curves are directional means; bands give the 10th–90th directional percentiles under consistent tractions. Each cumulative fraction uses the total internal energy of its own field or error as denominator.

![Figure 6](figures/F03_spectrum.png)

[PDF](figures/F03_spectrum.pdf) | [SVG](figures/F03_spectrum.svg) | [PNG](figures/F03_spectrum.png)

## Figure 7

**Figure 7. Accuracy gained by correcting B at fixed weights.** (a,b) Mean directional energy excess and field-based sensitivity error during Chebyshev smoothing on five cells. (c) M1 with no correction, eight smoothing steps, coarse correction followed by eight steps, and eight steps on each side of the coarse correction. The mean excesses are 13.5%, 4.68%, 0.230% and 0.186%, with 0, 8, 8 and 16 smoothing steps in total. Dots show means and caps the 90th percentile. The two methods with eight smoothing steps differ by one additional coarse solve. The \(Q_1(17)\) representation contains 5,601 coefficient columns for 165,927 internal degrees of freedom. (d) Mean energy excess versus steps per smoothing stage; the complete cycle uses twice this count. All panels use consistent tractions, fixed retained displacements and \(\alpha=30\). In (a,b), the step axis is linear from zero to one and logarithmic thereafter.

![Figure 7](figures/F04_correction.png)

[PDF](figures/F04_correction.pdf) | [SVG](figures/F04_correction.svg) | [PNG](figures/F04_correction.png)

## Figure 8

**Figure 8. Global compliance and local sensitivity in assembled cell pairs.** The target uses a learned operator and the neighbour exact condensation. (a,b) Maximum errors over six face loads for 25 model–configuration combinations; sensitivity is also maximised over both cells. Blank entries indicate unavailable results. (c) S8 on H1/x under target-face (T), neighbour-face (N) and macro-cut (C) tractions, each in the x, y and z directions; sensitivity refers to the target cell. (d) B on M1/x under target-face loads, comparing the full reconstructed response with fields obtained using the exact retained displacement or exact extension. These replacements probe separate effects on the sensitivity vector. Dashed lines mark 3%; cut-face loads are shown separately from the six-face-load comparison.

![Figure 8](figures/F05_assembly.png)

[PDF](figures/F05_assembly.pdf) | [SVG](figures/F05_assembly.svg) | [PNG](figures/F05_assembly.png)

## Figure 9

**Figure 9. Participation-weighted compliance error and local sensitivity.** Each point is one load for one model–configuration combination: 192 observations from the 25 model–configuration combinations in Figure 8. Filled markers denote face loads and open markers macro-cut loads. (a) Compliance error against \(\beta=\sum_mw_m\varepsilon_m\), with \(w_m=q_m^TS_mq_m/C\) and \(\varepsilon_m=q_m^T(\widehat S_m-S_m)q_m/(q_m^TS_mq_m)\), evaluated at the exact assembled retained displacement. Only the learned target contributes to \(\beta\). (b) Compliance and target-cell sensitivity errors under the same loads; the annotation identifies B on U1/x under the neighbour-z load. Dashed lines in (a,b) denote equality. (c,d) The two response errors versus the target's exact energy participation.

![Figure 9](figures/F10_energy_participation.png)

[PDF](figures/F10_energy_participation.pdf) | [SVG](figures/F10_energy_participation.svg) | [PNG](figures/F10_energy_participation.png)

## Figure 10

**Figure 10. Response errors caused by restricting box-face displacements.** Both cells of H1/x use exact operators and Bernstein degree \(r\) on every box face, with unrestricted non-box cut-band coordinates. (a) Reduced coordinate count; the dashed line denotes the 32,991-coordinate full representation. (b,c) Maximum compliance and target-cell sensitivity errors over the three target-face loads or all six target- and neighbour-face loads. Macro-cut tractions are excluded from both sets. Errors are relative to the full retained-space solution; horizontal reference lines mark 3%.

![Figure 10](figures/F06_bernstein.png)

[PDF](figures/F06_bernstein.pdf) | [SVG](figures/F06_bernstein.svg) | [PNG](figures/F06_bernstein.png)

## Figure 11

**Figure 11. Preparation and application cost of predictor D.** (a–c) Time per complete batch of one, 16 and 64 vectors on four cells, comparing exact interior solves with learned sparse-matrix (CSR) and fused implementations. The fp32 factor uses three fp64 iterative-refinement steps (IR). (d) Interior factorisation, network caching and preparation of the reusable learned action. (e) Sparse-stiffness storage and free-memory changes associated with the exact factor and learned state; the allocation measures are not additive. Measurements use an NVIDIA GeForce RTX 5090, with three timed repetitions after one warm-up. Memory is expressed in GiB. The supplementary geometry key identifies G1–G4.

![Figure 11](figures/F07_cost.png)

[PDF](figures/F07_cost.pdf) | [SVG](figures/F07_cost.svg) | [PNG](figures/F07_cost.png)

## Figure S02

**Figure S02. Distributions of geometry-level directional energy errors.** Each point is one geometry's directional mean in the original orientation; horizontal marks are population medians. The nodal-force, spring-support, single-face-force, polynomial and multiscale classes contain 80 geometries each. Consistent traction, single-face consistent traction, stiffness-scaled support and neighbour-induced displacement classes contain 20, 20, 19 and 15 geometries, respectively. P0 has no records in these four enriched classes. Marker shape and colour identify the predictor; deterministic horizontal offsets separate overlapping observations. All panels use the same logarithmic error range.

![Figure S02](figures/S02_distributions.png)

[PDF](figures/S02_distributions.pdf) | [SVG](figures/S02_distributions.svg) | [PNG](figures/S02_distributions.png)

## Figure S03A

**Figure S03A. Smoothing from learned and zero internal fields.** (a,b) Mean directional energy excess for consistent-traction and nodal-force responses; (c,d) corresponding field-based sensitivity errors. Both initialisations prescribe the same retained displacement. Solid curves with filled markers start from predictor B; dashed curves with open markers start from zero internal displacement. Zero-start sensitivity is recorded only at 32 steps. All corrections use \(\alpha=30\). The step axis is linear between zero and one and logarithmic thereafter.

![Figure S03A](figures/S03A_smoothing.png)

[PDF](figures/S03A_smoothing.pdf) | [SVG](figures/S03A_smoothing.svg) | [PNG](figures/S03A_smoothing.png)

## Figure S03B

**Figure S03B. Recorded coarse representations and correction sequences.** Rows correspond to U1, M1 and M2; columns use consistent-traction and nodal-force responses. Six coarse representations are compared under four initialisation and smoothing sequences, with eight steps in each pre- or post-smoothing stage. Dots indicate directional means and caps the 90th percentile. Dashed and dotted references denote B alone and B followed by one smoothing stage. \(Q_1\), \(Q_2\) and PU denote trilinear, quadratic and linearly enriched partition-of-unity generating families. The first label number identifies grid resolution and the lower number counts columns after internal restriction and screening. For the structurally redundant PU family, these counts do not establish an independent-space dimension, and the plotted solve results do not verify exact-projection properties. Appendix F.1 explains the rank and solve conditions; Table ST04 gives all statistics. Coarse updates preserve every retained coordinate.

![Figure S03B](figures/S03B_coarse_spaces.png)

[PDF](figures/S03B_coarse_spaces.pdf) | [SVG](figures/S03B_coarse_spaces.svg) | [PNG](figures/S03B_coarse_spaces.png)

## Figure S04

**Figure S04. Field-based sensitivity-error diagnostics.** (a) Paired mean energy and sensitivity errors for consistent-traction and nodal-force responses, using six B cells and five C cells. (b) Consistent-traction linear-term norm share \(\|D_1\|_F/(\|D_1\|_F+\|D_2\|_F)\), where \(D_1+D_2\) is the sensitivity-error matrix over all eight design components and evaluated directions. (c,d) Shares of absolute elementwise sensitivity-error contributions and element counts in four mutually exclusive material-volume-fraction groups for B under consistent tractions. Each error group sums absolute contributions over its elements, design components and directions before normalisation by the total. Open markers and hatched bars in (a,b) identify C; its H2 observation is unavailable.

![Figure S04](figures/S04_sensitivity_diagnostics.png)

[PDF](figures/S04_sensitivity_diagnostics.pdf) | [SVG](figures/S04_sensitivity_diagnostics.svg) | [PNG](figures/S04_sensitivity_diagnostics.png)

## Figure S05

**Figure S05. Recorded assembled iterative solves.** Rows show a cell pair and repeated-cell \(2\times2\times2\) and \(3\times3\times3\) arrays; columns show iteration counts, times and relative residuals. Filled markers give recursive residuals and open markers explicitly recomputed residuals using the operator applied in the same run. Crosses identify the 300 s time limit, and the dashed residual reference is \(10^{-8}\). Diag, \(K_{PP}\), Add, Bal and Def denote diagonal, assembled retained-block, additive, balanced two-level and deflated preconditioning, respectively. Here \(K_{PP}\) denotes the fine action \(\mathbb K_{PP}^{-1}\) on the assembled retained system; the precise actions are given in Supplementary Note S1. The learned operator uses predictor D. These historical RTX 5090 runs enabled TF32 convolution. The pair joins G1 and G3, while each array repeats G3.

![Figure S05](figures/S05_iterative_solves.png)

[PDF](figures/S05_iterative_solves.pdf) | [SVG](figures/S05_iterative_solves.svg) | [PNG](figures/S05_iterative_solves.png)
