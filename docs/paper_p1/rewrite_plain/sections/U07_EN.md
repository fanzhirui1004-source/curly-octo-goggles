## 5. Numerical examples

The numerical examples cover single cells, assemblies of cells and thickness design. On single cells, the mean energy error of NICE over the 80 validation geometries is 0.074%. Applying the correction to the base network without retraining lowers the mean energy error under traction loads from 6.89% to 0.0965%. Under the same correction, a graph-harmonic initial field without trainable parameters leaves 5.4 to 265 times the error of NICE (Sections 5.3–5.5). After assembly, the compliance error follows the relation of Proposition 3, in which the energy error of each cell is weighted by its energy fraction. The local sensitivity, however, has to be checked separately (Sections 5.6–5.8). In design, an analysis of an eight-cell lattice with sensitivities is about ten times faster than a direct solution of the whole lattice (Section 5.9). For the optimised designs, the compliance and the gradient computed with NICE agree with those of exact static condensation (Section 5.10). Table 3 summarises the accuracy of NICE against exact static condensation in all assembled examples. Sections 5.1 and 5.2 give the settings of the examples and verify the reference model.

### 5.1. Geometries, variants and loads

The 80 validation geometries form four cut-severity groups of 20 cells each: uncut, lightly cut, moderately cut and heavily cut. Their thickness fields are uniform, affine or mixed trilinear, and their corner parameters range from 0.1762 to 0.6983 (Supplementary Table ST02). Each cell has at most one planar cut, whose normal is \((\cos\vartheta,\sin\vartheta,0)\) with \(0<\vartheta<\pi/4\). A heavy cut retains less than one third of the volume of the cell box, a moderate cut between one and two thirds, and a light cut more than two thirds. The cells used for detailed comparisons are labelled by group, with U, L, M and H for uncut, lightly, moderately and heavily cut cells. They are U1, U2, L1, M1, M2 and H1–H3 (Supplementary Section R1). Table 1 lists the discretisation and correction settings. Figure 4 shows four of the cells used for detailed comparisons.

![Figure 4](figures/F08_geometry.png)

**Figure 4. Representative validation geometries.** (a) Uncut cell U1, (b) moderately cut cell M1 and (c,d) heavily cut cells H1 and H2, shown at the same unit-box scale and from the same viewing direction. Light blue shows the material surface, dark blue the wall section on the cell faces and sand the section on the cut plane. The part removed by the cut is drawn in translucent grey, and the cells are viewed from the side of the cut. The percentages give the box volume that remains after the cut, relative to the unit box, before the intersection with the thin-wall material. The surfaces are reconstructed from Eq. (1) on a grid of 129 points per axis and serve only for visualisation.

Each validation geometry is evaluated with the test displacements of the nine load classes of Supplementary Table ST03. Appendix G.2 defines these classes. The classes comprise imposed polynomial and multiscale displacements, and the responses to equilibrated nodal point loads and to traction loads applied on all cell faces or on a single face. They also comprise responses with spring supports and displacements imposed by a neighbouring cell. Traction loads represent surface loads and the tractions exerted by neighbouring cells. They form the principal load class, with 64 test displacements per geometry. Equal nodal point loads also act on weakly connected nodes and serve as a severe test of the stabilised problem. In the five cells of Supplementary Table ST04, the ghost penalty carries on average less than 0.05% of the energy of the exact field under traction loads, but 49–81% under equal nodal point loads.

**Table 1. Discretisation and correction settings**

| Quantity | Setting |
|---|---|
| Reference geometry | Unit box; Schwarz-P-type implicit band with trilinear corner parameters and an optional planar cut |
| Displacement approximation | Continuous tensor-product \(Q_2\) solid elements on a Cartesian background mesh |
| Background resolution, validation geometries | \(n=32\) elements per axis; \(65\) Q2 nodes per axis |
| Isotropic material, validation geometries | Normalised Young's modulus \(E_Y=1\), Poisson's ratio \(\nu=0.3\) |
| Ghost-penalty coefficient, validation geometries | \(\gamma=10^{-4}\) |
| Retained DOFs | DOFs of the cell-face nodes of material patches of positive area on the cell faces; all DOFs of the active elements that carry a cut-surface patch of positive area (cut-plane elements) |
| Standard volume integration | \(4^3\) initial subcells; one local refinement of partially filled subcells; clipped Kuhn tetrahedra with quadrature-rule parameter 4 |
| Smoothing interval | \([b/30,b]\), with \(b\) equal to 1.05 times a 40-step power-iteration estimate of \(\lambda_{\max}(D^{-1}A)\) (Section 3.3) |
| Principal coarse space | Trilinear vector functions on a \(17^3\)-vertex grid, restricted to the interior DOFs |
| Finite-difference step of the thickness derivative | \(h_c=10^{-5}\tau_c\), with the active DOFs and the ghost-penalty contribution held fixed |
| Two-grid correction of NICE | 8 Chebyshev steps, coarse-grid correction with \(Q_1(17)\), 8 Chebyshev steps; same sequence in training and evaluation |
| Floating-point precision | Network in single precision; stiffness matrix–vector products and energies in double precision; correction in double precision in the accuracy studies, and in single precision in the timed analyses of Section 5.9 and in the optimisations of Section 5.10 (Appendix F.2) |

The mesh, material and stabilisation settings apply to all 80 validation geometries. Lengths are relative to the unit box, and the Young's modulus is normalised. For other correction sequences, the coarse-space dimensions and the numbers of smoothing steps are given with the comparisons in which they appear.

The main text compares three variants. Two of them are trained networks, listed in Table 2. The base network was trained without correction. NICE continues the training of the base network for 15,000 training steps, with the complete correction \(\mathcal W\) of Sections 3.3 and 3.4 inside the training loop (Section 3.5).

**Table 2. Variants compared in the numerical examples**

| Variant | Network parameters | Training set (geometries) | Correction in training | Correction at evaluation |
| --- | --- | ---: | --- | --- |
| **NICE** | Base network continued for 15,000 training steps | 591 | 8 / \(Q_1(17)\) / 8 | 8 / \(Q_1(17)\) / 8 |
| Base network | 40,000 training steps; parameters of step 30,000 selected | 305 | None | None |

The third variant is the base network with the correction \(\mathcal W\) applied at deployment, without retraining. It is denoted 'Base network + correction' and is evaluated in Sections 5.3, 5.5 and 5.6.

All variants use the network architecture of Section 3, with about \(6\times10^5\) trainable parameters (Supplementary Table ST15). NICE draws its training geometries from a set of 591 geometries, which contains 304 of the 305 training geometries of the base network. In its 15,000 training steps, NICE sees at most 153 of the 591 geometries (Supplementary Table ST01). Supplementary Note S8 reports two further continuations of the base network. They draw from the same set of 591 geometries and see the same geometries in the same order as NICE.

Twenty of the 80 validation geometries, 6 uncut and 14 cut, were used for checkpoint selection (Supplementary Table ST01). Statistics are therefore also given for the 60 geometries outside checkpoint selection. The test cells of the two-cell assemblies of Section 5.6 belong to these 20 selection geometries. Nine further cells outside checkpoint selection are assembled in that section as an additional check. NICE was designated the principal variant after all variants had been compared on the 80 validation geometries and the two-cell configurations (Supplementary Table ST01). This comparison included the two continuations of Supplementary Note S8. The sixteen cells of the lattices of Section 5.8 entered no selection step. Population statistics first average the energy error over the test displacements of each geometry and load class, and then weight the geometries equally (Appendix A.1). The population maximum is therefore the largest mean of a single geometry.

**Table 3. Accuracy of NICE against exact static condensation in the assembled examples.** Compliance error: \(|\widehat C/C-1|\). Sensitivity error: largest relative error \(e_s\) of the eight-component thickness-sensitivity vector of a cell. Gradient error: relative error of the lattice gradient with respect to the shared vertex parameters. Entries are maxima or ranges over the listed loads and designs. In the two-cell assemblies only the test cell is learned; in all other examples every cell is learned.

| Example | Section | Cells (cut) | Loads or designs | Compliance error (%) | Sensitivity or gradient error (%) |
| --- | --- | --- | --- | ---: | --- |
| Two-cell assemblies, seven selection cells | 5.6 | 2, 14 configurations | six face loads each | ≤ 0.056 | sensitivity ≤ 0.67 |
| Two-cell assemblies, cut-surface traction loads | 5.6 | 2, 7 configurations | three cut-surface traction loads each | ≤ 0.10 | sensitivity ≤ 0.16 |
| Two-cell assemblies, nine held-out cells | 5.6 | 2, 18 configurations | six face loads each | ≤ 0.27 | sensitivity ≤ 1.49 |
| Lattices with every cell learned, face loads | 5.8 | 8 (4), 8 (3) | three face loads | ≤ 0.015 | sensitivity ≤ 0.144; gradient 0.04–0.09 |
| Lattices with every cell learned, random loads | 5.8 | 8 (4), 8 (3) | three random loads | ≤ 0.069 | sensitivity ≤ 0.454; gradient 0.22–0.28 |
| Case A, thickness optimisation | 5.10 | 8 (4) | designs at iterations 0, 12, 23 | 0.011–0.028 | gradient 0.069–0.33 |
| Plate supported on its cut | 5.10 | 24 (8) | start, final, homogenisation and two uniform designs | 0.011–0.032 | gradient 0.029–0.32 |
| Plates of the scale study | 5.10 | 24 (8), 51 (12) | first design | 0.013, 0.011 | gradient 0.045, 0.047 |

### 5.2. Verification of the reference model and of the deployed operator

The reference model with \(n=32\) was verified by mesh refinement on seven cells. U1 was refined to \(n=40\), and M1 and M2 to \(n=48\) (Figure S01). H1, H2 and two further validation cells were refined to \(n=64\) (Supplementary Table ST10). Except in the heavily cut cell H2, refinement changes the compliance by at most 0.28% and the thickness sensitivities by at most 0.44%. In H2, the changes reach about 1% in the compliance and 3.6% in the sensitivity. On uncut cells and assemblies of uncut cells, the compliance of the discrete model agrees with that of an independent body-fitted discretisation with quadratic tetrahedra to 0.05–0.25% (Supplementary Note S1). Varying the ghost-penalty coefficient between \(10^{-5}\) and \(10^{-3}\) and refining the volume integration each change the compliance and the sensitivities by at most 0.12%. Supplementary Note S1 also verifies the finite-difference stiffness derivative and its step size (Figure S01(c)). All subsequent comparisons are made against this discrete reference model (Section 2.1).

On five cells, U2, M1, M2, H1 and H2, the deployed operator is symmetric and satisfies the work–energy identity, both to relative deviations of at most \(3\times10^{-8}\). This identity states that the work of the returned nodal forces on the retained displacements equals the energy of the recovered field. The deployed operator gives the rigid-body modes zero energy up to round-off. Its recovered field agrees with the field computed by the training implementation to within \(1.1\times10^{-7}\) (Supplementary Table ST04).

The Chebyshev smoothing of Section 3.3 is guaranteed not to increase the energy error when \(b\ge\lambda_{\max}(D^{-1}A)\). On all 80 validation geometries, the endpoint \(b\) obtained from the power iteration exceeds the converged largest eigenvalue by 2.1–5.0% (Appendix D). The Gershgorin bound is a guaranteed upper bound, but it is 4.7 to 19.4 times larger than \(b\), with a median of 18.4. Used as \(b\), it would place the smoothing interval far above the actual spectrum. On these geometries, the comparison with the converged largest eigenvalue supports the spectral condition numerically. It was not checked for the cells of the lattices and optimisations of Sections 5.8–5.10, which use the same power-iteration estimate. The symmetry and positive semidefiniteness of \(\widehat S\) and the ordering \(\widehat S\succeq S\) do not depend on this condition.
