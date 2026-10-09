## Table ST08. Complete results of the two-cell assemblies with a continuous-thickness neighbour

Maximum relative errors (%) of the compliance and of the sensitivity vector over the six face loads, compared with the 3% reference. Sensitivity maxima include both cells. The test cell uses the stated learned variant, and its neighbour is exact. Cell labels abbreviate the validation identifiers (Supplementary Section R1).

| Test cell | Configuration | Variant | Max. compliance error (%) | Max. sensitivity error (%) |
| --- | --- | --- | --- | --- |
| U1 | x | Base network | 0.465 | 5.830 |
| U1 | y | Base network | 0.427 | 5.390 |
| U2 | x | Base network | 0.105 | 1.996 |
| M1 | x | Base network | 4.384 | 12.153 |
| H1 | x | Base network | 1.180 | 2.512 |
| H1 | y | Base network | 1.379 | 1.820 |
| M2 | x | Base network | 0.571 | 2.539 |
| U1 | x | Uncorrected continuation | 0.401 | 4.885 |
| U1 | y | Uncorrected continuation | 0.388 | 4.723 |
| U2 | x | Uncorrected continuation | 0.104 | 1.781 |
| U2 | y | Uncorrected continuation | 0.119 | 1.906 |
| M1 | x | Uncorrected continuation | 4.119 | 11.365 |
| M1 | y | Uncorrected continuation | 3.458 | 10.929 |
| H1 | x | Uncorrected continuation | 1.061 | 2.469 |
| H1 | y | Uncorrected continuation | 1.223 | 1.734 |
| M2 | x | Uncorrected continuation | 0.593 | 2.874 |
| L1 | x | Uncorrected continuation | 0.174 | 1.894 |
| L1 | y | Uncorrected continuation | 0.182 | 1.646 |
| U1 | x | Smoothing-trained | 0.112 | 3.155 |
| U1 | y | Smoothing-trained | 0.111 | 3.166 |
| U2 | x | Smoothing-trained | 0.031 | 0.508 |
| U2 | y | Smoothing-trained | 0.038 | 0.782 |
| M1 | x | Smoothing-trained | 1.363 | 4.432 |
| M1 | y | Smoothing-trained | 1.115 | 3.974 |
| H1 | x | Smoothing-trained | 0.144 | 0.636 |
| H1 | y | Smoothing-trained | 0.237 | 0.561 |
| M2 | x | Smoothing-trained | 0.189 | 1.428 |
| M2 | y | Smoothing-trained | 0.213 | 2.545 |
| H3 | x | Smoothing-trained | 0.106 | 1.290 |
| H3 | y | Smoothing-trained | 0.132 | 0.678 |
| U1 | x | Base network + correction | 0.011 | 0.945 |
| U1 | y | Base network + correction | 0.011 | 0.904 |
| U2 | x | Base network + correction | 0.00123 | 0.142 |
| U2 | y | Base network + correction | 0.0014 | 0.158 |
| M1 | x | Base network + correction | 0.079 | 0.350 |
| M1 | y | Base network + correction | 0.066 | 0.339 |
| H1 | x | Base network + correction | 0.010 | 0.139 |
| H1 | y | Base network + correction | 0.013 | 0.105 |
| M2 | x | Base network + correction | 0.00492 | 0.083 |
| M2 | y | Base network + correction | 0.0034 | 0.075 |
| H3 | x | Base network + correction | 0.00609 | 0.132 |
| H3 | y | Base network + correction | 0.00894 | 0.202 |
| L1 | x | Base network + correction | 0.00193 | 0.065 |
| L1 | y | Base network + correction | 0.00181 | 0.087 |
| U1 | x | NICE | 0.00614 | 0.598 |
| U1 | y | NICE | 0.00681 | 0.669 |
| U2 | x | NICE | 0.00121 | 0.082 |
| U2 | y | NICE | 0.00159 | 0.073 |
| M1 | x | NICE | 0.056 | 0.161 |
| M1 | y | NICE | 0.048 | 0.206 |
| H1 | x | NICE | 0.0076 | 0.098 |
| H1 | y | NICE | 0.011 | 0.097 |
| M2 | x | NICE | 0.00595 | 0.136 |
| M2 | y | NICE | 0.00472 | 0.154 |
| H3 | x | NICE | 0.0064 | 0.124 |
| H3 | y | NICE | 0.010 | 0.196 |
| L1 | x | NICE | 0.00202 | 0.101 |
| L1 | y | NICE | 0.00197 | 0.078 |

Combinations without a row were not evaluated. All cells in this table belong to the 20 validation geometries used for checkpoint selection (Section 5.1). Table ST09 gives the responses of the same configurations under cut-surface tractions. For the base network on U1/x under the z traction on the neighbour face (circled in Figure S04(a–c)), the test cell carries 0.1274669% of the exact assembled energy and has a local energy error of 2.312168%. The product \(\beta=w\varepsilon\) is 0.002947249%, the compliance error 0.002827394% and the test-cell sensitivity error 5.82992%.

#### ST08b. Held-out two-cell configurations (NICE)

The cells lie outside checkpoint selection: the five validation geometries with NICE's largest single-cell errors and one randomly chosen evaluable cell per cut-severity group. Entries are maximum relative errors (%), over the six face loads for the compliance and over both cells for the thickness sensitivity. As in Table ST08, the neighbour is exact and the test cell learned. Cell labels are defined in Supplementary Section R1.

| Cell | Selection | Retained volume | x: compliance | x: sensitivity | y: compliance | y: sensitivity |
| --- | --- | --- | --- | --- | --- | --- |
| W1 | largest error, rank 1 | 0.571 | 0.2706 | 1.485 | 0.2220 | 1.312 |
| W2 | largest error, rank 2 | 0.267 | 0.1982 | 0.815 | 0.1703 | 1.344 |
| W3 | largest error, rank 3 | 0.287 | 0.1587 | 1.329 | 0.2489 | 1.419 |
| W4 | largest error, rank 4 | 0.266 | 0.0827 | 0.526 | 0.0753 | 0.971 |
| W5 | largest error, rank 5 | 0.681 | 0.0967 | 0.434 | 0.0878 | 0.563 |
| RU | random, uncut | 1.000 | 0.0010 | 0.181 | 0.0011 | 0.133 |
| RL | random, light | 0.779 | 0.0038 | 0.102 | 0.0051 | 0.102 |
| RM | random, moderate | 0.661 | 0.0101 | 0.103 | 0.0066 | 0.100 |
| RH | random, heavy | 0.078 | 0.0050 | 0.136 | 0.0066 | 0.218 |


## Table ST09. Responses under cut-surface tractions

Maximum relative errors (%) over the three cut-surface traction directions, for every variant and configuration of Table ST08 with a cut test cell. Combinations without a row were not evaluated. Sensitivity maxima include both cells. The test cell is learned and the neighbour exact. As in the comparison of cut-surface tractions in Section 5.6, values above the common 3% line are in bold.

| Variant | Cell | Configuration | Compliance error (%) | Sensitivity error (%) |
| --- | --- | --- | --- | --- |
| Base network | M1 | x | 2.880 | **9.924** |
| Base network | H1 | x | 0.445 | 2.066 |
| Base network | H1 | y | 0.651 | 1.275 |
| Base network | M2 | x | 0.560 | **3.155** |
| Uncorrected continuation | M1 | x | 2.768 | **9.827** |
| Uncorrected continuation | M1 | y | **6.870** | **14.053** |
| Uncorrected continuation | H1 | x | 0.399 | 1.825 |
| Uncorrected continuation | H1 | y | 0.613 | 1.200 |
| Uncorrected continuation | M2 | x | 0.574 | **3.252** |
| Uncorrected continuation | L1 | x | 0.355 | 1.660 |
| Uncorrected continuation | L1 | y | 1.141 | **3.702** |
| Smoothing-trained | M1 | x | 0.713 | **3.228** |
| Smoothing-trained | M1 | y | 2.417 | **5.191** |
| Smoothing-trained | H1 | x | 0.062 | 0.356 |
| Smoothing-trained | H1 | y | 0.130 | 0.587 |
| Smoothing-trained | M2 | x | 0.200 | 1.664 |
| Smoothing-trained | M2 | y | 1.117 | **3.924** |
| Smoothing-trained | H3 | x | 0.052 | 0.689 |
| Smoothing-trained | H3 | y | 0.237 | 1.354 |
| Base network + correction | M1 | x | 0.049 | 0.217 |
| Base network + correction | M1 | y | 0.124 | 0.305 |
| Base network + correction | H1 | x | 0.0044 | 0.096 |
| Base network + correction | H1 | y | 0.00709 | 0.022 |
| Base network + correction | M2 | x | 0.00468 | 0.053 |
| Base network + correction | M2 | y | 0.020 | 0.091 |
| Base network + correction | H3 | x | 0.00391 | 0.038 |
| Base network + correction | H3 | y | 0.011 | 0.042 |
| Base network + correction | L1 | x | 0.00479 | 0.039 |
| Base network + correction | L1 | y | 0.011 | 0.069 |
| NICE | M1 | x | 0.035 | 0.161 |
| NICE | M1 | y | 0.101 | 0.158 |
| NICE | H1 | x | 0.00338 | 0.061 |
| NICE | H1 | y | 0.00587 | 0.126 |
| NICE | M2 | x | 0.00603 | 0.072 |
| NICE | M2 | y | 0.027 | 0.154 |
| NICE | H3 | x | 0.00313 | 0.125 |
| NICE | H3 | y | 0.00982 | 0.129 |
| NICE | L1 | x | 0.00435 | 0.087 |
| NICE | L1 | y | 0.011 | 0.071 |

## Supplementary Note S1. Verification of the reference and of the thickness derivative

Figure S01 and Table ST10 verify the CutFEM reference under refinement of the background grid and under changes of the ghost-penalty coefficient. Table ST10b compares the reference with a body-fitted discretisation on uncut cells. The last paragraph of this note verifies the numerical thickness derivative.

For U1, M1 and M2, the compliance at \(n=32\) differs from the finest level of Figure S01(a) by at most 0.057% and the thickness sensitivity by at most 0.11%. H1 is clamped on \(x=0\) and loaded on \(y=0\), because its retained part has no material on its \(z\)-faces. At \(n=32\) it differs from \(n=48\) by up to 0.99% in compliance, for the load normal to the loaded face, and by 0.97% in sensitivity, but from \(n=64\) by only 0.07% and 0.08%. Between \(n=48\) and \(n=64\) the difference is 1.08%, so H1 does not converge monotonically (Figure S01(d), Table ST10). On U1, M1, M2 and H1, varying the ghost-penalty coefficient between \(10^{-5}\) and \(10^{-3}\) changes the compliance by at most 0.053% and the sensitivities by at most 0.12%. At \(\gamma=10^{-3}\), the penalty energy is \(1.2\times10^{-4}\) to \(5.6\times10^{-4}\) of the total. Refining the volume integration changes both quantities by at most 0.010%.

### Table ST10. Refinement of the reference to \(n=64\)

Single cells are clamped on one cell face and loaded on another face by unit traction loads in the three coordinate directions, and they are solved at \(n=24\) to 64 by a sparse direct solver. H2 is loaded instead by a body force with unit resultant, because its only cell face that carries material is the clamped one. Each entry gives the signed compliance deviation from \(n=64\) with the largest magnitude over the three loads, followed after the slash by the largest relative deviation of the thickness sensitivity vector, both in percent. H2 could not be meshed at \(n=56\). Elsewhere in this work, the reference solutions are computed at \(n=32\).

| Cell | \(n=24\) | \(n=32\) | \(n=40\) | \(n=48\) | \(n=56\) |
| --- | --- | --- | --- | --- | --- |
| H1 | -1.05 / 1.06 | +0.07 / 0.08 | +0.46 / 0.44 | +1.08 / 1.06 | +1.40 / 1.37 |
| H2 | -1.58 / 5.60 | -0.93 / 3.59 | -0.52 / 2.16 | -0.27 / 1.24 | — |
| Validation cell with the largest NICE error | -0.32 / 0.57 | -0.14 / 0.23 | +0.08 / 0.09 | -0.14 / 0.15 | -0.02 / 0.03 |
| Validation cell with the thinnest walls | -0.66 / 1.02 | -0.28 / 0.44 | -0.16 / 0.24 | -0.09 / 0.20 | +0.17 / 0.16 |

On uncut cells, the discrete model was also compared with an independent body-fitted discretisation (Table ST10b). This discretisation meshes the same implicit geometry with quadratic tetrahedra and shares no geometry, quadrature or stabilisation code with the discrete model.

### Table ST10b. Comparison with a body-fitted quadratic-tetrahedral discretisation on uncut cells

| Configuration | Loads | CutFEM vs finer body-fitted mesh: max. compliance difference (%) | Body-fitted discretisation: change between its two mesh levels (%) |
| --- | --- | ---: | ---: |
| Four single uncut cells | Six fixture load cases | 0.084–0.101 | 0.130–0.273 |
| One uncut cell | Twelve local tractions on finite areas | 0.253 | 0.502 |
| Assemblies of uncut cells: two cells, 2×2×2, 1×1×4 | Fixture load cases | 0.052–0.090 | 0.108–0.190 |

Figure S01(c) reports a step-refinement study of the numerical thickness derivative on four cells. Relative to the step used in the computations, \(10^{-5}\tau_c\), the sensitivity changes by at most \(2.6\times10^{-7}\) at \(10^{-3}\tau_c\), \(2.5\times10^{-9}\) at \(10^{-4}\tau_c\) and \(1.6\times10^{-10}\) at \(10^{-6}\tau_c\). The hundredfold reduction per decade is the second-order truncation error of the central difference. It is consistent with a derivative that is smooth within \(\pm10^{-3}\tau_c\) at the fixed active set on these cells. On the same cells, the central-difference stiffness derivative agrees to \(4\times10^{-8}\) with differences of the compliance recomputed at the perturbed designs. An element-level check over the partially filled elements of the eight detailed cells shows that the discrete moments do not everywhere preserve the nesting on which Eq. (H.7) relies. The element stiffness matrices are positive semidefinite to rounding. However, the exact derivative of the discrete moments at fixed clipping topology gives element derivative matrices with negative eigenvalues. In the uncut cells, the smallest ratio \(\lambda_{\min}/\max|\lambda|\) is \(-2.3\times10^{-3}\). In four of the six cut cells, between 2 and 11 partially filled elements (of 762 to 6,033) have a uniform-thickening derivative whose most negative eigenvalue exceeds \(10^{-6}\) of its largest eigenvalue in magnitude. These derivatives are indefinite, with a largest eigenvalue of at least \(10^{-3}\) of the largest magnitude. The central difference used in the computations agrees with this exact discrete derivative to \(2\times10^{-8}\).

## Supplementary Note S2. Ablation: Bernstein-restricted cell faces

This note gives the complete results of the ablation summarised in Section 5.7. Every cell uses its exact condensed stiffness matrix (the Schur complement), and only the representation of the retained cell-face displacements is reduced.

### S2.1. Restricted Galerkin system

The restricted space uses tensor-product Bernstein polynomials of degree \(r\) on the selected cell faces of each cell. Its global map \(G_r\) respects the DOFs shared between cells. DOFs of the cut-plane elements that do not lie on the cell faces keep identity columns. With the supports applied, the Galerkin system is

\[
\mathbb K_r=G_r^T\mathbb K G_r,\qquad
f_r=G_r^Tf_g,\qquad U_r=G_r\mathbb K_r^{-1}f_r.
\tag{S2.1}
\]

The comparison evaluates the exact condensed stiffness matrices of the cells within this restricted space of cell-face displacements. The interior is therefore exact, and every error comes from the restricted boundary. Two cases are examined: every cell face restricted, as in a lattice built entirely from such substructures, and only the shared interface restricted.

Boundary restriction and the interior correction \(\mathcal W\) act on different displacement spaces. Exact condensation followed by the restriction \(U=G_ry\) solves over a subspace of the space of retained displacements, and nested boundary spaces give a nondecreasing Ritz compliance. The correction keeps every retained DOF fixed and changes the interior displacements of \(u=FBU\). Its ordering holds because the two-grid cycle that defines \(H\) does not increase the energy error (Section 3.4), although two such interior spaces need not be nested. In neither case is a norm of the local sensitivity ordered.

### S2.2. Conditions

- Pairs in configuration x (Figure 9; Supplementary Note S4), each test cell with its exact neighbour of continuous thickness, for U1, M1, M2 and H1; the interface-only case was run for U1, M1 and M2.
- One cell per substructure, the fine-scale face traction loads used throughout, and no oversampling.
- Loads: the three tractions on the test-cell face and the three tractions on the neighbour face.
- The DOFs of the cut-plane elements that are not on the cell faces remain unrestricted, which favours the restricted model (controlled DOFs in Table ST11).
- The comparison concerns accuracy only and does not compare accuracy at equal cost.

### Table ST11. Bernstein restriction of the cell-face displacements

Each test cell and its neighbour of continuous thickness are assembled in configuration x with exact condensed stiffness matrices. Degree r applies to the restricted cell faces; the DOFs of the cut-plane elements keep their identity representation. Controlled DOFs include these unrestricted DOFs of the cut-plane elements. All errors are maxima over the stated load set (%), relative to the solution with all retained DOFs; the sensitivity columns refer to the test-cell vector. Test-face loads are the three unit traction loads on the loaded face of the test cell, and the six-load set adds the three tractions on the neighbour face. The maxima of the two load sets need not occur under the same load.

#### ST11a. Every cell face restricted

| Cell | r | Controlled DOFs | Test-face compliance (3 loads) | Test-face sensitivity (3 loads) | Six-load compliance (6 loads) | Six-load sensitivity (6 loads) |
| --- | --- | --- | --- | --- | --- | --- |
| U1 | 1 | 24 | 76.998 | 89.056 | 82.609 | 203.566 |
| U1 | 2 | 102 | 56.944 | 62.055 | 56.944 | 317.415 |
| U1 | 3 | 240 | 38.907 | 47.849 | 38.907 | 398.644 |
| U1 | 5 | 696 | 3.926 | 3.603 | 3.961 | 114.545 |
| U1 | 8 | 1,830 | 0.485 | 1.467 | 0.485 | 24.435 |
| M1 | 1 | 15,423 | 77.749 | 88.966 | 78.485 | 294.272 |
| M1 | 2 | 15,498 | 53.133 | 62.622 | 53.133 | 402.365 |
| M1 | 3 | 15,627 | 33.998 | 45.584 | 33.998 | 421.766 |
| M1 | 5 | 16,047 | 3.930 | 4.748 | 3.930 | 138.963 |
| M1 | 8 | 17,082 | 0.521 | 1.295 | 0.611 | 39.320 |
| M2 | 1 | 15,648 | 76.234 | 84.796 | 85.269 | 244.968 |
| M2 | 2 | 15,723 | 49.547 | 41.050 | 50.436 | 606.013 |
| M2 | 3 | 15,852 | 29.124 | 19.138 | 33.554 | 747.730 |
| M2 | 5 | 16,272 | 3.413 | 4.275 | 3.634 | 334.483 |
| M2 | 8 | 17,307 | 0.505 | 4.458 | 0.586 | 38.693 |
| H1 | 1 | 12,888 | 81.713 | 87.320 | 81.713 | 333.566 |
| H1 | 2 | 12,939 | 37.528 | 29.960 | 41.962 | 188.527 |
| H1 | 3 | 13,026 | 21.660 | 18.637 | 30.364 | 340.008 |
| H1 | 5 | 13,308 | 2.784 | 2.890 | 3.860 | 171.306 |
| H1 | 8 | 14,001 | 0.608 | 1.520 | 0.735 | 64.044 |

#### ST11b. Only the shared interface restricted

| Cell | r | Controlled DOFs | Test-face compliance (3 loads) | Test-face sensitivity (3 loads) | Six-load compliance (6 loads) | Six-load sensitivity (6 loads) |
| --- | --- | --- | --- | --- | --- | --- |
| U1 | 1 | 25,386 | 1.947 | 3.599 | 1.947 | 85.982 |
| U1 | 2 | 25,401 | 0.667 | 4.905 | 0.667 | 29.133 |
| U1 | 3 | 25,422 | 0.169 | 0.338 | 0.169 | 7.999 |
| U1 | 5 | 25,482 | 0.010 | 0.328 | 0.036 | 2.194 |
| U1 | 8 | 25,617 | 0.001 | 0.003 | 0.002 | 0.359 |
| M1 | 1 | 37,080 | 9.355 | 10.397 | 9.355 | 81.909 |
| M1 | 2 | 37,095 | 1.915 | 4.529 | 1.915 | 36.640 |
| M1 | 3 | 37,116 | 0.440 | 1.106 | 0.440 | 9.954 |
| M1 | 5 | 37,176 | 0.073 | 0.335 | 0.073 | 2.174 |
| M1 | 8 | 37,311 | 0.011 | 0.031 | 0.011 | 0.615 |
| M2 | 1 | 35,196 | 4.228 | 8.397 | 4.228 | 88.562 |
| M2 | 2 | 35,211 | 0.728 | 3.825 | 0.728 | 18.427 |
| M2 | 3 | 35,232 | 0.261 | 1.058 | 0.261 | 21.216 |
| M2 | 5 | 35,292 | 0.030 | 1.004 | 0.030 | 3.366 |
| M2 | 8 | 35,427 | 0.009 | 0.049 | 0.009 | 1.537 |

With all retained DOFs and the supports applied, the systems have 28,206 (U1), 39,996 (M1), 39,120 (M2) and 32,991 (H1) free DOFs.

