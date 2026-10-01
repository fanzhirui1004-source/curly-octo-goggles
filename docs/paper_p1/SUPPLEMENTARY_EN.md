# Supplementary material

Supplementary Notes, Tables and Figures are numbered in the order in which they appear in this supplement. Predictor labels follow the main text (Table 2); the key below also lists the archived run identifiers and the labels used before the revision. S8 and P0 are reported in this supplement only.

## R1. Predictor and geometry key

| Label | Former label | Numerical role | Archived run identifier |
| --- | --- | --- | --- |
| P0 (supplement only) | P0 | Earlier uncorrected predictor; separate training lineage (148 legacy geometries) | c_oh |
| Base network | B | Baseline predictor; starting weights of every continuation and of the fixed-weight corrections | v2L1 |
| Uncorrected | C | Continued weights, no correction in training or evaluation | A0_ctrl |
| S8 (supplement only) | S8 | Continued weights with eight smoothing steps in training on identity-view samples only (rotated training views bypassed the smoothing); evaluated with eight steps | A2_tail8 |
| Smoothing-trained | A2b | Continued weights trained through eight smoothing steps (all training views) | A2b_tail8 |
| NICE-post | B+W | Base network's weights evaluated with NICE's correction; no training through it | B2grid |
| NICE | A3 | Principal predictor, trained through the complete correction (8 / Q1(17) / 8) | A3_2grid |

The S8 predictor has its own learned weights. Applying eight smoothing steps to the base network in the fixed-weight study is a distinct comparison.

| Cell label | Geometry stratum | Archived geometry identifier |
| --- | --- | --- |
| U1 | Uncut | fresh_val_2000_full |
| U2 | Uncut | fresh_val_2001_full |
| M1 | Moderately cut | fresh_val_2003_d1_v1 |
| H1 | Heavily cut | fresh_val_2005_d1_v0 |
| M2 | Moderately cut | fresh_val_2006_d0_v1 |
| H2 | Heavily cut | fresh_val_2010_d0_v0 |
| H3 | Heavily cut | fresh_val_2002_d0_v0 |
| L1 | Lightly cut | fresh_val_2004_d0_v2 |

The x/y suffix identifies the neighbouring-cell configuration. The deployment geometries use the G1–G4 labels of Table ST18; the table below gives their archived identifiers.

| Benchmark label | Abbreviated geometry | Archived geometry identifier |
| --- | --- | --- |
| G1 | 0020-r2 | fresh_train_0020_cover01_r2 |
| G2 | 0020-r1 | fresh_train_0020_cover01_r1 |
| G3 | 0020-FULL | fresh_train_0020_full |
| G4 | 0007-FULL | fresh_train_0007_full |

## R2. Further diagnostic observations

The element-group diagnostic for M1 distributes sensitivity-error contributions across the cut material. Elements with volume fraction below 0.1 account for 15.5% of the absolute contributions, while the 0.5–0.999 group accounts for 49.5%. These groups are disjoint; overlapping retained-node and weak-support classifications are not additive groups. Figure S03 gives the corresponding element populations and contribution shares.

The role of the initial field is particularly clear in M1: after 32 smoothing steps, the dimensionless mean energy excess is 0.029452 from the learned field and 188.76 from a zero interior field, under identical retained displacements. These values are ratios, not percentages.

For the same cell, coarse correction followed by one eight-step smoothing stage gives 0.22991% mean energy excess with the trilinear space. Adding the pre-smoothing stage gives 0.18643%. With eight steps on each side, the quadratic coarse space gives 0.09001%, and linear partition-of-unity enrichment gives 0.036479%. The enriched representation retains 22,404 coefficient columns; this count is not a certified independent-space dimension (Appendix F.1). Table ST07 includes both loading classes and all recorded spaces.

For the base network in M1/x, the largest retained-displacement error over the six face loads is 14.198% in the exact Schur norm, and the largest energy excess at the exact retained displacement is 14.778%. These maxima accompany the full, field-only and solution-only sensitivity errors of Table ST11, which are separate replacements, not additive scalar contributions; like them, they are maxima over the six loads and need not occur under the same load.

Geometry counts refer to geometries with an available direction-class mean. The aggregate records do not retain the realised direction count for every geometry and class; reported percentiles summarise their respective recorded direction banks.

## Table ST01. Training and evaluation settings

| Arm | Training pool (geometries) | Training-time validation geometries | Run budget (updates) | Evaluated update / weights | Evaluation correction | New-validation views | New-validation geometries |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | 148 | 20 | 15,000 | 15,000 / EMA | None | 0, 17 | 80 |
| Base network | 305 | 40 | 40,000 | 30,000 / EMA | None | 0, 17 | 80 |
| Uncorrected | 591 | 40 | 15,000 | 15,000 / EMA | None | 0 | 80 |
| S8 | 591 | 40 | 15,000 | 15,000 / EMA | Eight-step smoothing | 0 | 80 |
| Smoothing-trained | 591 | 40 | 15,000 | 15,000 / EMA | Eight-step smoothing | 0 | 80 |
| NICE-post | 305 | 40 | — | 30,000 / EMA | 8 / Q1(17) / 8 | 0 | 80 |
| NICE | 591 | 40 | 15,000 | 15,000 / EMA | 8 / Q1(17) / 8 | 0, 17 | 80 |

The training pool is the number of training geometries recorded at the start of each run (the SPLIT event of its training log): 148 for P0, 305 for the base network, and 591 for Uncorrected, Smoothing-trained and NICE. The S8 run used the same split file as the other continuations; its SPLIT event was not extracted. The pool of 591 comprises 304 of the base network's 305 geometries (one cell that behaved as a near-mechanism was removed) and 287 produced later. With three geometries in the device pool and one replacement every 100 updates (Appendix G.3), a run of \(N\) updates visits at most \(N/100+3\) distinct geometries, i.e. at most 153 for each 15,000-update continuation; the base network visited all 305 geometries of its pool over its 40,000 updates. Uncorrected, S8, Smoothing-trained and NICE use the same seed and split, so they draw their geometries in the same order.

P0 uses its final (15,000-update) weights; the other rows use the selected weights. Uncorrected, S8, Smoothing-trained and NICE are separate continuations initialised from the selected weights of the base network; NICE-post evaluates those same weights with NICE's correction and has no training run of its own. S8 is evaluated with its own weights and an eight-step smoothing tail. The fixed-weight correction experiments use the base network. The new-validation set comprises 20 uncut cells and 20, 20 and 20 cells in the light-, middle- and heavy-cut strata. In the identity view, the five basic direction classes, force_c and face_c cover all 80 geometries, and support_k and glued cover 75. The view-17 comparison in Table ST03c comes from an earlier evaluation of the same geometries, in which force_c, face_c, support_k and glued cover 20, 20, 19 and 15 geometries.

For the base network, Uncorrected, S8, Smoothing-trained and NICE, checkpoint selection uses the bias-corrected EMA weights and both validation views 0 and 17, even where the new-validation table reports only view 0. Within a geometry family and view, let \(E_{fv}\) be the mean energy excess averaged over the selection classes, \(S_{fv}\) the mean relative sensitivity-vector error over classes with labels, and \(P_{fv}\) the class-average 90th percentile of directional energy excess. Each class statistic is first averaged over the available geometries in that family. The selection score is

\[
J_{\rm sel}=\frac12\sum_{v\in\{0,17\}}\frac1{|\mathcal F|}
\sum_{f\in\mathcal F}\left(E_{fv}+S_{fv}+\tfrac12P_{fv}\right).
\]

Families and the two views carry equal weight. The eight selection classes are `force`, `support`, `face`, `macro`, `grf`, `force_c`, `face_c` and `support_k`; absent classes are omitted and an absent sensitivity term contributes zero. The percentile term averages within-geometry percentiles rather than pooling all directions. No additional sensitivity-percentile term is used. Among eligible evaluations, the lowest finite score is selected.

- The base network was scored at 10,000, 20,000, 30,000 and 40,000 updates (\(J_{\rm sel}\) = 0.1057, 0.1065, 0.0905 and 0.0912); the weights of update 30,000 were selected. Every continued predictor starts from these weights.
- Uncorrected, S8 and NICE were scored at 7,500 and 15,000 updates (0.0944 and 0.0906; 0.0582 and 0.0553; 0.00281 and 0.00237); in each case the final checkpoint scored lower and was retained.
- Smoothing-trained was run with the same evaluation schedule and views (its run configuration); its selected weights are those of update 15,000. Its per-checkpoint scores are not in the archived records.

This selection criterion differs from the per-batch training loss in Eq. (17) and from the geometry-weighted statistics of the 80-geometry validation set. Twenty of those 80 geometries belong to the selection list (Section 5.1).

## Table ST02. Discrete operator and diagnostic definitions

The same retained coordinate convention is used for the learned extension, the variational readout, and assembly. Relative errors are dimensionless and are displayed as percentages unless indicated otherwise.

| Item | Definition or setting |
| --- | --- |
| Geometry | Unit-box P-type thin-wall cells with corner thickness parameters; FULL cells and plane-cut cells in retained-volume strata v0, v1, v2. |
| Elastic discretisation | Isotropic small-strain elasticity on active tensor-product Q2 hexahedra; body stiffness plus the prescribed ghost-penalty contribution. |
| Retained coordinates | All active box-face coefficients and the cut-band retained coefficients; node-major Cartesian displacement order. |
| Internal reference | \(A=K_{II}\); exact interior extension with the retained values prescribed. |
| Background coordinate convention | 32 background elements and 65 Q2 node positions per axis. |
| Standard moment evaluator | \(4^3\) initial subcells per active element, one local refinement of partial subcells, and clipped Kuhn tetrahedra with rule parameter 4. |
| Material and stabilisation parameters | For all validation geometries: \(E_Y=1,\nu=0.3,\gamma=0.0001\). |
| Learned readout | \(\widehat S=F^TKF\), with \(F=\widehat E\) for the uncorrected network. |
| Energy excess | \(\varepsilon(q)=q^T(\widehat S-S)q/(q^TSq)\). Geometry means average directions first; population means weight geometries equally. |
| Compliance error | \(e_C=\lvert\widehat C/C-1\rvert\), with \(C=f_g^TU\). |
| Sensitivity error | \(e_s=\lVert\widetilde{\boldsymbol s}-\boldsymbol s\rVert_2/\lVert\boldsymbol s\rVert_2\); the field-based estimate has one component for each corner thickness parameter. |
| Spectrum | \(Av_j=\lambda_j Dv_j\), \(D=\operatorname{diag}(A)\); cumulative fractions of the internal error or exact internal-field energy, with separate denominators. |
| Fixed correction | Retained values fixed; Chebyshev relaxation and an interior Galerkin coarse correction. Two-sided sequences use k steps before and k steps after the coarse correction. |
| Assembly diagnostic | Learned target cell joined to its exact continuous-thickness neighbour; x and y denote the adjacent-cell configuration. |
| Six-load | Maximum compliance and sensitivity errors over the six face loads are each at most 3%; cut-traction loads are tabulated separately. |
| Iterative residuals | Recursive PCG residual and \(\max_j\lVert f_j-\mathbb K_{\rm run}\widehat U_j\rVert_2/\lVert f_j\rVert_2\), recomputed with the same operator used in that run. |

### Validation geometry domain

The 80 geometries use independently generated thickness fields, with 20 uniform, 30 affine and 30 mixed trilinear fields. The generator constrains every corner parameter to \([0.17520160,0.69933962]\), the corner span to at most 0.47, and the maximum reference-coordinate gradient norm to at most 0.47. In blocks of eight, a uniform-field-equivalent centre volume fraction is stratified between 0.1 and 0.4; nonuniform affine or trilinear shapes are scaled within these constraints. This centre-density parameter is a sampling coordinate, not the material fraction after cutting.

Canonical cut normals are \((\cos\vartheta,\sin\vartheta,0)\). The generator stratifies \(\vartheta\) over the two halves of \((0,\pi/4)\) and the retained macro-box volume \(v_{\mathcal B}\) over thirds of \((0,1)\). Two of every eight validation fields are uncut and the other six occupy the angle–volume strata. Training and validation are drawn from separate random streams; the present table describes the 80-geometry validation set.

| Stratum | Geometries | Generation interval for \(v_{\mathcal B}\) | Observed \(v_{\mathcal B}\) | Observed corner-parameter range |
| --- | ---: | --- | --- | --- |
| Uncut | 20 | 1 | 1 | 0.1762–0.6902 |
| Light cut | 20 | \((2/3,1)\) | 0.6765–0.9996 | 0.1853–0.6972 |
| Moderate cut | 20 | \((1/3,2/3)\) | 0.3437–0.6605 | 0.1867–0.6983 |
| Heavy cut | 20 | \((0,1/3)\) | 0.01326–0.3195 | 0.1768–0.6946 |

The cut volumes refer to the box before intersecting it with the TPMS band. Geometry identifiers and cube-orbit identifiers are unique within this validation set. The selected spectral and assembly cases are identified in the benchmark key; they are reported as diagnostic cases rather than a random sample for population inference.

## Table ST03. Identity-view energy excess by direction class

Entries are geometry-equal mean / 90th percentile / maximum of geometry-level direction means, in percent. The maximum is not a worst individual direction. An em dash denotes a class absent from that model's result. NICE-post denotes the base network's weights evaluated with NICE's correction. For NICE under consistent tractions, the 5,120 individual sampled directions of the 80 geometries have a 95th percentile of 0.331%, a 99th percentile of 0.693% and a maximum of 1.24% (3,840 directions of the 60 geometries outside weight selection: 0.355%, 0.734%, 1.24%).

| Class | Geometries per evaluated arm | P0 | Base network | Uncorrected | S8 | Smoothing-trained | NICE-post | NICE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| force | 80 | 7.387 / 19.508 / 109.813 | 5.036 / 12.038 / 70.122 | 4.607 / 10.750 / 62.465 | 0.672 / 1.887 / 4.398 | 0.525 / 1.301 / 3.658 | 0.092 / 0.180 / 0.683 | 0.058 / 0.117 / 0.325 |
| support | 80 | 8.078 / 13.959 / 153.172 | 5.332 / 9.811 / 99.857 | 4.878 / 9.493 / 90.741 | 0.761 / 1.864 / 3.408 | 0.638 / 1.601 / 3.552 | 0.084 / 0.168 / 0.852 | 0.055 / 0.132 / 0.372 |
| face | 80 | 3.203 / 6.296 / 33.992 | 2.486 / 5.209 / 22.139 | 2.233 / 4.045 / 17.635 | 0.218 / 0.435 / 2.638 | 0.165 / 0.341 / 1.808 | 0.072 / 0.112 / 1.990 | 0.037 / 0.056 / 0.724 |
| macro | 80 | 0.938 / 1.533 / 2.669 | 0.842 / 1.448 / 2.636 | 0.827 / 1.446 / 2.584 | 0.255 / 0.487 / 1.148 | 0.203 / 0.382 / 0.935 | 0.022 / 0.059 / 0.165 | 0.015 / 0.034 / 0.118 |
| grf | 80 | 2.125 / 3.191 / 4.747 | 2.038 / 3.104 / 4.582 | 2.009 / 3.075 / 4.550 | 0.311 / 0.709 / 1.172 | 0.251 / 0.552 / 0.933 | 0.029 / 0.072 / 0.180 | 0.025 / 0.060 / 0.143 |
| force_c | 80 | — | 6.886 / 17.331 / 48.628 | 6.334 / 16.142 / 41.962 | 1.513 / 3.982 / 9.161 | 1.280 / 3.453 / 7.817 | 0.097 / 0.239 / 0.905 | 0.074 / 0.213 / 0.651 |
| face_c | 80 | — | 3.150 / 6.726 / 36.337 | 2.806 / 6.290 / 25.525 | 0.569 / 1.277 / 3.059 | 0.483 / 1.140 / 2.326 | 0.043 / 0.102 / 0.319 | 0.032 / 0.078 / 0.210 |
| support_k | 75 | — | 4.177 / 9.695 / 28.817 | 3.941 / 8.831 / 26.175 | 1.135 / 3.022 / 6.714 | 0.938 / 2.390 / 5.246 | 0.077 / 0.193 / 0.632 | 0.057 / 0.169 / 0.468 |
| glued | 75 | — | 4.593 / 11.227 / 38.128 | 4.422 / 10.378 / 42.030 | 1.160 / 3.373 / 5.852 | 0.953 / 2.696 / 4.629 | 0.078 / 0.217 / 0.555 | 0.060 / 0.172 / 0.384 |

### ST03b. Force/support geometry-stratum means (%)

| Stratum | Geometries | P0 force/support | Base network force/support | Uncorrected force/support | S8 force/support | Smoothing-trained force/support | NICE-post force/support | NICE force/support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FULL | 20 | 1.052 / 1.815 | 0.908 / 1.413 | 0.879 / 1.332 | 0.110 / 0.278 | 0.072 / 0.225 | 0.034 / 0.044 | 0.013 / 0.022 |
| Light cut (v2) | 20 | 4.593 / 4.142 | 3.239 / 3.049 | 3.021 / 2.832 | 0.665 / 0.777 | 0.523 / 0.661 | 0.088 / 0.072 | 0.059 / 0.050 |
| Middle cut (v1) | 20 | 6.166 / 5.265 | 4.720 / 3.930 | 4.506 / 3.810 | 0.912 / 0.856 | 0.723 / 0.698 | 0.103 / 0.075 | 0.070 / 0.050 |
| Heavy cut (v0) | 20 | 17.738 / 21.089 | 11.278 / 12.938 | 10.021 / 11.540 | 0.999 / 1.133 | 0.783 / 0.966 | 0.141 / 0.145 | 0.090 / 0.098 |

### ST03c. View dependence: identity / view 17 means (%)

This comparison uses an earlier evaluation in both views, in which the four enriched classes cover the numbers of geometries given in the second column; their identity-view means therefore differ from those of Table ST03. Of the continued predictors, only NICE was also evaluated in view 17, on all 80 geometries with the validation directions of Table ST03 (identity / view 17 means, %): force 0.058 / 0.059, support 0.055 / 0.056, face 0.037 / 0.034, macro 0.015 / 0.016, grf 0.025 / 0.026, force_c 0.074 / 0.083, face_c 0.032 / 0.034.

| Class | Geometries | P0 | Base network |
| --- | --- | --- | --- |
| force | 80 | 7.387 / 8.107 | 5.036 / 5.271 |
| support | 80 | 8.078 / 8.678 | 5.332 / 5.598 |
| face | 80 | 3.203 / 3.223 | 2.486 / 2.417 |
| macro | 80 | 0.938 / 1.011 | 0.842 / 0.890 |
| grf | 80 | 2.125 / 2.240 | 2.038 / 2.119 |
| force_c | 20 | — | 6.624 / 7.493 |
| face_c | 20 | — | 3.838 / 4.249 |
| support_k | 19 | — | 3.350 / 3.760 |
| glued | 15 | — | 6.647 / 7.207 |

### ST03d. Geometries outside weight selection

Table ST03 restricted to the 60 validation geometries that entered neither training nor checkpoint selection (the other 20 entered checkpoint selection, Table ST01). Same statistics as Table ST03.

| Class | Geometries per evaluated arm | P0 | Base network | Uncorrected | S8 | Smoothing-trained | NICE-post | NICE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| force | 60 | 7.777 / 19.809 / 109.813 | 5.298 / 12.657 / 70.122 | 4.852 / 12.232 / 62.465 | 0.684 / 1.887 / 4.398 | 0.540 / 1.382 / 3.658 | 0.093 / 0.168 / 0.683 | 0.059 / 0.117 / 0.325 |
| support | 60 | 9.143 / 20.581 / 153.172 | 5.939 / 15.397 / 99.857 | 5.346 / 12.008 / 90.741 | 0.765 / 1.844 / 3.408 | 0.643 / 1.601 / 3.552 | 0.088 / 0.168 / 0.852 | 0.056 / 0.112 / 0.372 |
| face | 60 | 2.910 / 4.802 / 29.496 | 2.287 / 3.844 / 21.945 | 2.045 / 3.996 / 15.673 | 0.218 / 0.483 / 2.638 | 0.161 / 0.403 / 1.808 | 0.082 / 0.116 / 1.990 | 0.040 / 0.057 / 0.724 |
| macro | 60 | 0.939 / 1.533 / 2.669 | 0.846 / 1.448 / 2.636 | 0.831 / 1.446 / 2.584 | 0.255 / 0.477 / 1.148 | 0.203 / 0.382 / 0.935 | 0.022 / 0.061 / 0.165 | 0.015 / 0.034 / 0.118 |
| grf | 60 | 2.132 / 3.191 / 4.747 | 2.047 / 3.104 / 4.582 | 2.018 / 3.075 / 4.550 | 0.311 / 0.714 / 1.172 | 0.252 / 0.559 / 0.933 | 0.030 / 0.094 / 0.180 | 0.025 / 0.068 / 0.143 |
| force_c | 60 | — | 6.973 / 18.411 / 48.628 | 6.517 / 18.407 / 41.962 | 1.542 / 4.045 / 9.161 | 1.308 / 3.553 / 7.817 | 0.101 / 0.239 / 0.905 | 0.077 / 0.213 / 0.651 |
| face_c | 60 | — | 2.920 / 6.761 / 15.615 | 2.688 / 6.645 / 13.653 | 0.589 / 1.403 / 3.059 | 0.499 / 1.226 / 2.326 | 0.045 / 0.114 / 0.319 | 0.033 / 0.078 / 0.210 |
| support_k | 56 | — | 4.458 / 13.372 / 28.817 | 4.200 / 13.014 / 26.175 | 1.160 / 3.121 / 6.714 | 0.958 / 2.680 / 5.246 | 0.081 / 0.192 / 0.632 | 0.058 / 0.163 / 0.468 |
| glued | 60 | — | 4.080 / 10.717 / 25.558 | 3.850 / 10.552 / 22.814 | 1.081 / 2.971 / 5.852 | 0.886 / 2.569 / 4.629 | 0.074 / 0.196 / 0.555 | 0.055 / 0.166 / 0.384 |

## Table ST04. Operator verification in the deployed arithmetic

Columns: \(\lambda_{\max}\) of \(D^{-1}K_{II}\) by Lanczos; the power-iteration estimate \(b\) used by the smoothing; the Gershgorin bound; the maximum relative asymmetry of \(Q^T\widehat SQ\); the maximum relative difference between returned work and field energy; the deployed-versus-training field difference; the maximum rigid-body energy ratio; the ghost-penalty share of the exact field energy (consistent tractions / nodal forces); and the mean \(\delta\) and \(\kappa\) of the base network and of NICE under consistent tractions. Here \(\delta^2=d_I^TDd_I/u_I^TDu_I\) and \(\kappa=(d^TKd/d_I^TDd_I)/(u^TKu/u_I^TDu_I)\) with \(D=\operatorname{diag}(K_{II})\), i.e. the weighting \(W=\operatorname{diag}(0,D)\) of Appendix B.1. The \(b\) values of H2, H1, M2 and M1 come from a host evaluation, whose seeded generator gives a different power-iteration start from the GPU runs; the endpoint used by all GPU runs is the one verified on all 80 validation geometries in Appendix D. U2 is a GPU record. Data: `evidence/p1_checks_cpu.json` (H2, H1, M2, M1) and `evidence/p1_checks_u2.json` (U2); see Supplementary Note S2.

| cell | lambda_max (Lanczos) | b = 1.05 x power | margin | Gershgorin | sym | action-energy | deployed vs training | rigid energy | ghost share force_c / force | δ, κ base network (force_c) | δ, κ NICE (force_c) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H2 | 3.992 | 4.152 | 4.0% | 43.9 | 6.3e-09 | 4.5e-09 | 4.6e-09 | 1.7e-11 | 0.00048 / 0.49 | 0.71%, 7089 | 0.051%, 582 |
| H1 | 4.994 | 5.191 | 3.9% | 78.2 | 5.6e-09 | 4.0e-09 | 4.2e-08 | 1.6e-12 | 7.9e-05 / 0.81 | 1.23%, 124 | 0.186%, 33 |
| M2 | 5.027 | 5.228 | 4.0% | 100 | 8.7e-09 | 3.4e-09 | 9.4e-08 | 2.4e-12 | 3.6e-05 / 0.66 | 1.15%, 294 | 0.156%, 163 |
| M1 | 4.979 | 5.193 | 4.3% | 99.5 | 7.3e-09 | 4.4e-09 | 9.3e-08 | 1.4e-11 | 0.00012 / 0.58 | 1.26%, 866 | 0.179%, 426 |
| U2 | 4.897 | 5.106 | 4.3% | 71.7 | 3.5e-09 | 1.8e-09 | 1.1e-07 | 2.9e-13 | 3.6e-05 / 0.79 | 0.74%, 85 | 0.107%, 47 |

## Table ST05. Same-trace sensitivity and spectral diagnostics

### ST05a. Sensitivity estimates

Energy and sensitivity errors are directional means (%). The first-order share is the Frobenius norm of the linear error array divided by the sum of the Frobenius norms of the linear and quadratic arrays (%); each array includes all eight corners and all evaluated directions in that cell and class.

| Arm | Cell | Class | Mean energy excess | Mean sensitivity error | 90th-percentile sensitivity error | First-order share |
| --- | --- | --- | --- | --- | --- | --- |
| Base network | U1 | force_c | 1.270 | 0.664 | 0.778 | 42.887 |
| Base network | U1 | force | 0.742 | 1.018 | 1.449 | 55.124 |
| Base network | U2 | force_c | 0.402 | 0.588 | 0.693 | 35.658 |
| Base network | U2 | force | 0.506 | 1.008 | 1.772 | 71.565 |
| Base network | M1 | force_c | 13.512 | 14.128 | 19.184 | 7.164 |
| Base network | M1 | force | 7.156 | 4.822 | 10.493 | 23.516 |
| Base network | H1 | force_c | 1.814 | 1.909 | 3.221 | 28.403 |
| Base network | H1 | force | 1.842 | 0.923 | 1.565 | 35.199 |
| Base network | M2 | force_c | 3.643 | 5.159 | 7.351 | 9.339 |
| Base network | M2 | force | 2.175 | 1.103 | 2.019 | 41.390 |
| Base network | H2 | force_c | 34.954 | 75.072 | 156.803 | 0.786 |
| Base network | H2 | force | 19.507 | 13.649 | 38.851 | 36.322 |
| Uncorrected | U1 | force_c | 1.165 | 0.937 | 1.138 | 35.750 |
| Uncorrected | U1 | force | 0.779 | 0.766 | 0.981 | 59.185 |
| Uncorrected | U2 | force_c | 0.397 | 0.560 | 0.658 | 35.988 |
| Uncorrected | U2 | force | 0.515 | 1.062 | 1.773 | 72.929 |
| Uncorrected | M1 | force_c | 13.009 | 13.275 | 18.119 | 7.456 |
| Uncorrected | M1 | force | 6.841 | 4.442 | 9.663 | 26.413 |
| Uncorrected | H1 | force_c | 1.699 | 1.894 | 3.223 | 28.046 |
| Uncorrected | H1 | force | 1.877 | 1.013 | 1.630 | 34.751 |
| Uncorrected | M2 | force_c | 3.988 | 5.432 | 7.485 | 9.548 |
| Uncorrected | M2 | force | 2.105 | 1.357 | 2.192 | 49.454 |

### ST05b. Energy fractions in the lowest 200 generalised interior modes

| Cell | Class | Error fraction: mean / p10 / p90 (%) | Exact-field fraction: mean / p10 / p90 (%) |
| --- | --- | --- | --- |
| U1 | force_c | 24.019 / 20.530 / 27.123 | 8.498 / 8.230 / 8.755 |
| U1 | force | 5.043 / 3.486 / 7.430 | 7.977 / 6.906 / 8.860 |
| M1 | force_c | 24.452 / 20.027 / 28.532 | 4.870 / 4.574 / 5.161 |
| M1 | force | 22.378 / 17.917 / 26.173 | 6.644 / 4.770 / 9.198 |
| M2 | force_c | 18.399 / 14.496 / 22.067 | 4.620 / 3.887 / 5.957 |
| M2 | force | 11.411 / 7.955 / 16.414 | 6.578 / 4.832 / 8.138 |
| H2 | force_c | 15.216 / 12.681 / 19.023 | 4.705 / 1.983 / 6.199 |
| H2 | force | 21.837 / 13.953 / 34.960 | 4.877 / 2.383 / 6.832 |

The two fractions use their respective internal-energy denominators. Their cumulative curves are shown in Figure 6. The archived eigenvalues can also be compared with the interval used in the separate fixed-weight smoothing diagnostic of the base network:

| Cell | \(\lambda_1\) | \(\lambda_{200}\) | Diagnostic lower endpoint \(a=b/30\) | Recorded modes below \(a\), out of 200 |
| --- | --- | --- | --- | --- |
| U1 | 0.000402235 | 0.00979835 | 0.174937 | 200 |
| M1 | 0.000574199 | 0.0112575 | 0.173294 | 200 |
| M2 | 0.000935594 | 0.0117352 | 0.174182 | 200 |
| H2 | 0.0213148 | 0.416328 | 0.137632 | 66 |

The spectral and smoothing diagnostics use the same cell identifiers, predictor and specified discrete construction, with matching internal-coordinate counts. The intervals come from the separate smoothing runs: the recorded upper endpoint includes the 1.05 safety factor, and \(a=b/30\). These estimates differ from those of the correction wrapper and are not certified spectral bounds. Individual directional attenuation and stiffness-content hashes were not recorded, so the comparison supports a cell-level spectral interpretation.

## Table ST06. Complete recorded smoothing cases

The network rows use the base network. The retained trace is identical for network and zero interior initialisations. Entries are mean [90th percentile] directional errors (%). A zero interior start sets only the internal displacement to zero. The same five cells and both direction classes are included.

### ST06a. Energy excess

| Cell | Class | Network, k=0 | Network, k=8 | Network, k=32 | Zero interior, k=32 |
| --- | --- | --- | --- | --- | --- |
| U1 | force_c | 1.2704 [1.4799] | 0.41971 [0.51235] | 0.27998 [0.35499] | 2680.9 [3717.6] |
| U1 | force | 0.74211 [0.78789] | 0.1063 [0.12755] | 0.049733 [0.066014] | 409.72 [564.75] |
| M1 | force_c | 13.512 [19.634] | 4.685 [7.5373] | 2.9452 [4.74] | 18876 [26535] |
| M1 | force | 7.1563 [10.711] | 2.3037 [3.4374] | 1.4523 [2.1642] | 10417 [14953] |
| H1 | force_c | 1.814 [2.5294] | 0.25081 [0.33593] | 0.059403 [0.08861] | 186.98 [316.28] |
| H1 | force | 1.842 [3.3328] | 0.19516 [0.42805] | 0.047822 [0.11395] | 386.19 [629.7] |
| M2 | force_c | 3.6428 [5.158] | 1.0382 [1.49] | 0.61088 [0.97633] | 5434.1 [8866.3] |
| M2 | force | 2.1747 [3.2578] | 0.46478 [0.71357] | 0.2508 [0.39958] | 2279.1 [3703.7] |
| H2 | force_c | 34.954 [56.196] | 0.20515 [0.32813] | 0.0033313 [0.006318] | 0.42123 [0.57258] |
| H2 | force | 19.507 [36.537] | 0.22768 [0.34841] | 0.0074193 [0.011726] | 0.20928 [0.34236] |

### ST06b. Field-based sensitivity error

| Cell | Class | Network, k=0 | Network, k=8 | Network, k=32 | Zero interior, k=32 |
| --- | --- | --- | --- | --- | --- |
| U1 | force_c | 0.6639 [0.77847] | 0.52592 [0.69671] | 0.45509 [0.63825] | 1671.3 [2121.5] |
| U1 | force | 1.0177 [1.4489] | 0.23011 [0.35671] | 0.14741 [0.19763] | 93.161 [142.59] |
| M1 | force_c | 14.128 [19.184] | 2.7552 [4.1122] | 1.6339 [2.4573] | 10302 [14185] |
| M1 | force | 4.8221 [10.493] | 0.78839 [1.5521] | 0.5492 [0.84882] | 3203.8 [5893.7] |
| H1 | force_c | 1.9092 [3.2207] | 0.26139 [0.48401] | 0.14959 [0.30252] | 163.15 [223.26] |
| H1 | force | 0.9233 [1.5645] | 0.14313 [0.24897] | 0.051681 [0.082265] | 114.31 [197.95] |
| M2 | force_c | 5.1589 [7.3505] | 0.83675 [1.2052] | 0.5753 [0.78197] | 2952.9 [4584.9] |
| M2 | force | 1.1026 [2.0194] | 0.15273 [0.26638] | 0.08977 [0.16216] | 303.46 [686.58] |
| H2 | force_c | 75.072 [156.8] | 0.77195 [1.4896] | 0.010708 [0.023337] | 1.9551 [2.8917] |
| H2 | force | 13.649 [38.851] | 0.75099 [1.4264] | 0.073808 [0.14745] | 0.69083 [1.2398] |

## Table ST07. Interior coarse-space and smoothing comparisons

### ST07a. Eight steps per smoothing stage

Entries are mean [90th percentile] energy excess (%). C denotes one coarse correction and T one eight-step smoothing stage. All network rows use the base network's selected weights; the zero interior reference uses the same retained values. The size column counts coarse coefficient columns remaining after the support and diagonal-energy screens. For the PU rows, structural dependencies in the generating functions and the absence of an archived rank/solve-accuracy check prevent interpreting this count as an independent-space dimension; see Appendix F.1.

| Cell | Class | Coarse space | Surviving coarse coefficient columns | Net + C | Net + C + T | Net + T + C + T | Zero + T + C + T |
| --- | --- | --- | --- | --- | --- | --- | --- |
| U1 | force_c | Q1_9 | 1,728 | 0.87589 [1.0044] | 0.13597 [0.15856] | 0.096621 [0.11469] | 582.87 [746.75] |
| U1 | force_c | PU_9 | 6,912 | 0.53246 [0.60943] | 0.02491 [0.027969] | 0.015406 [0.017553] | 93.269 [108.72] |
| U1 | force_c | Q1_17 | 7,950 | 0.5889 [0.67776] | 0.037909 [0.043611] | 0.02671 [0.031486] | 133.13 [158.19] |
| U1 | force_c | Q2_17 | 10,452 | 0.45492 [0.52339] | 0.018783 [0.021303] | 0.012546 [0.014527] | 89.035 [103.75] |
| U1 | force_c | PU_17 | 31,800 | 0.27807 [0.32057] | 0.0074368 [0.0088953] | 0.0065266 [0.0079824] | 12.104 [15.996] |
| U1 | force_c | Q1_33 | 40,983 | 0.29109 [0.33079] | 0.0090665 [0.010395] | 0.0078425 [0.0093791] | 21.384 [27.345] |
| U1 | force | Q1_9 | 1,728 | 0.67304 [0.70534] | 0.066091 [0.072088] | 0.038737 [0.043165] | 89.796 [114.12] |
| U1 | force | PU_9 | 6,912 | 0.60974 [0.63812] | 0.050323 [0.053281] | 0.027803 [0.03114] | 17.806 [20.626] |
| U1 | force | Q1_17 | 7,950 | 0.60874 [0.63931] | 0.050277 [0.052953] | 0.027829 [0.031141] | 22.456 [27.314] |
| U1 | force | Q2_17 | 10,452 | 0.58831 [0.61599] | 0.048835 [0.051937] | 0.027186 [0.030536] | 17.159 [19.758] |
| U1 | force | PU_17 | 31,800 | 0.50662 [0.54204] | 0.039204 [0.043384] | 0.021406 [0.025149] | 4.4879 [5.6473] |
| U1 | force | Q1_33 | 40,983 | 0.45524 [0.50058] | 0.031736 [0.035242] | 0.014305 [0.016225] | 4.2175 [5.3685] |
| M1 | force_c | Q1_9 | 1,275 | 7.439 [10.515] | 1.0016 [1.4953] | 0.8125 [1.2317] | 4530.7 [6359.6] |
| M1 | force_c | PU_9 | 5,100 | 4.2489 [5.8755] | 0.16633 [0.22693] | 0.11402 [0.16383] | 729.23 [1070.1] |
| M1 | force_c | Q1_17 | 5,601 | 4.6803 [6.5368] | 0.22991 [0.33235] | 0.18643 [0.27983] | 1097.2 [1600.1] |
| M1 | force_c | Q2_17 | 7,401 | 3.5958 [4.9894] | 0.12522 [0.17305] | 0.09001 [0.12817] | 631.41 [927.81] |
| M1 | force_c | PU_17 | 22,404 | 2.0056 [2.7628] | 0.04841 [0.069908] | 0.036479 [0.051356] | 163.6 [238.41] |
| M1 | force_c | Q1_33 | 27,207 | 1.9826 [2.739] | 0.051622 [0.071595] | 0.047665 [0.069164] | 254.92 [370.44] |
| M1 | force | Q1_9 | 1,275 | 4.17 [6.2314] | 0.52699 [0.81966] | 0.41642 [0.6526] | 2344.7 [3181.6] |
| M1 | force | PU_9 | 5,100 | 2.5741 [3.7062] | 0.12599 [0.1791] | 0.078984 [0.1172] | 384.02 [570.53] |
| M1 | force | Q1_17 | 5,601 | 2.8067 [4.0798] | 0.15789 [0.23819] | 0.11392 [0.1776] | 558.09 [807.53] |
| M1 | force | Q2_17 | 7,401 | 2.2196 [3.1669] | 0.10132 [0.14109] | 0.06647 [0.097108] | 341.7 [492.96] |
| M1 | force | PU_17 | 22,404 | 1.3478 [1.8254] | 0.05649 [0.072894] | 0.035344 [0.049353] | 72.462 [114.85] |
| M1 | force | Q1_33 | 27,207 | 1.2562 [1.7009] | 0.045462 [0.056742] | 0.033306 [0.043036] | 116.6 [179.66] |
| M2 | force_c | Q1_9 | 984 | 2.1915 [3.1082] | 0.20405 [0.29624] | 0.13504 [0.1968] | 1617.3 [2562.3] |
| M2 | force_c | PU_9 | 3,936 | 1.3433 [1.9306] | 0.041654 [0.05888] | 0.021112 [0.029566] | 276.48 [424.28] |
| M2 | force_c | Q1_17 | 4,557 | 1.2521 [1.814] | 0.04538 [0.064972] | 0.027309 [0.038377] | 324.11 [506.68] |
| M2 | force_c | Q2_17 | 5,751 | 1.1722 [1.6942] | 0.032985 [0.047125] | 0.01695 [0.024094] | 238.17 [364.67] |
| M2 | force_c | PU_17 | 18,228 | 0.5528 [0.79721] | 0.012062 [0.017306] | 0.0075972 [0.010715] | 30.656 [44.479] |
| M2 | force_c | Q1_33 | 23,199 | 0.54679 [0.78933] | 0.011767 [0.017181] | 0.0087034 [0.01215] | 53.556 [79.929] |
| M2 | force | Q1_9 | 984 | 1.5947 [2.4459] | 0.16628 [0.28435] | 0.10121 [0.17679] | 552.63 [894.43] |
| M2 | force | PU_9 | 3,936 | 1.1993 [1.7578] | 0.095141 [0.15843] | 0.055364 [0.0926] | 90.85 [143.8] |
| M2 | force | Q1_17 | 4,557 | 1.1608 [1.7224] | 0.10256 [0.169] | 0.061274 [0.10242] | 106.64 [166.08] |
| M2 | force | Q2_17 | 5,751 | 1.0837 [1.6123] | 0.087778 [0.1467] | 0.051133 [0.085148] | 79.888 [127.32] |
| M2 | force | PU_17 | 18,228 | 0.71966 [1.0802] | 0.077675 [0.12553] | 0.044293 [0.07325] | 10.536 [17.558] |
| M2 | force | Q1_33 | 23,199 | 0.61821 [0.89137] | 0.049731 [0.074383] | 0.025119 [0.040226] | 17.939 [28.357] |

### ST07b. Shorter two-sided smoothing sequences

| Cell | Class | Steps on each side | Q1_9 mean [p90] (%) | Q1_17 mean [p90] (%) | PU_9 mean [p90] (%) |
| --- | --- | --- | --- | --- | --- |
| U1 | force_c | 2 | 0.28124 [0.33175] | 0.13385 [0.15343] | 0.11137 [0.12737] |
| U1 | force | 2 | 0.16983 [0.17926] | 0.14219 [0.14565] | 0.14314 [0.14567] |
| M1 | force_c | 2 | 2.3292 [3.4188] | 1.0178 [1.4665] | 0.86128 [1.2212] |
| M1 | force | 2 | 1.2227 [1.8318] | 0.59017 [0.86672] | 0.51585 [0.73952] |
| U1 | force_c | 4 | 0.15339 [0.17997] | 0.054177 [0.062172] | 0.04076 [0.046451] |
| U1 | force | 4 | 0.07691 [0.084069] | 0.059814 [0.062339] | 0.060538 [0.06364] |
| M1 | force_c | 4 | 1.328 [1.9919] | 0.42211 [0.61995] | 0.32643 [0.47049] |
| M1 | force | 4 | 0.68988 [1.0641] | 0.2529 [0.38508] | 0.20707 [0.30643] |

## Table ST08. Energy excess of different starting fields under the same correction

Mean directional energy excess (%) of the base network, of NICE and of starting fields under the same correction (fixed retained displacement, 32 validation directions per class). Columns give the starting field and the correction (smoothing steps per stage / coarse space / smoothing steps). Harmonic and zero starting fields receive the exact rigid-body split. Data: `evidence/p1_checks_cpu.json`, `evidence/p1_checks_u2.json`.

| cell | class | Base network | NICE | Base network + 8/Q1/8 (NICE-post) | harmonic + 8/Q1/8 | zero + 8/Q1/8 | Base network + 32/Q1/32 | harmonic + 32/Q1/32 | zero + 32/Q1/32 |
|---|---|---|---|---|---|---|---|---|---|
| H2 | force_c | 35 | 0.0148 | 0.0117 | 0.0806 | 28.3 | 0.000182 | 0.000614 | 0.0156 |
| H2 | force | 19.5 | 0.0145 | 0.0249 | 0.712 | 13.7 | 0.000333 | 0.0246 | 0.0056 |
| H1 | force_c | 1.81 | 0.0115 | 0.0131 | 1.28 | 39.9 | 0.0013 | 0.199 | 2.92 |
| H1 | force | 1.84 | 0.0187 | 0.024 | 2.78 | 46.5 | 0.00202 | 0.399 | 3.79 |
| M2 | force_c | 3.64 | 0.0358 | 0.0273 | 6.72 | 324 | 0.00691 | 1.52 | 38.9 |
| M2 | force | 2.17 | 0.0404 | 0.0613 | 7.77 | 107 | 0.00939 | 1.33 | 12.4 |
| M1 | force_c | 13.5 | 0.131 | 0.186 | 17.2 | 1.1e+03 | 0.0727 | 4.11 | 185 |
| M1 | force | 7.16 | 0.0835 | 0.114 | 13.5 | 558 | 0.0397 | 2.64 | 86.9 |
| U2 | force_c | 0.402 | 0.00465 | 0.00424 | 1.23 | 47.2 | 0.0009 | 0.38 | 5.51 |
| U2 | force | 0.506 | 0.00701 | 0.0121 | 6.4 | 11 | 0.00166 | 1.25 | 1.19 |

### ST08b. Smoothing budget: 8, 16, 32 and 64 steps per stage

Mean directional energy excess (%) after the correction with \(k\) smoothing steps before and after the Q1(17) coarse solve, for three starting fields and six detailed cells, including U1 (32 validation directions per class, fixed retained displacement). Data: `evidence/p1_checks_cpu.json`, `evidence/p1_checks_u2.json` and, for U1, `evidence/p1_checks.json`.

| cell | class | starting field | 8 steps | 16 steps | 32 steps | 64 steps |
|---|---|---|---|---|---|---|
| H2 | force_c | Base network | 0.0117 | 0.00163 | 0.000182 | 3.65e-06 |
| H2 | force_c | harmonic | 0.0806 | 0.00482 | 0.000614 | 1.22e-05 |
| H2 | force_c | zero | 28.3 | 0.284 | 0.0156 | 0.000315 |
| H2 | force | Base network | 0.0249 | 0.00311 | 0.000333 | 6.23e-06 |
| H2 | force | harmonic | 0.712 | 0.181 | 0.0246 | 0.000518 |
| H2 | force | zero | 13.7 | 0.134 | 0.0056 | 0.000109 |
| H1 | force_c | Base network | 0.0131 | 0.00487 | 0.0013 | 0.000173 |
| H1 | force_c | harmonic | 1.28 | 0.547 | 0.199 | 0.0345 |
| H1 | force_c | zero | 39.9 | 10.7 | 2.92 | 0.437 |
| H1 | force | Base network | 0.024 | 0.00905 | 0.00202 | 0.00022 |
| H1 | force | harmonic | 2.78 | 1.31 | 0.399 | 0.05 |
| H1 | force | zero | 46.5 | 13.9 | 3.79 | 0.607 |
| M2 | force_c | Base network | 0.0273 | 0.0135 | 0.00691 | 0.00351 |
| M2 | force_c | harmonic | 6.72 | 3.53 | 1.52 | 0.485 |
| M2 | force_c | zero | 324 | 104 | 38.9 | 14.1 |
| M2 | force | Base network | 0.0613 | 0.0292 | 0.00939 | 0.00203 |
| M2 | force | harmonic | 7.77 | 3.99 | 1.33 | 0.229 |
| M2 | force | zero | 107 | 33.3 | 12.4 | 4.45 |
| M1 | force_c | Base network | 0.186 | 0.12 | 0.0727 | 0.0393 |
| M1 | force_c | harmonic | 17.2 | 8.13 | 4.11 | 1.43 |
| M1 | force_c | zero | 1.1e+03 | 420 | 185 | 81.8 |
| M1 | force | Base network | 0.114 | 0.0695 | 0.0397 | 0.0212 |
| M1 | force | harmonic | 13.5 | 6.63 | 2.64 | 0.66 |
| M1 | force | zero | 558 | 202 | 86.9 | 38.2 |
| U1 | force_c | Base network | 0.0267 | 0.0176 | 0.0118 | 0.00783 |
| U1 | force_c | harmonic | 4.19 | 2.65 | 1.87 | 1.33 |
| U1 | force_c | zero | 133 | 47.7 | 22 | 11.9 |
| U1 | force | Base network | 0.0279 | 0.014 | 0.005 | 0.00135 |
| U1 | force | harmonic | 15.6 | 7.48 | 2.13 | 0.306 |
| U1 | force | zero | 22.5 | 7.91 | 3.26 | 1.52 |
| U2 | force_c | Base network | 0.00424 | 0.00202 | 0.0009 | 0.000377 |
| U2 | force_c | harmonic | 1.23 | 0.693 | 0.38 | 0.194 |
| U2 | force_c | zero | 47.2 | 14 | 5.51 | 2.32 |
| U2 | force | Base network | 0.0121 | 0.00536 | 0.00166 | 0.000282 |
| U2 | force | harmonic | 6.4 | 3.53 | 1.25 | 0.204 |
| U2 | force | zero | 11 | 3.4 | 1.19 | 0.409 |

## Table ST09. Complete continuous-neighbour assembly results

Maximum relative compliance and field-based sensitivity-vector errors over the six face loads defining the joint criterion. Sensitivity maxima include both cells. The target cell uses the specified learned arm and its neighbour is exact. Cell labels abbreviate the validation identifiers (R1). The joint criterion is 3% for both errors.

| Target cell | Configuration | Arm | Max. compliance error (%) | Max. sensitivity error (%) | PCG iterations | Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| U1 | x | Base network | 0.465 | 5.830 | 11 | Above criterion |
| U1 | y | Base network | 0.427 | 5.390 | 11 | Above criterion |
| U2 | x | Base network | 0.105 | 1.996 | 11 | Pass |
| M1 | x | Base network | 4.384 | 12.153 | 15 | Above criterion |
| H1 | x | Base network | 1.180 | 2.512 | 13 | Pass |
| H1 | y | Base network | 1.379 | 1.820 | 13 | Pass |
| M2 | x | Base network | 0.571 | 2.539 | 15 | Pass |
| U1 | x | Uncorrected | 0.401 | 4.885 | 11 | Above criterion |
| U1 | y | Uncorrected | 0.388 | 4.723 | 11 | Above criterion |
| U2 | x | Uncorrected | 0.104 | 1.781 | 11 | Pass |
| U2 | y | Uncorrected | 0.119 | 1.906 | 11 | Pass |
| M1 | x | Uncorrected | 4.119 | 11.365 | 15 | Above criterion |
| M1 | y | Uncorrected | 3.458 | 10.929 | 15 | Above criterion |
| H1 | x | Uncorrected | 1.061 | 2.469 | 13 | Pass |
| H1 | y | Uncorrected | 1.223 | 1.734 | 13 | Pass |
| M2 | x | Uncorrected | 0.593 | 2.874 | 15 | Pass |
| L1 | x | Uncorrected | 0.174 | 1.894 | 16 | Pass |
| L1 | y | Uncorrected | 0.182 | 1.646 | 16 | Pass |
| U1 | x | S8 | 0.139 | 3.894 | 7 | Above criterion |
| U1 | y | S8 | 0.140 | 3.747 | 7 | Above criterion |
| U2 | x | S8 | 0.031 | 1.115 | 6 | Pass |
| U2 | y | S8 | 0.038 | 1.106 | 6 | Pass |
| M1 | x | S8 | 1.677 | 5.829 | 9 | Above criterion |
| M1 | y | S8 | 1.391 | 5.555 | 9 | Above criterion |
| H1 | x | S8 | 0.178 | 1.216 | 9 | Pass |
| H1 | y | S8 | 0.270 | 0.880 | 8 | Pass |
| M2 | x | S8 | 0.188 | 1.887 | 8 | Pass |
| U1 | x | Smoothing-trained | 0.112 | 3.155 | 7 | Above criterion |
| U1 | y | Smoothing-trained | 0.111 | 3.166 | 7 | Above criterion |
| U2 | x | Smoothing-trained | 0.031 | 0.508 | 6 | Pass |
| U2 | y | Smoothing-trained | 0.038 | 0.782 | 6 | Pass |
| M1 | x | Smoothing-trained | 1.363 | 4.432 | 9 | Above criterion |
| M1 | y | Smoothing-trained | 1.115 | 3.974 | 9 | Above criterion |
| H1 | x | Smoothing-trained | 0.144 | 0.636 | 8 | Pass |
| H1 | y | Smoothing-trained | 0.237 | 0.561 | 8 | Pass |
| M2 | x | Smoothing-trained | 0.189 | 1.428 | 8 | Pass |
| M2 | y | Smoothing-trained | 0.213 | 2.545 | 9 | Pass |
| H3 | x | Smoothing-trained | 0.106 | 1.290 | 9 | Pass |
| H3 | y | Smoothing-trained | 0.132 | 0.678 | 10 | Pass |
| U1 | x | NICE-post | 0.011 | 0.945 | 5 | Pass |
| U1 | y | NICE-post | 0.011 | 0.904 | 5 | Pass |
| U2 | x | NICE-post | 0.00123 | 0.142 | 4 | Pass |
| U2 | y | NICE-post | 0.0014 | 0.158 | 4 | Pass |
| M1 | x | NICE-post | 0.079 | 0.350 | 6 | Pass |
| M1 | y | NICE-post | 0.066 | 0.339 | 6 | Pass |
| H1 | x | NICE-post | 0.010 | 0.139 | 6 | Pass |
| H1 | y | NICE-post | 0.013 | 0.105 | 6 | Pass |
| M2 | x | NICE-post | 0.00492 | 0.083 | 6 | Pass |
| M2 | y | NICE-post | 0.0034 | 0.075 | 6 | Pass |
| H3 | x | NICE-post | 0.00609 | 0.132 | 6 | Pass |
| H3 | y | NICE-post | 0.00894 | 0.202 | 7 | Pass |
| L1 | x | NICE-post | 0.00193 | 0.065 | 7 | Pass |
| L1 | y | NICE-post | 0.00181 | 0.087 | 7 | Pass |
| U1 | x | NICE | 0.00614 | 0.598 | 5 | Pass |
| U1 | y | NICE | 0.00681 | 0.669 | 5 | Pass |
| U2 | x | NICE | 0.00121 | 0.082 | 4 | Pass |
| U2 | y | NICE | 0.00159 | 0.073 | 4 | Pass |
| M1 | x | NICE | 0.056 | 0.161 | 7 | Pass |
| M1 | y | NICE | 0.048 | 0.206 | 7 | Pass |
| H1 | x | NICE | 0.0076 | 0.098 | 6 | Pass |
| H1 | y | NICE | 0.011 | 0.097 | 6 | Pass |
| M2 | x | NICE | 0.00595 | 0.136 | 7 | Pass |
| M2 | y | NICE | 0.00472 | 0.154 | 8 | Pass |
| H3 | x | NICE | 0.0064 | 0.124 | 7 | Pass |
| H3 | y | NICE | 0.010 | 0.196 | 7 | Pass |
| L1 | x | NICE | 0.00202 | 0.101 | 8 | Pass |
| L1 | y | NICE | 0.00197 | 0.078 | 8 | Pass |

The comparison contains seven configurations for the base network, eleven for Uncorrected, nine for S8, twelve for Smoothing-trained and fourteen each for NICE-post and NICE. The nine configurations U1/x,y, U2/x,y, M1/x,y, H1/x,y and M2/x are common to Uncorrected, S8, Smoothing-trained, NICE-post and NICE; the base network was evaluated on seven of them (no U2/y, no M1/y). Uncorrected and S8 satisfy both 3% criteria in the same five of these nine configurations, and Uncorrected also on L1/x and L1/y; Smoothing-trained fails the same four configurations as Uncorrected and S8; NICE-post and NICE satisfy both criteria in all fourteen. Missing model/configuration combinations have no row. All cells in this table belong to the 20 validation geometries used for weight selection (Section 5.1). Cut-traction responses of the same configurations are in Table ST10. Held-out configurations are in Table ST09b.

#### ST09b. Held-out two-cell configurations (NICE)

Cells outside weight selection, fixed before evaluation: the five validation geometries with NICE's largest single-cell errors and one random cell per stratum. Maximum relative errors (%) over the six face loads (compliance) and over both cells (eight-corner sensitivity); the neighbour is exact and the target learned, as in Table ST09.

| Cell | Selection | Retained volume | x: compliance | x: sensitivity | y: compliance | y: sensitivity |
| --- | --- | --- | --- | --- | --- | --- |
| fresh_val_2051 | worst 1 | 0.571 | 0.2706 | 1.485 | 0.2220 | 1.312 |
| fresh_val_2045 | worst 2 | 0.267 | 0.1982 | 0.815 | 0.1703 | 1.344 |
| fresh_val_2074 | worst 3 | 0.287 | 0.1587 | 1.329 | 0.2489 | 1.419 |
| fresh_val_2021 | worst 4 | 0.266 | 0.0827 | 0.526 | 0.0753 | 0.971 |
| fresh_val_2063 | worst 5 | 0.681 | 0.0967 | 0.434 | 0.0878 | 0.563 |
| fresh_val_2032 | random, uncut | 1.000 | 0.0010 | 0.181 | 0.0011 | 0.133 |
| fresh_val_2047 | random, light | 0.779 | 0.0038 | 0.102 | 0.0051 | 0.102 |
| fresh_val_2078 | random, moderate | 0.661 | 0.0101 | 0.103 | 0.0066 | 0.100 |
| fresh_val_2053 | random, heavy | 0.078 | 0.0050 | 0.136 | 0.0066 | 0.218 |


## Table ST10. Cut-traction responses outside the six-load criterion

Maximum relative errors (%) over the three cut-surface traction directions, for every recorded arm and configuration with a cut target (Table ST09). Sensitivity maxima include both cells. The target is learned and the neighbour exact. The 3% reference is not applied to these loads in the main text; values above it are marked in bold.

| Arm | Cell | Configuration | Compliance error (%) | Sensitivity error (%) |
| --- | --- | --- | --- | --- |
| Base network | M1 | x | 2.880 | **9.924** |
| Base network | H1 | x | 0.445 | 2.066 |
| Base network | H1 | y | 0.651 | 1.275 |
| Base network | M2 | x | 0.560 | **3.155** |
| Uncorrected | M1 | x | 2.768 | **9.827** |
| Uncorrected | M1 | y | **6.870** | **14.053** |
| Uncorrected | H1 | x | 0.399 | 1.825 |
| Uncorrected | H1 | y | 0.613 | 1.200 |
| Uncorrected | M2 | x | 0.574 | **3.252** |
| Uncorrected | L1 | x | 0.355 | 1.660 |
| Uncorrected | L1 | y | 1.141 | **3.702** |
| S8 | M1 | x | 0.912 | **4.445** |
| S8 | M1 | y | 2.862 | **6.760** |
| S8 | H1 | x | 0.078 | 0.839 |
| S8 | H1 | y | 0.158 | 0.372 |
| S8 | M2 | x | 0.200 | 2.207 |
| Smoothing-trained | M1 | x | 0.713 | **3.228** |
| Smoothing-trained | M1 | y | 2.417 | **5.191** |
| Smoothing-trained | H1 | x | 0.062 | 0.356 |
| Smoothing-trained | H1 | y | 0.130 | 0.587 |
| Smoothing-trained | M2 | x | 0.200 | 1.664 |
| Smoothing-trained | M2 | y | 1.117 | **3.924** |
| Smoothing-trained | H3 | x | 0.052 | 0.689 |
| Smoothing-trained | H3 | y | 0.237 | 1.354 |
| NICE-post | M1 | x | 0.049 | 0.217 |
| NICE-post | M1 | y | 0.124 | 0.305 |
| NICE-post | H1 | x | 0.0044 | 0.096 |
| NICE-post | H1 | y | 0.00709 | 0.022 |
| NICE-post | M2 | x | 0.00468 | 0.053 |
| NICE-post | M2 | y | 0.020 | 0.091 |
| NICE-post | H3 | x | 0.00391 | 0.038 |
| NICE-post | H3 | y | 0.011 | 0.042 |
| NICE-post | L1 | x | 0.00479 | 0.039 |
| NICE-post | L1 | y | 0.011 | 0.069 |
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

## Table ST11. Assembly sensitivity replacement diagnostics

Sensitivity after assembly depends on the interaction between field recovery and the assembled retained displacement. For the base network in M1/x, the largest full sensitivity error over the six face loads is 12.2%; evaluating the learned extension at the exact retained displacement (field-only) gives 15.6%, and evaluating the exact extension at the learned retained displacement (solution-only) gives 19.3%. The full error is smaller than either replacement error. The vector expansion of Eq. (H.2) accounts for this: the extension error, the change of the retained displacement and their mixed term enter the recovered field together, so the replacement norms cannot be added as scalar error contributions (Section 5.7). Tables ST12 and ST13 give the load-specific responses of S8 and of the base network.

Maximum errors (%) over the six face loads defining the joint criterion; maxima in separate columns may occur at different loads. Full, field-only, and solution-only values are separate nonlinear replacement diagnostics. The trace error is the relative exact-Schur norm, not its square. No per-load record of the field-only and solution-only values is archived.

| Arm | Cell | Configuration | Full sensitivity error | Field-only error | Solution-only error | Trace error | Energy excess at exact trace |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S8 | U1 | x | 3.894 | 2.460 | 1.556 | 1.544 | 1.108 |
| S8 | U1 | y | 3.747 | 2.376 | 1.527 | 1.606 | 1.047 |
| S8 | M1 | x | 5.829 | 3.320 | 7.839 | 6.580 | 5.101 |
| S8 | M1 | y | 5.555 | 3.094 | 7.623 | 6.402 | 4.940 |
| S8 | M2 | x | 1.887 | 0.727 | 2.421 | 2.706 | 1.137 |
| Base network | U1 | x | 5.830 | 2.674 | 3.305 | 2.413 | 2.312 |
| Base network | U1 | y | 5.390 | 2.603 | 2.966 | 2.322 | 2.056 |
| Base network | M1 | x | 12.153 | 15.602 | 19.269 | 14.198 | 14.778 |
| Base network | M1 | y | 11.439 | 15.532 | 18.966 | 14.329 | 14.671 |
| Base network | M2 | x | 2.539 | 3.371 | 4.841 | 4.434 | 2.352 |

## Table ST12. Matched face-load responses for the smoothed variant

S8 on the target cell, exact neighbour, configuration x. Each row uses one load and reports both cell sensitivity-vector errors. Energy participation refers to the target in the exact assembled solution. Relative quantities are percentages.

| Target | Load | Compliance error | Target sensitivity error | Neighbour sensitivity error | Target energy participation |
| --- | --- | --- | --- | --- | --- |
| H1 | T-x | 0.178113 | 1.21556 | 0.0755185 | 64.2733 |
| H1 | T-y | 0.0730244 | 0.843527 | 0.0145867 | 40.803 |
| H1 | T-z | 0.0319646 | 0.502716 | 0.00906095 | 27.3735 |
| H1 | N-x | 0.000504597 | 0.442032 | 0.000822323 | 0.217835 |
| H1 | N-y | 0.00082195 | 0.345192 | 0.00205754 | 0.308471 |
| H1 | N-z | 4.06479e-05 | 0.885999 | 5.96007e-05 | 0.0360864 |
| U1 | T-x | 0.13928 | 1.18517 | 0.00831909 | 35.0385 |
| U1 | T-y | 0.0448159 | 1.30571 | 0.00536493 | 11.542 |
| U1 | T-z | 0.0313817 | 1.10598 | 0.00456525 | 11.6406 |
| U1 | N-x | 0.00254548 | 2.37052 | 0.00246372 | 0.355414 |
| U1 | N-y | 0.00351229 | 1.28801 | 0.00481203 | 0.873919 |
| U1 | N-z | 0.00136724 | 3.89419 | 0.00186976 | 0.127467 |
| M1 | T-x | 1.6765 | 4.51959 | 1.37179 | 51.2504 |
| M1 | T-y | 1.41865 | 5.82917 | 0.942721 | 31.068 |
| M1 | T-z | 0.416357 | 2.99719 | 0.137715 | 21.2573 |
| M1 | N-x | 0.00212258 | 0.914003 | 0.0056564 | 0.354364 |
| M1 | N-y | 0.0178155 | 1.05549 | 0.0460078 | 0.742265 |
| M1 | N-z | 0.000598185 | 1.82925 | 0.000963143 | 0.102043 |

## Table ST13. Load-specific compliance weighting for the base network

U1/x under the neighbour-face z traction. The base network is used on the target and its neighbour is exact. The bound is evaluated from dimensionless ratios before percentage conversion.

| Target participation (%) | Local energy excess (%) | Product bound (%) | Compliance error (%) | Target sensitivity error (%) |
| --- | --- | --- | --- | --- |
| 0.1274669 | 2.312168 | 0.002947249 | 0.002827394 | 5.82992 |

## Supplementary Note S1. Geometry visualisation

The surfaces in Figure 1 are sampled on a grid with 97 positions per unit-box axis using the eight corner band parameters and cut-plane data of U1, M1, H1 and H2. The displayed percentages describe the retained macro-domain volume, before intersection with the thin-wall material. This surface sampling is used for visualisation; the mechanical discretisation has 32 background elements per axis and continuous Q2 displacement functions.

## Supplementary Note S2. Local correction records and operator verification

These records support Sections 5.2, 5.4 and 5.5 of the main text. Tables ST04 and ST08 are generated by `evidence/summarize_results.py` from `evidence/p1_checks_cpu.json` and `evidence/p1_checks_u2.json` (script `p1_checks.py`). Table ST04 checks the operator in the deployed arithmetic; Table ST08 compares starting fields under the same correction. Figure S01 verifies the CutFEM reference against background refinement, the ghost-penalty coefficient and the finite-difference step of the moment derivatives.

### Table ST14. Refinement of the reference to \(n=64\)

Single cells clamped on one box face and loaded by unit consistent tractions in the three directions on another face, solved at \(n=24\) to 64 by a sparse direct solver. Entries: signed compliance deviation from \(n=64\) with the largest magnitude over the three loads / largest relative deviation of the eight-corner sensitivity vector, both in percent. H2 could not be meshed at \(n=56\). Production references use \(n=32\).

| Cell | \(n=24\) | \(n=32\) | \(n=40\) | \(n=48\) | \(n=56\) |
| --- | --- | --- | --- | --- | --- |
| H1 | -1.05 / 1.06 | +0.07 / 0.08 | +0.46 / 0.44 | +1.08 / 1.06 | +1.40 / 1.37 |
| H2 | -1.58 / 5.60 | -0.93 / 3.59 | -0.52 / 2.16 | -0.27 / 1.24 | — |
| fresh_val_2051 (largest NICE error) | -0.32 / 0.57 | -0.14 / 0.23 | +0.08 / 0.09 | -0.14 / 0.15 | -0.02 / 0.03 |
| fresh_val_2074 (thinnest walls) | -0.66 / 1.02 | -0.28 / 0.44 | -0.16 / 0.24 | -0.09 / 0.20 | +0.17 / 0.16 |

### Table ST15. Element-level eigenvalues of the thickness derivative

For every partially filled element of the eight detailed cells and each of the eight corners, the element derivative \(K_{e,c}=\sum_m M_{em,c}T_m\) was formed from two derivatives of the discrete moments: the exact derivative at fixed clipping topology (forward-mode differentiation of the moment integrator) and the production central difference (step \(10^{-5}\tau_c\)). Entries are the smallest ratio \(\lambda_{\min}/\max|\lambda|\) over elements, for single corners and for uniform thickening \(\sum_cK_{e,c}\), and the number of elements with a ratio below \(-10^{-6}\), summed over the eight corners and for uniform thickening. A ratio of −1 means that the element derivative is negative semidefinite. The element stiffnesses \(K_e\) themselves have ratios of at least \(-9\times10^{-16}\) in every cell. The last column is the largest relative difference, over corners, between the two moment derivatives of all partially filled elements. Elements whose central-difference ratio is −1 while their exact ratio is not must therefore carry derivatives of at most this relative size.

| Cell | Partially filled elements | Smallest ratio, exact derivative: single corner | Smallest ratio, exact derivative: uniform | Elements below \(-10^{-6}\), exact: corners / uniform | Smallest ratio, central difference: single corner | Moment derivatives, exact vs central difference |
| --- | --- | --- | --- | --- | --- | --- |
| U1 | 7,584 | \(-2.6\times10^{-5}\) | \(-2.6\times10^{-5}\) | 56 / 7 | \(-7.1\times10^{-3}\) | \(1\times10^{-8}\) |
| U2 | 7,360 | \(-2.3\times10^{-3}\) | \(-2.3\times10^{-3}\) | 768 / 96 | \(-1.7\times10^{-2}\) | \(5\times10^{-10}\) |
| L1 | 6,033 | −1 | −1 | 80 / 10 | −1 | \(9\times10^{-9}\) |
| M1 | 4,992 | −1 | −1 | 25 / 3 | −1 | \(5\times10^{-10}\) |
| M2 | 3,534 | \(-9.2\times10^{-10}\) | \(-9.1\times10^{-10}\) | 0 / 0 | −1 | \(5\times10^{-9}\) |
| H1 | 762 | −1 | −1 | 82 / 11 | −1 | \(4\times10^{-9}\) |
| H2 | 278 | \(-5.0\times10^{-11}\) | \(-5.0\times10^{-11}\) | 0 / 0 | −1 | \(2\times10^{-8}\) |
| H3 | 1,946 | −1 | −1 | 16 / 2 | −1 | \(5\times10^{-9}\) |


## Supplementary Note S3. Enriched partition-of-unity coarse spaces as a numerical observation

The enriched spaces of Appendix F.1 (PU_9, PU_17 in Table ST07 and Figure S05) are not used by any reported predictor. Their entries record what the archived solves produced and are not presented as verified Galerkin projections.

The enriched generating functions have a specific coefficient redundancy. The trilinear nodal basis reproduces linear coordinates, so

\[
\sum_v N_v(x)=1,\qquad \sum_v N_v(x)x_v=x,
\qquad \sum_v N_v(x)(x_j-x_{v,j})=0.
\]

Thus taking \(a_v=0\) and the same slope matrix \(B_v=B\) at every vertex produces the zero displacement field. Restriction to internal coordinates preserves this identity. Support and diagonal-energy screens do not certify independence of the surviving columns. The archived PU records give column counts and field-error statistics, but no rank-revealing representation or coarse-equation residual. Their values in Table ST07 are therefore retained as numerical observations of those solves, rather than verification of the full-rank projection assumptions. The reported Q1(17) result is the principal coarse-correction result.

Coefficient redundancy does not preclude energy minimisation over the coarse range. Since \(A\succ0\), \(\ker(V^TAV)=\ker V\), and \(V^Tr_I\) is orthogonal to this kernel. An exactly solved compatible coarse equation therefore defines a unique displacement correction even when its coefficient vector is nonunique. Establishing that property for the archived numerical PU solve requires the corresponding representation and solve-accuracy evidence.

## Supplementary Note S4. Ablation of the retained representation: Bernstein-restricted box faces

This note gives the complete results of the ablation summarised in Section 5.8. It is an ablation of the present pipeline: every cell operator is the exact Schur complement, and only the representation of the retained box-face displacements is reduced. It is not a model or a reproduction of reduced-boundary learned substructures [Huang et al. (2023)](https://doi.org/10.1016/j.eml.2023.102041), [Huang et al. (2024)](https://doi.org/10.1016/j.jmps.2024.105893), [Guo et al. (2026a)](https://doi.org/10.1016/j.cma.2026.118955), [Guo et al. (2026b)](https://arxiv.org/abs/2607.22019v1), which control the boundary error by refining the partition, enriching the boundary interpolation (cubic Bézier faces with 56 control points per substructure) or oversampling local bases joined by an overlapping partition of unity. The ablation makes no statement about the accuracy of those methods.

### S4.1. Restricted Galerkin system

The restricted space uses tensor-product Bernstein polynomials of degree \(r\) on the selected box faces of each cell. Its global map \(G_r\) respects the coordinates shared between cells, while cut-band coordinates outside the box trace retain identity columns. The supported Galerkin system is

\[
\mathbb K_r=G_r^T\mathbb K G_r,\qquad
f_r=G_r^Tf_g,\qquad U_r=G_r\mathbb K_r^{-1}f_r.
\tag{S4.1}
\]

The comparison evaluates exact local Schur operators within this restricted trace space, so the interior is exact and every error comes from the restricted boundary. Two variants are examined: every box face restricted, as in a lattice built entirely from such substructures, and only the shared interface restricted. Nested boundary spaces give a nondecreasing Ritz compliance (Appendix J.8). The interior correction of Sections 4.2 and 4.3 acts on a different trial space: it leaves every retained coordinate intact.

### S4.2. Conditions

- Pairs in configuration x (Figure 4; Supplementary Note S6), each target with its exact continuous-thickness neighbour, for U1, M1, M2 and H1; the interface-only variant was run for U1, M1 and M2.
- One cell per substructure, the fine-scale consistent face tractions used throughout, and no oversampling.
- Loads: the three target-face tractions, the three neighbour-face tractions and, for cut targets, the three cut-surface tractions.
- The non-box cut-band coordinates remain unrestricted, which favours the restricted model: at \(r=1\) the restricted M1 pair still controls 15,423 coordinates, whereas the uncut U1 pair controls only its 24 corner coordinates, against 28,206 free coordinates of the full retained space.
- The comparison concerns accuracy, not accuracy at equal cost.

### S4.3. Results

With every box face restricted, degree one gives compliance errors of 76–82% under the three target-face loads alone and 78–85% over the six face loads on U1, M1, M2 and H1. The error decreases with degree to about 4% at \(r=5\) and 0.49–0.74% at \(r=8\), where the restricted problem retains, on H1, 14,001 of 32,991 free coordinates. The local sensitivity converges more slowly: at \(r=8\) the maximum target-cell sensitivity error is 1.3–4.5% over the target-face loads and 24–64% when the neighbour-face loads are included. The loads applied to the neighbour deform the shared face in patterns that a polynomial of degree eight does not resolve.

Restricting only the shared interface removes most of the compliance error (U1, M1 and M2: 1.95–9.4% at \(r=1\), 0.17–0.44% at \(r=3\)), but the sensitivity error under the six face loads remains 8–21% at \(r=3\) and 2.2–3.4% at \(r=5\). A restricted interface is therefore adequate for compliance at low degree, but not for the local sensitivity under neighbour loads. Keeping the complete retained space removes this component of the error, at the cost of a coarse model with tens of thousands of coordinates per cell. The maxima refer to their respective load sets and need not occur under the same load.

### Table ST16. Bernstein restriction of the box-face displacements

Ablation of the present pipeline, not a reproduction of PIML-type substructures (Supplementary Note S4). Each target cell and its continuous-thickness neighbour are assembled in configuration x using exact cell operators. Degree r applies to the restricted box faces; cut-band coefficients retain their identity representation. Controlled DOFs include these unrestricted cut-band coefficients. All errors are maxima in the stated load set (%), relative to the full retained-space solution; sensitivity columns refer to the target-cell vector. Target-face loads are the three unit consistent tractions on the target's loaded face, the six-load set adds the three neighbour-face tractions, and the all-load set adds the three macro-cut tractions.

#### ST16a. Every box face restricted

| Cell | r | Controlled DOFs | Target-face compliance (3 loads) | Target-face sensitivity (3 loads) | Six-load compliance (6 loads) | Six-load sensitivity (6 loads) | All-load compliance (9 loads) | All-load sensitivity (9 loads) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1 | 1 | 24 | 76.998 | 89.056 | 82.609 | 203.566 | — | — |
| U1 | 2 | 102 | 56.944 | 62.055 | 56.944 | 317.415 | — | — |
| U1 | 3 | 240 | 38.907 | 47.849 | 38.907 | 398.644 | — | — |
| U1 | 5 | 696 | 3.926 | 3.603 | 3.961 | 114.545 | — | — |
| U1 | 8 | 1,830 | 0.485 | 1.467 | 0.485 | 24.435 | — | — |
| M1 | 1 | 15,423 | 77.749 | 88.966 | 78.485 | 294.272 | 78.485 | 294.272 |
| M1 | 2 | 15,498 | 53.133 | 62.622 | 53.133 | 402.365 | 53.133 | 402.365 |
| M1 | 3 | 15,627 | 33.998 | 45.584 | 33.998 | 421.766 | 33.998 | 421.766 |
| M1 | 5 | 16,047 | 3.930 | 4.748 | 3.930 | 138.963 | 3.930 | 138.963 |
| M1 | 8 | 17,082 | 0.521 | 1.295 | 0.611 | 39.320 | 0.611 | 39.320 |
| M2 | 1 | 15,648 | 76.234 | 84.796 | 85.269 | 244.968 | 85.269 | 244.968 |
| M2 | 2 | 15,723 | 49.547 | 41.050 | 50.436 | 606.013 | 50.885 | 606.013 |
| M2 | 3 | 15,852 | 29.124 | 19.138 | 33.554 | 747.730 | 33.649 | 747.730 |
| M2 | 5 | 16,272 | 3.413 | 4.275 | 3.634 | 334.483 | 4.933 | 334.483 |
| M2 | 8 | 17,307 | 0.505 | 4.458 | 0.586 | 38.693 | 0.586 | 45.351 |
| H1 | 1 | 12,888 | 81.713 | 87.320 | 81.713 | 333.566 | 81.713 | 333.566 |
| H1 | 2 | 12,939 | 37.528 | 29.960 | 41.962 | 188.527 | 41.962 | 188.527 |
| H1 | 3 | 13,026 | 21.660 | 18.637 | 30.364 | 340.008 | 30.364 | 340.008 |
| H1 | 5 | 13,308 | 2.784 | 2.890 | 3.860 | 171.306 | 3.860 | 171.306 |
| H1 | 8 | 14,001 | 0.608 | 1.520 | 0.735 | 64.044 | 0.778 | 64.044 |

#### ST16b. Only the shared interface restricted

| Cell | r | Controlled DOFs | Target-face compliance (3 loads) | Target-face sensitivity (3 loads) | Six-load compliance (6 loads) | Six-load sensitivity (6 loads) | All-load compliance (9 loads) | All-load sensitivity (9 loads) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1 | 1 | 25,386 | 1.947 | 3.599 | 1.947 | 85.982 | — | — |
| U1 | 2 | 25,401 | 0.667 | 4.905 | 0.667 | 29.133 | — | — |
| U1 | 3 | 25,422 | 0.169 | 0.338 | 0.169 | 7.999 | — | — |
| U1 | 5 | 25,482 | 0.010 | 0.328 | 0.036 | 2.194 | — | — |
| U1 | 8 | 25,617 | 0.001 | 0.003 | 0.002 | 0.359 | — | — |
| M1 | 1 | 37,080 | 9.355 | 10.397 | 9.355 | 81.909 | 9.355 | 81.909 |
| M1 | 2 | 37,095 | 1.915 | 4.529 | 1.915 | 36.640 | 1.915 | 36.640 |
| M1 | 3 | 37,116 | 0.440 | 1.106 | 0.440 | 9.954 | 0.440 | 9.954 |
| M1 | 5 | 37,176 | 0.073 | 0.335 | 0.073 | 2.174 | 0.073 | 2.174 |
| M1 | 8 | 37,311 | 0.011 | 0.031 | 0.011 | 0.615 | 0.011 | 0.615 |
| M2 | 1 | 35,196 | 4.228 | 8.397 | 4.228 | 88.562 | 4.228 | 88.562 |
| M2 | 2 | 35,211 | 0.728 | 3.825 | 0.728 | 18.427 | 0.728 | 18.427 |
| M2 | 3 | 35,232 | 0.261 | 1.058 | 0.261 | 21.216 | 0.261 | 21.216 |
| M2 | 5 | 35,292 | 0.030 | 1.004 | 0.030 | 3.366 | 0.030 | 3.366 |
| M2 | 8 | 35,427 | 0.009 | 0.049 | 0.009 | 1.537 | 0.009 | 1.537 |

The supported systems on the full retained space have 28,206 (U1), 39,996 (M1), 39,120 (M2) and 32,991 (H1) free DOFs. U1 has no cut band, so its corner-linear restriction controls only the 24 corner coordinates. Six-load and all-load maxima coincide only when the maximizing load belongs to both sets. The interface-only variant was run for U1, M1 and M2. Data: `evidence/piml4_all_*.json`, `evidence/piml4_interface_*.json` (historical file names; the records are Bernstein-restriction ablations of the present pipeline, not PIML computations).

## Supplementary Note S5. Cost records: whole-lattice direct solution and per-cell condensation

This note gives the complete records of Table 5 (Table ST17) and the per-cell cost comparison formerly in the main text (Table ST18). The four-cell lattices are the two layers \(z=0\) and \(z=1\) of the \(2\times2\times2\) block of Section 5.9, taken with their thickness corners unchanged: each layer holds two uncut cells and two cut cells with retained volume fractions 0.616 and 0.252. All lattices are clamped on the face \(y=\min\) and loaded by unit consistent tractions on the face \(y=\max\) in the three Cartesian directions; the direct solution additionally solves three random loads.

The direct solution assembles the full cut-cell stiffness of every cell, retained and interior degrees of freedom, into one global matrix. The retained degrees of freedom are numbered and coupled exactly as in the learned lattice, with the same clamp, free set and load vectors, and the interior degrees of freedom of each cell follow the free retained ones. The matrix is scaled symmetrically by its diagonal and factorised by MKL PARDISO 2026.1 on an AMD EPYC 9654 host, as a symmetric positive definite Cholesky factorisation of the upper triangle (mtype 2) with 16 threads and with one thread, and for the four-cell lattices also as an unsymmetric LU factorisation with 16 threads. The PARDISO phases are called directly with explicitly set parameters (nested-dissection ordering, iparm(2) = 3, selected on the uncut cell G3 among the tested orderings); each factorisation has its own symbolic analysis, and all six loads are solved at once. Relative residuals \(\|Ku-f\|/\|f\|\) are recomputed with the unscaled matrix. PARDISO memory is the sum of its permanent and factorisation storage (iparm(16) + iparm(17)); peak process memory is the maximum resident set size of the process, which also holds the assembled matrix and load vectors. Cholesky and LU give the same compliance to a relative difference of \(4\times10^{-10}\).

Conventional exact condensation (route (b) of Table 5) processes the cells one at a time on the same host with 16 threads: cell setup and assembly as in the direct solution, then the dense condensed matrix of the cell on its retained degrees of freedom by PARDISO's Cholesky factorisation with the Schur-complement option (iparm(36)), after which the cell is released. The condensed lattice system is solved exactly by a block Cholesky factorisation in substructuring order: for each cell, its private retained block is factorised and eliminated with dense kernels, the dense interface matrix over the retained coordinates shared by several cells is factorised and solved, and the private coordinates are recovered by back-substitution. The condensed cell matrices are held until their elimination.

SciPy 1.18's default sparse direct solver (SuperLU, one thread, COLAMD and MMD orderings) failed with a memory error while factorising a single cell of 325,404 degrees of freedom, and the two- and four-cell lattices, in each case at 4.1 to 5.3 GiB of resident memory, far below the available memory; it is therefore not included in Table 5.

The learned route runs one design iteration of the deployed implementation with NICE on one NVIDIA GeForce RTX 5090: front end (cell construction, stiffness and moment assembly, network input and encoding, and a warm-up application that prepares the correction), assembly of the lattice and of \(K_{PP}\), preconditioner setup (the balanced two-level action of Supplementary Note S6), conjugate gradients for the three consistent loads to a recursive relative residual of \(10^{-6}\) (reached at \(8.6\times10^{-7}\) to \(9.8\times10^{-7}\)), and the field-based sensitivities \(\widetilde s_c\) of Eq. (8) for the three loads, obtained by reverse-mode differentiation of the moment integrals at the fixed recovered fields. In this timed route the network, and also the correction's smoothing and coarse solve, run in single precision, while the stiffness products of the condensed action are in double precision; the accuracy results of Sections 5.2–5.9 use a double-precision correction (Appendix F.3). Both routes read the same generated cell geometries; geometry generation is not timed in either. The eight-cell runs keep the operator state of four cells on the GPU and stream the remainder from host memory.

The compliance errors in Table ST17e (and in Table 5) are those of the timed run and include the algebraic error of its \(10^{-6}\) solve (Eq. (18)). For the eight-cell lattices they exceed the operator errors of Section 5.9, which solved six loads to a recursive relative residual of \(10^{-10}\) (recomputed residual of the exact reference at most \(1.3\times10^{-10}\)), by 1.4–5% of their value and by 47% for the z-load of the \(3\times3\times1\) layer (0.0151% against 0.0103%); unlike the Section 5.9 values, they exceed the participation-weighted bound of Eq. (7). The single-precision correction itself does not change the operator accuracy: repeating the \(2\times2\times2\) solve of Section 5.9 with the correction in single precision (to \(10^{-10}\)) changes its compliance errors by less than \(3\times10^{-9}\) (record `evidence/lat_hetero222_stream.json`).

All memory values are in GiB (\(2^{30}\) bytes): GPU memory is the peak memory allocated by the process, host memory the resident set size after the front end, and PARDISO memory the reported kilobytes taken as 1024 bytes.

### Table ST17. Whole-lattice direct solution and learned route: dimensions, phases, memory and compliance

#### ST17a. Lattice dimensions

| Lattice | Cells (cut) | Cell DOFs (min–max) | Total DOFs | Free retained DOFs | Interior DOFs | Stored nonzeros, upper triangle |
| --- | --- | --- | --- | --- | --- | --- |
| 2×2×1, z=0 | 4 (2) | 102,786–328,608 | 957,888 | 77,310 | 880,578 | 132,719,531 |
| 2×2×1, z=1 | 4 (2) | 97,230–302,772 | 884,940 | 71,046 | 813,894 | 125,748,434 |
| 2×2×2 | 8 (4) | 97,230–328,608 | 1,833,474 | 139,002 | 1,694,472 | 258,181,146 |
| 3×3×1 | 8 (3) | 83,220–336,162 | 2,113,611 | 143,685 | 1,969,926 | 296,791,884 |

#### ST17b. Direct solution on the host: phases (s) and memory (GiB)

| Lattice | Factorisation | Threads | Cells: setup + assembly | Global assembly and scaling | Analysis | Factorisation | Solution, 6 loads | Total | PARDISO memory | Peak process memory | Max. relative residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2×2×1, z=0 | Cholesky | 16 | 229.9 | 21.9 | 26.0 | 193.1 | 13.5 | 484.4 | 32.0 | 35.1 | 3.3e-10 |
| 2×2×1, z=0 | LU | 16 | 215.0 | 20.3 | 27.8 | 333.3 | 17.3 | 613.6 | 63.6 | 69.9 | 3.2e-11 |
| 2×2×1, z=0 | Cholesky | 1 | 777.3 | 19.8 | 115.8 | 1,500.0 | 12.1 | 2,425.0 | 31.4 | 34.8 | 4.0e-10 |
| 2×2×1, z=1 | Cholesky | 16 | 184.2 | 17.6 | 20.2 | 116.9 | 10.4 | 349.2 | 28.8 | 32.0 | 4.5e-10 |
| 2×2×1, z=1 | LU | 16 | 191.2 | 19.3 | 25.8 | 240.7 | 31.7 | 508.6 | 57.4 | 63.4 | 3.8e-11 |
| 2×2×1, z=1 | Cholesky | 1 | 685.6 | 18.8 | 106.1 | 1,309.4 | 11.3 | 2,131.3 | 28.3 | 31.6 | 5.2e-10 |
| 2×2×2 | Cholesky | 16 | 379.5 | 36.0 | 53.1 | 368.2 | 31.0 | 867.8 | 66.4 | 71.1 | 1.5e-10 |
| 2×2×2 | Cholesky | 1 | 1,402.1 | 39.3 | 254.5 | 4,140.3 | 27.7 | 5,863.9 | 65.5 | 70.6 | 1.8e-10 |
| 3×3×1 | Cholesky | 16 | 431.3 | 42.3 | 50.4 | 481.5 | 36.0 | 1,041.6 | 80.9 | 85.9 | 2.6e-09 |
| 3×3×1 | Cholesky | 1 | 1,669.9 | 45.4 | 324.4 | 6,059.1 | 32.4 | 8,131.0 | 79.8 | 85.2 | 2.8e-09 |

Total: sum of the preceding phases. PARDISO memory: permanent plus factorisation storage (iparm(16) + iparm(17)) reported by the analysis, which agreed with the value after the numerical factorisation within 0.1% where both were recorded. Records: `evidence/host_direct/lat2_<lattice>_*.json` (16 threads) and `evidence/host_direct_1thread/lat1_<lattice>_chol.json` (one thread). With 32 threads on the same host, the factorisation of the four lattices takes 105.5, 87.4, 275.7 and 395.1 s instead of 193.1, 116.9, 368.2 and 481.5 s, and the whole direct solution 378.5, 341.1, 823.4 and 955.5 s instead of 484.4, 349.2, 867.8 and 1,041.6 s, because cell setup and assembly do not speed up (records `evidence/host_direct_32threads/lat2t32_<lattice>_chol.json`).

#### ST17c. Conventional exact condensation on the host: phases (s) and memory (GiB)

| Lattice | Cells: front end | Cells: Schur complement | Condensed solve | Total | Condensed cell matrices held (GiB) | Dense interface (GiB) | Shared retained DOFs | Peak process memory (GiB) | Compliance vs. (a) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2×2×1, z=0 | 210.6 | 274.3 | 86.8 | 588.8 | 17.8 | 1.1 | 12,327 | 27.9 | 5e-10 |
| 2×2×1, z=1 | 263.4 | 279.2 | 71.4 | 631.7 | 14.9 | 0.9 | 11,103 | 24.7 | 4e-10 |
| 2×2×2 | 393.6 | 392.5 | 133.9 | 949.7 | 32.8 | 8.0 | 32,784 | 40.1 | 2e-10 |
| 3×3×1 | 462.0 | 555.0 | 165.7 | 1,217.9 | 33.4 | 7.6 | 31,845 | 42.2 | 7e-10 |

Condensed solve: private elimination, interface factorisation and solution, and back-substitution for the six loads. Total also includes the per-cell lattice geometry, the matrix scaling and data movement between phases (17 to 35 s). Compliance vs. (a): largest relative difference over the six loads from the whole-lattice direct solution. Records: `evidence/host_condensed/latcond_<lattice>.json`.

#### ST17d. Learned route: phases of one design iteration (s) and memory (GiB)

| Lattice | Front end | Lattice and \(K_{PP}\) assembly | Preconditioner setup | Conjugate-gradient solve, 3 loads | Iterations | Sensitivities | Total | Peak GPU memory | Host memory after front end | Operator state streamed from the host | Record |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2×2×1, z=0 | 5.47 | 0.07 | 3.53 | 27.41 | 114 | 1.88 | 38.65 | 6.34 | 2.42 | 0.00 | `learned_hlat221a.json` |
| 2×2×1, z=1 | 5.14 | 0.11 | 3.45 | 26.96 | 119 | 1.85 | 37.74 | 6.09 | 2.40 | 0.00 | `learned_hlat221b.json` |
| 2×2×2 | 9.73 | 0.13 | 6.62 | 60.66 | 129 | 3.69 | 81.24 | 8.50 | 3.88 | 1.28 | `d5_off.json` |
| 3×3×1 | 11.03 | 0.10 | 7.38 | 87.08 | 165 | 4.11 | 110.12 | 9.19 | 4.69 | 2.09 | `learned_hlat331.json` |

#### ST17e. Compliance under the three consistent face loads

| Lattice | Reference | Reference compliance, x / y / z | Learned compliance, timed run, x / y / z | Learned error of the timed run (%), x / y / z |
| --- | --- | --- | --- | --- |
| 2×2×1, z=0 | direct, Cholesky and LU | 302.1935 / 51.6935 / 635.7739 | 302.1486 / 51.6874 / 635.7085 | −0.0148 / −0.0118 / −0.0103 |
| 2×2×1, z=1 | direct, Cholesky and LU | 404.8880 / 68.6089 / 833.2975 | 404.7993 / 68.5981 / 833.1795 | −0.0219 / −0.0157 / −0.0142 |
| 2×2×2 | exact condensation (Section 5.9) | 138.4184 / 24.5936 / 90.9945 | 138.3990 / 24.5908 / 90.9857 | −0.0140 / −0.0112 / −0.0097 |
| 3×3×1 | exact condensation (Section 5.9) | 254.9993 / 44.3539 / 1393.2987 | 254.9599 / 44.3478 / 1393.0878 | −0.0154 / −0.0137 / −0.0151 |

Data: `evidence/lat_direct_smoke.json`, `evidence/lat_direct_hlat221a_m11.json`, `evidence/lat_direct_hlat221b_m2_11.json`, `evidence/lat_direct_hlat222.json`, `evidence/lat_direct_hlat331.json` (direct solution); `evidence/learned_hlat221a.json`, `evidence/learned_hlat221b.json`, `evidence/d5_off.json` (\(2\times2\times2\)), `evidence/learned_hlat331.json` (learned route); `evidence/hlat221a.json`, `evidence/hlat221b.json` (four-cell layouts); `evidence/lat_hetero222_A3.json`, `evidence/lat_hetero331_A3.json` (exact references of Section 5.9). Scripts: `lat_direct_cpu.py` (direct solution) and `lat_scale.py` (learned route) of the code archive.

### Table ST18. Per-cell cost of conventional condensation and NICE

Four deployment cells, two cut and two uncut (R1). Front end: geometry preprocessing plus cell setup, moment integration and stiffness assembly, on the GPU for NICE and on the host for the conventional route. Condensation: network encoding plus correction setup (smoothing interval and coarse factorisation) for NICE; symbolic analysis and Cholesky factorisation of the interior by MKL PARDISO (16 threads, AMD EPYC 9654) for the conventional route. Memory, in GiB: device memory held by NICE's network state and correction, or PARDISO's factorisation storage (iparm(17)); neither includes the cell stiffness \(K\), which both routes hold (last column, GPU storage format). Application: \(\widehat Sq\) or \(Sq\) for a batch of 1, 16 or 64 retained vectors, the latter by interior forward and backward substitution. Explicit \(S\): dense condensed matrix by PARDISO's Schur-complement option (iparm(36)); time and size of the dense matrix. Data: `evidence/bench_A3_cells.jsonl` (NICE), `evidence/host_direct/host_summary.json` (host).

| Cell | DOFs / retained | Front end (s): NICE / host | Condensation (s): NICE / host | Memory (GiB): NICE state / host factor | Application, 1 / 16 / 64 vectors (ms): NICE | Application (ms): host | Explicit \(S\), host (s / GiB) | \(K\) (GiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G1 (cut) | 177,507 / 24,636 | 5.3 / 27.3 | 0.72 / 10.5 | 0.74 / 2.89 | 54 / 531 / 923 | 136 / 665 / 2,011 | 48 / 4.5 | 1.39 |
| G2 (cut) | 67,224 / 18,858 | 3.0 / 10.9 | 0.16 / 1.9 | 0.25 / 0.60 | 25 / 205 / 388 | 32 / 154 / 484 | 11 / 2.6 | 0.51 |
| G3 (uncut) | 289,494 / 17,508 | 6.1 / 45.0 | 0.78 / 27.3 | 1.20 / 6.18 | 84 / 877 / 1,481 | 248 / 1,308 / 4,043 | 76 / 2.3 | 2.32 |
| G4 (uncut) | 404,148 / 25,920 | 6.5 / 47.0 | 1.09 / 48.9 | 1.37 / 9.42 | 99 / 1,052 / 1,828 | 435 / 2,144 / 6,780 | 163 / 5.0 | 2.71 |

## Supplementary Note S6. Global preconditioning and two-cell load configurations

### S6.1. Balanced two-level preconditioner

Let \(\mathbb A\) be the supported assembled operator used in a run, either the exact or learned one. This notation is distinct from the local interior stiffness \(A=K_{II}\). Let
\(\mathbb K_{PP}=\sum_mB_m^TK_{PP,m}B_m\) on the free retained coordinates. The stiffness-block fine action is \(\mathcal B_f=\mathbb K_{PP}^{-1}\); the diagonal alternative uses \(\operatorname{diag}(\mathbb K_{PP})^{-1}\). It does not require local Neumann solves.

For a global retained-coordinate coarse basis \(Z_g\), the ideal coarse action is
\(Q_g=Z_g(Z_g^T\mathbb A Z_g)^{-1}Z_g^T\), after removal of dependent columns. The balanced action recorded as BNN is

\[
\mathcal M^{-1}=Q_g+(I-Q_g\mathbb A)\mathcal B_f(I-\mathbb A Q_g).
\]

The reported q1r basis multiplies macro-grid trilinear vertex functions by three translations and three rotations about each vertex, then restricts the resulting fields to the supported retained coordinates. This global coarse basis differs from the interior basis \(V\) used to correct local extensions. The lattice solves of Sections 5.9 and 5.10, exact and learned, use this balanced action with the stiffness-block fine action and the q1r basis (recorded setting `bnn:kpp:q1r`).

The supplied implementation forms \(A_g=Z_g^T\mathbb A Z_g\), records its relative asymmetry, symmetrises it, and scales it by its diagonal. It retains positive scaled eigenvalues exceeding \(10^{-10}\) times the largest eigenvalue and uses the corresponding normalised columns \(W\) so that \(Q_g=WW^T\). The stored product \(\mathbb A W\) supplies the two projections. This spectral selection concerns the preconditioner coarse action and leaves the local fine stiffnesses and condensed targets unchanged. The ideal symmetric formula above describes the algorithm; a finite-precision operator may additionally exhibit the action/energy discrepancy discussed in Appendix J.6.

The additive variant applies \(Q_g+\mathcal B_f\). The deflated variants use a coarse initial solution and a projected fine correction. The reference inequality \(\mathbb K_{PP}\succeq\mathbb K\) holds for exact condensation of the specified positive-semidefinite cell matrices; it does not imply \(\mathbb K_{PP}\succeq\widehat{\mathbb K}\) for an arbitrary learned extension.

### S6.2. Two-cell diagnostic configuration

The target cell occupies \([0,1]^3\). In configuration x, the neighbour is translated by \((-1,0,0)\), its far face \(x=-1\) is clamped, and the six face loads act on the plane \(y=0\). In configuration y, the neighbour is translated by \((0,-1,0)\), its far face \(y=-1\) is clamped, and the face loads act on \(x=0\). The six cases comprise the three Cartesian traction directions applied to the target face and the same three directions applied to the neighbour face. The consistent-load implementation integrates the Q2 surface shape functions, normalizes each nodal load to unit resultant before support elimination, and then eliminates clamped entries. Three similarly normalised consistent tractions on the target cut surface are reported separately when present. Non-box cut-band coordinates remain private free variables. The continuous-thickness neighbour shares the prescribed thickness values on the common face. The neighbour is an uncut cell; where the target's cut plane meets the shared face, the pair is therefore a diagnostic assembly rather than a physically cut specimen. Box-face coordinates of the two cells are identified by their background-grid position and displacement component, and the reference and learned assemblies use the same maps \(B_m\); a face coordinate present in one cell only remains a coordinate of that cell.

The pair accuracy calculation uses the Cholesky factor of the exact assembled reference stiffness as the PCG preconditioner. The supplied implementation uses a relative recursive-residual tolerance of \(10^{-10}\), and the reported runs allow at most 400 iterations. The saved result summaries retain the iteration count but omit the residual returned by the solver. These settings concern the pair accuracy comparison.

The implementation also contains a uniform-nodal-force branch. The comparisons identified as consistent face loads use the integrated branch; the two branches must retain distinct load definitions in any reuse of the records.

### S6.3. Instrumented solves and difference quotients of the surrogate

The lattices of Section 5.9 and their two \(2\times2\times1\) layers were re-solved with NICE in double-precision correction arithmetic, to a recursive residual of \(10^{-10}\), recording at the final iterate and at the recursive-residual levels \(10^{-3}\) to \(10^{-9}\): the signed residual work \(\bar U^T\rho\) with \(\rho=f_g-y(\bar U)\) and \(y\) the applied learned action; the action–energy inconsistency \(\omega=\sum_m\omega_m\), \(\omega_m=\widehat q_m^T\widehat S_m\widehat q_m-\bar u_m^TK_m\bar u_m\); the residual of the identity \(C-\bar C=\sum_ma_m+\bar U^T\rho+\omega\) (Eq. 18 and Appendix J.6); and the dual-norm bound \(|\bar U^T\rho|\le\sqrt{\bar U^Ty(\bar U)}\sqrt{\rho^T\mathbb K^{-1}\rho}\), valid because \(\widehat{\mathbb S}\succeq\mathbb S\), with \(\rho^T\mathbb K^{-1}\rho\) computed with the exact assembled operator. All quantities are relative to the exact compliance. Repeating the two eight-cell runs with the correction in single precision changes the compliance errors by less than \(4\times10^{-8}\) and the gradient errors by less than \(10^{-6}\).

The gradient metrics compare the field-based sensitivities with the exact ones after aggregation over the shared lattice vertices (the sum of the cells' corner sensitivities at each vertex), which is the gradient an optimiser with shared thickness variables receives; loads are held fixed. For the complete surrogate derivative, every cell's learned operator was rebuilt at \(\tau\pm h\tau_ce_c\) for each corner and applied to the solved retained displacements at fixed \(\widehat q\) (the action-energy difference quotient \(D_{\mathrm{act}}\)); rebuilding at the unperturbed design reproduces the operator's energies to \(8\times10^{-8}\) for the lattice cells and \(7\times10^{-7}\) for the pair cells. A rebuild counts as switched when it changes a binary node feature, a fringe hyperedge, the active element set or the coarse-factor shift relative to the unperturbed build.

#### Table ST19. Residual work, dual-norm bound, field-based gradient and difference quotients of the surrogate

Face = the three consistent face loads; random = the three random loads. Gradient error: \(\|\widetilde s_g-s_g\|/\|s_g\|\) over the shared vertex parameters, in percent; cosine and largest component error over all six loads. \(D_{\mathrm{act}}\): relative error against the exact vertex gradient under the face loads at step \(h\) (in units of \(\tau_c\)), in percent. Switched rebuilds: number of the 49 rebuilds per cell (three steps, both signs and eight corners, plus one unperturbed) that change a discrete choice, range over cells.

| Lattice | Compliance error, face (%) | \(\max\lvert\bar U^T\rho\rvert/C\) | Dual-norm bound / \(C\) | \(\max\lvert\omega\rvert/C\) | Identity residual / \(C\) | Gradient error, face / random (%) | Smallest cosine | Largest component error (%) | \(D_{\mathrm{act}}\) at \(h=3\times10^{-3}\) / \(10^{-3}\) / \(3\times10^{-4}\) (%) | Switched rebuilds per cell |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \(2\times2\times1\), \(z=0\) | 0.0145 | \(2.3\times10^{-8}\) | \(6.8\times10^{-6}\) | \(5.0\times10^{-9}\) | \(4.6\times10^{-9}\) | 0.022–0.048 / 0.22–0.27 | 0.9999998 | 0.47 | 0.29 / 0.65 / 1.53 | 31–48 |
| \(2\times2\times1\), \(z=1\) | 0.0215 | \(5.3\times10^{-8}\) | \(7.9\times10^{-6}\) | \(9.5\times10^{-9}\) | \(1.2\times10^{-8}\) | 0.067–0.130 / 0.29–0.35 | 0.9999993 | 0.68 | 0.21 / 0.27 / 0.35 | 32–47 |
| \(2\times2\times2\) | 0.0137 | \(2.9\times10^{-9}\) | \(4.7\times10^{-6}\) | \(2.4\times10^{-9}\) | \(5.2\times10^{-9}\) | 0.052–0.069 / 0.24–0.27 | 0.9999996 | 0.47 | 0.18 / 0.29 / 0.59 | 31–48 |
| \(3\times3\times1\) | 0.0147 | \(2.5\times10^{-8}\) | \(1.4\times10^{-5}\) | \(7.0\times10^{-9}\) | \(8.9\times10^{-9}\) | 0.041–0.088 / 0.22–0.28 | 0.9999996 | 0.64 | 0.19 / 0.23 / 0.35 | 21–48 |

On the fourteen development pair configurations of Section 5.6, the same instrumentation gives \(|\bar U^T\rho|/C\le1.8\times10^{-8}\), a dual-norm bound of at most \(2.0\times10^{-5}\), \(|\omega|/C\le1.7\times10^{-8}\), field-based eight-corner sensitivity errors of 0.07–0.67% with cosines of at least 0.999998, and \(D_{\mathrm{act}}\) errors of 0.05–0.27% at \(h=10^{-3}\) that likewise do not decrease with the step.

One-corner sweeps test the smoothness of the surrogate across these switches. In the two-cell configurations M1/x and U1/y, one corner parameter of the target cell was varied over up to ±10% of its value (4, 27 and 9 designs on three sweeps, on the largest- and the median-sensitivity corner), every design with its own regenerated geometry and NICE operator, and the compliance change between neighbouring designs was compared with the change predicted by the trapezoidal rule from the sensitivities at both ends; the same comparison was made for the exact discrete model at every tenth design. Every interval on M1/x and 24 of the 26 intervals on U1/y change at least one discrete choice, most of them the active elements, ghost faces and retained coordinates. There the surrogate departs from the predicted change by up to 0.25% (M1/x), 0.17% and 0.085% (U1/y) of the compliance, and the exact model by the same amount: 0.230, 0.252 and 0.228% against 0.228, 0.249 and 0.225% on M1/x, and identical to three significant digits on U1/y. On the two U1/y intervals without a switch the departure is \(7\times10^{-8}\). At the reference designs the surrogate's compliance error stays at its bound \(\beta\) (\(6.0\times10^{-4}\) on M1/x, \(7.2\times10^{-5}\) on U1/y). The non-smoothness of the surrogate objective is therefore that of the discrete model, which an optimiser driven by exact condensation meets in the same way.

## Supplementary Note S7. Network settings and parameter counts

This note complements Appendix G with the settings of the learned extension that are needed to rebuild it. Table ST20 lists them; the values were read from the archived run configurations (`evidence/meta_p1.json`, `evidence/meta_c_oh.json`) and from the model definition of the archived source snapshot. The parameter counts were obtained by instantiating that model definition with the recorded settings. P0 differs from the other predictors in the entries marked "not used by P0" and in its class weights.

### Table ST20. Architecture, coefficient bounds, initialisation and parameter counts

| Item | Setting |
| --- | --- |
| Latent displacement channels, heads | 32 channels; four gather–mix–scatter heads per local interaction |
| Local interaction layers | Four element and four ghost-face layers before the latent hierarchy (alternating), four and four after it, then four weak-region element and four weak-region face layers: 8 ordinary element, 4 weak-region element and 12 face layers |
| Latent hierarchy | Three coarse levels (33, 17 and 9 grid positions per axis); two residual \(3\times3\times3\) convolutions per level on each pass (12 in total); additive skips with a learned scalar \(\sigma_\ell\) per level |
| Geometry embeddings | 64 components; two element–node message-passing rounds; all encoders and heads are two affine layers separated by GELU, hidden width 64 |
| Slot embeddings | Two learned tables of \(27\times8\) values: one for the element stencils (ordinary and weak-region layers) and one for the ghost-face stencils. For each incidence, the element or face embedding (64), the node embedding (64) and the slot's 8 values are concatenated (136 inputs) |
| Coefficient scaling | Raw gather/scatter head outputs are multiplied by 0.2 |
| Coefficient bounds \(a_{\max}\) | Fixed per group: gather and scatter role of each of the 8 element, 12 face and 4 weak-region layers (48 groups), and restriction and prolongation of each of the 3 levels (6 groups), plus the knee \(k_b\): 55 stored values. Each \(a_{\max}\) is twice the largest magnitude of its coefficient group recorded in a calibration pass of the geometry branch over training geometries, so with \(k_b=0.5\) the map is the identity up to that recorded maximum. Convolution coefficients are not bounded (\(2\operatorname{sigmoid}\)). The numerical values are part of the stored model state. Not used by P0 |
| \(k_b\) | 0.5 for the base network, Uncorrected, S8, Smoothing-trained and NICE (and hence NICE-post) |
| Stiffness-share scattering, Eq. (G.1) | One learned \(\lambda_{\rm mix}\) per element, face and weak-region element layer (24 values), initialised at zero, i.e. at plain incidence averaging. Not used by P0 |
| Optional five-feature node extension | Disabled for every predictor |
| Initialisation at the start of the training lineage | Channel maps \(W_{\ell h}\): \(0.5\,\mathcal N(0,1)/\sqrt{32}\); \(W_{\rm in}\): \(\mathcal N(0,1)/\sqrt3\); \(W_{\rm out}\): \(0.1\,\mathcal N(0,1)/\sqrt{32}\); convolution kernels: \(0.5\,\mathcal N(0,1)/\sqrt{27\cdot32}\); slot tables: \(0.1\,\mathcal N(0,1)\); \(\sigma_\ell=1\); affine layers: PyTorch default. The reported predictors are warm-started: the base network from an earlier checkpoint trained on 148 legacy geometries, the continuations from the base network's selected weights |
| Parameter counts, geometry branch | Element encoder 12,288; node encoder 9,024; message passing 49,664; face encoder 12,608; slot tables 432; element, face and weak-region coefficient heads 12,928, 15,008 and 10,848; restriction/prolongation heads 12,870; convolution-coefficient heads 37,440. Total 173,110 |
| Parameter counts, displacement branch | Element channel maps 32,768; face channel maps 49,152; weak-region element channel maps 16,384; convolution kernels 331,776; \(W_{\rm in}\) and \(W_{\rm out}\) 96 each; skip scalars 3; \(\lambda_{\rm mix}\) 24. Total 430,299 |
| Stored values | 603,409 trainable parameters and 55 fixed bound values: 603,464 in total |
| Training settings | Batches of 16 directions; one geometry per update from a device pool of three, one pool replacement every 100 updates; Adam with a one-cycle schedule (peak \(3\times10^{-4}\), 5% warm-up, cosine decay, final division factor 100); gradient norm clipped at one; EMA decay 0.9997 with bias correction; sensitivity weight 1; eight difficult-direction candidates and four block iterations when a geometry is loaded; class weights as in Appendix G.3 (P0: `force` 0.25, `support` 0.15, `face` 0.20, `macro` 0.10, `grf` 0.15, `adv` 0.15); seed 0 for every run |

Run identifiers, checkpoints and the evaluation records of every table are listed in R1 and in the data lines of the tables.

## Supplementary Note S8. Definition of the illustrative matrix example

The re-equilibration example (example 1 of Appendix J.9) uses two retained and three internal coordinates. Its matrices are

\[
A=\begin{bmatrix}3&.2&.1\\.2&2&-.1\\.1&-.1&1.4\end{bmatrix},\qquad
S=\begin{bmatrix}1.4&-.2\\-.2&.9\end{bmatrix},\qquad
L=\begin{bmatrix}.2&-.1\\.3&.25\\-.2&.15\end{bmatrix},
\]
\[
K=\begin{bmatrix}S+L^TAL&-L^TA\\-AL&A\end{bmatrix},\quad
E=\begin{bmatrix}I_2\\L\end{bmatrix},\quad
H_0=\begin{bmatrix}.3&-.2\\-.1&.4\\.2&.1\end{bmatrix},\quad
F_t=E+t\begin{bmatrix}0_{2\times2}\\H_0\end{bmatrix}.
\]

The supported three-coordinate assembly uses

\[
B=\begin{bmatrix}1&0&0\\0&1&0\end{bmatrix},\quad
G=B^TSB+\begin{bmatrix}.5&0&-.2\\0&.4&-.1\\-.2&-.1&1.1\end{bmatrix},\quad
f=(1,-.4,.8)^T,
\]
\[
\widehat G_t=G+t^2 B^TH_0^TAH_0B,\qquad
U=G^{-1}f,\quad \widehat U_t=\widehat G_t^{-1}f.
\]

Eight symmetric derivative matrices are generated once with NumPy's default generator and seed 620260926: for each corner, draw a \(5\times5\) standard-normal matrix \(R_c\), then set \(D_c=(R_c+R_c^T)/16\). These matrices provide algebraic sensitivity directions; the positive-semidefinite nested-thickening counterexamples are given separately in examples 2 and 3 of Appendix J.9. The saved sequence is \(t=0.1,0.05,0.025,0.0125,0.00625\). The final two entries give the following log-two slopes:

| Quantity | Saved slope |
| --- | --- |
| Condensed operator error | 2.000000 |
| Retained solution error | 1.999941 |
| Compliance gap | 1.999949 |
| Full local field error | 0.999626 |
| Field-based sensitivity error | 1.001084 |
| Retained-solution error energy | 3.999886 |

These entries reproduce the previously saved algebraic check; they are not TPMS observations or new mechanics experiments.

## Supplementary Note S9. Design optimisation records

This note gives the settings and complete records of the thickness optimisations of Section 5.11: the optimiser and its constraints (S9.1, Table ST21); the \(2\times2\times2\) block of Section 5.9, optimised with NICE and, from the same start, with exact condensation (case A; S9.2, Table ST22); the cut plates B1 and B2, the designs Hom-\(y\) and Hom-\(z\) obtained from a homogenised model, and the NICE optimisations X-\(y\) and X-\(z\) continued from them (S9.3, Table ST23); the geometry-generation fallback and the discrete switches between iterations (S9.4, Table ST24); checks of the analysis route (S9.5, Table ST25); and the scale demonstration and the exact verification of the plate designs (S9.6). All NICE runs use one RTX 5090. Compliance is given in the units of Section 5.1 (\(E_Y=1\), unit cell), memory in GiB (\(2^{30}\) bytes), and times are wall-clock times of complete design iterations, geometry generation included, unless stated otherwise. The geometry of the start design was generated beforehand and is not included at iteration 0 of any run; the same holds at iteration 17 of the exact twin and, in part, at iteration 5 of B1 and B2, where geometry generated earlier was reused.

### S9.1. Optimiser and constraints

The design variables are the corner thickness parameters at the lattice vertices, \(\boldsymbol\tau_g\). Every cell meeting at a vertex takes its value, \(\boldsymbol\tau_m=\boldsymbol\tau_m(\boldsymbol\tau_g)\), so that the thickness field is continuous across shared faces. The objective is the compliance \(\widehat C=f_g^T\bar U\) of the NICE lattice under one consistent face traction of unit resultant. The optimiser receives the field-based estimate of its gradient, \(\sum_m(\partial\boldsymbol\tau_m/\partial\boldsymbol\tau_g)^T\widetilde{\boldsymbol s}_m\), with \(\widetilde s_c\) from Eq. (8) for every corner of every cell; this estimates the exact gradient and is not the complete derivative of the surrogate compliance (Eq. (9), Section 6.2). The vertices in the plane of the loaded face keep their initial values. The thickness field on that face, the material part of the face over which the traction is integrated, and hence the nodal load, then do not depend on the design, and no load-derivative term \(2f_{g,c}^T\widehat U\) arises (Appendix H).

The volume \(V\) is the sum of the zeroth element moments of all cells, that is, the material volume of the discrete model. It is bounded by \(V^*=0.8\,V(\boldsymbol\tau^0)\) or, for the runs continued from the homogenisation designs, by the absolute bound of B1 and B2. The corner span and the gradient norm of every cell's trilinear thickness field are limited to 0.45 and the parameters to [0.18, 0.69], inside the training limits of 0.47 and [0.1752, 0.6993] (Table ST02). The span constraints are linear in the parameters. The gradient norm is evaluated at every corner from the three edge differences; this bounds it over the whole cell, because the largest gradient norm of a trilinear field is attained at a corner. Over all designs analysed, the largest span was 0.4500 and the largest gradient norm 0.4518, and the parameters stayed in [0.1800, 0.6900], so every analysed cell lay within the training limits.

The update is the standard method of moving asymptotes [Svanberg (1987)](https://doi.org/10.1002/nme.1620240207): one convex separable approximation per analysis, solved by a primal–dual interior-point method, with no line search and no conservativeness test (Table ST21). The optimiser thus uses the gradient estimate only to build its approximation and never tests the decrease of \(\widehat C\) along a search direction, where the inconsistency discussed in Section 6.2 would matter. In the runs from an initial design (case A, B1 and B2) the initial volume exceeds the bound by 25%, and the move limit of 0.0255 per iteration brings it to the bound at iteration 4. Meanwhile the compliance rises as material is removed: in case A to 37.86 at iteration 4 (Table ST22a), in B1 to 129.69 at iteration 3 and in B2 to 2,129.57 at iteration 4. In the first three or four iterations the elastic variables of the subproblem are positive (at most 0.19). The optimisation stops when the largest parameter change falls below \(10^{-3}\) or the relative change of the objective stays below \(10^{-4}\) in three consecutive iterations. Neither the gradient norm nor a KKT residual is used for stopping, since the estimate is not the gradient of \(\widehat C\) and \(\widehat C\) changes its discrete model between iterations (S9.4).

Every iteration regenerates the geometry of every cell from its current corner parameters, with the geometry generator used for all cells of this study, and rebuilds every learned operator. The lattice is solved with the arithmetic and the preconditioner of the timed route of Table 5: preconditioned conjugate gradients with the balanced two-level preconditioner of Supplementary Note S6.1 to a recursive relative residual of \(10^{-6}\), with the network and the correction in single precision. The solve starts from the previous iteration's solution, matched coordinate by coordinate by absolute grid position, displacement component and private cut-band flag (unmatched coordinates start at zero) and scaled by the energy-optimal factor \(f_g^TX_0/(X_0^T\widehat{\mathbb K}X_0)\); at iteration 0 of every run, and at iteration 16 of case A and iteration 5 of B1 and B2, it started from zero. In case A all cells stay on the GPU; in the plates, cells stay on the GPU while its allocated memory is below 22 GiB and the others are streamed from host memory, as for the eight-cell lattices of Supplementary Note S5. The sensitivities and the volume gradient are obtained in one reverse-mode pass through the moment integrals at the fixed recovered field; the vertex sensitivities agree with those from the central moment differences of Eq. (H.6) to \(4.6\times10^{-7}\) (Table ST25). The optimisation runs do not use the four implementation options of the timed run of Table 5 (fused per-tetrahedron moment kernels, template-based coarse Galerkin matrices, a sparse coarse space and element gathers; record `evidence/d5_off.json`), and their phase times (Table ST22c) are therefore not comparable with those of Table ST17d.

Geometry generation failed in five design iterations over all runs: the geometry generator could not certify the material patches of an element (Appendix A.1). The fallback of Table ST21 then multiplied the free vertices of the failing cells by \(1+\epsilon\), with \(\epsilon=\pm10^{-4}\), then \(\pm10^{-3}\) and \(\pm3\times10^{-3}\) in turn, regenerated every cell sharing them, and analysed and continued from the perturbed design. It succeeded at its first or third perturbation (\(\epsilon=10^{-4}\) or \(10^{-3}\)) and changed a corner parameter by at most \(5.8\times10^{-4}\) (Table ST24a). At the first failures (iteration 16 of case A, iteration 5 of B1 and B2), perturbations of the failing cell's own corner parameters by up to \(10^{-8}\) had not resolved them (Table ST24a).

### Table ST21. Optimiser, constraints and analysis settings

| Item | Setting |
| --- | --- |
| Design variables | Corner thickness parameters at the lattice vertices \(\boldsymbol\tau_g\), each shared by the cells meeting there; \(\boldsymbol\tau_m=\boldsymbol\tau_m(\boldsymbol\tau_g)\) copies the vertex values to the corners of cell \(m\). Case A: 27 vertices, 18 free; plates: 74 vertices, 64 free |
| Fixed vertices | The vertices in the plane of the loaded face keep their initial values (9 in case A, 10 in the plates), so that the thickness field on that face, the material part of the face and its consistent nodal load do not change with the design (Appendix H: no load-derivative term) |
| Objective | Compliance \(\widehat C=f_g^T\bar U\) under one consistent face traction of unit resultant, divided by its value at the first iteration |
| Gradient | Field-based estimate \(\sum_m(\partial\boldsymbol\tau_m/\partial\boldsymbol\tau_g)^T\widetilde{\boldsymbol s}_m\), \(\widetilde s_c\) from Eq. (8) of every cell; reverse-mode differentiation of the moment integrals at the fixed recovered field. The complete surrogate derivative of Eq. (9) is not used. Exact twin: exact sensitivities from the exact field with the moment derivatives of Eq. (H.6) |
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

Over every design analysed on the fine scale (all iterations of all runs of Tables ST22 and ST23), the largest corner span is 0.4500, the largest gradient norm 0.4518 and the corner parameters lie in [0.1800, 0.6900]. Records: `evidence/opt/` (arguments, histories and logs of every run); the optimiser and the fallback are part of the code archive.

### S9.2. Case A: NICE optimisation, exact twin and exact checks

Case A is the \(2\times2\times2\) block of Section 5.9, with its graded initial corner parameters (0.252 to 0.535) and its four cut cells of 62% and 25% retained volume, clamped on the face \(y=\min\) and loaded by the consistent traction in \(y\) on the face \(y=\max\). Of its 27 vertices, the 9 in the loaded face are fixed and 18 are design variables; \(V^*=0.8\,V(\boldsymbol\tau^0)=1.03558\). The exact twin repeats the optimisation from the same start with the same settings, geometry regeneration and constraints, but with exact condensation: dense exact condensed matrices of every cell, the assembled solve to a recursive relative residual of \(10^{-10}\), exact sensitivities with the moment derivatives of Eq. (H.6), and a cold start. The NICE run was checked with the exact model at iterations 0, 12 and 23. At iteration 0 the design is that of Section 5.9, and the compliances, 24.59081 (NICE) and 24.59356 (exact), are the \(y\)-load values of Table ST17e.

The NICE run stopped after 24 iterations and the twin after 23, both by the objective-change rule (Tables ST22a and ST22c). Both paths first thin the block to the volume bound, the NICE compliance rising from 24.59 to 37.86 at iteration 4, and then redistribute material. Up to iteration 4 the two runs analyse the same design: while the volume bound is violated, every free parameter moves by the move limit or to the lower bound. At these iterations the difference of the two compliances is the surrogate error on the same design, which grows from −0.011% to −0.023% as material is removed. From iteration 5 the paths separate, with corner parameters differing by at most \(8.8\times10^{-4}\), and the NICE compliance lies 0.016–0.030% below the twin's, except at iterations 16 and 19, where the fallback perturbed the NICE design (−0.037% and −0.10%, the latter with the volume 0.036% above the bound). Both final designs reach the lower bound 0.18, have a largest parameter of 0.624, and have the gradient-norm constraint active (0.4500) and the span below its limit (0.444). Their corner parameters differ by at most 0.0051 (root mean square 0.0011), and the exact compliance of the NICE design, 24.110504, is \(1.25\times10^{-5}\) below that of the twin's design, 24.110806. The final compliance is 2.0% below that of the initial design, which holds 25% more material.

In the exact checks (Table ST22b), the surrogate error of the NICE compliance is −0.0112%, −0.0182% and −0.0279% at iterations 0, 12 and 23, and the error of the vertex gradient 0.069%, 0.17% and 0.33%, with cosines of at least 0.9999966 and the signs of all 18 components correct; the component errors are at most \(4.1\times10^{-3}\) of the largest gradient component (95th percentile at most \(2.8\times10^{-3}\)). Both errors increase along the path and remain below 0.03% and 0.33%. Over all iterations the recomputed residual lies between \(5.3\times10^{-5}\) and \(3.2\times10^{-4}\) and the signed residual work \(\lvert\bar U^T\rho\rvert\) is at most \(2.7\times10^{-8}\) of the compliance, as in Section 5.9. Table ST22b also lists a KKT residual computed with the exact gradient, \(\lVert s_g+\lambda\nabla V\rVert/\lVert s_g\rVert\) over the parameters strictly inside the bounds, with the volume multiplier \(\lambda\) fitted by least squares. It omits the multipliers of the span and gradient-norm constraints; since the gradient-norm constraint is active at the final design, its value there, 0.249, overstates the stationarity residual and is not a measure of optimality.

A NICE iteration took 141 s on average (128–171 s), 3,380 s for all 24 iterations together, with peaks of 15.7 GiB of GPU and 3.5 GiB of host memory. The longest iteration and the largest number of conjugate-gradient iterations, 170.7 s and 170 at iteration 16, occur where the solve started from zero and the geometry needed a second generation attempt (Tables ST22a and ST24a). The twin, which forms the dense exact condensed matrices of every cell on the same GPU, took 469 s per iteration and 10,793 s for its 23 iterations, with 17.3 GiB of GPU and 80.0 GiB of host memory. The twin is a verification run; its times are given for completeness, and the cost comparison of the exact and learned routes is that of Section 5.10.

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

The plates are a single layer of \(8\times4\) cells, cut by a plane that runs from the bottom-right corner to the top edge at two cells from the left end. Of the 32 cells, 24 remain: 16 uncut and 8 cut, two each with retained volume fractions 0.917, 0.667, 0.333 and 0.083. The layout is stored with its two in-plane axes interchanged, a cube-symmetry image of the plate, so that the cut normal lies at \(\vartheta=33.69^\circ\), within the range \(0<\vartheta<\pi/4\) of the validation geometries (Section 5.1), rather than at its image, 56.31°; the network is not equivariant under the cube symmetries (Section 6.4). The plate is clamped along its long edge of 8 cells and loaded on the opposite face, which after the cut is two cells wide, by a consistent traction of unit resultant: in the plane along the long edge (B1) or normal to the plate, bending the single layer (B2). Of the 74 vertices, the 10 in the plane of the loaded face are fixed and 64 are design variables; the start is uniform, \(\tau=0.40\), and \(V^*=0.8\,V(\boldsymbol\tau^0)=3.65526\).

B1 stopped after 23 iterations by the objective-change rule and B2 after 30 by the parameter-change rule (Table ST23a). With 20% less material than the uniform start, the NICE compliance falls by 7.7% for B1 (94.341 to 87.089) and by 8.3% for B2 (1,467.743 to 1,346.354). Both final designs span the full parameter range [0.180, 0.690], with the span and gradient-norm constraints active. An iteration took 574 s (B1) and 639 s (B2) on average, most of it in the conjugate-gradient solve (294 and 362 s, 138–295 iterations) and the sensitivities (160 and 159 s); the iterations took 3.7 and 5.3 h in total, with at most 24.3 GiB of GPU and 5.0 GiB of host memory. The recomputed residual reached \(3.8\times10^{-3}\) (B1) and \(6.7\times10^{-3}\) (B2), while the signed residual work was at most \(3.1\times10^{-7}\) of the compliance. All plate values are NICE values; their exact verification is pending (S9.6).

For comparison with a homogenisation-based graded design, the effective elasticity tensor \(C^H(\tau)\) of the uncut cell with uniform corner parameter was computed by periodic homogenisation on the same discrete model (\(n=32\), Q2 elements, the stabilised stiffness \(K\) with its ghost penalty). Nodes on opposite faces of the cell are identified, the fluctuation is periodic with one node fixed, and six unit macroscopic strains give \(C^H_{ij}=u_i^TKu_j\) for the unit cell; the material volume fraction \(V^H(\tau)\) is the sum of the zeroth element moments. At twelve thicknesses from 0.18 to 0.70 (Table ST23b), the tensor is cubic to \(3.1\times10^{-13}\) and the periodic fluctuations are in equilibrium to \(2.2\times10^{-13}\); cubic splines in \(\tau\) interpolate \(C^H_{11}\), \(C^H_{12}\), \(C^H_{44}\) and \(V^H\) (Figure S06). The macroscale model meshes the plate with Q1 hexahedra, six per cell and axis (4,464 elements). At every quadrature point it interpolates the local parameter \(\tau(\mathbf x)\) trilinearly from the corners of its cell and evaluates \(C^H(\tau(\mathbf x))\) and \(V^H(\tau(\mathbf x))\); it integrates the cut by a finite-cell rule (\(4^3\) sub-points in intersected elements, the void part carrying \(10^{-6}\,C^H(0.4)\)); and it applies the same clamped face and a uniform traction of unit resultant on the material part of the loaded face. It was optimised with adjoint sensitivities and the same MMA settings, constraints, fixed vertices and volume fraction. At the uniform design its volume agrees with the fine-scale material volume to \(1.2\times10^{-7}\).

At the uniform design the macroscale model underestimates the fine-scale NICE compliance by 26.7% for the in-plane load and by 36.6% for bending (Table ST23c). The plate, one cell thick and four cells wide, offers little separation of scales, most of all through its thickness; the error was not analysed further. The macroscale optimisations stopped after 23 and 27 iterations. Evaluated with NICE on the fine scale, the homogenisation design Hom-\(y\) has a compliance 1.07% above that of B1, with a volume 0.040% above the bound, and Hom-\(z\) one 0.34% below that of B2, with a volume 0.045% below the bound; for these graded designs the macroscale volume model no longer equals the fine-scale material volume exactly. The corner parameters of the B and Hom designs have correlation coefficients of 0.906 (\(y\)) and 0.969 (\(z\)) and differ by 0.074 and 0.045 in root mean square. Continued with NICE for 12 iterations under the absolute volume bound of B1 and B2, the homogenisation designs reach 87.141 (X-\(y\), 0.060% above B1) and 1,334.697 (X-\(z\), 0.87% below B2). Neither continuation met the stopping rule; the last parameter change of X-\(y\) was still close to the move limit (0.0254 of 0.0255). For each load the three final designs thus lie within 1.1% (\(y\)) and 0.9% (\(z\)) of each other in NICE compliance, and for bending the optimisation from the uniform design ended 0.87% above the design reached from the homogenisation start. In the exact model the bending comparisons hold to within 0.002 percentage points (S9.6, Table ST27). Differences of a few hundredths of a percent, such as the 0.060% between X-\(y\) and B1, are comparable to the surrogate errors of the checked plate designs (0.013–0.038%); the in-plane designs were not checked, and their ranking is left open.

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

Prediction error: \((C_{\rm macro}-\widehat C)/\widehat C\) at the uniform design \(\tau=0.40\). B: B1 or B2; Hom: Hom-\(y\) or Hom-\(z\); X: X-\(y\) or X-\(z\). Relative differences of the final compliances as stated, e.g. Hom vs B \((\widehat C_{\rm Hom}-\widehat C_B)/\widehat C_B\). Corner correlation and RMS difference over the \(24\times8\) cell-corner parameters. The macroscale volume of the uniform design, 4.569075, agrees with the fine-scale material volume 4.569074 to 1.2e-07.

### S9.4. Geometry-generation perturbations and discrete switches

Table ST24a lists the five applications of the geometry-generation fallback. They occurred in the uncut cell 010 of case A at iterations 16 and 19; in both plates at iteration 5, in the cut cell 060 (retained volume 0.917) and, after the perturbation by \(-10^{-4}\), also in the cut cell 160 (0.083); and in the uncut cell 200 of X-\(z\) at iteration 2. The exact twin, whose path differs slightly from the NICE path, needed none. Each perturbed design was generated after one or three perturbations of the fallback, changing a corner parameter by at most \(5.8\times10^{-4}\). At the first failures in case A, B1 and B2, the earlier cell-local scaling of the failing cell by \(1+\epsilon\) (\(\epsilon=10^{-9}\), \(-10^{-9}\), \(10^{-8}\)) had failed.

Table ST24b counts, between consecutive iterations, the cells whose discrete description changed. With the corner parameters moving by up to the move limit, the active elements, ghost faces, retained coordinates and the network's binary node features change in many cells at every step: in the median step all 8 cells of case A, 20 of the 24 cells of B1 and 17 of B2 change at least one of the counts, and never fewer than 3, 16 and 10; in the cross-starts the median is 17 and 9 cells. The coarse-factor shift changed in at most two cells (case A) and four cells (plates) per step. Each iteration therefore evaluates the objective on a different discrete model, and the derivatives of Section 3.3 hold only within each model. Across such switches the discrete reference compliance itself can jump, a non-smoothness that exact condensation meets in the same way, and the surrogate adds at most its own error level (Section 3.3, Supplementary Note S6.3). In case A the NICE and exact compliances differ by 0.011–0.023% on the common design of iterations 0–4 and, after the paths separate, by 0.016–0.030% at the unperturbed iterations (Table ST22a).

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

Three checks, each over the first one to three iterations of case A, support the analysis route of S9.1 (Table ST25). (i) The implementation of the optimisation runs, with the correction in single precision, gives the compliance of the implementation used for the accuracy results of Sections 5.2–5.9, with the correction in double precision, to \(1.7\times10^{-6}\), with the same conjugate-gradient iteration counts and a vertex gradient within \(1.8\times10^{-4}\). (ii) Reverse-mode differentiation of the moment integrals reproduces the vertex gradient obtained from the central moment differences of Eq. (H.6) to \(4.6\times10^{-7}\), in 47 s instead of 95 s. (iii) The warm start reduces the conjugate-gradient iterations from 127 to 102 and from 132 to 108 at iterations 1 and 2, by 18–20%, and the solve time by 17–18%, with compliances equal to \(2.1\times10^{-8}\); 99.98% and 99.995% of the free retained coordinates were found in the previous solution.

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

Each comparison runs the case A optimisation for up to three iterations with both settings. Deployed route: the implementation of the optimisation runs (Table ST21), network and correction in single precision, cold start, sensitivities from central moment differences (Eq. (H.6)) unless stated; the warm-start run uses reverse-mode sensitivities. Double-precision correction route: the implementation of the accuracy results of Sections 5.2–5.9, with the correction in double precision and operator state held in host memory between applications (peak host memory 30.7 GiB, against 3.4 GiB for the deployed route); the two routes therefore differ in implementation, correction precision and operator placement, and the time of the second is not a cost baseline. Vertex-gradient difference: \(\lVert\widetilde s_{g,1}-\widetilde s_{g,2}\rVert/\lVert\widetilde s_{g,2}\rVert\) over all vertices. Matched: fraction of free retained coordinates found in the previous solution; scale: energy-optimal factor of the warm start. At iteration 0 the warm-start run has no previous solution.

### S9.6. Scale demonstration and exact verification of the plates

**Storage of the streamed cell state.** For lattices whose operator state does not fit on the GPU, the state of the remaining cells is streamed from host memory (Supplementary Note S5). For the scale demonstration it is stored packed without loss: the lower-triangular coarse factors as their triangles and the integer index arrays as 32-bit instead of 64-bit integers. On the \(2\times2\times2\) block with every cell streamed, packing reduces the streamed state from 4.50 to 3.45 GiB (23%) and needs 0.40 GiB more peak GPU memory; the conjugate gradients take 129 iterations in both cases. Packing stores the same values; the compliances under the three face loads differ by at most \(1.1\times10^{-8}\) and the sensitivities by \(5.1\times10^{-7}\), at the level of the nondeterministic summation order of the GPU reductions.

**Scale demonstration.** Plates with the proportions of B1 and B2 (short side : long side 1 : 2, one cell thick), the planar cut of B1 and B2 scaled with the plate, and the clamp and in-plane load of B1 were generated with 24, 51, 88, 110 and 135 cells (Table ST26); all cut cells have the four retained volume fractions of B1 and B2. Each run starts from the uniform \(\tau=0.40\) with \(V^*=0.8\,V(\boldsymbol\tau^0)\) and the settings of Table ST21, and was stopped after four analyses (three MMA updates); the 51-cell run has two analyses. Cells are held on the GPU up to 4 GiB of operator state and streamed from host memory beyond it, with the packed storage above; the assembled retained stiffness \(K_{PP}\) of the preconditioner (Supplementary Note S6.1) is factorised on the GPU with NVIDIA cuDSS. The 24-cell plate is plate B1: its compliances at the first four iterations agree with those of B1 (94.341 at the start and 129.69 at iteration 3). Degrees of freedom are those of the initial design: the cut finite-element model of all cells, with nodes on shared faces counted once, and the retained coordinates of the assembled lattice without the clamped face. GPU memory is the largest memory in use on the device, sampled every 30 s with nvidia-smi while the run was the only process on the GPU; it includes the cuDSS factor, which the allocator statistics of PyTorch (13.3–20.1 GiB for the same runs) do not. Host memory is the peak resident memory of the main process; the geometry generation runs in separate processes. Times include geometry generation for every cell and, as for all optimisation runs, use no fused integration kernels (S9.1). Time per design iteration grows from 534 s at 24 cells to 2,606 s at 110 cells, about in proportion to the number of cells (22.2 s to 24.2 s per cell), with the conjugate gradients taking 46–48% of it; the number of conjugate-gradient iterations stays between 138 and 174. At 110 cells the device memory was fully in use (31.4 of 31.4 GiB). At 135 cells the factorisation of \(K_{PP}\) failed in its analysis phase for lack of device memory; the run stopped before its first solve. At the 23 to 30 iterations taken by B1 and B2, an optimisation of the 110-cell plate would take about 17 to 22 h; this is an extrapolation, not a measured run. The accuracy of NICE at these sizes is not verified against the exact reference.

**Table ST26. Scale demonstration.** Plates with the geometry and load of B1, uniform start, 4 GiB GPU budget for resident cell operators. DOFs: degrees of freedom of the cut finite-element model / free retained coordinates of the assembled lattice (initial design). Time: mean over the analyses (range), with mean phase times, in s. Residual: largest recomputed relative residual / largest \(|\bar U^T\rho|/\widehat C\). Memory: peak GPU memory in use (nvidia-smi) / peak host memory of the main process, GiB. Records: `evidence/opt/scale/scale_summary.json`, `evidence/opt/scale/scale_dofs.json`.

\(^{a}\) Largest 30-s samples before the failure; host memory from the RSS samples.

| Cells / cut | DOFs: cut model / free retained | Design variables | Analyses | Time per design iteration (s) and phases | PCG iterations | Residual / residual work | Memory, GPU / host (GiB) |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| 24 / 8 | 6.51 M / 0.388 M | 64 | 4 | 534 (505–566): geometry 22, front end 78, preconditioner 24, PCG 247, sensitivities 160 | 138–169 | 3.8e-04 / 2.9e-08 | 18.2 / 14.7 |
| 51 / 12 | 14.57 M / 0.780 M | 128 | 2 | 1,215 (1,177–1,254): geometry 41, front end 182, preconditioner 54, PCG 574, sensitivities 358 | 156–168 | 5.8e-04 / 2.2e-08 | 20.0 / 32.7 |
| 88 / 16 | 25.85 M / 1.305 M | 212 | 4 | 2,126 (2,097–2,178): geometry 52, front end 319, preconditioner 92, PCG 1,022, sensitivities 633 | 158–174 | 9.6e-04 / 5.2e-08 | 28.9 / 62.5 |
| 110 / 18 | 32.70 M / 1.618 M | 262 | 4 | 2,606 (2,524–2,745): geometry 59, front end 407, preconditioner 116, PCG 1,210, sensitivities 800 | 142–172 | 1.2e-03 / 2.2e-08 | 31.4 / 78.4 |
| 135 / 20 | 40.32 M / 1.963 M | 316 | 0 | Failed in the analysis phase of the \(K_{PP}\) factorisation on the GPU (cuDSS: allocation failed) | — | — | 24.1 / 83.8\(^{a}\) |

**Exact verification of the plate designs.** Six plate designs were analysed with exact condensation as in Table ST22b: B1 and B2 at the uniform start and at their final designs, the latter with the exact vertex gradient (central moment differences, Eq. (H.6)), and the bending designs Hom-\(z\) and X-\(z\) (Table ST27). The dense exact condensed matrices of the 103 distinct cells were formed with PARDISO's Schur-complement option and verified column-wise against interior solves (16 columns per cell, relative difference at most 1.4e-15, asymmetry at most 7.2e-17); the assembled systems were solved to a recursive relative residual of \(10^{-10}\). The surrogate error is negative in all six designs, between −0.013% and −0.038%, against −0.011% to −0.028% in case A; it is largest for the bending designs. At the final designs of B1 and B2 the gradient errors, 0.22% and 0.32%, and the component errors (95th percentile 0.26% of the largest component) match those of case A, and every one of the 64 components has the exact sign. Relative to the exact compliance of the uniform start, the macroscale model underestimates the compliance by 26.7% (in plane) and 36.6% (bending), as relative to NICE. In the exact model, Hom-\(z\) lies 0.34% and X-\(z\) 0.87% below B2 and X-\(z\) 0.53% below Hom-\(z\) (NICE: 0.34%, 0.87%, 0.53%), and the reductions from the uniform start are 7.68% (B1) and 8.26% (B2) (NICE: 7.69%, 8.27%). The in-plane designs Hom-\(y\) and X-\(y\) and the intermediate designs were not checked. The KKT residual of Table ST22b is not reported for the plates, since the gradient-norm constraints active at the final designs are not included in it.

**Table ST27. Exact checks of the plate designs.** Exact condensation of every cell (Schur-complement route, column-verified), assembled solve to a recursive relative residual of \(10^{-10}\). Surrogate error: \(\widehat C/C-1\). Gradient: vertex gradient with exact sensitivities (central moment differences) against the field-based NICE estimate; final designs of B1 and B2 only. Records: `evidence/opt/exact_plates/runs/*/check_exact_*.json`, summary `evidence/opt/exact_plates/EXACT_PLATES.json`.

| Design | Iteration | Exact \(C\) | \(\widehat C\) | Surrogate error (%) | Gradient error (%) | Cosine | Component error / \(\max\lvert g\rvert\) (%): median / 95th percentile / max | Sign agreement (variables) | Exact PCG iterations / recomputed residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| B1 | 0 (start) | 94.353 | 94.341 | -0.0132 | — | — | — | — | 243 / 9.8e-11 |
| B1 | 22 (final) | 87.109 | 87.089 | -0.0229 | 0.219 | 0.9999982 | 0.066 / 0.260 / 0.287 | 1.000 (64) | 340 / 8.8e-11 |
| B2 | 0 (start) | 1,468.052 | 1,467.743 | -0.0210 | — | — | — | — | 252 / 1.9e-10 |
| B2 | 29 (final) | 1,346.854 | 1,346.354 | -0.0372 | 0.317 | 0.9999954 | 0.052 / 0.259 / 0.335 | 1.000 (64) | 416 / 2.0e-10 |
| Hom-\(z\) | final (macroscale iteration 27) | 1,342.299 | 1,341.784 | -0.0384 | — | — | — | — | 405 / 1.9e-10 |
| X-\(z\) | 11 (last) | 1,335.198 | 1,334.697 | -0.0375 | — | — | — | — | 399 / 1.9e-10 |

## Supplementary figures

![Figure S01](figures/S06_reference_verification.png)

**Figure S01. Verification of the CutFEM reference.** Single cells U1, M1, M2 and H1, clamped on one box face and loaded by unit consistent tractions on another face in the three Cartesian directions. (a) Largest relative change over the three loads of compliance (filled, solid) and eight-corner sensitivity (open, dashed) against the finest background resolution (\(n=40\) for U1, 48 otherwise); production uses \(n=32\). On H1 the successive compliance increments do not yet decrease between \(n=40\) and 48; refinement to \(n=64\) (Table ST14) shows that H1 does not converge monotonically, so the difference from \(n=48\) is not an estimate of the \(n=32\) error. (b) The same quantities when the ghost-penalty coefficient is changed from its production value \(10^{-4}\). (c) Largest relative change of the sensitivity when the finite-difference step of the moment derivatives is changed from its production value \(h=10^{-5}\tau_c\) (filled), and largest relative difference between central compliance differences and the sensitivity at the production step (open). The change is at most \(2.6\times10^{-7}\) at \(10^{-3}\tau_c\) and falls a hundredfold per decade of step. Refinement to \(n=64\) of H1, H2 and two further validation cells: Table ST14. Data: `evidence/ref_valid.json`, `evidence/ref_valid_h1.json`; script `figures_src/fig_refconv.py`.

![Figure S02](figures/S01_distributions.png)

**Figure S02. Distributions of geometry-level directional energy errors.** Each point is one validation geometry's directional mean in the identity view; horizontal bars are population medians. The nodal-force, spring-support, single-face-force, polynomial, multiscale, consistent-traction and single-face consistent-traction classes contain 80 geometries, and the stiffness-scaled support and neighbour-induced displacement classes 75, the same populations as Table ST03. P0 was evaluated only on the first five classes (n/a elsewhere). The base network and Uncorrected use the markers and colours of the main-text figures, S8 a purple diamond; deterministic horizontal offsets separate overlapping observations. All panels share the logarithmic error axis. Data: `evidence/newval_c_oh.json` (P0) and `evidence/newval2_<run>.json` (base network, Uncorrected, S8); script `figures_src/fig_s01_distributions.py`.

![Figure S03](figures/S03_sensitivity_diagnostics.png)

**Figure S03. Field-based sensitivity-error diagnostics.** (a) Paired mean energy and sensitivity errors for consistent-traction and nodal-force responses, using six cells of the base network and five of Uncorrected. (b) Consistent-traction linear-term norm share \(\|D_1\|_F/(\|D_1\|_F+\|D_2\|_F)\), where \(D_1+D_2\) is the sensitivity-error matrix over all eight design components and evaluated directions. (c,d) Shares of absolute elementwise sensitivity-error contributions and element counts in four mutually exclusive material-volume-fraction groups for the base network under consistent tractions. Each error group sums absolute contributions over its elements, design components and directions before normalisation by the total. Filled markers identify the base network and open markers and hatched bars in (a,b) Uncorrected (B and C in the figure legend); the H2 observation of Uncorrected is unavailable.

![Figure S04](figures/S02A_smoothing.png)

**Figure S04. Smoothing from learned and zero internal fields.** (a,b) Mean directional energy excess for consistent-traction and nodal-force responses; (c,d) corresponding field-based sensitivity errors. Both initialisations prescribe the same retained displacement. Solid curves with filled markers start from the base network; dashed curves with open markers start from zero internal displacement. Zero-start sensitivity is recorded only at 32 steps. All corrections use \(a=b/30\). The step axis is linear between zero and one and logarithmic thereafter.

![Figure S05](figures/S02B_coarse_spaces.png)

**Figure S05. Recorded coarse representations and correction sequences.** Rows correspond to U1, M1 and M2; columns use consistent-traction and nodal-force responses. "Network" denotes the base network. Six coarse representations are compared under four initialisation and smoothing sequences, with eight steps in each pre- or post-smoothing stage. Dots indicate directional means and caps the 90th percentile. Dashed and dotted references denote the base network alone and the base network followed by one smoothing stage. \(Q_1\), \(Q_2\) and PU denote trilinear, quadratic and linearly enriched partition-of-unity generating families. The first label number identifies grid resolution and the lower number counts columns after internal restriction and screening. For the structurally redundant PU family, these counts do not establish an independent-space dimension, and the plotted solve results do not verify exact-projection properties. Appendix F.1 and Supplementary Note S3 explain the rank and solve conditions; Table ST07 gives all statistics. Coarse updates preserve every retained coordinate.

![Figure S06](figures/S06_homogenised_law.png)

**Figure S06. Homogenised law of the uniform-thickness cell.** (a) Effective elasticity tensor \(C^H_{11}\), \(C^H_{12}\), \(C^H_{44}\) (Voigt notation, engineering shear strains, \(E_Y=1\), \(\nu=0.3\)), cubic to \(3.1\times10^{-13}\); (b) material volume fraction \(V^H\), the material volume of the unit cell from the zeroth element moments. Markers: periodic homogenisation on the discrete model (\(n=32\), Q2 elements, ghost penalty) at 12 thicknesses from 0.18 to 0.70 (Table ST23b); lines: the cubic splines in \(\tau\) used by the macroscale model. Data: `evidence/opt/FACTS_6_11.json` (homog_law); script `figures_src/fig_opt.py`.
