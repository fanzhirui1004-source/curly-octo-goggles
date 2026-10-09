### 3.3. Chebyshev smoothing

Equation (7) expresses the energy error of a cell through its interior residual, and Eq. (14) expresses the compliance error as the energy of the error of the recovered field. Both suggest reducing the interior force imbalance of the network prediction while the retained displacements are held fixed. Polynomial smoothing [Adams et al. (2003)](https://doi.org/10.1016/s0021-9991(03)00194-3) and energy minimisation on a coarse grid reduce different components of this imbalance.

Let \(D=\operatorname{diag}(A)\) be the diagonal of the interior stiffness matrix \(A\). The generalised eigenvectors \(v_j\) satisfy \(Av_j=\lambda_jDv_j\) and \(v_j^TDv_k=\delta_{jk}\), and they are ordered by increasing eigenvalue. With the coefficients \(c_j=v_j^TDd_I\) of the interior error \(d_I\) of Section 3.1, the error energy and its fraction in the first \(\ell\) modes are

\[
d_I^TAd_I=\sum_j\lambda_jc_j^2,\qquad
\chi_\ell(q)=\frac{\sum_{j=1}^{\ell}\lambda_jc_j^2}{d_I^TAd_I}.
\tag{10}
\]

The fraction \(\chi_\ell\) gives the part of the error energy that lies in the first \(\ell\) Jacobi-scaled interior modes. The corresponding fraction for the exact interior field \(u_I\) is defined in the same way, with \(u_I^TAu_I\) as its total energy.

The network prediction is an initial approximation to the solution of \(Au_I=-K_{IP}q\). Chebyshev semi-iteration with Jacobi preconditioning [Golub & Varga (1961)](https://doi.org/10.1007/BF01386013) improves this approximation with \(q\) held fixed. The number of iterations is prescribed, and the coefficients depend on the geometry but not on \(q\), so the corrected recovery operator remains linear. Smoothing a learned initial approximation in this way resembles the smoothing of a tentative prolongation in smoothed aggregation [Vaněk et al. (1996)](https://doi.org/10.1007/BF02238511). It also resembles hybrid solvers in which relaxation removes the high-frequency error that a learned prediction leaves [Zhang, E. et al. (2024)](https://doi.org/10.1038/s42256-024-00910-x).

After \(k\) smoothing steps, the interior error \(d_I\) becomes \(\Phi_kd_I\), where \(\Phi_k=p_k(D^{-1}A)\) and \(p_k\) is the error polynomial of degree \(k\). Smoothing therefore multiplies each coefficient \(c_j\) in Eq. (10) by \(p_k(\lambda_j)\). The Chebyshev polynomial is constructed for a target interval \([a,b]\) with \(0<a<b\). It does not increase the error in the \(A\)-energy norm if the actual positive spectrum of \(D^{-1/2}AD^{-1/2}\) lies in \((0,b]\). Modes with eigenvalues below \(a\) can decay slowly, which motivates the complementary coarse-grid correction of Section 3.4. The correction sequences in this paper use \(a=b/30\), where \(b\) is a power-iteration estimate of the largest eigenvalue of \(D^{-1}A\), increased by 5%. Section 5.2 verifies that \(b\) bounds the spectrum. Appendix D gives the polynomial, its recurrence, the estimate of the interval and the contraction bound.

### 3.4. Coarse-grid correction and the two-grid cycle

Let \(\widehat u_I\) be the current approximation of the interior displacements, either the network prediction or a smoothed iterate, and let \(r_I=(K\widehat u)_I\) be its interior residual. The coarse basis \(V\in\mathbb R^{i\times n_c}\) is assumed to have full column rank; Appendix F.1 treats a basis without it. With the coarse-grid matrix \(A_c=V^TAV\) and \(b_r=V^Tr_I\), minimising the energy over \(\widehat u_I+\operatorname{range}V\) gives

\[
\widehat u_I^{\rm c}=\widehat u_I-VA_c^{-1}b_r,\qquad
d_I^{\rm c}=(I_i-VA_c^{-1}V^TA)d_I,\qquad
\|d_I\|_A^2-\|d_I^{\rm c}\|_A^2=b_r^TA_c^{-1}b_r.
\tag{11}
\]

The coarse-grid correction enforces interior equilibrium for every virtual displacement in the coarse space. The last relation in Eq. (11) gives the energy removed by this projection [Xu (1992)](https://doi.org/10.1137/1034116). The principal coarse basis, denoted \(Q_1(17)\), consists of trilinear vector functions on a grid of \(17^3\) vertices, restricted to the interior DOFs (Appendix F.1). All coarse spaces act on the interior DOFs only, so the retained displacements stay fixed.

The two-grid cycle places the coarse-grid correction between two smoothing stages of \(k\) steps each. This cycle is the correction \(\mathcal W\) of Section 3.1. Let \(C_V=I_i-VA_c^{-1}V^TA\) be the error operator of the coarse-grid correction, and let \(H_{\rm net}=J_I(\widehat E-E)\) be the interior error of the network recovery operator. The two-grid error operator \(\Phi_kC_V\Phi_k\) [Hackbusch (1985)](https://doi.org/10.1007/978-3-662-02427-0), [Trottenberg et al. (2001)](https://shop.elsevier.com/books/multigrid/trottenberg/978-0-12-701070-0), [Xu & Zikatanov (2002)](https://doi.org/10.1090/S0894-0347-02-00398-3), [Falgout et al. (2005)](https://doi.org/10.1002/nla.437) maps \(H_{\rm net}\) to the interior error of \(F\). By Eq. (6), the error of the condensed stiffness is then

\[
H=\Phi_kC_V\Phi_kH_{\rm net},\qquad
\widehat S-S=H^TAH.
\tag{12}
\]

**Proposition 2 (correction ordering).** Let the network recovery operator \(\widehat E\) satisfy (A2), let the coarse solve be exact or use the coarse inverse with the nonnegative shift of Appendix F.1 (Eq. (D.6)), and let \(\|\Phi_k\|_A\le1\). Then, under (A1), \(S\preceq\widehat S\preceq\widehat S_{\rm net}\), where \(\widehat S_{\rm net}=\widehat E^TK\widehat E\) is the condensed stiffness without correction (Appendix D.1). The same ordering holds for any linear correction that is fixed for a given geometry and whose interior error map \(\Theta\) satisfies \(\Theta^TA\Theta\preceq A\).

The correction therefore cannot increase the error in the \(A\)-energy norm. The assembled stiffness matrices inherit the ordering of Proposition 2, and their inverses are ordered in reverse (Appendix C). As a result, the correction moves the assembled compliance towards the reference value (Appendix D.1). Proposition 2 guarantees the ordering but not the size of the reduction (Appendix D). Section 5.5 measures this reduction.

**Remark 1.** The guarantee of Proposition 2 rests on the two conditions on the smoothing and on the coarse solve, and it covers the energy only. If the spectrum of \(D^{-1}A\) extends above \(a+b\), the Chebyshev polynomial amplifies these modes, and the cycle can increase the energy error (Appendix D). Changing the number \(k\) of smoothing steps changes the polynomial. The ordering therefore holds for each \(k\) separately, and the reduction need not be monotone in \(k\). For an approximate coarse inverse \(G\), the correction does not increase the energy error if \(2G-GA_cG\succeq0\) (Eq. (D.6)). The nonnegative shift satisfies this condition, whereas an arbitrary approximation need not. The ordering does not apply to the sensitivity. In the example of Appendix D.1, a projection halves the energy of the interior error. The same projection changes the sensitivity difference \(\widetilde s_c-s_c\) of Eq. (17) from zero to \(2t-t^2\), whereas the reference sensitivity is \(-1\).

### 3.5. Training

The network is trained on sets of test displacements. Prescribed polynomial and multiscale displacements cover a broad range of spatial variation. Responses to equilibrated nodal point loads, traction loads and spring supports, together with displacements imposed by a neighbouring cell, provide test displacements that arise from mechanical loading. The rigid-body components are removed, and each test displacement is normalised with the exact condensed stiffness to \(q_j^TSq_j=1\). For a batch of \(B\) test displacements, the loss function is

\[
\mathcal L(\theta)=\frac1B\sum_{j=1}^{B}\log(q_j^T\widehat S q_j)
 +w_s\frac1{|\mathcal J_s|}\sum_{j\in\mathcal J_s}
 \frac{\|\widetilde{\boldsymbol s}_j-\boldsymbol s_j\|_2^2}{\|\boldsymbol s_j\|_2^2}.
\tag{13}
\]

Because \(q_j^TSq_j=1\), the first term is the mean logarithm of the energy ratio \(q_j^T\widehat Sq_j/(q_j^TSq_j)\). By Proposition 1, for a recovery operator that reproduces the retained displacements, this term is smallest when the recovered field equals the exact field for every sampled test displacement. The second term compares the thickness sensitivities \(\widetilde{\boldsymbol s}_j\) computed from the recovered field (Section 4.2) with the reference sensitivities \(\boldsymbol s_j\). Both are eight-component vectors, and the term is evaluated on the subset \(\mathcal J_s\) of test displacements that have reference sensitivities. All trained networks reported in this paper use \(w_s=1\). The gradients of the loss with respect to the network parameters \(\theta\) pass through the recovered field to the geometry encoders, the coefficient heads and the linear displacement maps. Test displacements with large energy ratios are added during training. They are found by a block search on the complement of the rigid-body modes, which iterates eight candidate test displacements at once (Appendix G.3). The geometries are augmented with the 48 symmetries of the cube (Appendix G). Table 2 lists the variants, and Supplementary Table ST01 gives the settings for training and model selection.

NICE is trained with the operator that is used at deployment. The loss of Eq. (13) is evaluated with the corrected recovery operator \(F=\mathcal W\widehat E\), where \(\mathcal W\) is the two-grid cycle of Table 1. This cycle is written 8 / \(Q_1(17)\) / 8, which denotes 8 pre-smoothing steps, the coarse-grid correction with \(Q_1(17)\) and 8 post-smoothing steps. Because \(\mathcal W\) is linear and fixed for a given geometry, the gradient passes through the smoothing recurrence and the coarse-grid correction to the network parameters, as in solver-in-the-loop training [Um et al. (2020)](https://proceedings.neurips.cc/paper/2020/hash/43e4e6a6f341e00671e123714de019a8-Abstract.html). The smoothing interval and the factorisation of the coarse-grid matrix depend only on \(K\) and are recomputed for each geometry. The correction therefore adds no trainable parameters.

### 3.6. Applying the condensed stiffness

The quantities that depend on the geometry are computed once for each geometry and reused for all retained displacements. They are the element moments, the geometry encoding of the network with its coefficients and maps, the smoothing interval and the factorisation of the coarse-grid matrix. Supplementary Table ST13 gives the cost of this preparation. Each product of \(\widehat S\) with a vector of retained displacements then requires one forward pass of the network, one two-grid cycle and one product with \(K\). It also requires the transposes of the cycle and of the network (Appendices E–F). Neither \(\widehat E\) nor \(\widehat S\) is formed as a matrix. Each cell stores its geometry-dependent coefficients and the factorisation of its coarse-grid matrix. By comparison, a dense \(\widehat S\) for a cell with the median number of retained DOFs, 23,604, would take 4.2 GiB in double precision.

**Algorithm 1. Action of the corrected condensed stiffness.**

1. Compute \(\widehat Eq\) with the network, and apply the two-grid cycle \(\mathcal W\) with the retained displacements held fixed to obtain \(\widehat u=Fq\).
2. Compute \(y=K\widehat u\) and return \(F^Ty\) by applying \(\mathcal W^T\) and then the transpose of the network recovery operator.
3. Assemble these cell actions through Eq. (3). After the global solve, recover the cell displacements \(F_mB_m\widehat U\) to evaluate the displacement field and the sensitivities.
