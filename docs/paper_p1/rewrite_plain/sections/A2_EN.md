## Appendix D. Polynomial smoothing and coarse projections

For \(0<a<b\), the degree-\(k\) Chebyshev error polynomial and its energy bound are

\[
p_k(t)=\frac{T_k((a+b-2t)/(b-a))}{T_k((a+b)/(b-a))}.
\tag{D.1}
\]

\[
d_I^{\rm sm}=p_k(D^{-1}A)d_I,\qquad
\|d_I^{\rm sm}\|_A^2\le\psi_k^2\|d_I\|_A^2,\qquad
\psi_k=\max_{\lambda\in\sigma(D^{-1/2}AD^{-1/2})}|p_k(\lambda)|.
\tag{D.2}
\]

Here \(D=\operatorname{diag}(A)\).

Write \(\bar\lambda=(a+b)/2\), \(\eta_s=(b-a)/2\), and \(\zeta=\bar\lambda/\eta_s>1\) for the centre and half-width of the smoothing interval and their ratio. Starting from \(u_0\), keep its retained entries fixed and define the interior residual with the sign convention \(r_j=(Ku_j)_I\). A recurrence producing the polynomial in Eq. (D.1) is

\[
\begin{aligned}
\varrho_0&=1/\zeta,& z_0&=-D^{-1}r_0/\bar\lambda,\\
u_{j+1}&=u_j+J_I^Tz_j,&
\varrho_{j+1}&=(2\zeta-\varrho_j)^{-1},\\
z_{j+1}&=\varrho_{j+1}\varrho_j z_j
 -\frac{2\varrho_{j+1}}{\eta_s}D^{-1}r_{j+1}.
\end{aligned}
\tag{D.3}
\]

The last line is evaluated only when another step is needed. Zero steps return the initial field. To verify the polynomial, observe that \(\varrho_j=T_j(\zeta)/T_{j+1}(\zeta)\), and apply \(T_{j+2}(x)=2xT_{j+1}(x)-T_j(x)\). After the first step the error is \(d_1=(I-D^{-1}A/\bar\lambda)d_0\), and the same recurrence gives \(d_k=p_k(D^{-1}A)d_0\).

The matrices \(D^{-1}A\) and \(D^{-1/2}AD^{-1/2}\) are similar, \(p_k(D^{-1}A)\) is self-adjoint in the \(A\)-inner product, and

\[
\|p_k(D^{-1}A)d\|_A^2
=\sum_j\lambda_j p_k(\lambda_j)^2c_j^2,
\qquad c_j=v_j^TDd.
\tag{D.4}
\]

This proves Eq. (D.2). If \(a\le\lambda\le b\), the argument of \(T_k\) lies in \([-1,1]\), where \(|T_k|\le1\). For \(0<\lambda<a\), it lies between 1 and \(\zeta\), where \(T_k\) increases from 1 to \(T_k(\zeta)\). Hence \(|p_k(\lambda)|\le1\) throughout \((0,b]\). For \(b<\lambda\le a+b\) the argument lies in \([-\zeta,-1)\), where \(|T_k|\le T_k(\zeta)\), so \(|p_k(\lambda)|\le1\) on \((0,a+b]\), with equality at \(a+b\). The polynomial therefore does not increase the error in the \(A\)-energy norm when the actual positive spectrum is contained in \((0,a+b]\), and in particular when it is contained in \((0,b]\). Modes below \(a\) can contract slowly, and eigenvalues above \(a+b\) are amplified.

The operational upper endpoint is estimated using a random power vector, 40 iterations of \(D^{-1}A\), Euclidean normalisation, and a factor of 1.05; the interval is computed once per geometry. The estimate sets the polynomial, while the result in Eq. (D.2) is conditional on the actual spectrum. A finite power estimate multiplied by a safety factor only estimates the endpoint and does not establish the containment used above. For the 80 validation geometries the operational endpoint was therefore compared with \(\lambda_{\max}(D^{-1}A)\) computed by Lanczos iteration with full reorthogonalisation on \(D^{-1/2}AD^{-1/2}\) (78–150 steps; relative Ritz residual at most \(9.6\times10^{-4}\), median \(2.2\times10^{-6}\)). The operational endpoint exceeds the converged value by 2.1–5.0% (median 3.9%) on every geometry, which supports the containment required by Eq. (D.2) on all evaluated cells. A converged Ritz value is a numerical estimate, not a certified bound. The strict Gershgorin bound \(\max_i\sum_j|A_{ij}|/A_{ii}\) is 4.7–19.4 times the operational endpoint (median 18.4); used as \(b\), it would stretch the smoothing interval by that factor and slow the contraction of the upper spectrum accordingly.

For a full-column-rank coarse basis \(V\), with \(A_c=V^TAV\succ0\), define \(Q_V=VA_c^{-1}V^TA\). Direct multiplication gives \(Q_V^2=Q_V\) and \(Q_V^TA=AQ_V\). Thus \(Q_V\) is an \(A\)-orthogonal projection and \(C_V=I-Q_V\) is its complementary projection. With \(b_r=V^TAd\),

\[
\|d\|_A^2=\|C_Vd\|_A^2+\|Q_Vd\|_A^2,
\qquad\|Q_Vd\|_A^2=b_r^TA_c^{-1}b_r.
\tag{D.5}
\]

This proves Eq. (11). Applying Eq. (D.2) before and after this projection yields \(\|\Phi_kC_V\Phi_kd\|_A\le\psi_k^2\|d\|_A\). Because \(\psi_k<1\) whenever the positive spectrum lies in \((0,b]\), the cycle is an energy contraction, which gives the operator ordering stated in Section 3.4 (Appendix D.1). For the spectra of U1, M1 and M2 (\(\lambda_1\approx4\)–\(9\times10^{-4}\) against \(a\approx0.17\); Supplementary Table ST05b), however, the energy contraction factor \(\psi_8^4\) of the cycle is 0.97–0.99. The estimate therefore guarantees little more than that the cycle does not increase the energy error. It uses no approximation property of the coarse space, and the reductions by one to three orders of magnitude in Section 5.5 are empirical. A two-grid convergence estimate would require an approximation property of the \(Q_1\) space for walls about one element thick [Xu & Zikatanov (2002)](https://doi.org/10.1090/S0894-0347-02-00398-3). This is the standard energy-projection mechanism of subspace correction [Xu (1992)](https://doi.org/10.1137/1034116).

### D.1. Corrections, orderings and approximate coarse inverses (proof of Proposition 2)

A linear correction that is fixed for a given geometry acts on \(H\) through \(H_{\rm new}=\Theta H\). If \(\Theta^TA\Theta\preceq A\), then

\[
S\preceq\widehat S_{\rm new}\preceq\widehat S_{\rm old}.
\]

Thus the exact assembled compliance increases monotonically towards the reference, and the total reconstructed energy error of Eq. (14) decreases; in particular, repeating a fixed cycle cannot increase the energy error. Changing the smoothing degree, or stopping the recurrence of Eq. (D.3) at an intermediate step, changes the polynomial, so the energy error need not decrease monotonically in \(k\). The sensitivity error is not ordered by this matrix inequality. For example, take \(K=I_3\), \(P=\{1\}\), \(u=(1,0,0)^T\) and the positive semidefinite derivative \(K_{,c}=vv^T\), \(v=(1,1,-1)^T\), of the nested family \(K(\tau)=I+\tau vv^T\). Projecting the second component out of the interior error \(d=(0,t,t)^T\) halves its energy from \(2t^2\) to \(t^2\), whereas the sensitivity discrepancy of Eq. (17) changes from zero to \(2t-t^2\) against a reference sensitivity of \(-1\) (0.19 at \(t=0.1\)).

For a symmetric approximate inverse \(G\) in place of \(A_c^{-1}\) in the coarse-grid correction \(C_V=I-VA_c^{-1}V^TA\) of Eq. (D.5), the energy removed by the correction is

\[
\|d\|_A^2-\|d-VGV^TAd\|_A^2
=b_r^T(2G-GA_cG)b_r,\quad b_r=V^TAd.
\tag{D.6}
\]

Consequently, \(2G-GA_cG\succeq0\) suffices for the correction not to increase the energy error. A nonnegative shift \(G=(A_c+\Lambda)^{-1}\), with \(\Lambda\succeq0\) and \(A_c+\Lambda\succ0\), satisfies this condition, since \(2G-GA_cG=G(A_c+2\Lambda)G\), although it is not the exact projection. Neither Eq. (D.6) nor this condition requires \(V\) to have full column rank. If \(A_c\) is singular because the columns of \(V\) are dependent, a shift \(\Lambda=\xi\operatorname{diag}(A_c)\) with \(\xi>0\) still gives \(A_c+\Lambda\succ0\), because every column kept in \(V\) carries positive energy. This statement concerns the coarse inverse while the prescribed fine-grid stiffness \(K\) remains fixed. If a probed matrix differs from \(A_c\), contraction depends on the approximate inverse relative to the true \(A_c\).

## Appendix E. Transpose of the recovery operator

All transposes below use the Euclidean pairing of the stored displacement and nodal-force vectors. Write \(M_I=J_I^TJ_I\). Transposing Eq. (9) gives

\[
\widehat E^Ty=J_Py+C_R^TR^TM_Iy
 +\Pi_P^T\mathcal N_\theta(\eta)^TM_Iy.
\tag{E.1}
\]

The final overwrite of the retained values therefore keeps the direct retained contribution and masks the field sent through the network transpose. The geometry coefficients are held fixed during this displacement transpose.

Set \(\varphi_k(t)=[1-p_k(t)]/t\), with the polynomial continuation at zero. The full-field smoothing map \(\mathcal T_k\) has the block representation

\[
\mathcal T_k=
\begin{bmatrix}
I_p&0\\
-\varphi_k(D^{-1}A)D^{-1}K_{IP}&p_k(D^{-1}A)
\end{bmatrix},\qquad
\mathcal T_k^Ty=
\begin{bmatrix}
y_P-K_{PI}\varphi_k(D^{-1}A)D^{-1}y_I\\
D p_k(D^{-1}A)D^{-1}y_I
\end{bmatrix}.
\tag{E.2}
\]

The final interior field equals its equilibrium value plus \(p_k(D^{-1}A)\) times the initial error. Replacing \((I-p_k(D^{-1}A))A^{-1}\) by \(\varphi_k(D^{-1}A)D^{-1}\) yields the first expression. The second follows from \((D^{-1}A)^TD=A=D(D^{-1}A)\), which implies \(p_k(D^{-1}A)^T=Dp_k(D^{-1}A)D^{-1}\) and the corresponding identity for \(\varphi_k\). Although the forward map leaves the retained displacements unchanged, its transpose contributes a force on the retained DOFs through the upper block in Eq. (E.2).

For the full-field coarse-grid correction,

\[
\mathcal W_c=I_{n_a}-J_I^TVA_c^{-1}V^TJ_IK,
\qquad
\mathcal W_c^T=I_{n_a}-KJ_I^TVA_c^{-1}V^TJ_I.
\tag{E.3}
\]

A cycle \(\mathcal W=\mathcal T_{\rm post}\mathcal W_c\mathcal T_{\rm pre}\) therefore uses the reverse sequence \(\mathcal W^T=\mathcal T_{\rm pre}^T\mathcal W_c^T\mathcal T_{\rm post}^T\). The resulting reaction is

\[
\widehat S q=\widehat E^T\mathcal W^TK\mathcal W\widehat E q.
\tag{E.4}
\]

The same formulas hold for the symmetric shifted inverse of Appendix F.1 when it replaces the coarse inverse consistently in both actions.

## Appendix F. Coarse space, factorisation, and arithmetic

### F.1. Coarse basis and factorisation

The principal coarse basis \(V\) consists of trilinear (\(Q_1\)) vector shape functions on a \(17^3\)-vertex grid over the reference box, with three displacement DOFs per vertex. The shape functions are evaluated at the active background-node coordinates and restricted to the interior DOFs \(I\); columns with an absolute interior-support sum at most \(10^{-14}\) are removed. Alternative coarse spaces (\(Q_1\) on \(9^3\) and \(33^3\) vertices, \(Q_2\) on \(17^3\) nodes and enriched partition-of-unity spaces) are compared in Supplementary Table ST06.

The coarse matrix \(V^TAV\) is recovered from coloured stiffness probes. With a stencil reach of four fine-grid node spacings and \(h_g\) fine-node spacings per coarse-vertex spacing, the probe radius is \(R_g=2+\lceil4/h_g\rceil\). Colours use vertex coordinates modulo \(2R_g+1\) and the displacement component, and the resulting colour separation resolves every interacting coarse pair. The recovered matrix is Jacobi scaled, with the corresponding column scaling of \(V\), and factorised by Cholesky, trying diagonal shifts \(\xi\) in the order \(0,10^{-12},10^{-10},10^{-8},10^{-6},10^{-4}\). The selected value, the coarse-factor shift, is stored with the geometry factorisation. With the correction in single precision, as in the optimisation runs of Section 5.10, a factor is accepted only if its smallest squared pivot is at least \(10^{-7}\) of the unit diagonal. Otherwise the next shift is tried. This keeps the forward and transposed coarse solves consistent for cut cells near the lower thickness bound. A shift replaces the exact projection by the approximate inverse \(G_s=(A_s+\xi I)^{-1}\) of the scaled Galerkin matrix \(A_s\). In the unscaled basis this is the shift \(\Lambda=\xi\operatorname{diag}(V^TAV)\) of Appendix D.1. The support screen removes only columns that are numerically zero, so \(V\) need not have full column rank. A positive shift then still gives \(A_s+\xi I\succ0\). Since \(2G_s-G_sA_sG_s=G_s(A_s+2\xi I)G_s\succeq0\), Eq. (D.6), with the scaled basis in place of \(V\), shows that the shifted coarse-grid correction still does not increase the \(A\)-energy error, with or without full column rank. This requires the forward and transposed actions to use the same solve.

### F.2. Arithmetic

The network's interior displacements, the rigid-body reconstruction and the prescribed retained entries use single precision (float32). Quadratic energies and sparse stiffness products use double precision (float64), and convolutions use true float32 arithmetic, without TF32 or lower precisions. The correction, including the smoothing and the coarse algebra, runs in double precision in training and in the accuracy studies of Sections 5.2–5.8. It runs in single precision in the timed lattice route of Table 5 and in the optimisations of Section 5.10. The fixed-network correction studies of Section 5.5 (Table 4 and Supplementary Tables ST05a, ST06 and ST07) apply the same correction in double precision to the single-precision network field. For the base network they estimate the smoothing interval from their own power-iteration start, which is also the interval compared with the spectrum in Section 5.4 (Supplementary Table ST05b). Sensitivity contractions use float32 element products with float64 accumulation. These arithmetic choices approximate the real linear maps of the preceding derivations.

