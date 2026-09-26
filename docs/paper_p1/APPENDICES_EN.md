## Appendix A. Discrete construction and metric definitions

### A.1. Active coefficients and element integration

The active background elements use tensor-product \(Q_2\) displacements, with 27 nodes and 81 displacement degrees of freedom per element. Active coefficients are stored in a fixed node-major \(x,y,z\) order. The retained box set is the union of the nine face nodes of every certified positive-area material patch. The cut set is the union of all 27 nodes of each active element carrying a positive-area macro-cut patch. Their union is deduplicated in active-node order. The off-plane cut-band coefficients are retained because their basis functions determine displacement and virtual work on the cut plane.

The discrete stiffness is assembled as

\[
K(\eta)=\sum_e L_e^TK_e(\eta)L_e
 +\gamma\sum_{f\in\mathcal F_g}L_f^TG_f^TG_fL_f,
\qquad
K_e(\eta)=\sum_{\alpha\in\{0,\ldots,4\}^3}M_{e\alpha}(\eta)T_\alpha.
\tag{A.1}
\]

Here \(L_e,L_f\) extract element and ghost-face coefficients. The fixed elasticity templates \(T_\alpha\in\mathbb R^{81\times81}\) use the physical basis derivatives and isotropic Lamé constants \(\lambda_L=E_Y\nu/[(1+\nu)(1-2\nu)]\) and \(\mu_L=E_Y/[2(1+\nu)]\). The 125 moments \(M_{e\alpha}\) integrate local monomials with coordinatewise exponents from zero to four over the material part of an element. The ghost-face set, its factors and coefficient specify the stabilisation contribution. The sum defines the stabilised discrete energy, including its ghost-penalty contribution.

The ghost-face set \(\mathcal F_g\) consists of internal background faces whose two neighbouring elements are active and are not both certified as completely filled with material. These faces lie within each substructure. For background-element width \(h=1/n\), the unit-coefficient stabilisation bilinear form is

\[
g_h(v,u)=(\lambda_0+2\mu_0)\sum_{f\in\mathcal F_g}\sum_{j=1}^{2}
h^{2j-1}\int_f [\partial_{n_f}^{j}v]\cdot[\partial_{n_f}^{j}u] \,\mathrm dA.
\]

The derivative uses a common positive coordinate normal on both neighbouring elements, and the jump is the value on the first element minus that on the second. Integration covers the complete background face. The fixed templates use \(E_0=1\), \(\nu_0=0.3\), and their Lamé constants \(\lambda_0,\mu_0\); these match the material values of the 80 validation geometries. Each face integral uses a tensor-product three-point Gauss rule in its two tangential coordinates. Stacking the square-root-weighted jump evaluations for both derivative orders and all three displacement components gives \(G_f\in\mathbb R^{54\times135}\), acting on the 45 distinct nodes of the two-element patch. Thus the second term in Eq. (A.1) is the matrix representation of \(\gamma g_h\).

All 80 validation geometries use \(n=32\), normalised Young's modulus \(E_Y=1\), Poisson's ratio \(\nu=0.3\), and ghost-penalty coefficient \(\gamma=10^{-4}\), with length expressed relative to the unit reference box. This gives 65 Q2 node positions per axis; the network's three transfer levels use 33, 17 and 9 positions per axis. The moment evaluator begins with \(4^3\) subcells per element and refines partial subcells once, then clips Kuhn tetrahedra using the tetrahedral rule parameter four. Appendix F.4 gives the full integration sequence. Training populations and the historical deployment benchmark are identified separately.

### A.2. Response metrics and aggregation

With \(K_{,c}=\partial K/\partial\tau_c\) on the stated fixed-coordinate design interval, the metrics are

\[
\begin{aligned}
\varepsilon(q)&=\frac{\widehat u^TK\widehat u}{q^TSq}-1,
&C&=f_g^TU,\quad\widehat C=f_g^T\widehat U,\\
e_C&=\left|\frac{\widehat C}{C}-1\right|,
&\widetilde s_c(\widehat u)&=-\widehat u^TK_{,c}\widehat u,\\
e_s&=\frac{\|\widetilde{\boldsymbol s}-\boldsymbol s\|_2}{\|\boldsymbol s\|_2},
&\boldsymbol s&=\big(-u^TK_{,c}u\big)_{c=1}^{8}.
\end{aligned}
\tag{A.2}
\]

The local same-retained-displacement comparison uses \(u=Eq\). The assembled comparison uses the recovered fields at the respective global equilibria, whose retained displacements generally differ. All relative denominators are nonzero. A geometry-level energy score averages over the evaluated directions in one class; population means then give equal weight to each available geometry. A population maximum of geometry means is distinct from a worst-direction operator error.

The pair-assembly comparison joins a learned target to an exact neighbour with matched thickness parameters on the common face. Its six face-load cases define the selected 3% joint compliance/sensitivity criterion, with sensitivity maxima taken over both cells. Cut-traction cases are reported separately. Tables 2 and 3 and the complete supplementary result tables identify the populations and loading sets.

## Appendix B. Variational identity, rigid kernel, and directional norms

Let the selectors satisfy \(J_P^TJ_P+J_I^TJ_I=I_{n_a}\), and assume the symmetric stiffness in Eq. (2) has \(A\succ0\). For any field \(v\) with \(J_Pv=q\), there is a unique \(w\in\mathbb R^i\) such that \(v=Eq+J_I^Tw\). Since \(J_IKE=0\),

\[
v^TKv=q^TSq+w^TAw.
\tag{B.1}
\]

This proves the energy-minimising property and uniqueness of \(Eq\). Applying the same expansion to a linear admissible extension \(F=E+J_I^TH\) yields Eq. (9). Moreover, \(r_I=J_IKFq=AHq\), proving both expressions in Eq. (10). These identities require trace admissibility and the specified symmetric stiffness; a contraction property of the approximation is a separate condition.

Suppose additionally that \(\ker K=\operatorname{range}R\), \(R_P\) has rank six, and \(FR_P=R\). The exact field associated with \(R_Pa\) is \(Ra\), because it has that retained trace and satisfies internal equilibrium. Thus \(ER_P=R\) and \(HR_P=0\). Rigid reproduction gives \(\widehat S R_P=F^TKR=0\). Conversely, if \(q^T\widehat S q=0\), positive semidefiniteness implies \(Fq\in\ker K\). Write \(Fq=Ra\); selecting the retained entries gives \(q=R_Pa\). Hence

\[
\ker S=\ker\widehat S=\operatorname{range}R_P.
\tag{B.2}
\]

The retained rigid coefficient map and deformation projector are

\[
C_R=(R_P^TR_P)^{-1}R_P^T,\qquad
\Pi_P=I_p-R_PC_R,\qquad q_d=\Pi_Pq.
\tag{B.3}
\]

Rigid reproduction and symmetry give

\[
\widehat S R_P=0,\qquad R_P^T\widehat S=0.
\tag{B.4}
\]

For the construction in Eq. (5), \(C_RR_P=I_6\) and \(\Pi_PR_P=0\). A linear raw displacement map sends zero to zero, so \(\widehat ER_P=R\). Corrections driven by the internal residual leave this field unchanged. These observations establish the two-sided rigid annihilation in Eq. (B.4) without altering any stiffness eigenvalue.

Let \(Z\in\mathbb R^{p\times(p-6)}\) have orthonormal columns spanning the retained rigid complement, and define \(S_*=Z^TSZ\succ0\). The all-direction energy error on this space is

\[
\varepsilon_*=\sup_{q\perp R_P,\ q\ne0}\varepsilon(q)
=\|A^{1/2}HZ S_*^{-1/2}\|_2^2,
\qquad\mu_*=1+\varepsilon_*.
\tag{B.5}
\]

Let \(q=Za\), with \(Z^TZ=I\) spanning the rigid complement. Its relative energy excess is

\[
\frac{a^TZ^TH^TAHZa}{a^TS_*a}
=\frac{\|A^{1/2}HZ S_*^{-1/2}b\|_2^2}{\|b\|_2^2},
\qquad b=S_*^{1/2}a.
\tag{B.6}
\]

Taking the supremum proves Eq. (B.5). The basis \(Z\) is used for analysis; the operator application does not construct these whitened coordinates.

### B.1. Displacement magnitude and directional stiffness

The energy metric assigns different significance to displacement errors of equal magnitude. To make that dependence explicit, fix the retained rigid gauge by taking \(q\perp R_P\), let \(d=Fq-Eq\), and introduce a displacement norm induced by \(W\succ0\):

\[
\delta_W(q)=\frac{\|d\|_W}{\|Eq\|_W},\qquad
\mathcal R_{K,W}(v)=\frac{v^TKv}{v^TWv},\qquad
\kappa_W(q,d)=\frac{\mathcal R_{K,W}(d)}{\mathcal R_{K,W}(Eq)}.
\tag{B.7}
\]

For nonzero error, cancellation of the displacement norms gives

\[
\boxed{\varepsilon(q)=\delta_W(q)^2\kappa_W(q,d).}
\tag{B.8}
\]

We use the full-field Euclidean norm, \(W=I_{n_a}\). The factor \(\kappa_W\) compares the stiffness content of the error and the equilibrium response; it is not a matrix condition number. An error concentrated in stiffer deformation than the loaded response has a large \(\kappa_W\), making even a small relative displacement error mechanically significant. Thus an energy target \(\varepsilon_{\rm tar}\) requires \(\delta_W\le\sqrt{\varepsilon_{\rm tar}/\kappa_W}\). Correction must address this energy content as well as displacement magnitude.

For nonzero field error and nonzero reference energy, the product in Eq. (B.8) follows by cancellation of \(d^TWd\) and \(u^TWu\). Changing the displacement normalisation changes the corresponding amplification factor. An alternative based on the retained displacement is

\[
\delta_P=\frac{\|d_I\|_2}{\|q\|_2},\qquad
\kappa_P=\frac{d_I^TAd_I/\|d_I\|_2^2}{q^TSq/\|q\|_2^2},
\qquad\varepsilon(q)=\delta_P^2\kappa_P.
\tag{B.9}
\]

## Appendix C. Assembly and compliance ordering

For a global retained vector \(U\), minimising the sum of substructure energies over their disjoint internal variables separates into the individual minimisations in Eq. (B.1). The minimised quadratic form is \(\sum_m U^TB_m^TS_mB_mU\). This proves the equivalence of local condensation followed by assembly and elimination from the modular assembled system under the coupling assumptions in Section 2.3.

Let \(\mathbb D=\widehat{\mathbb K}-\mathbb K\). From Eq. (9),

\[
U^T\mathbb D U=\sum_m\|H_m B_mU\|_{A_m}^2\ge0.
\tag{C.1}
\]

If homogeneous supports make \(\mathbb K\succ0\), then both global systems are invertible and matrix inversion reverses their order. For any fixed nonzero load,

\[
\widehat C=f_g^T\widehat{\mathbb K}^{-1}f_g
\le C=f_g^T\mathbb K^{-1}f_g.
\tag{C.2}
\]

For completeness, the order reversal follows by setting \(Q=\mathbb K^{-1/2}\widehat{\mathbb K}\mathbb K^{-1/2}\succeq I\). Its eigenvalues are at least one, so \(Q^{-1}\preceq I\); a congruence gives the inverse inequality. If every substructure also satisfies \(0\preceq\widehat S_m-S_m\preceq\bar\varepsilon S_m\), applying the same argument to the upper and lower bounds gives

\[
\mathbb K\preceq\widehat{\mathbb K}\preceq(1+\bar\varepsilon)\mathbb K,
\qquad\frac{C}{1+\bar\varepsilon}\le\widehat C\le C,
\qquad\frac{C-\widehat C}{C}\le\frac{\bar\varepsilon}{1+\bar\varepsilon}.
\tag{C.3}
\]

The required local bound is uniform over directions. A mean over a finite bank has a different statistical meaning.

The assembled state error obeys the exact identity

\[
\widehat U-U=-\widehat{\mathbb K}^{-1}\mathbb D U.
\tag{C.4}
\]

Consequently, for a fixed finite assembly with uniformly stable support and bounded inverse, a family \(H_m=O(t)\) gives \(\mathbb D=O(t^2)\) and \(\widehat U-U=O(t^2)\) as \(t\to0\). This asymptotic statement holds with the exact discrete system fixed. Its constants depend on the assembly and do not prescribe which contribution dominates a finite-error sensitivity diagnostic.

## Appendix D. Polynomial smoothing and coarse projections

For \(0<a<b\), the degree-\(k\) Chebyshev error polynomial and its energy bound are

\[
p_k(t)=\frac{T_k((a+b-2t)/(b-a))}{T_k((a+b)/(b-a))}.
\tag{D.1}
\]

\[
d_I^{\rm sm}=p_k(D^{-1}A)d_I,\qquad
\|d_I^{\rm sm}\|_A^2\le\rho_k^2\|d_I\|_A^2,\qquad
\rho_k=\max_{\lambda\in\sigma(\widetilde A)}|p_k(\lambda)|.
\tag{D.2}
\]

Here \(D=\operatorname{diag}(A)\) and \(\widetilde A=D^{-1/2}AD^{-1/2}\).

Write \(M=D^{-1}A\), \(\theta=(a+b)/2\), \(\eta_s=(b-a)/2\), and \(\sigma=\theta/\eta_s>1\). Starting from \(u_0\), keep its retained entries fixed and define the internal residual with the sign convention \(r_j=(Ku_j)_I\). A recurrence producing the polynomial in Eq. (D.1) is

\[
\begin{aligned}
\varrho_0&=1/\sigma,& s_0&=-D^{-1}r_0/\theta,\\
u_{j+1}&=u_j+J_I^Ts_j,&
\varrho_{j+1}&=(2\sigma-\varrho_j)^{-1},\\
s_{j+1}&=\varrho_{j+1}\varrho_j s_j
 -\frac{2\varrho_{j+1}}{\eta_s}D^{-1}r_{j+1}.
\end{aligned}
\tag{D.3}
\]

The last line is evaluated only when another step is needed. Zero steps return the initial field. To verify the polynomial, observe that \(\varrho_j=T_j(\sigma)/T_{j+1}(\sigma)\), and apply \(T_{j+2}(x)=2xT_{j+1}(x)-T_j(x)\). The first error update is \(d_1=(I-M/\theta)d_0\), and the same recurrence gives \(d_k=p_k(M)d_0\).

The matrices \(M\) and \(\widetilde A=D^{-1/2}AD^{-1/2}\) are similar, and

\[
\|p_k(M)d\|_A^2
=\sum_j\lambda_j p_k(\lambda_j)^2c_j^2,
\qquad c_j=v_j^TDd.
\tag{D.4}
\]

This proves Eq. (D.2). If \(a\le\lambda\le b\), the argument of \(T_k\) lies in \([-1,1]\), where \(|T_k|\le1\). For \(0<\lambda<a\), it lies between 1 and \(\sigma\), where \(T_k\) increases from 1 to \(T_k(\sigma)\). Hence \(|p_k(\lambda)|\le1\) throughout \((0,b]\). The argument establishes contraction of the prescribed polynomial; it does not require every intermediate degree to improve monotonically.

The operational upper endpoint is estimated using a random power vector, 40 iterations of \(M\), Euclidean normalisation, and a factor of 1.05. The correction implementation fixes the random generator seed and caches the interval for the geometry. The separate smoothing diagnostic uses its own random start. The estimates are used to set the polynomial, while the result in Eq. (D.2) is conditional on its actual spectral values.

For a full-column-rank coarse basis, define \(Q_V=VA_c^{-1}V^TA\). Direct multiplication gives \(Q_V^2=Q_V\) and \(Q_V^TA=AQ_V\). Thus \(Q_V\) is an \(A\)-orthogonal projection and \(C_V=I-Q_V\) is its complementary projection. With \(b_r=V^TAd\),

\[
\|d\|_A^2=\|C_Vd\|_A^2+\|Q_Vd\|_A^2,
\qquad\|Q_Vd\|_A^2=b_r^TA_c^{-1}b_r.
\tag{D.5}
\]

This proves Eq. (16). Applying Eq. (D.2) before and after this projection yields \(\|P_kC_VP_kd\|_A\le\rho_k^2\|d\|_A\), which proves the \(\rho_k^4\) energy estimate and operator ordering in Eq. (17). This is the standard energy-projection mechanism of subspace correction [Xu (1992)](https://doi.org/10.1137/1034116).

## Appendix E. Transpose of the complete extension

All transposes below use the Euclidean pairing of the stored displacement and nodal-force coordinates. Write \(M_I=J_I^TJ_I\). Transposing Eq. (5) gives

\[
\widehat E^Ty=J_Py+C_R^TR^TM_Iy
 +\Pi_P^T\mathcal N_\theta(\eta)^TM_Iy.
\tag{E.1}
\]

The overwrite therefore selects the direct retained contribution and masks the field sent through the network transpose. The geometry coefficients are held fixed during this displacement transpose.

Set \(s_k(t)=[1-p_k(t)]/t\), with the polynomial continuation at zero. The full-field smoothing map \(\mathcal T_k\) has the block representation

\[
\mathcal T_k=
\begin{bmatrix}
I_p&0\\
-s_k(M)D^{-1}K_{IP}&p_k(M)
\end{bmatrix},\qquad
\mathcal T_k^Ty=
\begin{bmatrix}
y_P-K_{PI}s_k(M)D^{-1}y_I\\
D p_k(M)D^{-1}y_I
\end{bmatrix}.
\tag{E.2}
\]

Indeed, the final internal field is its equilibrium value plus \(p_k(M)\) times the initial error. Replacing \((I-p_k(M))A^{-1}\) by \(s_k(M)D^{-1}\) yields the first expression. The second follows from \(M^TD=DM\), which implies \(p_k(M)^T=Dp_k(M)D^{-1}\) and the corresponding identity for \(s_k\). Although the forward map preserves retained displacement, its transpose contributes a retained force through the upper block in Eq. (E.2).

For the full-field coarse correction,

\[
\mathcal W_c=I_{n_a}-J_I^TVA_c^{-1}V^TJ_IK,
\qquad
\mathcal W_c^T=I_{n_a}-KJ_I^TVA_c^{-1}V^TJ_I.
\tag{E.3}
\]

A cycle \(\mathcal W=\mathcal T_{\rm post}\mathcal W_c\mathcal T_{\rm pre}\) therefore uses the reverse sequence \(\mathcal W^T=\mathcal T_{\rm pre}^T\mathcal W_c^T\mathcal T_{\rm post}^T\). The complete reaction is

\[
\widehat S q=\widehat E^T\mathcal W^TK\mathcal W\widehat E q.
\tag{E.4}
\]

The same formulas hold for the symmetric shifted inverse defined below when it replaces the coarse inverse consistently in both actions. Iteration counts, coarse bases, and polynomial coefficients depend on geometry and remain fixed across input directions. An input-dependent iterative stopping rule would require a separate analysis of linearity and differentiation.

## Appendix F. Coarse spaces, factorisation, and arithmetic

### F.1. Explicit coarse-space study

Coarse tensor-product shape functions are evaluated at the active background-node coordinates and restricted to \(I\). Standard \(Q_1\) and \(Q_2\) vector spaces carry three displacement coefficients per coarse node. The enriched partition-of-unity space carries 12 coefficients per vertex through fields of the form \(\sum_vN_v(x)[a_v+B_v(x-x_v)]\), with \(a_v\in\mathbb R^3\) and \(B_v\in\mathbb R^{3\times3}\). The spaces examined use \(Q_1\) grids of 9, 17, or 33 vertices per axis, a \(Q_2\) grid of 17 nodes per axis, and enriched \(Q_1\) grids of 9 or 17 vertices per axis.

Columns with an absolute interior-support sum at most \(10^{-14}\) are removed. In the fixed-weight CPU study, the symmetric internal matrix is formed from its stored upper triangle. The coarse matrix is explicitly computed as \(V^TAV\) and symmetrised. Columns whose diagonal energy is at most \(10^{-12}\) times the largest coarse diagonal are then removed. The resulting sparse matrix is factorised with PARDISO, or SuperLU when PARDISO is unavailable. This construction adds no explicit diagonal shift. The projection formulas in Eqs. (16)–(17) apply to linearly independent surviving coarse columns and the exact Galerkin action. Interpreting a recorded numerical solve through those formulas additionally requires verification of its solve accuracy.

The enriched generating functions have a specific coefficient redundancy. The trilinear nodal basis reproduces linear coordinates, so

\[
\sum_v N_v(x)=1,\qquad \sum_v N_v(x)x_v=x,
\qquad \sum_v N_v(x)(x_j-x_{v,j})=0.
\]

Thus taking \(a_v=0\) and the same slope matrix \(B_v=B\) at every vertex produces the zero displacement field. Restriction to internal coordinates preserves this identity. Support and diagonal-energy screens do not certify independence of the surviving columns. The archived PU records give column counts and field-error statistics, but no rank-revealing representation or coarse-equation residual. Their values in Table ST04 are retained as numerical observations of those solves, rather than verification of the full-rank projection assumptions. The reported Q1(17) result is the principal coarse-correction result.

Coefficient redundancy does not preclude energy minimisation over the coarse range. Since \(A\succ0\), \(\ker(V^TAV)=\ker V\), and \(V^Tr_I\) is orthogonal to this kernel. An exactly solved compatible coarse equation therefore defines a unique displacement correction even when its coefficient vector is nonunique. Establishing that property for the archived numerical PU solve requires the corresponding representation and solve-accuracy evidence.

For each input direction, the study evaluates the exact field and the uncorrected network field once. It compares the network alone, a smoothing tail, a coarse update, coarse followed by smoothing, smoothing on both sides of the coarse update, and a zero-interior initialization followed by the same complete cycle. Its energy ratios use a recomputed teacher energy for the supplied direction. The fields are converted to float64 before correction.

### F.2. Differentiable correction implementation

The differentiable implementation constructs coarse support metadata and recovers a coarse matrix using coloured stiffness probes. Its configured stencil reach is four fine-grid node spacings. With \(h_g\) fine-node spacings per coarse-vertex spacing, the probe radius is \(R_g=2+\lceil4/h_g\rceil\); colours use vertex coordinates modulo \(2R_g+1\) and the component or enrichment slot. Recovery assumes the resulting colour separation resolves every interacting coarse pair. This setup supports the \(Q_1\) and enriched \(Q_1\) spaces. The explicit \(Q_2\) study uses the CPU construction in Appendix F.1.

The recovered matrix is Jacobi scaled with the corresponding column scaling of \(V\). Cholesky factorisation reads its lower triangle, trying diagonal shifts in the order \(0,10^{-12},10^{-10},10^{-8},10^{-6},10^{-4}\). The selected value is stored with the geometry factorisation. To characterize such a solve algebraically, let \(V_s\) denote the scaled basis, \(H_c=V_s^TAV_s\succeq0\), \(\xi\ge0\), \(H_c+\xi I\succ0\), \(B_c=(H_c+\xi I)^{-1}\), and \(b_s=V_s^Tr_I\). Then

\[
\|d_I\|_A^2-\|d_I-V_sB_cb_s\|_A^2
=b_s^TB_c(H_c+2\xi I)B_cb_s\ge0.
\tag{F.1}
\]

Expanding the squared norms gives \(b_s^T(2B_c-B_cH_cB_c)b_s\), and multiplication by \(H_c+\xi I\) establishes the equality. A positive shift changes the exact projection but preserves this energy decrease when the solved matrix is the stated scaled Galerkin matrix. The identity assumes that matrix equality, symmetry, and consistent forward and transpose solves.

### F.3. Precision and normalisation

The learned displacement extension, rigid-body reconstruction, and prescribed retained entries use float32. Quadratic energies and sparse stiffness products use float64. The differentiable correction implementation computes its relaxation and coarse algebra in float64 and returns to the input displacement dtype. The explicit inference transpose also crosses the float64/float32 boundary around its network action. Sensitivity contractions use float32 element products with float64 accumulation. These arithmetic choices approximate the real linear maps in the preceding derivations.

| Evaluation path | Initial field and correction | Condensed action |
|---|---|---|
| Fixed-weight coarse study (B) | Float32 network field converted to float64; explicit CPU Galerkin matrix and float64 correction | The saved comparison evaluates corrected field energies; it is separate from the deployment timing study. |
| Differentiable correction / S8 evaluation | Float32 learned field; relaxation algebra in float64; correction routines return to their input dtype | The correction and its complete transpose enter the stated variational action. S8 uses eight smoothing steps and no coarse solve. |
| Explicit inference implementation | Float32 network and rigid reconstruction; float64 stiffness product; dtype conversions before the transpose action | The complete transpose follows the actual configured correction sequence, with finite-precision consistency assessed separately. |

In the mixed-precision two-grid implementation, each smoothing routine returns its input dtype. Consequently a float32 pre-smoothed field can be rounded before entering the float64 coarse solve; the final corrected field returns to the original dtype. These casts belong to the numerical implementation rather than the real-linear maps used in the identities.

The role of actual retained values can be isolated algebraically. Suppose a computed field \(v\) has retained entries \(q'=J_Pv=q+b_P\), and set \(d=v-Eq\). Without assuming \(b_P=0\),

\[
v^TKv-q^TSq=d^TKd+2b_P^TSq.
\tag{F.2}
\]

Using the actual trace instead gives the nonnegative variational difference \(v^TKv-q'^TSq'\). In exact arithmetic applied to those actual vectors, with \(r_I=(Kv)_I\),

\[
v^TKv-1=r_I^TA^{-1}r_I+(q'^TSq'-1).
\tag{F.3}
\]

Thus a score formed by subtracting one also depends on the normalisation of the stored direction. Recomputed teacher denominators, stored unit-energy assumptions, and finite-precision quadratic forms are separate parts of a numerical energy diagnostic.

### F.4. Moment integration

The teacher's standard entry points initialize \(4^3\) subcells per active background element and refine partially occupied subcells once into \(2^3\) children. Thus the finest subcell width near the material boundary is \(h/8\). Fully occupied subcells contribute analytic tensor-product monomial moments. At the finest partial level, each subcell is divided into six Kuhn tetrahedra and clipped successively by the three interpolated inequalities defining the band and the macro cut. The tetrahedral quadrature helper is called with its order parameter set to four. The accumulated moments are multiplied by the physical Jacobian \((2n)^{-3}\). Refinement is local to partial subcells; it does not uniformly subdivide every active element to the finest level.

## Appendix G. Network coefficients and directional training

### G.1. Geometry features and displacement maps

The architecture in Figure 3 separates geometry-dependent coefficient generation from displacement propagation. Let \(N\) denote the number of active nodes, \(N_P\) the number of retained nodes and \(B\) the number of simultaneous displacement directions. Then \(n_a=3N\), \(p=3N_P\), and the deformational input \(q_d=\Pi_Pq\) has shape \(p\times B\). The fine displacement features have shape \(N\times B\times32\); a coarse latent level with \(N_\ell\) participating vertices has shape \(N_\ell\times B\times32\). Geometry embeddings have 64 components and carry no displacement-direction dimension. Element, face, retained-node and grid-transfer incidences are constructed before the network is evaluated.

**Geometry encoding.** Each element has 126 input features: material volume fraction, its logarithm, and the remaining 124 moments normalised by material volume. The 11 node features contain retained, box, cut-band and weak-support indicators, a normalised logarithmic stiffness diagonal, and six sine/cosine coordinate entries. The stiffness summary is the Frobenius norm of the node's diagonal \(3\times3\) block. A node is designated weakly supported when this norm is less than 0.01 times its median over active nodes. These features supply information about both material occupancy and its mechanical support.

The element encoder maps 126 inputs to a 64-component embedding. Mean aggregation of incident element embeddings, concatenated with the 11 node features, supplies the 75-input node encoder. Two residual message-passing rounds then update elements from the mean embeddings of their 27 nodes and update nodes from their incident elements. Each update uses the current embedding and the aggregated neighbouring embedding as a 128-component input. A ghost-face embedding is formed from the two adjacent element embeddings and a three-component indicator of the face axis. All encoders and coefficient heads use two affine layers separated by GELU, with hidden width 64. The B, C and S8 configurations use the base features described here; their optional five-feature extension is disabled.

**Local linear interactions.** The three retained displacement components are lifted by a shared matrix \(W_{\rm in}\in\mathbb R^{3\times32}\), while internal features are initially zero. Denote this initial feature field by \(X^0(q)\), with its dependence on \(q\) occurring through \(q_d\). For each element or face stencil \(t\), local node slot \(s\) and head \(h\), the interaction first gathers and mixes channels:

\[
Z_{th}=\left(\sum_{s=1}^{27}a_{tsh}(\eta)X_{i(t,s)}\right)W_{\ell h},
\qquad
\Delta X_i=\sum_{(t,s):i(t,s)=i}\sum_{h=1}^{4}
\omega_{ts}\,b_{tsh}(\eta)Z_{th}.
\]

Here \(W_{\ell h}\in\mathbb R^{32\times32}\) is a shared learned channel map for layer \(\ell\) and head \(h\). The scalar gather and scatter coefficients \(a\) and \(b\) are generated separately from the element or face embedding, the incident node embedding and an eight-dimensional slot embedding. Their 136-component concatenation identifies both the local material context and the position within the stencil. The four heads provide distinct gather--mix--scatter contributions on the same fixed incidence. Equation (4) adds their update to the incoming state and restores the prescribed retained features after every local interaction.

The element stencil contains all 27 \(Q_2\) nodes. The learned ghost-face stencil contains 18 owner-side nodes and nine neighbour-side nodes, ordered consistently by face direction. It defines a learned communication map on a face neighbourhood; the full mechanical ghost-penalty contribution remains in \(K\).

The node scatter accounts for the number and relative stiffness of incident contributions. If \(k_{es}\) is the Frobenius norm of the element's diagonal \(3\times3\) block at slot \(s\), the element weights are

\[
\pi_{es}=\frac{k_{es}}{\sum_{(e',s')\mapsto i}k_{e's'}},
\qquad
\omega_{es}=\frac{1-\lambda_{\rm mix}}{\deg(i)}
 +\lambda_{\rm mix}\pi_{es},\qquad(e,s)\mapsto i.
\tag{G.1}
\]

The face weights use the corresponding ghost-contribution diagonal blocks and face incidences. When all incident stiffness norms at a node vanish, the stiffness share is defined by inverse incidence count. Each layer has its own learned \(\lambda_{\rm mix}\), without a convex-interval constraint. This weighting changes coefficients on the prescribed scatter map. For the additional weak-region layers, a stencil is selected when it contains a weak node; its weights retain the normalisation of the full element or face incidence set.

**Multilevel propagation.** The latent hierarchy enlarges the spatial range of communication while retaining the fine field through additive skips. Let \(t_{ia}\) be the fixed trilinear incidence weight between a fine node \(i\) and a coarse vertex \(a\). Positive geometry coefficients \(r_i\) and \(p_i\) define restriction and prolongation componentwise:

\[
(\mathcal R X)_a=\frac{\sum_i t_{ia}r_iX_i}{\sum_i t_{ia}r_i},
\qquad
(\mathcal P X_c)_i=p_i\frac{\sum_a t_{ia}X_{c,a}}{\sum_a t_{ia}}.
\]

Only coarse vertices reached by the stored trilinear incidences participate. The denominators depend on geometry alone, so both maps are linear in displacement. Restriction and prolongation are separately parameterised. Their relationship does not impose symmetry on the raw extension; the transpose in Eq. (6) is obtained from the complete composition of operations.

On each coarse level, a residual convolution takes the form

\[
X\leftarrow X+g_{\ell j}(\eta)\odot\operatorname{Conv}_{\ell j}(X).
\]

Each convolution has a \(3\times3\times3\) kernel, 32 input and output channels, unit padding and no bias or displacement activation. Participating latent nodes are inserted into the corresponding background grid, with unused positions set to zero; the convolution output is sampled back at those nodes. The geometry branch supplies one multiplicative coefficient per participating node, channel and convolution. Its coarse embeddings are obtained by fixed trilinear weighted averaging.

The full displacement sequence is specified below. Grid sizes count background positions per axis, whereas the participating node counts depend on geometry. Every row carries 32 displacement channels until the final projection.

| Stage | Operations in execution order |
|---|---|
| Fine input, 65 grid | Lift \(q_d\) on retained nodes, set internal features to zero; apply four element-then-face interaction pairs; save \(S_0\). |
| Downward, 33 grid | Restrict 65→33; apply two residual convolutions; save \(S_1\). |
| Downward, 17 grid | Restrict 33→17; apply two residual convolutions; save \(S_2\). |
| Downward, 9 grid | Restrict 17→9; apply two residual convolutions. |
| Upward, 9 grid | Apply two residual convolutions; prolong 9→17; add \(S_2\). |
| Upward, 17 grid | Apply two residual convolutions; prolong 17→33; add \(S_1\). |
| Upward, 33 grid | Apply two residual convolutions; prolong 33→65; add \(S_0\). |
| Fine output, 65 grid | Restore retained features; apply four element-then-face interaction pairs, then four weak-region element-then-face pairs; project 32→3. |

At each upward step the combination is \(X=S_\ell+\sigma_\ell\mathcal P_\ell X_c\), with a learned scalar \(\sigma_\ell\). Thus convolution precedes prolongation, the skips are additive, and the 9-position grid receives two downward and two upward convolutions. There are twelve coarse convolution updates in total. The fine retained values are restored after the hierarchy returns to the active nodes. These latent operations communicate within the extension; the mechanical operator still accepts and returns all \(p\) retained coordinates. The Galerkin correction of Section 5 instead uses the internal stiffness and equilibrium residual.

**Coefficient bounds and output.** The local heads produce separate gather/scatter values for eight ordinary element layers, twelve face layers and four additional weak-region element layers, each with four heads and 27 slots. Raw head outputs are multiplied by 0.2. For a finite positive group bound \(a_{\max}\), the coefficient map is

\[
\mathcal C(z)=
\begin{cases}
z,&|z|\le k_ba_{\max},\\
\operatorname{sgn}(z)\left[k_ba_{\max}+(1-k_b)a_{\max}
\tanh\!\left(\dfrac{|z|-k_ba_{\max}}{(1-k_b)a_{\max}}\right)\right],&|z|>k_ba_{\max}.
\end{cases}
\]

The B, C and S8 configurations set \(k_b=0.5\). An infinite bound denotes the identity map. Bounds are fixed model-state arrays indexed by coefficient type, layer and gather/scatter role. Positive transfer coefficients apply the corresponding bound to \(\operatorname{softplus}(z)\) and add \(10^{-3}\); convolution coefficients use \(2\operatorname{sigmoid}(z)\). All these nonlinearities act on geometry-derived coefficients. The channel matrices, convolution kernels and skip scalars are shared learned parameters, whereas the coefficient-head outputs vary with geometry.

A bias-free matrix \(W_{\rm out}\in\mathbb R^{32\times3}\) returns the fine features to nodal displacement in the prescribed coordinate order. Rigid reconstruction and retained-value restoration then produce Eq. (5). The energy action Eq. (6) uses this complete map, together with the prescribed correction, and its transpose; it therefore determines the force from the same displacement representation.

**Frame consistency.** For an orthogonal cube transformation, the deterministic signed permutation maps satisfy

\[
K'=T_aKT_a^T,\qquad E'=T_aET_P^T,\qquad S'=T_PST_P^T.
\tag{G.2}
\]

The augmented geometry and vector field are passed through the network and the predicted displacement is returned to the original frame for mechanical evaluation. These relations describe the transformed target and the frame-consistency condition. Sampling the cube views trains the geometry-conditioned extension towards that condition; displacement linearity alone does not establish rotational equivariance.

### G.2. Direction banks

The normalised retained directions are

\[
q_j=\frac{\Pi_P\widetilde q_j}
{\sqrt{(\Pi_P\widetilde q_j)^TS(\Pi_P\widetilde q_j)}}.
\tag{G.3}
\]


The direction classes used by the trainer have the following mechanical definitions.

| Class | Construction before rigid removal and energy normalisation |
|---|---|
| `force` | Equilibrated nodal loads from smooth plane waves or localized Gaussian patches, with a subset loading the cut boundary. |
| `force_c` | Consistently integrated self-equilibrated tractions on material box faces; a subset also loads the cut face. Equilibrium is imposed in traction-quadrature space. |
| `face` / `face_c` | Single-face equilibrated nodal loads or consistently integrated tractions. |
| `support` | Responses with soft spring support on one box face and equilibrated loads on other faces. |
| `support_k` | Consistent loads with springs scaled by the local stiffness diagonal and a logarithmically sampled factor in \([0.3,3]\); the cut face remains unloaded. |
| `macro` | Equal sampling of uniform strain, quadratic, and cubic imposed displacement fields. |
| `grf` | Multiscale displacement fields spanning 0.5–24 spatial cycles across the reference box. |
| `glued` | Traces induced by a neighbouring cell with shared interface degrees of freedom and far-face clamping or springs; one quarter of samples load only the neighbour. |
| `adv` | Directions selected by the generalised-energy search described below. |

The isolated teacher removes the rigid component and normalizes the resulting retained direction by Eq. (G.3). Glued responses therefore supply a contextual direction while the energy and sensitivity labels refer to the target cell's operator. Banks store the normalised vectors in float32.

### G.3. Optimisation and difficult-direction search

The v2L1 (B), A0 (C), A2 (S8), A2b and A3 configurations use batches of 16 directions, update the geometry every step, keep three geometries in the device pool, and replace a pool entry every 100 steps. Their nominal class weights, in the order `force`, `force_c`, `support`, `support_k`, `face`, `face_c`, `macro`, `grf`, `glued`, `adv`, are \(0.10,0.10,0.075,0.075,0.10,0.10,0.10,0.125,0.075,0.15\). Weights are renormalized over available classes and assigned by systematic quotas.

Optimisation uses Adam, a one-cycle learning-rate schedule with peak \(3\times10^{-4}\), 5% warm-up, cosine decay, and final division factor 100. The gradient norm is clipped at one. Selected weights use an exponential moving average with decay 0.9997 and zero-initialization bias correction. Table 2 identifies the model roles, and Table ST12 gives the training and weight-selection settings for each arm. The energy term in Eq. (8) uses a \(10^{-12}\) lower clamp inside the logarithm; the reported sensitivity loss is the squared relative eight-corner norm with unit weight. The optional bank-spectrum loss has zero default weight in these configurations.

A3 and A2b continue B from its selected weights for 15,000 steps with the same schedule, class weights and training population as C, so that the three continuations differ only in the correction used in the forward map of Eq. (8): none for C, eight Chebyshev steps for A2b, and the eight-step, \(Q_1(17)\), eight-step sequence for A3. The correction is recomputed for every geometry of the device pool (smoothing interval by power iteration, coarse matrix by coloured probing, Appendix F.2) and is treated as a fixed linear map of the network output. Gradients of Eq. (8) are propagated through the smoothing recurrence and the coarse solve by automatic differentiation in double precision; the transposes used at evaluation (Appendix E) are the explicit adjoints of the same operations. The weights are selected at step 15,000 on the validation list described in Section 6.1.

The adversarial search initializes eight candidate directions and performs four block iterations when a geometry is loaded. Applying \(S^{-1}_{\perp}\) uses an equilibrated Neumann solve: six deterministically selected displacement pins remove rigid freedom, and the result is projected into the retained rigid complement. A bank search instead forms \(G=Q_b^TSQ_b\) and \(\widehat G=(FQ_b)^TK(FQ_b)\), solves a generalised eigenproblem after flooring the search metric at \(10^{-3}\lambda_{\max}(G)\), and renormalizes selected candidates with the original \(G\). The floor belongs to the search coordinates. Candidate energy ratios remain directional observations; their maximization does not supply an all-direction upper bound.

## Appendix H. Sensitivity identities and design intervals

The exact condensed derivative and the fixed-displacement energy derivative are

\[
S_{,c}=E^TK_{,c}E,\qquad
\frac{\partial}{\partial\tau_c}\left(\tfrac12q^TSq\right)
=\tfrac12(Eq)^TK_{,c}(Eq).
\tag{H.1}
\]

For an assembly, the full local field difference is

\[
F\widehat q-Eq=(F-E)q+E(\widehat q-q)+(F-E)(\widehat q-q).
\tag{H.2}
\]


On a differentiable interval with fixed active topology and coordinate maps, differentiating \(J_PE=I_p\) gives \(J_PE_{,c}=0\). Since \(J_IKE=0\), both terms \(E_{,c}^TKE\) and \(E^TKE_{,c}\) vanish. Differentiating \(S=E^TKE\) therefore proves Eq. (H.1).

For a supported equilibrium, differentiating \(\mathbb K U=f_g\) at a fixed load gives \(U_{,c}=-\mathbb K^{-1}\mathbb K_{,c}U\) and hence \(C_{,c}=-U^T\mathbb K_{,c}U\). With fixed gathers and Eq. (H.1), this becomes the sum of \(-u_m^TK_{m,c}u_m\) over affected cells. For an isolated energy-normalised direction, this label holds the base-design force fixed. The assembled sensitivity labels likewise hold the base-design nodal load fixed, including loads generated from consistent tractions. Reassembling a prescribed surface traction over a changing material boundary introduces the load-derivative term stated below; differentiating the direction normalisation also defines a different quantity.

Expanding the field-based quadratic estimate at \(u+d\) proves Eq. (13). In the Euclidean norm, it implies

\[
|\widetilde s_c-s_c|
\le2\|K_{,c}u\|_2\|d\|_2+\|K_{,c}\|_2\|d\|_2^2.
\tag{H.3}
\]

For the corner-thickness band in Eq. (1), the nonnegative trilinear shape functions make the material domains nested as one thickness parameter increases. With fixed basis functions and ghost stabilisation, exact integration gives \(K_{,c}\succeq0\) and nonpositive equilibrium compliance derivatives. The linear cross term in Eq. (13) can have either sign, while the quadratic term is nonpositive; their relative importance depends on derivative-weighted alignment and error amplitude. General design parameterizations can instead produce indefinite derivative matrices. The numerical moment-difference implementation is specified below. In an assembly, substituting Eq. (H.2) into the same quadratic expansion includes both the change of trace and its interaction with the extension error. Field-only and solution-only relative errors are therefore not additive scalar contributions.

For the surrogate operator, direct differentiation gives

\[
\widehat S_{,c}=F_{,c}^TKF+F^TK_{,c}F+F^TKF_{,c}.
\tag{H.4}
\]

Apply the equilibrium compliance derivative to the assembled surrogate and use \(J_PF_{,c}=0\). The two extension terms become \(2(F_{I,c}\widehat q)^Tr_I\), proving Eq. (14). Its magnitude is bounded by

\[
|\widehat C_{,c}-\widetilde s_c|
\le2\|F_{I,c}\widehat q\|_A\|r_I\|_{A^{-1}}
\tag{H.5}
\]

for a single affected cell, with a sum of such bounds for several cells. This bound involves the extension's design derivative as well as its equilibrium residual. It provides no fixed error order in the field amplitude without a corresponding assumption on that derivative. If the load depends on design, the full compliance derivative also contains \(2f_{g,c}^T\widehat U\); changes of coordinate, assembly, or support maps contribute their own chain-rule terms.

In the teacher, moment derivatives are approximated by

\[
M_{e\alpha,c}\approx
\frac{M_{e\alpha}(\boldsymbol\tau+h_ce_c)-M_{e\alpha}(\boldsymbol\tau-h_ce_c)}{2h_c},
\qquad h_c=10^{-5}\tau_c.
\tag{H.6}
\]

The active set is fixed for this calculation, and the elasticity templates and ghost contribution are held constant. The moment quadrature is recomputed at each perturbed thickness, so its clipping and integration branches may change. The resulting matrices represent a numerical design derivative on that prescribed topology. Training differentiates the field-based sensitivity loss with respect to the predicted field and network parameters. Computing the complete design derivative in Eq. (14) additionally differentiates the geometry-dependent extension and correction maps.

## Appendix I. Residual lower diagnostics and a reduced-trace comparison

For any nonzero internal test vector \(z\), Cauchy–Schwarz applied to \(A^{1/2}z\) and \(A^{-1/2}r_I\) gives

\[
\Delta(q):=q^T(\widehat S-S)q=r_I^TA^{-1}r_I
\ge L_z:=\frac{(z^Tr_I)^2}{z^TAz}.
\tag{I.1}
\]

A preconditioned residual, a coarse approximation, or a Krylov iterate can supply \(z\). Evaluating the final one-vector quotient with the original \(A\) makes the inequality independent of how that vector was generated. For \(e_h=\widehat u^TK\widehat u\) and \(e_h-L_z>0\), the map \(x\mapsto x/(e_h-x)\) is increasing on the relevant interval. Therefore

\[
\varepsilon(q)\ge\frac{L_z}{e_h-L_z},\qquad
\mu_*\ge1+\frac{L_z}{e_h-L_z}.
\tag{I.2}
\]

If the exact reference energy is known, \(L_z/(q^TSq)\) gives a sharper lower estimate. A large value detects an inaccurate direction; a small value alone does not bound the error from above. These are real-arithmetic inequalities, with numerical evaluation requiring the stated original quadratic forms.

The reduced-trace comparison in Section 6 uses a tensor-product Bernstein space of degree \(r\) on selected box faces. Its global map \(G_r\) respects shared coordinates, while cut-band coordinates outside the box trace retain identity columns. The supported Galerkin system is

\[
\mathbb K_r=G_r^T\mathbb K G_r,\qquad
f_r=G_r^Tf_g,\qquad U_r=G_r\mathbb K_r^{-1}f_r.
\tag{I.3}
\]

The comparison evaluates exact local Schur operators within this restricted trace space. Restriction may be applied to all box faces or only shared interfaces. It isolates the effect of boundary representation in the specified assembly; local interior correction in Section 5 leaves those retained coordinates intact. Boundary-space reduction is also used in physics-informed local shape-function methods [Huang et al. (2023)](https://doi.org/10.1016/j.eml.2023.102041), but the Bernstein construction here is defined by Eq. (I.3) and the face selections of this comparison.


## Appendix J. Further variational and mechanical analysis

This appendix uses the notation of the main text and preceding appendices to develop the variational analysis of the prescribed discrete system. Analytical matrix examples in Appendix J.9 illustrate the resulting error orders and mechanical relationships.

### J.1. Assumptions and mechanical target

In this appendix, \(f\equiv f_g\) denotes the assembled retained load.

For each cell, \(K=K^T\succeq0\), the retained and internal coordinate sets \(P,I\) are fixed, and \(A=K_{II}\succ0\). The exact extension is defined by \(E_P=I\) and \(E_I=-A^{-1}K_{IP}\). A linear extension \(F\) has \(F_P=I\). Set \(H=F_I-E_I\), so \(F=E+J_I^TH\), \(S=E^TKE\), and \(\widehat S=F^TKF\). All transposes include the entire extension and correction. Interior body loads are zero; equivalently, the load functional factors through the retained coordinates. Consistent boundary loads have this property when all coefficients with nonzero boundary virtual work are retained.

The cell interiors are disjoint, all intercell couplings act through the retained coordinates, and the assembly maps \(B_m\) include fixed homogeneous supports. The supported exact assembled stiffness \(\mathbb K=\sum_m B_m^TS_mB_m\) is positive definite. The same local stabilised matrices define both the reference and surrogate assemblies. The applied global load \(f\ne0\) is fixed. These assumptions distinguish the modular reference from a different monolithic discretisation that introduces additional intercell stabilisation.

Design statements additionally use a differentiable interval on which the active coordinates, retained coordinates, assembly maps, supports, and load are fixed. Stiffness derivatives include every varying term in the chosen discrete specification. In the present frozen sensitivity implementation, moment differences vary the body stiffness while the selected ghost matrix and its coefficient are fixed. The algebraic field-error expansions also hold for the saved symmetric finite-difference derivative matrices; interpreting them as exact design derivatives requires convergence to the derivative of that discrete system.

The full retained nodal representation need not be a minimal independent basis of surface traces. The cut band keeps the active coefficients needed for the cut-surface load functional. This sufficiency, and containment of all intercell couplings in \(P\), is the property used by condensation and assembly.

### J.2. Why the complete transpose gives a quadratic operator error

Internal stationarity gives \(J_IKE=0\). Expanding the quadratic form therefore proves

\[
\widehat S-S=H^TAH,\qquad r_I=(KFq)_I=AHq.
\]

Both cross terms vanish. The stabilised discrete-energy excess at fixed \(q\) is one half of \(q^TH^TAHq\); the reported energy excess uses the quadratic energy without the factor one half.

The retained block of the unbalanced discrete force, used alone, would be

\[
(KF)_P=S+K_{PI}H.
\]

The additional variational reaction is \(F_I^T(KF)_I=(E_I+H)^TAH\). Since \(E_I^TA=-K_{PI}\), its term \(E_I^TAH\) cancels \(K_{PI}H\). Thus

\[
(KF)_P+F_I^T(KF)_I=S+H^TAH.
\tag{J.1}
\]

The complete transpose supplies both reciprocity and cancellation of the first-order stiffness error. A retained-force extraction without this term generally has an \(O(H)\) error and need not be symmetric.

If \(\ker K=\operatorname{range}R\), \(R_P\) has full column rank, and \(FR_P=R\), then \(\ker\widehat S=\operatorname{range}R_P\). Indeed, \(q^T\widehat Sq=0\) implies \(Fq\in\ker K\); its retained part gives \(q=R_Pa\). Conversely, rigid reproduction gives \(\widehat S R_P=0\), and symmetry gives the left nullspace. The conclusion excludes extra modes without adding stiffness. Positive semidefiniteness alone does not imply rigid reproduction; rigid reproduction alone does not exclude extra modes in a reference \(K\) with a larger kernel.

For any global vector \(V\), the quadratic error is a sum of nonnegative local errors. Consequently \(\widehat{\mathbb K}\succeq\mathbb K\succ0\). This also proves supported solvability even if some local cells are only semidefinite. A high condition number can still increase the cost and sensitivity of the numerical solve.

### J.3. Compliance as the full reconstructed error energy

Let \(U=\mathbb K^{-1}f\), \(\widehat U=\widehat{\mathbb K}^{-1}f\), \(q_m=B_mU\), \(\widehat q_m=B_m\widehat U\), \(u_m=E_mq_m\), and \(\widehat u_m=F_m\widehat q_m\). Write

\[
\Delta=\widehat{\mathbb K}-\mathbb K
=\sum_m B_m^TH_m^TA_mH_mB_m.
\]

For conforming cell fields, use the energy form \(a(v,v)=\sum_m v_m^TK_mv_m\). Shared retained coefficients are counted through their separate cell energy contributions. Exact local equilibrium and the global equation give \(a(u,v)=f^TV_P\), where \(V_P\) denotes the free assembled trace. At the surrogate equilibrium, \(a(\widehat u,\widehat u)=f^T\widehat U=\widehat C\). Hence

\[
\boxed{C-\widehat C=a(\widehat u-u,\widehat u-u)
=\|\widehat U-U\|_{\mathbb K}^2
+\sum_m\|H_m\widehat q_m\|_{A_m}^2.}

\]

The second equality follows from
\(\widehat u_m-u_m=E_m(\widehat q_m-q_m)+J_{I,m}^TH_m\widehat q_m\).
The two terms are \(K_m\)-orthogonal because \(J_{I,m}K_mE_m=0\). Thus compliance underestimation measures the total energy error of the reconstructed field, whereas the retained-solution error accounts for only one part of it. In particular,

\[
\frac{\|\widehat U-U\|_{\mathbb K}}{\|U\|_{\mathbb K}}
\le\sqrt{\frac{C-\widehat C}{C}}.
\]

This nonasymptotic bound is valid without a small-error assumption. The sharper asymptotic retained-solution order is derived below.

Define \(T=\mathbb K^{-1/2}\Delta\mathbb K^{-1/2}\succeq0\), \(z=\mathbb K^{-1/2}f\), \(\rho=\|T\|_2\), and

\[
\beta=\frac{U^T\Delta U}{C}
=\sum_{m:q_m^TS_mq_m>0}w_m\varepsilon_m(q_m),
\qquad w_m=\frac{q_m^TS_mq_m}{C}.
\]

For zero-energy cells with exact rigid reproduction, the corresponding numerator is also zero and its contribution is defined directly as zero. Spectral calculus gives

\[
\frac{C-\widehat C}{C}
=\frac{z^TT(I+T)^{-1}z}{z^Tz},\qquad
\boxed{\frac{\beta}{1+\rho}\le\frac{C-\widehat C}{C}
\le\frac{\beta}{1+\beta}\le\beta.}

\]

The lower bound uses \(\lambda/(1+\lambda)\ge\lambda/(1+\rho)\). For the sharper upper bound, insert \(V=\alpha U\) in the maximum principle

\[
\widehat C=\max_V\{2f^TV-V^T\widehat{\mathbb K}V\}.
\]

Optimizing the scalar gives \(\alpha=(1+\beta)^{-1}\) and \(\widehat C\ge C/(1+\beta)\). The familiar participation-weighted bound \(e_C\le\beta\) follows. This is a load-specific statement: \(w_m\) and \(\varepsilon_m\) are evaluated at the same exact assembled trace.

A uniform local inequality \(0\preceq\widehat S_m-S_m\preceq\varepsilon_*S_m\) implies \(T\preceq\varepsilon_*I\), and hence \(e_C\le\varepsilon_*/(1+\varepsilon_*)\) for every load. Rigid reproduction extends a quotient-space bound to arbitrary local traces.

Now take \(H_m(t)=tH_{0,m}\) and hold the reference matrices and maps fixed. Set \(\mathbb D_0=\sum_m B_m^TH_{0,m}^TA_mH_{0,m}B_m\). Expanding the inverse at zero yields

\[
\begin{aligned}
\widehat{\mathbb K}_t&=\mathbb K+t^2\mathbb D_0,\\
\widehat U_t-U&=-t^2\mathbb K^{-1}\mathbb D_0U+O(t^4),\\
C-\widehat C_t&=t^2U^T\mathbb D_0U+O(t^4),\\
\widehat u_{m,t}-u_m
&=tJ_{I,m}^TH_{0,m}q_m
-t^2E_mB_m\mathbb K^{-1}\mathbb D_0U+O(t^3).
\end{aligned}
\tag{J.2}

\]

Thus the retained displacement and compliance errors are \(O(t^2)\), the local full-field error is generically \(O(t)\), and the retained-error contribution in Eq. (11) is \(O(t^4)\). If \(\mathbb D_0U=0\), then \(H_{0,m}q_m=0\) for every affected cell, and this load is reproduced exactly for all \(t\). Uniform estimates across a family of geometries require uniform coercivity and bounded maps; fixed-geometry big-O constants do not supply those estimates automatically.

### J.4. From energy to eight-corner sensitivity

For a fixed retained \(q\), set \(u=Eq\), \(d=J_I^THq\), and \(\mathcal E=d^TKd=(Hq)^TA(Hq)\). Let \(D_c=K_{,c}\), \(c=1,\ldots,8\), be symmetric. Define the reference vector \(s_c=-u^TD_cu\), and assume \(\|\boldsymbol s\|_2>0\). Direct expansion gives

\[
\widetilde s_c-s_c=-2d^TD_cu-d^TD_cd.
\]

Let \(b_c=J_ID_cu\), \(D_{II,c}=J_ID_cJ_I^T\), and let the rows of \(\mathsf B_s\in\mathbb R^{8\times i}\) be \(b_c^TA^{-1/2}\). Define

\[
L=\|\mathsf B_s\|_2,\qquad
Q=\left(\sum_{c=1}^8\|A^{-1/2}D_{II,c}A^{-1/2}\|_2^2\right)^{1/2}.
\]

Cauchy–Schwarz and the quadratic-form operator bound give

\[
\boxed{\|\widetilde{\boldsymbol s}-\boldsymbol s\|_2
\le2L\sqrt{\mathcal E}+Q\mathcal E,
\qquad e_s\le\frac{2L\sqrt{q^TSq}}{\|\boldsymbol s\|_2}\sqrt\varepsilon
+\frac{Qq^TSq}{\|\boldsymbol s\|_2}\varepsilon.}

\tag{J.3}
\]

All eight components are included, and the denominator is the norm of the exact eight-vector for that particular direction. The constants distinguish derivative coupling, interior coercivity, and reference sensitivity scale. For fixed \(q\) and \(H=tH_0\), the linear coefficient vanishes precisely when \((H_0q)^Tb_c=0\) for every corner. It vanishes for all interior errors if \(J_ID_cE q=0\) for every corner. A useful special case is \(D_c=\alpha_cK\): internal stationarity removes the linear term and the relative sensitivity error equals the energy excess when the reference vector is nonzero.

For a re-equilibrated assembly, the quadratic expansion remains exact with \(d_m=\widehat u_m-u_m\), but \(d_m\) now includes a changed retained trace. Eq. (J.2) gives, componentwise,

\[
\widetilde s_{m,c}(t)-s_{m,c}
=-2t(J_I^TH_{0,m}q_m)^TD_cu_m
+t^2\{2(E_mB_m\mathbb K^{-1}\mathbb D_0U)^TD_cu_m
-(J_I^TH_{0,m}q_m)^TD_c(J_I^TH_{0,m}q_m)\}+O(t^3).
\]

The leading term is the same as the fixed-trace term. The solution-only replacement is \(O(t^2)\); its coefficient can be large on a soft assembly or relative to a small local reference. If only one cell has a learned extension, an exact neighbouring cell has no \(O(t)\) reconstruction term, although its solved trace can change at order \(t^2\).

There is also a global-to-local bound. Suppose \(K_m\) has its fixed rigid kernel, each \(D_{m,c}\) annihilates that kernel, and

\[
\Gamma_m=\left(\sum_c
\|K_{m,*}^{-1/2}D_{m,c,*}K_{m,*}^{-1/2}\|_2^2\right)^{1/2}<\infty
\]

on its rigid complement. Set \(E_m^{\rm loc}=u_m^TK_mu_m=w_mC>0\), \(\chi_m=\|\boldsymbol s_m\|_2/E_m^{\rm loc}>0\), and \(e_C=(C-\widehat C)/C\). The field expansion and Eq. (11) yield

\[
\boxed{e_{s,m}\le\frac{\Gamma_m}{\chi_m}
\left(2\sqrt{\frac{e_C}{w_m}}+\frac{e_C}{w_m}\right).}
\tag{J.4}
\]

Indeed, the local error energy is at most \(e_CC\). Substitution into
\(\|\delta\boldsymbol s_m\|_2\le\Gamma_m(2\sqrt{E_m^{\rm loc}a_m(d_m,d_m)}+a_m(d_m,d_m))\)
proves the result. This bound makes the roles of small participation and small reference sensitivity explicit. The frozen TPMS records do not measure \(\Gamma_m\), \(\chi_m\), or the local error energy, so Eq. (J.4) is a quantitative theoretical explanation, not a fitted attribution of a reported percentage.

Replacement diagnostics are not an additive sensitivity decomposition. If \(a=(F-E)q\), \(b=E(\widehat q-q)\), and \(c=(F-E)(\widehat q-q)\), then the full field error is \(a+b+c\). In addition to the field-only and solution-only vector differences, its sensitivity discrepancy contains
\(-2c^TD_cu-c^TD_cc-2a^TD_cb-2a^TD_cc-2b^TD_cc\).
Knowing only the three output norms does not determine these terms or their angles.

### J.5. The complete design derivative and its order

Differentiate \(J_IKE=0\) at fixed coordinates to obtain

\[
A E_{I,c}=-J_IK_{,c}E,\qquad S_{,c}=E^TK_{,c}E.
\]

The exact assembly compliance derivative is \(C_{,c}=-U^T\mathbb K_{,c}U\). For the surrogate, differentiation of its own equilibrium gives

\[
\widehat C_{,c}
=-\sum_m\left[\widehat u_m^TK_{m,c}\widehat u_m
+2(F_{I,m,c}\widehat q_m)^Tr_{I,m}\right].
\]

Only affected cells contribute. The design derivative of the solved trace has already been eliminated using equilibrium; the residual term differentiates \(F\) at fixed trace. It includes geometry-conditioned coefficients and every design-dependent correction operation.

Equivalently, differentiate the variational stiffness error:

\[
\boxed{(\widehat S-S)_{,c}
=H_{,c}^TAH+H^TA_{,c}H+H^TAH_{,c}.}

\tag{J.5}
\]

Its norm is bounded by \(2\|A\|\|H\|\|H_{,c}\|+\|A_{,c}\|\|H\|^2\). Thus a value error \(H=O(t)\) by itself does not imply a quadratic derivative error. A uniformly \(C^1\)-small extension error, \(H=tH_0(\tau)\) with bounded \(H_0,H_{0,c}\), gives \((\widehat S-S)_{,c}=O(t^2)\). With uniformly bounded exact solutions and inverse stiffnesses, this also yields \(\widehat C_{,c}-C_{,c}=O(t^2)\). One useful exact splitting is

\[
\widehat C_{,c}-C_{,c}
=-\widehat U^T\Delta_{,c}\widehat U
-(\widehat U-U)^T\mathbb K_{,c}(\widehat U+U).
\]

The \(O(t)\) field-estimate term can cancel against the residual term. At the same trace, \(J_IK_{,c}Eq=-AE_{I,c}q\). For \(H=tH_0\), the linear field-estimate error is \(+2t(H_0q)^TAE_{I,c}q\), while the linear residual-chain contribution is \(-2t(E_{I,c}q)^TAH_0q\). This cancellation explains how a first-order field estimate and a second-order complete derivative can coexist.

Under a fixed basis and exact integration, increasing one band parameter \(\tau_c\) with nonnegative \(Q_1\) shape functions enlarges the material domain. With fixed ghost stabilisation, this gives \(K_{,c}\succeq0\), \(S_{,c}\succeq0\), and \(C_{,c}\le0\). The field estimate is then also nonpositive for any field. Pointwise variational stiffness dominance does not itself enforce monotonicity of the surrogate with respect to design: the residual-chain term can change the sign of its complete derivative. This is a separate mechanical consistency question from symmetry and positive definiteness at a fixed design. Numerical moment derivatives inherit the monotonicity statement only when they represent the same nested-domain integration rule.


More explicitly, for \(h>0\), \(N_c^{Q_1}(x)\ge0\) implies \(\Omega(\tau)\subseteq\Omega(\tau+h e_c)\). For any discrete coefficient vector \(v\),
\[
v^T[K(\tau+h e_c)-K(\tau)]v
=\int_{\Omega(\tau+h e_c)\setminus\Omega(\tau)}
\epsilon(v):\mathsf C:\epsilon(v)\,dx\ge0.
\tag{J.6}
\]
Here the material tensor and basis are fixed and the ghost contribution cancels. Taking the differentiable limit proves positive semidefiniteness of the thickness derivative. A centred difference of stiffness matrices assembled from exactly nested domains is also PSD. Changes of adaptive integration subdivision, approximate moments, or independently selected stabilisation can interrupt that discrete nesting relation; fixed active topology alone does not verify its numerical preservation. No derivative eigenvalues or step-refinement study are saved in the selected sensitivity records.

This positive structure strengthens the sensitivity interpretation without requiring an indefinite derivative. Put \(a_c=u^TK_{,c}u=-s_c\ge0\), \(\zeta_c=d^TK_{,c}d\ge0\). Positive-semidefinite Cauchy–Schwarz gives
\[
|\widetilde s_c-s_c|\le2\sqrt{a_c\zeta_c}+\zeta_c,\qquad
\|\widetilde{\boldsymbol s}-\boldsymbol s\|_2
\le2\sqrt{\sum_c a_c\zeta_c}+\|\boldsymbol\zeta\|_2.
\tag{J.7}
\]
If \(\gamma_c=\|A^{-1/2}(K_{,c})_{II}A^{-1/2}\|_2\), then \(\zeta_c\le\gamma_c\mathcal E\), giving \(2\sqrt{\sum_c a_c\gamma_c}\sqrt{\mathcal E}+\|\boldsymbol\gamma\|_2\mathcal E\). The quadratic term in the signed sensitivity discrepancy is nonpositive; the cross term can have either sign. The reference eight-vector has nonpositive components. A small reference norm reflects small derivative-weighted local strain energy. For a general variable that moves a cut or redistributes material, an indefinite derivative is possible; that generality is unnecessary to explain the present corner-thickening mechanism.

If \(f=f(\tau)\), the compliance formula acquires \(2f_{,c}^TU\). If \(B_m\) varies, derivatives of \(B_m^TS_mB_m\) must be included. Fixed active topology does not, on its own, imply differentiability of quadrature branches or a moving-load map. These changes are different derivative problems rather than modifications of Eq. (J.5).

### J.6. Finite iteration residual and energy/action consistency

Let \(\bar U\) be an approximate solution of the same fixed variational operator, \(\rho=f-\widehat{\mathbb K}\bar U\), \(\bar C=f^T\bar U\), and \(\bar u_m=F_mB_m\bar U\). The exact identity becomes

\[
\boxed{C-\bar C
=a(\bar u-u,\bar u-u)+\bar U^T\rho,
\quad
J(\bar U)=\bar C+\bar U^T\rho\le\widehat C\le C.}

\]

The error of the corrected functional is \(\widehat C-J(\bar U)=\rho^T\widehat{\mathbb K}^{-1}\rho\), and \(\widehat C-\bar C=\widehat U^T\rho\). Thus the work \(f^T\bar U\) can violate the variational compliance ordering if the algebraic residual is appreciable, while the concave work functional retains a quadratic residual error. A Euclidean residual bound converts through

\[
\frac{|\bar U^T\rho|}{C}
\le\frac{\|\bar U\|_2\|f\|_2}{C}\frac{\|\rho\|_2}{\|f\|_2},
\]

with a load- and stiffness-dependent constant.

If an applied action \(y(\bar U)\) is not numerically identical to the variational energy, define
\(\omega=\bar U^Ty(\bar U)-\sum_m\bar u_m^TK_m\bar u_m\)
and \(\rho=f-y(\bar U)\). Then the first identity acquires an additional \(+\omega\). This separates solve residual from action/energy inconsistency. A recursive Krylov residual need not equal this applied-action residual. The deployment benchmark for predictor D records both residual norms. Evaluating the terms in Eq. (18) additionally requires the signed residual work and \(\omega\), which are not stored in those records.

### J.7. Directional training, coverage, rotations, and spectra

On the rigid complement, let \(S_*=Z^TSZ\succ0\) and
\(T_*=S_*^{-1/2}Z^TH^TAHZ S_*^{-1/2}\).
For an energy-normalised direction, \(x_j=S_*^{1/2}Z^Tq_j\) has unit Euclidean norm and \(\varepsilon_j=x_j^TT_*x_j\). Define its coverage matrix \(M_N=N^{-1}\sum_jx_jx_j^T\). Then

\[
\overline\varepsilon=\operatorname{tr}(T_*M_N),\qquad
M_N\succeq\alpha I, \alpha>0
\ \Longrightarrow\ 
\varepsilon_*\le\operatorname{tr}(T_*)\le\overline\varepsilon/\alpha.
\tag{J.8}
\]

If the directions do not span the complement, choose a unit \(v\) orthogonal to them and \(T_*=Mvv^T\). Every sampled error is zero and the worst error is \(M\), arbitrarily large. Such a PSD perturbation can be realised as \(H^TAH\) if there is at least one interior degree of freedom. Thus retaining all coordinates in P defines the approximation target, while adequate direction coverage controls its learned accuracy.

The logarithmic loss is nonnegative for the exact variational construction and behaves as \(\log(1+\varepsilon)=\varepsilon+O(\varepsilon^2)\) near zero. It weights large directional errors differently from their arithmetic mean. A finite average log loss \(L_N\) only supplies the weak sample bound \(\sum_j\varepsilon_j\le e^{NL_N}-1\); it has no unsampled-direction implication without coverage. With exact reference normalisation, Ritz searches give attained Rayleigh quotients and hence lower bounds on the true largest quotient. Geometry-averaged direction means, their population maxima, and the full operator supremum are different statistics.

For an orthogonal signed-permutation action, consistent changes \(K'=T_aKT_a^T\), \(q'=T_Pq\), \(E'=T_aET_P^T\), and \(B'_m=T_PB_mT_g^T\) preserve energies, nullspaces and assembly. A correspondingly transformed extension has the same property. Training on transformed examples encourages this relation but does not prove that an unconstrained learned map satisfies it on every group element. Geometry and loading transform together, and fields are returned to the original frame before comparison.

The identity \(\varepsilon=\delta^2\kappa\) uses the same full-field displacement norm for both Rayleigh quotients. Its denominator measures the physical response in the chosen trace direction. The smoother instead acts on the spectrum of \(A v=\lambda Dv\), \(D=\operatorname{diag}A\), with trace fixed. These concepts of softness are distinct. For example, \(A=\operatorname{diag}(\epsilon,1)\) has a very soft unscaled direction, while \(D^{-1/2}AD^{-1/2}=I\). An internal eigenmode also has zero retained trace and is not itself a free-cell Schur response. Consequently, cumulative Jacobi-scaled low-mode error energy identifies slowly relaxed components; it does not determine the measured \(\kappa\), a continuum bending-mode label, or a model-capacity lower bound.

### J.8. Corrections preserve the target and can improve different quantities differently

A geometry-fixed linear correction acts on \(H\) through \(H_{\rm new}=TH\). If \(T^TAT\preceq A\), then

\[
S\preceq\widehat S_{\rm new}\preceq\widehat S_{\rm old}.
\]

Thus exact assembled compliance increases monotonically toward the reference, and Eq. (11)'s total reconstructed energy error decreases. Neither a signed component of displacement nor the field-based sensitivity norm is ordered by this matrix inequality.

For Chebyshev smoothing, \(p_k(D^{-1}A)\) is self-adjoint in the \(A\)-inner product, and its contraction factor is the maximum of \(|p_k(\lambda)|\) over the actual positive spectrum. If the positive spectrum is contained in \((0,b]\), the standard normalised polynomial targeting \([a,b]\) has modulus at most one: on \([a,b]\) use \(|T_k|\le1\); below \(a\), its transformed argument lies between 1 and the normalising argument, where \(T_k\) is monotone. Modes below \(a\) can contract slowly. A finite power estimate multiplied by a safety factor is an estimate of the endpoint, not this containment theorem. Outside the upper bound, amplification is possible.

For a full-rank interior basis \(V\), set \(A_V=V^TAV\succ0\). Then \(C_V=I-VA_V^{-1}V^TA\) is an exact \(A\)-orthogonal projection onto the complement of the coarse space. The energy removed is \((V^Tr)^TA_V^{-1}(V^Tr)\). For a symmetric approximate inverse \(G\), the precise condition is

\[
\|e\|_A^2-\|e-VGV^TAe\|_A^2
=b^T(2G-GA_VG)b,\quad b=V^TAe.
\tag{J.9}
\]

Consequently \(2G-GA_VG\succeq0\) suffices for nonexpansiveness. For an accurate \(A_V\), a nonnegative shift \(G=(A_V+\Lambda)^{-1}\), \(\Lambda\succeq0\), satisfies this condition, although it is not the exact projection. This statement concerns the coarse inverse while the prescribed fine-grid stiffness \(K\) remains fixed. If a probed matrix differs from \(A_V\), contraction depends on the approximate inverse relative to the true \(A_V\).

In a pre-smooth/coarse/post-smooth cycle, \(H_{\rm tg}=P_kC_VP_kH_0\) and the energy bound is \(\|H_{\rm tg}q\|_A^2\le\rho_k^4\|H_0q\|_A^2\). At the operator-application level the entire affine field update, including the retained forcing, is transposed in reverse order. Geometry-fixed coefficients and a fixed iteration count ensure linearity. A fixed number of ordinary CG iterations generally does not: its coefficients depend on the right-hand side. Arbitrary direction-dependent stopping also need not define one linear extension.

The benefit of learning under a fixed correction budget is measured by the error surviving the correction: \(\|TH_{\rm net}q\|_A^2\) versus \(\|TH_{\rm start}q\|_A^2\). Initial errors with the same energy can leave different corrected errors. This connects direction-sensitive training to the numerical correction actually used.

Boundary-space restriction and interior correction affect different trial spaces. Exact interior condensation followed by a restriction \(U=Ry\) solves over a subspace of the retained coordinates; nested boundary spaces give nondecreasing Ritz compliance. Interior correction preserves all retained coordinates and changes the interior graph \(u=FBU\); its matrix ordering comes from contraction of \(H\), even though two such graph spaces need not be nested. In both cases compliance can approach the reference monotonically while a local sensitivity norm does not.

### J.9. Illustrative matrix examples

The following matrix examples isolate the roles of error amplitude, derivative coupling, and design regularity.

1. **Orders under re-equilibration.** A two-retained/three-internal SPD cell is coupled to a three-coordinate supported assembly. Five halved values of \(t\) verify the variational identity, Eq. (11), Eq. (12) and the sensitivity expansion. The saved final log-two slopes are approximately 2 for the operator, retained displacement and compliance, 1 for the full field and field-based sensitivity, and 4 for the retained-error energy in Eq. (11). Supplementary Note S2 gives the matrices, parameter sequence, and saved slope values for this illustrative algebraic example.

2. **Energy decreases while sensitivity error increases, even for a PSD derivative.** At the base design take \(K=I_3\), \(P=\{1\}\), \(u=(1,0,0)^T\), \(v=(1,1,-1)^T\), and \(K_{,c}=vv^T\succeq0\). The family \(K(\tau)=I+\tau vv^T\) is SPD for \(\tau>-1/3\) and increases with \(\tau\). Start with error \(d=(0,t,t)^T\) and project out the second coordinate. Its energy falls from \(2t^2\) to \(t^2\), while the sensitivity discrepancy changes from zero to \(2t-t^2\). The reference sensitivity is \(-1\), so this is also a relative-error counterexample. At \(t=.1\), the energies are .02 and .01 and the sensitivity discrepancies are zero and .19. The construction has the positive derivative structure of a nested thickening family.

3. **First-order field estimate, second-order total derivative with a PSD stiffness derivative.** Let
\[
K(\tau)=\begin{bmatrix}1+\tau&\tau\\\tau&1+\tau\end{bmatrix},\quad
E=(1,-\tau/(1+\tau))^T,\quad F_t=E+(0,t)^T.
\]
Then \(S=(1+2\tau)/(1+\tau)\) and \(\widehat S=S+(1+\tau)t^2\). For \(f=1\) at \(\tau=0\), the exact derivative is \(-1\), the field estimate is \(-(1+t)^2/(1+t^2)^2\), and the residual-chain term is \(2t/(1+t^2)^2\). Their sum is \(-1/(1+t^2)\), whose error is \(t^2/(1+t^2)\). The \(O(t)\) terms cancel while the stiffness derivative is positive semidefinite.

4. **Small values without small derivatives.** For a fixed load \(f=1\), let \(K=I_2\), \(E=(1,0)^T\), and \(F_t(\tau)=(1,t+\tau)^T\). At \(\tau=0\), the compliance gap is \(C-\widehat C_t=t^2/(1+t^2)\), whereas the signed derivative error is
\[
\left.\partial_\tau(\widehat C_t-C)\right|_{\tau=0}
=-\frac{2t}{(1+t^2)^2}.
\]
Uniform value convergence also does not imply derivative convergence: take \(H_t(\tau)=t\sin(\tau/t^2+\pi/4)\). At \(\tau=0\), the compliance gap is \(O(t^2)\), but
\[
\left.\partial_\tau(\widehat C_t-C)\right|_{\tau=0}
=-\frac{1}{(1+t^2/2)^2}\longrightarrow-1.
\]
Each family is differentiable; neither is uniformly \(C^1\)-small. The derivative of the positive gap \(C-\widehat C_t\) has the opposite sign.

5. **Fixed-design structure does not imply design monotonicity.** For \(K(c)=cI_2\), \(c>0\), \(E=(1,0)^T\), and \(F=(1,h(c))^T\), one has \(\widehat S=c(1+h^2)\ge S=c\). At \(c=1\), \(h=.1\), \(h'=-10\), \(\widehat S'=-.99\) and \(\widehat C'>0\) for \(f=1\), although \(C'=-1\). The field estimate remains negative. This counterexample isolates design variation of the extension from its valid pointwise energy structure.

6. **Trace drift and normalisation.** If an implemented field has retained coordinates \(q+b\), then relative to the original input
\[
\widehat u^TK\widehat u-q^TSq
=2b^TSq+b^TSb+\|\widehat u_I-E_I(q+b)\|_A^2.
\tag{J.10}
\]
Thus a first-order trace or normalisation discrepancy can mask a small quadratic internal error. Relative to the actual retained vector, the nonnegative internal-energy identity remains the appropriate comparison. Equation (J.10) separates retained-coordinate drift from internal reconstruction error and specifies the quantities needed for a consistent energy comparison.
