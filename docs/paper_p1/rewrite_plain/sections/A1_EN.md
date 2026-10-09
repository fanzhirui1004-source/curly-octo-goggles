## Appendix A. Discrete construction and metric definitions

### A.1. Active degrees of freedom and element integration

A background element is active if its intersection with \(\Omega(\eta)\) has positive measure. This is decided with closed-form interval enclosures of \(\phi\) and \(\tau\) over the element clipped by the cut plane. The element is excluded if an enclosure shows that a constraint is violated everywhere, accepted if both constraints hold strictly at its centre or a vertex, and subdivided otherwise. The active background elements use tensor-product \(Q_2\) displacements, with 27 nodes and 81 displacement degrees of freedom per element. The retained cell-face node set is the union of the nine face nodes of every certified positive-area material patch. The cut node set is the union of all 27 nodes of each active element carrying a positive-area cut-surface patch. Their union is deduplicated in active-node order. The DOFs of the cut-plane elements that lie off the cut plane are also retained, because their basis functions determine the displacement and the virtual work on the cut plane.

The discrete stiffness is assembled as

\[
K(\eta)=\sum_e L_e^TK_e(\eta)L_e
 +\gamma\sum_{f\in\mathcal F_g}L_f^TG_f^TG_fL_f,
\qquad
K_e(\eta)=\sum_{\alpha\in\{0,\ldots,4\}^3}M_{e\alpha}(\eta)T_\alpha.
\tag{A.1}
\]

Here \(L_e,L_f\) extract element and ghost-face DOFs. The fixed elasticity templates \(T_\alpha\in\mathbb R^{81\times81}\) use the physical basis derivatives and isotropic Lamé constants \(\lambda_L=E_Y\nu/[(1+\nu)(1-2\nu)]\) and \(\mu_L=E_Y/[2(1+\nu)]\). The 125 moments \(M_{e\alpha}\) integrate local monomials with coordinatewise exponents from zero to four over the material part of an element. The ghost-face set, its factors and coefficient specify the stabilisation contribution. The sum defines the stabilised discrete energy, including its ghost-penalty contribution.

The ghost-face set \(\mathcal F_g\) consists of interior background faces whose two neighbouring elements are active and are not both certified as completely filled with material. These faces lie within each substructure. For background-element width \(h=1/n\), the unit-coefficient stabilisation bilinear form is

\[
g_h(v,u)=(\lambda_{\rm ref}+2\mu_{\rm ref})\sum_{f\in\mathcal F_g}\sum_{j=1}^{2}
h^{2j-1}\int_f [\partial_{n_f}^{j}v]\cdot[\partial_{n_f}^{j}u] \,\mathrm dA.
\]

The derivative uses a common positive coordinate normal on both neighbouring elements, and the jump is the value on the first element minus that on the second. Integration covers the complete background face. The stabilisation form \(g_h\), and hence \(G_f\), uses fixed reference values \(E_{\rm ref}=1\), \(\nu_{\rm ref}=0.3\) and their Lamé constants \(\lambda_{\rm ref},\mu_{\rm ref}\); these equal the material values \(E_Y,\nu\) of the 80 validation geometries. Each face integral uses a tensor-product three-point Gauss rule in its two tangential coordinates. Stacking the square-root-weighted jump evaluations for both derivative orders and all three displacement components gives \(G_f\in\mathbb R^{54\times135}\), acting on the 45 distinct nodes of the two-element patch. Thus the second term in Eq. (A.1) is the matrix representation of \(\gamma g_h\).

The moments are integrated on \(4^3\) initial subcells per active background element. Partially occupied subcells are refined once into \(2^3\) children, so that the finest subcell width near the material boundary is \(h/8\). Refinement is local to partially occupied subcells and does not subdivide every active element uniformly. Fully occupied subcells contribute analytic tensor-product monomial moments. At the finest partial level, each subcell is divided into six Kuhn tetrahedra and clipped successively by the three interpolated inequalities defining the implicit band and the cut plane. The clipped tetrahedra are integrated with a conical Gauss–Jacobi product rule of four points per coordinate direction (64 points per tetrahedron). The accumulated moments are multiplied by the physical Jacobian \((2n)^{-3}\). Every quadrature weight is positive. The integrand that the templates expand in monomials is \(B^T\mathsf CB\), where \(B\) is the strain–displacement matrix and \(\mathsf C\) the elasticity tensor. Each numerically integrated \(K_e\) is therefore a positive combination of positive semidefinite matrices, and \(K\succeq0\) holds for the numerical moments as well.

Geometry generation accepts a cell only if its active background elements form a single face-connected set carrying retained cell-face DOFs. This establishes the connectivity of the stabilised discrete model, in which material connected only through partially filled elements is coupled through shared basis functions and the ghost penalty. Every training, validation, neighbour and lattice cell used in this work passes this check. This check yields the rigid-body kernel assumed in Section 2.2, provided the element integration is exact and the elasticity tensor is positive definite. Suppose \(v^TKv=0\). Both terms of Eq. (A.1) are positive semidefinite, so each vanishes. On every active element the material part has positive measure, so the \(Q_2\) field \(v_h\) has zero symmetric gradient on a set of positive measure. Since \(\nabla^{\rm s}v_h\) is polynomial, it then vanishes on the whole element, and \(v_h\) is rigid there. Two face-adjacent active elements share the nine nodes of their common face, which are not collinear, so their rigid motions coincide. Face-connectivity therefore makes \(v_h\) one global rigid motion. Conversely, a global rigid field is affine, so it has zero strain and zero jumps of its first and second normal derivatives, and the ghost term vanishes on it. Hence \(\ker K=\operatorname{range}R\). With numerical moments, the same conclusion requires the discrete element matrices to keep this kernel. That is an assumption, supported by the rigid-body energy ratios of Table ST04.

For the population statistics of Section 5, the energy error of a geometry is the mean over the evaluated test displacements of one load class. Population means then give equal weight to each available geometry.

## Appendix B. Variational identity, rigid-body kernel and energy-error measures

### B.1. Assumptions

In Appendices B and C, \(f\equiv f_g\) denotes the assembled retained load.

For each cell, \(K=K^T\succeq0\), the retained and interior DOF sets \(P,I\) are fixed, and \(A=K_{II}\succ0\). The exact recovery \(E\) is defined by \(E_P=I_p\) and \(E_I=-A^{-1}K_{IP}\). A linear recovery operator \(F\) has \(F_P=I_p\). Set \(H=F_I-E_I\), so \(F=E+J_I^TH\), \(S=E^TKE\), and \(\widehat S=F^TKF\). All transposes are taken of the complete recovery operator, including the correction. Interior body loads are zero; equivalently, the load functional depends only on the retained DOFs. Boundary loads integrated consistently have this property when all DOFs with nonzero boundary virtual work are retained.

The cell interiors are disjoint, all intercell couplings act through the retained DOFs, and the assembly maps \(B_m\) include fixed homogeneous supports. The supported exact assembled stiffness \(\mathbb K=\sum_m B_m^TS_mB_m\) is positive definite. The same local stabilised matrices define both the reference and the learned assemblies. The applied global load \(f\ne0\) is fixed.

Design statements additionally use a differentiable design interval on which the following discrete choices are fixed: the active elements; the ghost-face set \(\mathcal F_g\); the retained and interior DOF sets \(P,I\); the assembly maps \(B_m\) and the supports; the load; the network's binary node indicators and the sets of element and face stencils of weakly connected regions (Appendix G.1); and the coarse-column support screen and the coarse-factor shift \(\xi\) (Appendix F.1). A change in any of them is a switch of the discrete model, and derivatives are not taken across it. Stiffness derivatives include every varying term in the chosen discrete specification. In the sensitivities reported here, moment differences vary the body stiffness while the selected ghost matrix and its coefficient are fixed.

### B.2. Variational identity, transpose of the recovery operator and rigid-body kernel (proof of Proposition 1)

Let the selectors satisfy \(J_P^TJ_P+J_I^TJ_I=I_{n_a}\), and assume the symmetric stiffness in Eq. (2) has \(A\succ0\). For any field \(v\) with \(J_Pv=q\), there is a unique \(w\in\mathbb R^i\) such that \(v=Eq+J_I^Tw\). Since \(J_IKE=0\),

\[
v^TKv=q^TSq+w^TAw.
\tag{B.1}
\]

This proves the energy-minimising property and uniqueness of \(Eq\). Applying the same expansion to an admissible linear recovery operator \(F=E+J_I^TH\) yields Eq. (6). At fixed \(q\), the stabilised discrete strain energy of \(Fq\) therefore exceeds that of \(Eq\) by one half of \(q^TH^TAHq\). The reported energy error uses the quadratic energy without the factor one half. Moreover, \(r_I=J_IKFq=AHq\), which proves both expressions in Eq. (7).

Suppose additionally that \(\ker K=\operatorname{range}R\), \(R_P\) has rank six, and \(FR_P=R\). The exact field associated with \(R_Pa\) is \(Ra\), because it takes these retained displacements and satisfies interior equilibrium. Thus \(ER_P=R\) and \(HR_P=0\). Rigid reproduction gives \(\widehat S R_P=F^TKR=0\). Conversely, if \(q^T\widehat S q=0\), positive semidefiniteness implies \(Fq\in\ker K\). Write \(Fq=Ra\); selecting the retained entries gives \(q=R_Pa\). Hence

\[
\ker S=\ker\widehat S=\operatorname{range}R_P.
\tag{B.2}
\]

The inclusion \(\ker\widehat S\subseteq\ker S\) holds without these assumptions: \(\widehat S-S\succeq0\) and \(S\succeq0\), so \(q^T\widehat Sq=0\) implies \(q^TSq=0\) and hence \(Sq=0\). An approximate recovery operator therefore cannot add zero-energy modes to the condensed stiffness. A zero-energy mode other than the retained rigid-body modes could only belong to the discrete model itself. Appendix A.1 excludes such a mode under its connectivity condition and the assumption that the numerically integrated element matrices keep the rigid-body kernel. For the lattices of Table 5, the supported whole-lattice systems admitted a Cholesky factorisation, which excludes a zero-energy mode of the supported assembled system.

Rigid reproduction and symmetry give

\[
\widehat S R_P=0,\qquad R_P^T\widehat S=0.
\tag{B.3}
\]

For the construction in Eq. (9), \(C_RR_P=I_6\) and \(\Pi_PR_P=0\), and a linear raw displacement map sends zero to zero, so \(\widehat ER_P=R\). Corrections driven by the interior residual leave this field unchanged, so \(FR_P=R\) and Eq. (B.3) holds for the corrected recovery operator.

Let \(Z\in\mathbb R^{p\times(p-6)}\) have orthonormal columns spanning the orthogonal complement of the retained rigid-body modes, and define \(S_*=Z^TSZ\succ0\). The largest energy error over this space is

\[
\varepsilon_*=\sup_{q\perp R_P,\ q\ne0}\varepsilon(q)
=\|A^{1/2}HZ S_*^{-1/2}\|_2^2,
\qquad\mu_*=1+\varepsilon_*.
\tag{B.4}
\]

Let \(q=Za\), with \(Z^TZ=I\) spanning the rigid complement. Its relative energy error is

\[
\frac{a^TZ^TH^TAHZa}{a^TS_*a}
=\frac{\|A^{1/2}HZ S_*^{-1/2}b\|_2^2}{\|b\|_2^2},
\qquad b=S_*^{1/2}a.
\tag{B.5}
\]

Taking the supremum proves Eq. (B.4).

The retained block of the unbalanced discrete force, used alone, would be

\[
(KF)_P=S+K_{PI}H.
\]

The additional variational reaction is \(F_I^T(KF)_I=(E_I+H)^TAH\). Since \(E_I^TA=-K_{PI}\), its term \(E_I^TAH\) cancels \(K_{PI}H\). Thus

\[
(KF)_P+F_I^T(KF)_I=S+H^TAH.
\tag{B.6}
\]

The transpose \(F^T\) thus ensures reciprocity and cancels the first-order stiffness error. A retained-force extraction without this term generally has an \(O(H)\) error and need not be symmetric.

For any global vector, the quadratic error is the sum of nonnegative local errors in Eq. (C.1). Consequently \(\widehat{\mathbb K}\succeq\mathbb K\succ0\), which proves that the supported system is solvable even if some cells are only positive semidefinite.

### B.3. Displacement magnitude and stiffness ratio

The energy error weights displacement errors of equal magnitude differently. To make this dependence explicit, remove the rigid-body part of the retained displacements by taking \(q\perp R_P\), let \(d=Fq-Eq\), and introduce a displacement norm induced by \(\mathsf W\succ0\):

\[
\delta_{\mathsf W}(q)=\frac{\|d\|_{\mathsf W}}{\|Eq\|_{\mathsf W}},\qquad
\rho_{K,\mathsf W}(v)=\frac{v^TKv}{v^T\mathsf Wv},\qquad
\kappa_{\mathsf W}(q,d)=\frac{\rho_{K,\mathsf W}(d)}{\rho_{K,\mathsf W}(Eq)}.
\tag{B.7}
\]

For nonzero field error and nonzero reference energy, cancellation of \(d^T\mathsf Wd\) and \(u^T\mathsf Wu\), with \(u=Eq\), gives

\[
\boxed{\varepsilon(q)=\delta_{\mathsf W}(q)^2\kappa_{\mathsf W}(q,d).}
\tag{B.8}
\]

Section 5.4 and Table ST04 use \(\mathsf W=\operatorname{diag}(0,D)\) with \(D=\operatorname{diag}(K_{II})\), i.e. a Jacobi-weighted interior norm. This weighting is only positive semidefinite. The identity still holds, provided \(d_I\ne0\) and \(u_I\ne0\), because \(d\) vanishes on the retained DOFs; the condition \(q\perp R_P\) is then not needed. The factor \(\kappa_{\mathsf W}\) compares the stiffness content of the error with that of the equilibrium response; it is not a matrix condition number. An error concentrated in deformation stiffer than the loaded response has a large \(\kappa_{\mathsf W}\), so that even a small relative displacement error is mechanically significant.

The denominator of \(\kappa_{\mathsf W}\) measures the physical response to the chosen test displacement. The smoothing acts on the spectrum of \(Av=\lambda Dv\) with the retained displacements fixed, so the two notions of softness differ. For example, \(A=\operatorname{diag}(\epsilon_0,1)\) with \(0<\epsilon_0\ll1\) has a very soft unscaled mode, while \(D^{-1/2}AD^{-1/2}=I\). The cumulative Jacobi-scaled low-mode error energy (Eq. (10)) therefore identifies components that the smoothing reduces slowly, but does not determine the measured \(\kappa\).

## Appendix C. Assembly and compliance ordering

For a global retained vector \(U\), minimising the sum of substructure energies over their disjoint interior DOFs separates into the individual minimisations in Eq. (B.1). The minimised quadratic form is \(\sum_m U^TB_m^TS_mB_mU\). Under the coupling assumptions of Section 2.3, local condensation followed by assembly is therefore equivalent to eliminating the interior DOFs from the assembled system of all substructures.

Let \(\mathbb D=\widehat{\mathbb K}-\mathbb K\). From Eq. (6),

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

The order reversal follows by setting \(Q=\mathbb K^{-1/2}\widehat{\mathbb K}\mathbb K^{-1/2}\succeq I\). Its eigenvalues are at least one, so \(Q^{-1}\preceq I\); a congruence gives the inverse inequality.

### C.1. Compliance error as the error energy of the recovered field (proof of Proposition 3)

Let \(U=\mathbb K^{-1}f\), \(\widehat U=\widehat{\mathbb K}^{-1}f\), \(q_m=B_mU\), \(\widehat q_m=B_m\widehat U\), \(u_m=E_mq_m\), and \(\widehat u_m=F_m\widehat q_m\). The matrix \(\mathbb D=\widehat{\mathbb K}-\mathbb K\) of Eq. (C.1) equals \(\sum_m B_m^TH_m^TA_mH_mB_m\).

For conforming cell fields, use the energy form \(a(v,v)=\sum_m v_m^TK_mv_m\). Shared retained DOFs are counted through their separate cell energy contributions. Exact local equilibrium and the global equation give \(a(u,v)=f^TY\), where \(Y\) denotes the free assembled retained displacements of \(v\). At the equilibrium \(\widehat{\mathbb K}\widehat U=f\), \(a(\widehat u,\widehat u)=f^T\widehat U=\widehat C\). Hence

\[
\boxed{C-\widehat C=a(\widehat u-u,\widehat u-u)
=\|\widehat U-U\|_{\mathbb K}^2
+\sum_m\|H_m\widehat q_m\|_{A_m}^2.}
\]

The second equality follows from
\(\widehat u_m-u_m=E_m(\widehat q_m-q_m)+J_{I,m}^TH_m\widehat q_m\).
The two terms are \(K_m\)-orthogonal because \(J_{I,m}K_mE_m=0\). The underestimation of the compliance thus measures the total energy error of the recovered field, and the error of the retained solution accounts for only one part of it.

Define

\[
\beta=\frac{U^T\mathbb D U}{C}
=\sum_{m:q_m^TS_mq_m>0}w_m\varepsilon_m(q_m),
\qquad w_m=\frac{q_m^TS_mq_m}{C}.
\]

For zero-energy cells with exact rigid reproduction, the corresponding numerator is also zero and its contribution is defined directly as zero. Insert \(Y=\alpha U\) in the maximum principle

\[
\widehat C=\max_Y\{2f^TY-Y^T\widehat{\mathbb K}Y\}.
\]

Since \(U^T\widehat{\mathbb K}U=(1+\beta)C\), optimising the scalar gives \(\alpha=(1+\beta)^{-1}\) and \(\widehat C\ge C/(1+\beta)\), i.e. \((C-\widehat C)/C\le\beta/(1+\beta)\le\beta\). With Eq. (C.2), this proves Eq. (15). The statement is specific to the load, since \(w_m\) and \(\varepsilon_m\) are evaluated at the same exact assembled retained displacements.

### C.2. Inexact assembled solves

Let \(\bar U\) be an approximate solution of the same fixed variational operator, \(\rho=f-\widehat{\mathbb K}\bar U\), \(\bar C=f^T\bar U\), and \(\bar u_m=F_mB_m\bar U\). The exact identity becomes

\[
\boxed{C-\bar C
=a(\bar u-u,\bar u-u)+\bar U^T\rho,
\quad
J(\bar U)=\bar C+\bar U^T\rho\le\widehat C\le C.}
\]

The error of the corrected functional is \(\widehat C-J(\bar U)=\rho^T\widehat{\mathbb K}^{-1}\rho\), and \(\widehat C-\bar C=\widehat U^T\rho\). Thus the work \(f^T\bar U\) can violate the variational compliance ordering if the algebraic residual is appreciable, while the concave work functional retains a quadratic residual error.

If an applied action \(y(\bar U)\) is not numerically identical to the variational energy, define
\(\omega=\bar U^Ty(\bar U)-\sum_m\bar u_m^TK_m\bar u_m\)
and \(\rho=f-y(\bar U)\). Then the first identity acquires an additional \(+\omega\). This separates the residual of the solve from the inconsistency between the applied action and the energy. A recursive Krylov residual need not equal this applied-action residual. Evaluating the terms in Eq. (16) additionally requires the signed residual work and \(\omega\). For the two-cell configurations and the lattices, both are recorded together with the dual-norm bound \(|\bar U^T\rho|\le\sqrt{\bar U^T\mathbb K\bar U}\sqrt{\rho^T\mathbb K^{-1}\rho}\le\sqrt{\bar U^Ty(\bar U)-\omega}\,\sqrt{\rho^T\mathbb K^{-1}\rho}\); the second inequality holds because \(\widehat{\mathbb K}\succeq\mathbb K\) (Supplementary Note S4.3).

