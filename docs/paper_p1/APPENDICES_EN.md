## Appendix A. Discrete construction and metric definitions

### A.1. Active degrees of freedom and element integration

A background element is active if its intersection with \(\Omega(\eta)\) has positive measure. This is decided with closed-form interval enclosures of \(\phi\) and \(\tau\) over the element clipped by the cut plane: the element is excluded if an enclosure shows that a constraint is violated everywhere, accepted if both constraints hold strictly at its centre or a vertex, and subdivided otherwise. The active background elements use tensor-product \(Q_2\) displacements, with 27 nodes and 81 displacement degrees of freedom per element. Active DOFs are stored in a fixed node-major \(x,y,z\) order. The retained box set is the union of the nine face nodes of every certified positive-area material patch. The cut set is the union of all 27 nodes of each active element carrying a positive-area cut-surface patch. Their union is deduplicated in active-node order. The off-plane cut-band DOFs are retained because their basis functions determine displacement and virtual work on the cut plane.

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

All 80 validation geometries use \(n=32\), normalised Young's modulus \(E_Y=1\), Poisson's ratio \(\nu=0.3\), and ghost-penalty coefficient \(\gamma=10^{-4}\), with length expressed relative to the unit reference box. This gives 65 Q2 node positions per axis; the network's three transfer levels use 33, 17 and 9 positions per axis. The moment evaluator begins with \(4^3\) subcells per element and refines partial subcells once, then clips Kuhn tetrahedra using the tetrahedral rule parameter four. Appendix F.4 gives the full integration sequence. Geometry generation accepts a cell only if its active background elements form a single face-connected set carrying retained box-face DOFs; 10 of the 1,109 cells generated as training candidates failed this check, and every training, validation, neighbour and lattice cell used in this work passes it. Training populations are identified in Table ST01.

### A.2. Response metrics and aggregation

With \(K_{,c}=\partial K/\partial\tau_c\) on the stated design interval with fixed DOF sets, the metrics are

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

The pair-assembly comparison joins a learned target to an exact neighbour with matched thickness parameters on the common face. Its six face-load cases define the common 3% reference for compliance and sensitivity, with sensitivity maxima taken over both cells. Cut-traction cases are reported separately (Table ST10). Table 2 and Tables ST03, ST01 and ST09 identify the populations and loading sets, and Supplementary Note S6 specifies the two-cell configurations, the pair solver and the preconditioner of the lattice solves.

## Appendix B. Variational identity, rigid-body kernel, and directional norms

### B.1. Assumptions

In Appendices B and C, \(f\equiv f_g\) denotes the assembled retained load.

For each cell, \(K=K^T\succeq0\), the retained and interior DOF sets \(P,I\) are fixed, and \(A=K_{II}\succ0\). The exact extension is defined by \(E_P=I_p\) and \(E_I=-A^{-1}K_{IP}\). A linear extension \(F\) has \(F_P=I_p\). Set \(H=F_I-E_I\), so \(F=E+J_I^TH\), \(S=E^TKE\), and \(\widehat S=F^TKF\). All transposes include the entire extension and correction. Interior body loads are zero; equivalently, the load functional factors through the retained DOFs. Consistent boundary loads have this property when all DOFs with nonzero boundary virtual work are retained.

The cell interiors are disjoint, all intercell couplings act through the retained DOFs, and the assembly maps \(B_m\) include fixed homogeneous supports. The supported exact assembled stiffness \(\mathbb K=\sum_m B_m^TS_mB_m\) is positive definite. The same local stabilised matrices define both the reference and the learned assemblies. The applied global load \(f\ne0\) is fixed. These assumptions distinguish the modular reference from a different monolithic discretisation that introduces additional intercell stabilisation.

Design statements additionally use a differentiable interval on which the active and retained DOF sets, assembly maps, supports, and load are fixed. Stiffness derivatives include every varying term in the chosen discrete specification. In the sensitivities reported here, moment differences vary the body stiffness while the selected ghost matrix and its coefficient are fixed. The algebraic field-error expansions also hold for the symmetric finite-difference derivative matrices; interpreting them as exact design derivatives requires convergence to the derivative of that discrete system.

The full retained nodal representation need not be a minimal independent basis of surface traces. The cut band keeps the active DOFs needed for the cut-surface load functional. This sufficiency, and containment of all intercell couplings in \(P\), is the property used by condensation and assembly.

### B.2. Variational identity, transpose of the complete extension and rigid-body kernel

Let the selectors satisfy \(J_P^TJ_P+J_I^TJ_I=I_{n_a}\), and assume the symmetric stiffness in Eq. (2) has \(A\succ0\). For any field \(v\) with \(J_Pv=q\), there is a unique \(w\in\mathbb R^i\) such that \(v=Eq+J_I^Tw\). Since \(J_IKE=0\),

\[
v^TKv=q^TSq+w^TAw.
\tag{B.1}
\]

This proves the energy-minimising property and uniqueness of \(Eq\). Applying the same expansion to a linear admissible extension \(F=E+J_I^TH\) yields Eq. (4). At fixed \(q\), the stabilised discrete strain energy of \(Fq\) therefore exceeds that of \(Eq\) by one half of \(q^TH^TAHq\); the reported energy error uses the quadratic energy without the factor one half. Moreover, \(r_I=J_IKFq=AHq\), proving both expressions in Eq. (5). These identities require trace admissibility and the specified symmetric stiffness; a contraction property of the approximation is a separate condition.

Suppose additionally that \(\ker K=\operatorname{range}R\), \(R_P\) has rank six, and \(FR_P=R\). The exact field associated with \(R_Pa\) is \(Ra\), because it has that retained trace and satisfies interior equilibrium. Thus \(ER_P=R\) and \(HR_P=0\). Rigid reproduction gives \(\widehat S R_P=F^TKR=0\). Conversely, if \(q^T\widehat S q=0\), positive semidefiniteness implies \(Fq\in\ker K\). Write \(Fq=Ra\); selecting the retained entries gives \(q=R_Pa\). Hence

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

For the construction in Eq. (17), \(C_RR_P=I_6\) and \(\Pi_PR_P=0\). A linear raw displacement map sends zero to zero, so \(\widehat ER_P=R\). Corrections driven by the interior residual leave this field unchanged. These observations establish the two-sided rigid annihilation in Eq. (B.4) without altering any stiffness eigenvalue. Positive semidefiniteness alone does not imply rigid reproduction, and rigid reproduction alone does not exclude extra modes in a reference \(K\) with a larger kernel.

Let \(Z\in\mathbb R^{p\times(p-6)}\) have orthonormal columns spanning the retained rigid complement, and define \(S_*=Z^TSZ\succ0\). The all-direction energy error on this space is

\[
\varepsilon_*=\sup_{q\perp R_P,\ q\ne0}\varepsilon(q)
=\|A^{1/2}HZ S_*^{-1/2}\|_2^2,
\qquad\mu_*=1+\varepsilon_*.
\tag{B.5}
\]

Let \(q=Za\), with \(Z^TZ=I\) spanning the rigid complement. Its relative energy error is

\[
\frac{a^TZ^TH^TAHZa}{a^TS_*a}
=\frac{\|A^{1/2}HZ S_*^{-1/2}b\|_2^2}{\|b\|_2^2},
\qquad b=S_*^{1/2}a.
\tag{B.6}
\]

Taking the supremum proves Eq. (B.5). The basis \(Z\) is used for analysis; the operator application does not construct these whitened coordinates.

The retained block of the unbalanced discrete force, used alone, would be

\[
(KF)_P=S+K_{PI}H.
\]

The additional variational reaction is \(F_I^T(KF)_I=(E_I+H)^TAH\). Since \(E_I^TA=-K_{PI}\), its term \(E_I^TAH\) cancels \(K_{PI}H\). Thus

\[
(KF)_P+F_I^T(KF)_I=S+H^TAH.
\tag{B.7}
\]

The transpose \(F^T\) supplies both reciprocity and cancellation of the first-order stiffness error. A retained-force extraction without this term generally has an \(O(H)\) error and need not be symmetric.

For any global vector, the quadratic error is the sum of nonnegative local errors in Eq. (C.1). Consequently \(\widehat{\mathbb K}\succeq\mathbb K\succ0\). This also proves supported solvability even if some local cells are only semidefinite. A high condition number can still increase the cost and sensitivity of the numerical solve.

### B.3. Displacement magnitude and directional stiffness

The energy metric assigns different significance to displacement errors of equal magnitude. To make that dependence explicit, fix the retained rigid gauge by taking \(q\perp R_P\), let \(d=Fq-Eq\), and introduce a displacement norm induced by \(\mathsf W\succ0\):

\[
\delta_{\mathsf W}(q)=\frac{\|d\|_{\mathsf W}}{\|Eq\|_{\mathsf W}},\qquad
\mathcal R_{K,\mathsf W}(v)=\frac{v^TKv}{v^T\mathsf Wv},\qquad
\kappa_{\mathsf W}(q,d)=\frac{\mathcal R_{K,\mathsf W}(d)}{\mathcal R_{K,\mathsf W}(Eq)}.
\tag{B.8}
\]

For nonzero field error and nonzero reference energy, cancellation of \(d^T\mathsf Wd\) and \(u^T\mathsf Wu\), with \(u=Eq\), gives

\[
\boxed{\varepsilon(q)=\delta_{\mathsf W}(q)^2\kappa_{\mathsf W}(q,d).}
\tag{B.9}
\]

Section 5.4 and Table ST04 use \(\mathsf W=\operatorname{diag}(0,D)\) with \(D=\operatorname{diag}(K_{II})\), i.e. a Jacobi-weighted interior norm. This weighting is only positive semidefinite, but the identity still holds because \(d\) vanishes on the retained DOFs, and the rigid gauge is then not needed. The factor \(\kappa_{\mathsf W}\) compares the stiffness content of the error and the equilibrium response; it is not a matrix condition number. An error concentrated in stiffer deformation than the loaded response has a large \(\kappa_{\mathsf W}\), making even a small relative displacement error mechanically significant. Thus an energy target \(\varepsilon_{\rm tar}\) requires \(\delta_{\mathsf W}\le\sqrt{\varepsilon_{\rm tar}/\kappa_{\mathsf W}}\). Correction must address this energy content as well as displacement magnitude.

Changing the displacement normalisation changes the corresponding amplification factor. An alternative based on the retained displacement is

\[
\delta_P=\frac{\|d_I\|_2}{\|q\|_2},\qquad
\kappa_P=\frac{d_I^TAd_I/\|d_I\|_2^2}{q^TSq/\|q\|_2^2},
\qquad\varepsilon(q)=\delta_P^2\kappa_P.
\tag{B.10}
\]

In Eq. (B.9), the same displacement weighting enters both Rayleigh quotients, and the denominator of \(\kappa_{\mathsf W}\) measures the physical response in the chosen trace direction. The smoothing instead acts on the spectrum of \(A v=\lambda Dv\) with the trace fixed, so the two notions of softness are distinct. For example, \(A=\operatorname{diag}(\epsilon,1)\) has a very soft unscaled direction, while \(D^{-1/2}AD^{-1/2}=I\). An interior eigenmode also has zero retained trace and is not itself a free-cell Schur response. Consequently, cumulative Jacobi-scaled low-mode error energy identifies components that the smoothing reduces slowly; it does not determine the measured \(\kappa\), a continuum bending-mode label, or a model-capacity lower bound.

## Appendix C. Assembly and compliance ordering

For a global retained vector \(U\), minimising the sum of substructure energies over their disjoint interior DOFs separates into the individual minimisations in Eq. (B.1). The minimised quadratic form is \(\sum_m U^TB_m^TS_mB_mU\). This proves the equivalence of local condensation followed by assembly and elimination from the modular assembled system under the coupling assumptions in Section 2.3.

Let \(\mathbb D=\widehat{\mathbb K}-\mathbb K\). From Eq. (4),

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

For completeness, the order reversal follows by setting \(Q=\mathbb K^{-1/2}\widehat{\mathbb K}\mathbb K^{-1/2}\succeq I\). Its eigenvalues are at least one, so \(Q^{-1}\preceq I\); a congruence gives the inverse inequality. If every substructure also satisfies \(0\preceq\widehat S_m-S_m\preceq\varepsilon_* S_m\), applying the same argument to the upper and lower bounds gives

\[
\mathbb K\preceq\widehat{\mathbb K}\preceq(1+\varepsilon_*)\mathbb K,
\qquad\frac{C}{1+\varepsilon_*}\le\widehat C\le C,
\qquad\frac{C-\widehat C}{C}\le\frac{\varepsilon_*}{1+\varepsilon_*}.
\tag{C.3}
\]

The required local bound is uniform over directions. A mean over a finite set of sampled directions has a different statistical meaning.

The assembled state error obeys the exact identity

\[
\widehat U-U=-\widehat{\mathbb K}^{-1}\mathbb D U.
\tag{C.4}
\]

Consequently, for a fixed finite assembly with uniformly stable support and bounded inverse, a family \(H_m=O(t)\) gives \(\mathbb D=O(t^2)\) and \(\widehat U-U=O(t^2)\) as \(t\to0\) (Eq. (C.5)).

### C.1. Compliance error as the reconstructed error energy

Let \(U=\mathbb K^{-1}f\), \(\widehat U=\widehat{\mathbb K}^{-1}f\), \(q_m=B_mU\), \(\widehat q_m=B_m\widehat U\), \(u_m=E_mq_m\), and \(\widehat u_m=F_m\widehat q_m\). The matrix \(\mathbb D=\widehat{\mathbb K}-\mathbb K\) of Eq. (C.1) equals \(\sum_m B_m^TH_m^TA_mH_mB_m\).

For conforming cell fields, use the energy form \(a(v,v)=\sum_m v_m^TK_mv_m\). Shared retained DOFs are counted through their separate cell energy contributions. Exact local equilibrium and the global equation give \(a(u,v)=f^TV_P\), where \(V_P\) denotes the free assembled trace. At the equilibrium \(\widehat{\mathbb K}\widehat U=f\), \(a(\widehat u,\widehat u)=f^T\widehat U=\widehat C\). Hence

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

Define \(T=\mathbb K^{-1/2}\mathbb D\mathbb K^{-1/2}\succeq0\), \(z=\mathbb K^{-1/2}f\), and

\[
\beta=\frac{U^T\mathbb D U}{C}
=\sum_{m:q_m^TS_mq_m>0}w_m\varepsilon_m(q_m),
\qquad w_m=\frac{q_m^TS_mq_m}{C}.
\]

For zero-energy cells with exact rigid reproduction, the corresponding numerator is also zero and its contribution is defined directly as zero. Spectral calculus gives

\[
\frac{C-\widehat C}{C}
=\frac{z^TT(I+T)^{-1}z}{z^Tz},\qquad
\boxed{\frac{\beta}{1+\|T\|_2}\le\frac{C-\widehat C}{C}
\le\frac{\beta}{1+\beta}\le\beta.}
\]

The lower bound uses \(\lambda/(1+\lambda)\ge\lambda/(1+\|T\|_2)\). For the sharper upper bound, insert \(V=\alpha U\) in the maximum principle

\[
\widehat C=\max_V\{2f^TV-V^T\widehat{\mathbb K}V\}.
\]

Optimising the scalar gives \(\alpha=(1+\beta)^{-1}\) and \(\widehat C\ge C/(1+\beta)\). The share-weighted bound \(e_C\le\beta\) of Eq. (7) follows. This is a load-specific statement: \(w_m\) and \(\varepsilon_m\) are evaluated at the same exact assembled trace.

A uniform local inequality \(0\preceq\widehat S_m-S_m\preceq\varepsilon_*S_m\) implies \(T\preceq\varepsilon_*I\), which recovers the load-independent bound \(e_C\le\varepsilon_*/(1+\varepsilon_*)\) of Eq. (C.3). Rigid reproduction extends a quotient-space bound to arbitrary local traces.

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
\tag{C.5}
\]

Thus the retained displacement and compliance errors are \(O(t^2)\), the local full-field error is generically \(O(t)\), and the retained-error contribution in Eq. (6) is \(O(t^4)\). If \(\mathbb D_0U=0\), then \(H_{0,m}q_m=0\) for every affected cell, and this load is reproduced exactly for all \(t\). Uniform estimates across a family of geometries require uniform coercivity and bounded maps; fixed-geometry big-O constants do not supply those estimates automatically, nor do they determine which contribution dominates the sensitivity error at a finite error level.

### C.2. Inexact assembled solves

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
and \(\rho=f-y(\bar U)\). Then the first identity acquires an additional \(+\omega\). This separates solve residual from action/energy inconsistency. A recursive Krylov residual need not equal this applied-action residual. Evaluating the terms in Eq. (8) additionally requires the signed residual work and \(\omega\). For the pair configurations and lattices, both are recorded together with the dual-norm bound \(|\bar U^T\rho|\le\sqrt{\bar U^Ty(\bar U)}\sqrt{\rho^T\mathbb K^{-1}\rho}\), which holds because \(\widehat{\mathbb K}\succeq\mathbb K\) (Supplementary Note S6.3): at the final iterates, \(|\bar U^T\rho|\) and \(|\omega|\) stay below \(6\times10^{-8}\) of the compliance and the bound below \(2\times10^{-5}\), so the reported errors are those of the operator.

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

Write \(\bar\lambda=(a+b)/2\), \(\eta_s=(b-a)/2\), and \(\sigma=\bar\lambda/\eta_s>1\) for the centre and half-width of the target interval and their ratio. Starting from \(u_0\), keep its retained entries fixed and define the interior residual with the sign convention \(r_j=(Ku_j)_I\). A recurrence producing the polynomial in Eq. (D.1) is

\[
\begin{aligned}
\varrho_0&=1/\sigma,& z_0&=-D^{-1}r_0/\bar\lambda,\\
u_{j+1}&=u_j+J_I^Tz_j,&
\varrho_{j+1}&=(2\sigma-\varrho_j)^{-1},\\
z_{j+1}&=\varrho_{j+1}\varrho_j z_j
 -\frac{2\varrho_{j+1}}{\eta_s}D^{-1}r_{j+1}.
\end{aligned}
\tag{D.3}
\]

The last line is evaluated only when another step is needed. Zero steps return the initial field. To verify the polynomial, observe that \(\varrho_j=T_j(\sigma)/T_{j+1}(\sigma)\), and apply \(T_{j+2}(x)=2xT_{j+1}(x)-T_j(x)\). After the first step the error is \(d_1=(I-D^{-1}A/\bar\lambda)d_0\), and the same recurrence gives \(d_k=p_k(D^{-1}A)d_0\).

The matrices \(D^{-1}A\) and \(D^{-1/2}AD^{-1/2}\) are similar, \(p_k(D^{-1}A)\) is self-adjoint in the \(A\)-inner product, and

\[
\|p_k(D^{-1}A)d\|_A^2
=\sum_j\lambda_j p_k(\lambda_j)^2c_j^2,
\qquad c_j=v_j^TDd.
\tag{D.4}
\]

This proves Eq. (D.2). If \(a\le\lambda\le b\), the argument of \(T_k\) lies in \([-1,1]\), where \(|T_k|\le1\). For \(0<\lambda<a\), it lies between 1 and \(\sigma\), where \(T_k\) increases from 1 to \(T_k(\sigma)\). Hence \(|p_k(\lambda)|\le1\) throughout \((0,b]\), so the polynomial is nonexpansive in the \(A\)-energy norm when the actual positive spectrum is contained in \((0,b]\). Modes below \(a\) can contract slowly, and eigenvalues above \(b\) can be amplified.

The operational upper endpoint is estimated using a random power vector, 40 iterations of \(D^{-1}A\), Euclidean normalisation, and a factor of 1.05. The correction implementation fixes the random generator seed and caches the interval for the geometry. The separate fixed-weight smoothing study of the base network uses its own random start. The estimates are used to set the polynomial, while the result in Eq. (D.2) is conditional on its actual spectral values: a finite power estimate multiplied by a safety factor is an estimate of the endpoint, not the containment used above. For the 80 validation geometries we therefore compared the operational endpoint with \(\lambda_{\max}(D^{-1}A)\) computed by Lanczos iteration with full reorthogonalisation on \(D^{-1/2}AD^{-1/2}\) (78–150 steps; relative Ritz residual at most \(9.6\times10^{-4}\), median \(2.2\times10^{-6}\)). The operational endpoint exceeds the converged value by 2.1–5.0% (median 3.9%) on every geometry, so the containment required by Eq. (D.2) holds for all evaluated cells. The strict Gershgorin bound \(\max_i\sum_j|A_{ij}|/A_{ii}\) is 4.7–19.4 times the operational endpoint (median 18.4); used as \(b\), it would stretch the targeted interval by that factor and slow the contraction of the upper spectrum accordingly.

For a full-column-rank coarse basis \(V\), with \(A_c=V^TAV\succ0\), define \(Q_V=VA_c^{-1}V^TA\). Direct multiplication gives \(Q_V^2=Q_V\) and \(Q_V^TA=AQ_V\). Thus \(Q_V\) is an \(A\)-orthogonal projection and \(C_V=I-Q_V\) is its complementary projection. With \(b_r=V^TAd\),

\[
\|d\|_A^2=\|C_Vd\|_A^2+\|Q_Vd\|_A^2,
\qquad\|Q_Vd\|_A^2=b_r^TA_c^{-1}b_r.
\tag{D.5}
\]

This proves Eq. (14). Applying Eq. (D.2) before and after this projection yields \(\|\Phi_kC_V\Phi_kd\|_A\le\psi_k^2\|d\|_A\). Because \(\psi_k<1\) whenever the positive spectrum lies in \((0,b]\), the cycle is an energy contraction, which gives the operator ordering of Eq. (15) (Appendix D.1). For the spectra of U1, M1 and M2 (\(\lambda_1\approx4\)–\(9\times10^{-4}\) against \(a\approx0.17\); Supplementary Table ST05b), however, \(\psi_8^4\ge0.98\), so the estimate guarantees little more than non-expansion; it uses no approximation property of the coarse space, and the reductions by one to two orders of magnitude in Section 5.5 are empirical. A two-grid convergence estimate would require an approximation property of the \(Q_1\) space for walls about one element thick [Xu & Zikatanov (2002)](https://doi.org/10.1090/S0894-0347-02-00398-3). This is the standard energy-projection mechanism of subspace correction [Xu (1992)](https://doi.org/10.1137/1034116).

### D.1. Corrections, orderings and approximate coarse inverses

A geometry-fixed linear correction acts on \(H\) through \(H_{\rm new}=\Theta H\). If \(\Theta^TA\Theta\preceq A\), then

\[
S\preceq\widehat S_{\rm new}\preceq\widehat S_{\rm old}.
\]

Thus exact assembled compliance increases monotonically toward the reference, and Eq. (6)'s total reconstructed energy error decreases; in particular, repeating a fixed cycle cannot increase the energy error. Changing the smoothing degree, or stopping the recurrence of Eq. (D.3) at an intermediate step, changes the polynomial, so the energy error need not decrease monotonically in \(k\). Neither a signed component of displacement nor the field-based sensitivity norm is ordered by this matrix inequality.

For a symmetric approximate inverse \(G\) in place of \(A_c^{-1}\) in the coarse-grid correction \(C_V=I-VA_c^{-1}V^TA\) of Eq. (D.5), the precise condition is

\[
\|d\|_A^2-\|d-VGV^TAd\|_A^2
=b_r^T(2G-GA_cG)b_r,\quad b_r=V^TAd.
\tag{D.6}
\]

Consequently \(2G-GA_cG\succeq0\) suffices for nonexpansiveness. For an accurate \(A_c\), a nonnegative shift \(G=(A_c+\Lambda)^{-1}\), \(\Lambda\succeq0\), satisfies this condition, although it is not the exact projection. This statement concerns the coarse inverse while the prescribed fine-grid stiffness \(K\) remains fixed. If a probed matrix differs from \(A_c\), contraction depends on the approximate inverse relative to the true \(A_c\).

At the operator-application level the entire affine correction of the field, including the retained forcing, is transposed in reverse order, as in Eqs. (E.3)–(E.4). Geometry-fixed coefficients and a fixed iteration count ensure linearity. A fixed number of ordinary CG iterations generally does not: its coefficients depend on the right-hand side. Arbitrary direction-dependent stopping also need not define one linear extension.

The benefit of learning under a fixed correction budget is measured by the error surviving the correction: \(\|\Theta H_{\rm net}q\|_A^2\) versus \(\|\Theta H_{\rm start}q\|_A^2\). Initial errors with the same energy can leave different corrected errors. This connects direction-sensitive training to the numerical correction actually used.

Boundary-space restriction and interior correction affect different trial spaces. Exact interior condensation followed by a restriction \(U=G_ry\) solves over a subspace of the retained space; nested boundary spaces give nondecreasing Ritz compliance. Interior correction preserves all retained DOFs and changes the interior graph \(u=FBU\); its matrix ordering comes from contraction of \(H\), even though two such graph spaces need not be nested. In neither case is a local sensitivity norm ordered.

## Appendix E. Transpose of the complete extension

All transposes below use the Euclidean pairing of the stored displacement and nodal-force vectors. Write \(M_I=J_I^TJ_I\). Transposing Eq. (17) gives

\[
\widehat E^Ty=J_Py+C_R^TR^TM_Iy
 +\Pi_P^T\mathcal N_\theta(\eta)^TM_Iy.
\tag{E.1}
\]

The overwrite therefore selects the direct retained contribution and masks the field sent through the network transpose. The geometry coefficients are held fixed during this displacement transpose.

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

Indeed, the final interior field is its equilibrium value plus \(p_k(D^{-1}A)\) times the initial error. Replacing \((I-p_k(D^{-1}A))A^{-1}\) by \(\varphi_k(D^{-1}A)D^{-1}\) yields the first expression. The second follows from \((D^{-1}A)^TD=A=D(D^{-1}A)\), which implies \(p_k(D^{-1}A)^T=Dp_k(D^{-1}A)D^{-1}\) and the corresponding identity for \(\varphi_k\). Although the forward map preserves retained displacement, its transpose contributes a retained force through the upper block in Eq. (E.2).

For the full-field coarse-grid correction,

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

Coarse tensor-product shape functions are evaluated at the active background-node coordinates and restricted to \(I\). Standard \(Q_1\) and \(Q_2\) vector spaces carry three displacement DOFs per coarse node. The enriched partition-of-unity space carries 12 DOFs per vertex through fields of the form \(\sum_vN_v(x)[a_v+B_v(x-x_v)]\), with \(a_v\in\mathbb R^3\) and \(B_v\in\mathbb R^{3\times3}\). The spaces examined use \(Q_1\) grids of 9, 17, or 33 vertices per axis, a \(Q_2\) grid of 17 nodes per axis, and enriched \(Q_1\) grids of 9 or 17 vertices per axis.

Columns with an absolute interior-support sum at most \(10^{-14}\) are removed. In the fixed-weight CPU study, the symmetric interior matrix is formed from its stored upper triangle. The coarse matrix is explicitly computed as \(V^TAV\) and symmetrised. Columns whose diagonal energy is at most \(10^{-12}\) times the largest coarse diagonal are then removed. The resulting sparse matrix is factorised with PARDISO. This construction adds no explicit diagonal shift. The projection formulas in Eqs. (14)–(15) apply to linearly independent surviving coarse columns and the exact Galerkin action. Interpreting a recorded numerical solve through those formulas additionally requires verification of its solve accuracy.

The enriched generating functions are linearly dependent, and the surviving columns are not certified independent; the enriched entries of Table ST07 are therefore reported as numerical observations in Supplementary Note S3, and the \(Q_1(17)\) result is the principal coarse-correction result.

For each input direction, the study evaluates the exact field and the uncorrected network field once. It compares the network alone, a smoothing tail, a coarse-grid correction, a coarse-grid correction followed by smoothing, smoothing on both sides of the coarse-grid correction, and a zero-interior initialisation followed by the same complete cycle. Its energy ratios use a recomputed exact energy for the supplied direction. The fields are converted to float64 before correction.

### F.2. Differentiable correction implementation

The differentiable implementation constructs coarse support metadata and recovers a coarse matrix using coloured stiffness probes. Its configured stencil reach is four fine-grid node spacings. With \(h_g\) fine-node spacings per coarse-vertex spacing, the probe radius is \(R_g=2+\lceil4/h_g\rceil\); colours use vertex coordinates modulo \(2R_g+1\) and the component or enrichment slot. Recovery assumes the resulting colour separation resolves every interacting coarse pair. This setup supports the \(Q_1\) and enriched \(Q_1\) spaces. The explicit \(Q_2\) study uses the CPU construction in Appendix F.1.

The recovered matrix is Jacobi scaled with the corresponding column scaling of \(V\). Cholesky factorisation reads its lower triangle, trying diagonal shifts in the order \(0,10^{-12},10^{-10},10^{-8},10^{-6},10^{-4}\). The selected value is stored with the geometry factorisation (the coarse-factor shift); in the recorded setups it was 0 on six cells and \(10^{-12}\) on H1 (data archive record of the setup timings). On three lattice cells, the probed factor agrees with the element-assembled Galerkin factor to a relative \(1.3\times10^{-11}\) or better (data archive record of the probing test). To characterise such a solve algebraically, let \(V_s\) denote the scaled basis, \(A_s=V_s^TAV_s\succeq0\), \(\xi\ge0\), \(A_s+\xi I\succ0\), \(G_s=(A_s+\xi I)^{-1}\), and \(b_s=V_s^Tr_I\). Then

\[
\|d_I\|_A^2-\|d_I-V_sG_sb_s\|_A^2
=b_s^TG_s(A_s+2\xi I)G_sb_s\ge0.
\tag{F.1}
\]

Expanding the squared norms, as in Eq. (D.6) with basis \(V_s\), \(G=G_s\), and \(A_c\) replaced by \(A_s\), gives \(b_s^T(2G_s-G_sA_sG_s)b_s\), and multiplication by \(A_s+\xi I\) establishes the equality. A positive shift changes the exact projection but preserves this energy decrease when the solved matrix is the stated scaled Galerkin matrix. The identity assumes that matrix equality, symmetry, and consistent forward and transpose solves.

### F.3. Precision and normalisation

The learned displacement extension, rigid-body reconstruction, and prescribed retained entries use float32. Quadratic energies and sparse stiffness products use float64. The differentiable correction implementation computes its smoothing and coarse algebra in float64 in training and in the accuracy studies of Sections 5.2–5.8, and returns to the input displacement dtype; the timed lattice route of Table 5 runs the smoothing and the coarse solve in float32 (Supplementary Note S5). Convolutions use true float32 arithmetic; TF32 and lower precisions are not used. The explicit inference transpose also crosses the float64/float32 boundary around its network action. Sensitivity contractions use float32 element products with float64 accumulation. These arithmetic choices approximate the real linear maps in the preceding derivations.

| Evaluation path | Initial field and correction | Condensed action |
|---|---|---|
| Fixed-weight coarse study (base network) | Float32 network field converted to float64; explicit CPU Galerkin matrix and float64 correction | The saved comparison evaluates corrected field energies; it is separate from the deployment timing study. |
| Differentiable correction (training; accuracy evaluation of NICE, the corrected base network and Smoothing-trained) | Float32 learned field; smoothing and coarse algebra in float64; correction routines return to their input dtype | The correction and its transpose enter the stated variational action. Smoothing-trained uses eight smoothing steps and no coarse-grid correction. |
| Timed lattice route (Table 5) | Float32 network and rigid reconstruction; smoothing and coarse solve in float32; float64 stiffness product of the condensed action | Same operations and transpose as above; the fp32 correction changes the \(2\times2\times2\) compliance errors by less than \(3\times10^{-9}\) (Supplementary Note S5). |
| Explicit inference implementation | Float32 network and rigid reconstruction; float64 stiffness product; dtype conversions before the transpose action | The transpose \(F^T\) follows the actual configured correction sequence, with finite-precision consistency assessed separately. |

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

Thus a score formed by subtracting one also depends on the normalisation of the stored direction. Recomputed exact denominators, stored unit-energy assumptions, and finite-precision quadratic forms are separate parts of a numerically evaluated energy error.

### F.4. Moment integration

The reference integration initialises \(4^3\) subcells per active background element and refine partially occupied subcells once into \(2^3\) children. Thus the finest subcell width near the material boundary is \(h/8\). Fully occupied subcells contribute analytic tensor-product monomial moments. At the finest partial level, each subcell is divided into six Kuhn tetrahedra and clipped successively by the three interpolated inequalities defining the band and the cut plane. The tetrahedra are integrated with a rule of order four. The accumulated moments are multiplied by the physical Jacobian \((2n)^{-3}\). Refinement is local to partial subcells; it does not uniformly subdivide every active element to the finest level.

## Appendix G. Network coefficients and directional training

### G.1. Geometry features and displacement maps

The architecture in Figure 3 separates geometry-dependent coefficient generation from displacement propagation. Let \(N\) denote the number of active nodes, \(N_P\) the number of retained nodes and \(B\) the number of simultaneous displacement directions. Then \(n_a=3N\), \(p=3N_P\), and the deformational input \(q_d=\Pi_Pq\) has shape \(p\times B\). The fine displacement features have shape \(N\times B\times32\); a coarse latent level \(k\) with \(N^{(k)}\) participating vertices has shape \(N^{(k)}\times B\times32\), where \(k=1,2,3\) index the 33, 17 and 9 grids and \(k=0\) the fine 65 grid. Geometry embeddings, the 64-component element and node feature vectors of Section 4.4, carry no displacement-direction dimension. Element, face, retained-node and grid-transfer incidences are constructed before the network is evaluated.

**Geometry encoding.** Each element has 126 input features: material volume fraction, its logarithm, and the remaining 124 moments normalised by material volume. The 11 node features contain retained, box, cut-band and weak-support indicators, a normalised logarithmic stiffness diagonal, and six sine/cosine coordinate entries. The stiffness summary is the Frobenius norm of the node's diagonal \(3\times3\) block. A node is designated weakly supported when this norm is less than 0.01 times its median over active nodes. These features supply information about both material occupancy and its mechanical support.

The element encoder maps 126 inputs to a 64-component embedding. Mean aggregation of incident element embeddings, concatenated with the 11 node features, supplies the 75-input node encoder. Two residual message-passing rounds (the two rounds of exchange in Section 4.4) then update elements from the mean embeddings of their 27 nodes and update nodes from their incident elements. Each update uses the current embedding and the aggregated neighbouring embedding as a 128-component input. A ghost-face embedding is formed from the two adjacent element embeddings and a three-component indicator of the face axis. All encoders and coefficient heads use two affine layers separated by GELU, with hidden width 64. Every variant uses only the features described here.

**Local linear interactions.** The three retained displacement components are lifted by a shared matrix \(W_{\rm in}\in\mathbb R^{3\times32}\), while interior features are initially zero. Denote this initial feature field by \(X^0(q)\), with its dependence on \(q\) occurring through \(q_d\). For each element or face stencil \(t\), local node slot \(s\) and head \(h\), the interaction first gathers and mixes channels:

\[
Z_{th}=\left(\sum_{s=1}^{27}a_{tsh}(\eta)X_{i(t,s)}\right)W_{\ell h},
\qquad
\Delta X_i=\sum_{(t,s):i(t,s)=i}\sum_{h=1}^{4}
\omega_{ts}\,b_{tsh}(\eta)Z_{th}.
\]

Here \(W_{\ell h}\in\mathbb R^{32\times32}\) is a shared learned channel map for layer \(\ell\) and head \(h\). The scalar gather and scatter coefficients \(a\) and \(b\) are generated separately from the element or face embedding, the incident node embedding and an eight-dimensional slot embedding, a learned \(27\times8\) table indexed by the local slot (one table for element stencils and one for face stencils). Their 136-component concatenation identifies both the local material context and the position within the stencil. The four heads provide distinct gather--mix--scatter contributions on the same fixed incidence. Equation (16) adds their update to the incoming state and restores the prescribed retained features after every local interaction.

The element stencil contains all 27 \(Q_2\) nodes. The learned ghost-face stencil contains 18 owner-side nodes and nine neighbour-side nodes, ordered consistently by face direction. It defines a learned communication map on a face neighbourhood; the full mechanical ghost-penalty contribution remains in \(K\).

The node scatter accounts for the number and relative stiffness of incident contributions. If \(k_{es}\) is the Frobenius norm of the element's diagonal \(3\times3\) block at slot \(s\), the element weights are

\[
\pi_{es}=\frac{k_{es}}{\sum_{(e',s')\mapsto i}k_{e's'}},
\qquad
\omega_{es}=\frac{1-\lambda_{\rm mix}}{\deg(i)}
 +\lambda_{\rm mix}\pi_{es},\qquad(e,s)\mapsto i.
\tag{G.1}
\]

The face weights use the corresponding ghost-contribution diagonal blocks and face incidences. When all incident stiffness norms at a node vanish, the stiffness share is defined by inverse incidence count. Each layer has its own learned \(\lambda_{\rm mix}\), initialised at zero (plain incidence averaging), without a convex-interval constraint. This weighting changes coefficients on the prescribed scatter map. For the additional weak-region layers, a stencil is selected when it contains a weak node; its weights retain the normalisation of the full element or face incidence set.

**Multilevel propagation.** The latent hierarchy enlarges the spatial range of communication while retaining the fine field through additive skips. Let \(t_{ia}\) be the fixed trilinear incidence weight between a fine node \(i\) and a coarse vertex \(a\). Positive geometry coefficients \(r_i\) and \(p_i\) define restriction and prolongation componentwise:

\[
(\mathcal R X)_a=\frac{\sum_i t_{ia}r_iX_i}{\sum_i t_{ia}r_i},
\qquad
(\mathcal P X_c)_i=p_i\frac{\sum_a t_{ia}X_{c,a}}{\sum_a t_{ia}}.
\]

Only coarse vertices reached by the stored trilinear incidences participate. The denominators depend on geometry alone, so both maps are linear in displacement. Restriction and prolongation are separately parameterised. Their relationship does not impose symmetry on the raw extension; the transpose in Eq. (11) is obtained from the complete composition of operations.

On each coarse level, a residual convolution takes the form

\[
X\leftarrow X+g^{(k)}_j(\eta)\odot\operatorname{Conv}^{(k)}_j(X).
\]

Each convolution has a \(3\times3\times3\) kernel, 32 input and output channels, unit padding and no bias or displacement activation. Participating latent nodes are inserted into the corresponding background grid, with unused positions set to zero; the convolution output is sampled back at those nodes. The geometry branch supplies one multiplicative coefficient per participating node, channel and convolution. Its coarse embeddings are obtained by fixed trilinear weighted averaging.

The full displacement sequence is specified below. Grid sizes count background positions per axis, whereas the participating node counts depend on geometry. Every row carries 32 displacement channels until the final projection.

| Stage | Operations in execution order |
|---|---|
| Fine input, 65 grid | Lift \(q_d\) on retained nodes, set interior features to zero; apply four element-then-face interaction pairs; save \(Y^{(0)}\). |
| Downward, 33 grid | Restrict 65→33; apply two residual convolutions; save \(Y^{(1)}\). |
| Downward, 17 grid | Restrict 33→17; apply two residual convolutions; save \(Y^{(2)}\). |
| Downward, 9 grid | Restrict 17→9; apply two residual convolutions. |
| Upward, 9 grid | Apply two residual convolutions; prolong 9→17; add \(Y^{(2)}\). |
| Upward, 17 grid | Apply two residual convolutions; prolong 17→33; add \(Y^{(1)}\). |
| Upward, 33 grid | Apply two residual convolutions; prolong 33→65; add \(Y^{(0)}\). |
| Fine output, 65 grid | Restore retained features; apply four element-then-face interaction pairs, then four weak-region element-then-face pairs; project 32→3. |

At the upward step that returns to level \(k\in\{0,1,2\}\), the combination is \(X=Y^{(k)}+\sigma^{(k)}\mathcal P^{(k)}X_c\), where \(X_c\) is the field on level \(k+1\) and \(\sigma^{(k)}\) is a learned scalar. Thus convolution precedes prolongation, the skips are additive, and the 9-position grid receives two downward and two upward convolutions. There are twelve coarse convolution updates in total. The fine retained values are restored after the hierarchy returns to the active nodes. These latent operations communicate within the extension; the mechanical operator still accepts and returns all \(p\) retained DOFs. The coarse-grid (Galerkin) correction of Section 4.3 instead uses the interior stiffness and the interior equilibrium residual.

**Coefficient bounds and output.** The local heads produce separate gather/scatter values for eight ordinary element layers, twelve face layers and four additional weak-region element layers, each with four heads and 27 slots. Raw head outputs are multiplied by 0.2. For a finite positive group bound \(a_{\max}\), the coefficient map is

\[
\mathcal C(z)=
\begin{cases}
z,&|z|\le k_ba_{\max},\\
\operatorname{sgn}(z)\left[k_ba_{\max}+(1-k_b)a_{\max}
\tanh\!\left(\dfrac{|z|-k_ba_{\max}}{(1-k_b)a_{\max}}\right)\right],&|z|>k_ba_{\max}.
\end{cases}
\]

All variants of Table 2 set \(k_b=0.5\); each \(a_{\max}\) is twice the largest magnitude of its coefficient group recorded in a calibration pass over training geometries, so the map is the identity up to that recorded maximum. Table ST20 (Supplementary Note S7) lists the groups, the initialisation and the parameter count of every block. An infinite bound denotes the identity map. Bounds are fixed model-state arrays indexed by coefficient type, layer and gather/scatter role. Positive transfer coefficients apply the corresponding bound to \(\operatorname{softplus}(z)\) and add \(10^{-3}\); convolution coefficients use \(2\operatorname{sigmoid}(z)\). All these nonlinearities act on geometry-derived coefficients. The channel matrices, convolution kernels and skip scalars are shared learned parameters, whereas the coefficient-head outputs vary with geometry.

A bias-free matrix \(W_{\rm out}\in\mathbb R^{32\times3}\) returns the fine features to nodal displacement in the prescribed coordinate order. Rigid reconstruction and retained-value restoration then produce Eq. (17). The energy action Eq. (11) uses this complete map, together with the correction \(\mathcal W\), and its transpose; it therefore determines the force from the same displacement representation.

**Frame consistency.** For an orthogonal cube transformation, the deterministic signed permutation maps satisfy

\[
K'=T_aKT_a^T,\qquad E'=T_aET_P^T,\qquad S'=T_PST_P^T.
\tag{G.2}
\]

Here \(T_a\in\mathbb R^{n_a\times n_a}\) and \(T_P\in\mathbb R^{p\times p}\) are the signed permutation matrices that the cube transformation induces on the active and the retained DOFs of the cell, and primes denote quantities of the transformed cell. The augmented geometry and vector field are passed through the network and the predicted displacement is returned to the original frame for mechanical evaluation. These relations describe the transformed target and the frame-consistency condition. Sampling the cube views trains the geometry-conditioned extension towards that condition; displacement linearity alone does not establish rotational equivariance.

### G.2. Direction sets

The normalised retained directions are

\[
q_j=\frac{\Pi_Pq_j^{\rm raw}}
{\sqrt{(\Pi_Pq_j^{\rm raw})^TS(\Pi_Pq_j^{\rm raw})}},
\tag{G.3}
\]
where \(q_j^{\rm raw}\) is the direction produced by its class before rigid-body removal and energy normalisation.

The direction classes used by the trainer have the following mechanical definitions.

| Class | Construction before rigid removal and energy normalisation |
|---|---|
| `force` | Equilibrated nodal loads from smooth plane waves or localised Gaussian patches, with a subset loading the cut boundary. |
| `force_c` | Consistently integrated self-equilibrated tractions on material box faces; a subset also loads the cut face. Equilibrium is imposed in traction-quadrature space. |
| `face` / `face_c` | Single-face equilibrated nodal loads or consistently integrated tractions. |
| `support` | Responses with soft spring support on one box face and equilibrated loads on other faces. |
| `support_k` | Consistent loads with springs scaled by the local stiffness diagonal and a logarithmically sampled factor in \([0.3,3]\); the cut face remains unloaded. |
| `macro` | Equal sampling of uniform strain, quadratic, and cubic imposed displacement fields. |
| `grf` | Multiscale displacement fields spanning 0.5–24 spatial cycles across the reference box. |
| `glued` | Traces induced by a neighbouring cell with shared interface degrees of freedom and far-face clamping or springs; one quarter of samples load only the neighbour. |
| `adv` | Directions selected by the generalised-energy search described below. |

The exact reference computation for an isolated cell removes the rigid-body component and normalises the resulting retained direction by Eq. (G.3). Glued responses therefore supply a contextual direction while the energy and sensitivity labels refer to the target cell's operator. The direction sets store the normalised vectors in single precision.

### G.3. Optimisation and difficult-direction search

The variants of Table 2 use batches of 16 directions and change the geometry at every step; three geometries of the training set are held on the GPU, and one of them is replaced every 100 steps. Their nominal class weights, in the order `force`, `force_c`, `support`, `support_k`, `face`, `face_c`, `macro`, `grf`, `glued`, `adv`, are \(0.10,0.10,0.075,0.075,0.10,0.10,0.10,0.125,0.075,0.15\). The class weights are renormalised over the available classes and assigned by systematic quotas. A run of \(N_{\rm st}\) training steps therefore visits at most \(N_{\rm st}/100+3\) distinct geometries of its training set, which contains 305 geometries for the base network and 591 for the continuations (Table ST01).

Optimisation uses Adam, a one-cycle learning-rate schedule with peak \(3\times10^{-4}\), 5% warm-up, cosine decay, and final division factor 100. The gradient norm is clipped at one. Model selection and evaluation use an exponential moving average of the network parameters, with decay 0.9997 and zero-initialisation bias correction. Table 2 identifies the variants, and Table ST01 gives the training and model-selection settings of each. The energy term in Eq. (18) uses a \(10^{-12}\) lower clamp inside the logarithm, and the sensitivity term, the squared relative error of the thickness sensitivity vector, has unit weight.

NICE and the Smoothing-trained variant continue the base network from its selected parameters for 15,000 training steps with the same schedule, class weights, training set and seed as the Uncorrected continuation, so that the three continuations differ only in the correction used in the forward map of Eq. (18): none for Uncorrected, eight Chebyshev smoothing steps for Smoothing-trained, and the eight-step, \(Q_1(17)\), eight-step sequence for NICE. The correction is recomputed for every geometry held on the GPU (smoothing interval by power iteration, coarse matrix by coloured probing, Appendix F.2) and is treated as a fixed linear map of the network output. Gradients of Eq. (18) are propagated through the smoothing recurrence and the coarse-grid correction by automatic differentiation in double precision; the transposes used at evaluation (Appendix E) are the explicit adjoints of the same operations. Model selection (Section 5.1, Table ST01) compared the network parameters after 7,500 and 15,000 training steps of each continuation and after 10,000, 20,000, 30,000 and 40,000 steps of the base network; the continuations use those after 15,000 steps and the base network those after 30,000.

The adversarial search initialises eight candidate directions and performs four block iterations whenever a geometry enters the set held on the GPU. Applying \(S^{-1}_{\perp}\) uses an equilibrated Neumann solve: six deterministically selected displacement pins remove the rigid-body motion, and the result is projected into the retained rigid complement. A search within a stored direction set instead forms \(G=Q_b^TSQ_b\) and \(\widehat G=(FQ_b)^TK(FQ_b)\), solves a generalised eigenproblem after flooring the search metric at \(10^{-3}\lambda_{\max}(G)\), and renormalises selected candidates with the original \(G\). The floor applies only within the search. Candidate energy ratios remain directional observations; their maximisation does not supply an all-direction upper bound.

### G.4. Direction coverage, symmetry and spectra

With \(Z\) and \(S_*=Z^TSZ\succ0\) as in Eq. (B.5), let
\(T_*=S_*^{-1/2}Z^TH^TAHZ S_*^{-1/2}\).
For an energy-normalised direction, \(x_j=S_*^{1/2}Z^Tq_j\) has unit Euclidean norm and, by Eq. (B.6), \(\varepsilon_j=x_j^TT_*x_j\). For \(N_q\) sampled directions, define the coverage matrix \(M_{N_q}=N_q^{-1}\sum_{j=1}^{N_q}x_jx_j^T\). Then

\[
\overline\varepsilon=\operatorname{tr}(T_*M_{N_q}),\qquad
M_{N_q}\succeq\alpha I, \alpha>0
\ \Longrightarrow\ 
\varepsilon_*\le\operatorname{tr}(T_*)\le\overline\varepsilon/\alpha.
\tag{G.4}
\]

If the directions do not span the complement, choose a unit \(v\) orthogonal to them and \(T_*=\Theta\,vv^T\) with \(\Theta>0\). Every sampled error is zero and the worst error is \(\Theta\), arbitrarily large. Such a PSD perturbation can be realised as \(H^TAH\) if there is at least one interior degree of freedom. Thus retaining all DOFs in P defines the approximation target, while adequate direction coverage controls its learned accuracy.

The logarithmic loss is nonnegative for the exact variational construction and behaves as \(\log(1+\varepsilon)=\varepsilon+O(\varepsilon^2)\) near zero. It weights large directional errors differently from their arithmetic mean. A finite average log loss \(L_{N_q}\) over these directions only supplies the weak sample bound \(\sum_j\varepsilon_j\le e^{N_qL_{N_q}}-1\); it has no unsampled-direction implication without coverage. With exact reference normalisation, Ritz searches give attained Rayleigh quotients and hence lower bounds on the true largest quotient. Geometry-averaged direction means, their population maxima, and the full operator supremum are different statistics.

For an orthogonal signed-permutation action, the transformations in Eq. (G.2) together with \(q'=T_Pq\) and \(B'_m=T_PB_mT_g^T\), where \(T_g\) is the signed permutation that the same transformation of the assembly induces on the free global retained displacement \(U\), preserve energies, nullspaces and assembly. A correspondingly transformed extension has the same property. Training on transformed examples (Appendix G.1) encourages this relation but does not prove that an unconstrained learned map satisfies it on every group element.

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

On a differentiable interval with fixed active topology and fixed selectors \(J_P\) and \(J_I\), differentiating \(J_PE=I_p\) gives \(J_PE_{,c}=0\). Since \(J_IKE=0\), both terms \(E_{,c}^TKE\) and \(E^TKE_{,c}\) vanish. Differentiating \(S=E^TKE\) therefore proves Eq. (H.1).

For a supported equilibrium, differentiating \(\mathbb K U=f_g\) at a fixed load gives \(U_{,c}=-\mathbb K^{-1}\mathbb K_{,c}U\) and hence \(C_{,c}=-U^T\mathbb K_{,c}U\). With fixed gathers and Eq. (H.1), this becomes the sum of \(-u_m^TK_{m,c}u_m\) over affected cells. For an isolated energy-normalised direction, this label holds the base-design force fixed. The assembled sensitivity labels likewise hold the base-design nodal load fixed, including loads generated from consistent tractions. Reassembling a prescribed surface traction over a changing material boundary introduces the load-derivative term stated below; differentiating the direction normalisation also defines a different quantity.

Expanding the field-based quadratic estimate at \(u+d\) proves Eq. (9). In the Euclidean norm, it implies

\[
|\widetilde s_c-s_c|
\le2\|K_{,c}u\|_2\|d\|_2+\|K_{,c}\|_2\|d\|_2^2.
\tag{H.3}
\]

For the corner-thickness band in Eq. (1), exact integration with fixed basis functions and ghost stabilisation gives \(K_{,c}\succeq0\) and nonpositive equilibrium compliance derivatives; Appendix H.2 proves this (Eq. H.10) and shows that the discrete moments of this study preserve it only approximately. The linear cross term in Eq. (9) can have either sign, while under exact integration the quadratic term is nonpositive; their relative importance depends on derivative-weighted alignment and error amplitude. The numerical moment-difference implementation is specified below. In an assembly, substituting Eq. (H.2) into the same quadratic expansion includes both the change of trace and its interaction with the extension error (Appendix H.1).

For the condensed stiffness \(\widehat S=F^TKF\), direct differentiation gives

\[
\widehat S_{,c}=F_{,c}^TKF+F^TK_{,c}F+F^TKF_{,c}.
\tag{H.4}
\]

Apply the equilibrium compliance derivative to the assembly of learned substructures and use \(J_PF_{,c}=0\). The two extension terms become \(2(F_{I,c}\widehat q)^Tr_I\), proving Eq. (10). Its magnitude is bounded by

\[
|\widehat C_{,c}-\widetilde s_c|
\le2\|F_{I,c}\widehat q\|_A\|r_I\|_{A^{-1}}
\tag{H.5}
\]

for a single affected cell, with a sum of such bounds for several cells. This bound involves the extension's design derivative as well as its equilibrium residual. It provides no fixed error order in the field amplitude without a corresponding assumption on that derivative. If the load depends on design, the full compliance derivative also contains \(2f_{g,c}^T\widehat U\), and correspondingly \(2f_{g,c}^TU\) for the exact assembly; changes of the DOF selectors, assembly maps or support maps contribute their own chain-rule terms, including the derivatives of \(B_m^TS_mB_m\) when \(B_m\) varies. Fixed active topology does not, on its own, imply differentiability of quadrature branches or a moving-load map.

In the reference computation, moment derivatives are approximated by

\[
M_{e\alpha,c}\approx
\frac{M_{e\alpha}(\boldsymbol\tau+h_ce_c)-M_{e\alpha}(\boldsymbol\tau-h_ce_c)}{2h_c},
\qquad h_c=10^{-5}\tau_c.
\tag{H.6}
\]

The active set is fixed for this calculation, and the elasticity templates and ghost contribution are held constant. The moment quadrature is recomputed at each perturbed thickness, so its clipping and integration branches may change. The resulting matrices represent a numerical design derivative on that prescribed topology. Training differentiates the field-based sensitivity loss with respect to the predicted field and network parameters. Computing the complete design derivative in Eq. (10) additionally differentiates the geometry-dependent extension and correction maps.

### H.1. Sensitivity error bounds

For a fixed retained \(q\), set \(u=Eq\), \(d=J_I^THq\), and \(\mathcal E=d^TKd=(Hq)^TA(Hq)\). The derivatives \(K_{,c}\), \(c=1,\ldots,8\), are symmetric. Define the reference vector \(\boldsymbol s\) with components \(s_c=-u^TK_{,c}u\), and assume \(\|\boldsymbol s\|_2>0\). Eq. (9) then reads

\[
\widetilde s_c-s_c=-2d^TK_{,c}u-d^TK_{,c}d.
\]

Let \(b_c=J_IK_{,c}u\), \(K_{II,c}=J_IK_{,c}J_I^T\), and let the rows of \(\mathsf B_s\in\mathbb R^{8\times i}\) be \(b_c^TA^{-1/2}\). Define

\[
L=\|\mathsf B_s\|_2,\qquad
Q=\left(\sum_{c=1}^8\|A^{-1/2}K_{II,c}A^{-1/2}\|_2^2\right)^{1/2}.
\]

Cauchy–Schwarz and the quadratic-form operator bound give

\[
\boxed{\|\widetilde{\boldsymbol s}-\boldsymbol s\|_2
\le2L\sqrt{\mathcal E}+Q\mathcal E,
\qquad e_s\le\frac{2L\sqrt{q^TSq}}{\|\boldsymbol s\|_2}\sqrt\varepsilon
+\frac{Qq^TSq}{\|\boldsymbol s\|_2}\varepsilon.}
\tag{H.7}
\]

All eight components are included, and the denominator is the norm of the exact sensitivity vector for that particular direction. The constants distinguish derivative coupling, interior coercivity, and reference sensitivity scale. For fixed \(q\) and \(H=tH_0\), the linear coefficient vanishes precisely when \((H_0q)^Tb_c=0\) for every corner. It vanishes for all interior errors if \(J_IK_{,c}E q=0\) for every corner. A useful special case is \(K_{,c}=\alpha_cK\): interior equilibrium removes the linear term and the relative sensitivity error equals the energy error when the reference vector is nonzero.

For a re-equilibrated assembly, the quadratic expansion remains exact with \(d_m=\widehat u_m-u_m\), but \(d_m\) now includes a changed retained trace. Eq. (C.5) gives, componentwise,

\[
\begin{aligned}
\widetilde s_{m,c}(t)-s_{m,c}
={}&-2t(J_I^TH_{0,m}q_m)^TK_{m,c}u_m\\
&+t^2\{2(E_mB_m\mathbb K^{-1}\mathbb D_0U)^TK_{m,c}u_m\\
&\qquad-(J_I^TH_{0,m}q_m)^TK_{m,c}(J_I^TH_{0,m}q_m)\}+O(t^3).
\end{aligned}
\]

The leading term is the same as the fixed-trace term. The solution-only replacement is \(O(t^2)\); its coefficient can be large on a soft assembly or relative to a small local reference. If only one cell has a learned extension, an exact neighbouring cell has no \(O(t)\) reconstruction term, although its solved trace can change at order \(t^2\).

There is also a global-to-local bound. Suppose \(K_m\) has its fixed rigid-body kernel, each \(K_{m,c}\) annihilates that kernel, and

\[
\Gamma_m=\left(\sum_c
\|K_{m,*}^{-1/2}K_{m,c,*}K_{m,*}^{-1/2}\|_2^2\right)^{1/2}<\infty
\]

on its rigid complement. Since \(u_m^TK_mu_m=q_m^TS_mq_m=w_mC>0\), set \(\chi_m=\|\boldsymbol s_m\|_2/(w_mC)>0\) and \(e_C=(C-\widehat C)/C\). The field expansion and Eq. (6) yield

\[
\boxed{e_{s,m}\le\frac{\Gamma_m}{\chi_m}
\left(2\sqrt{\frac{e_C}{w_m}}+\frac{e_C}{w_m}\right).}
\tag{H.8}
\]

Indeed, the local error energy is at most \(e_CC\). Substitution into
\(\|\delta\boldsymbol s_m\|_2\le\Gamma_m\big(2\sqrt{w_mC\,a_m(d_m,d_m)}+a_m(d_m,d_m)\big)\)
proves the result. This bound makes the roles of a small energy share and a small reference sensitivity explicit. The recorded results do not measure \(\Gamma_m\), \(\chi_m\), or the local error energy, so Eq. (H.8) is a quantitative theoretical explanation, not a fitted attribution of a reported percentage.

The field-only and solution-only replacements of Table ST11 do not form an additive sensitivity decomposition. Denote the three terms of Eq. (H.2) by \(d^{(1)}=(F-E)q\), \(d^{(2)}=E(\widehat q-q)\) and \(d^{(3)}=(F-E)(\widehat q-q)\), so that the full field error is \(d^{(1)}+d^{(2)}+d^{(3)}\). In addition to the field-only and solution-only vector differences, its sensitivity discrepancy contains
\(-2d^{(3)T}K_{,c}u-d^{(3)T}K_{,c}d^{(3)}-2d^{(1)T}K_{,c}d^{(2)}-2d^{(1)T}K_{,c}d^{(3)}-2d^{(2)T}K_{,c}d^{(3)}\).
Knowing only the three output norms does not determine these terms or their angles.

### H.2. The complete design derivative and the sign of the thickness derivative

Differentiating \(J_IKE=0\) with fixed selectors and \(J_PE_{,c}=0\) gives

\[
A E_{I,c}=-J_IK_{,c}E,
\]

while the condensed derivative \(S_{,c}=E^TK_{,c}E\) is Eq. (H.1). The exact assembly compliance derivative \(C_{,c}=-U^T\mathbb K_{,c}U\) and, from the equilibrium of the assembly with \(\widehat S\), Eq. (10) were derived above: \(\widehat C_{,c}\) is the sum over affected cells of \(\widetilde s_{m,c}-2(F_{I,m,c}\widehat q_m)^Tr_{I,m}\). The design derivative of the solved trace has already been eliminated using equilibrium; the residual term differentiates \(F\) at fixed trace. It includes geometry-conditioned coefficients and every design-dependent correction operation.

Equivalently, with \(K_{II,c}\) as in Appendix H.1, differentiate the variational stiffness error:

\[
\boxed{(\widehat S-S)_{,c}
=H_{,c}^TAH+H^TK_{II,c}H+H^TAH_{,c}.}
\tag{H.9}
\]

Its norm is bounded by \(2\|A\|\|H\|\|H_{,c}\|+\|K_{II,c}\|\|H\|^2\). Thus a value error \(H=O(t)\) by itself does not imply a quadratic derivative error. A uniformly \(C^1\)-small extension error, \(H=tH_0(\tau)\) with bounded \(H_0,H_{0,c}\), gives \((\widehat S-S)_{,c}=O(t^2)\). With uniformly bounded exact solutions and inverse stiffnesses, this also yields \(\widehat C_{,c}-C_{,c}=O(t^2)\). One useful exact splitting is

\[
\widehat C_{,c}-C_{,c}
=-\widehat U^T\mathbb D_{,c}\widehat U
-(\widehat U-U)^T\mathbb K_{,c}(\widehat U+U),
\]
where \(\mathbb D=\widehat{\mathbb K}-\mathbb K\) as in Eq. (C.1).

The \(O(t)\) field-estimate term can cancel against the residual term. At the same trace, \(J_IK_{,c}Eq=-AE_{I,c}q\). For \(H=tH_0\), the linear field-estimate error is \(+2t(H_0q)^TAE_{I,c}q\), while the linear residual-chain contribution is \(-2t(E_{I,c}q)^TAH_0q\). This cancellation explains how a first-order field estimate and a second-order complete derivative can coexist.

Under a fixed basis and exact integration, increasing one band parameter \(\tau_c\) with nonnegative \(Q_1\) shape functions enlarges the material domain. With fixed ghost stabilisation, this gives \(K_{,c}\succeq0\), \(S_{,c}\succeq0\), and \(C_{,c}\le0\). The field estimate is then also nonpositive for any field. Pointwise variational stiffness dominance does not itself enforce monotonicity of the surrogate compliance with respect to design: the residual-chain term can change the sign of its complete derivative. This is a separate mechanical consistency question from symmetry and positive definiteness at a fixed design. Numerical moment derivatives inherit the monotonicity statement only when they represent the same nested-domain integration rule.

More explicitly, for \(h>0\), \(N_c^{Q_1}(x)\ge0\) implies \(\Omega(\tau)\subseteq\Omega(\tau+h e_c)\). For any discrete coefficient vector \(v\), with interpolated displacement field \(v_h\),
\[
v^T[K(\tau+h e_c)-K(\tau)]v
=\int_{\Omega(\tau+h e_c)\setminus\Omega(\tau)}
\nabla^{\rm s}v_h:\mathsf C:\nabla^{\rm s}v_h\,dx\ge0.
\tag{H.10}
\]
Here \(\nabla^{\rm s}\) is the symmetric gradient, the elasticity tensor \(\mathsf C\) and the basis are fixed, and the ghost contribution cancels. Taking the differentiable limit proves positive semidefiniteness of the thickness derivative. A centred difference of stiffness matrices assembled from exactly nested domains is also PSD. Changes of adaptive integration subdivision, approximate moments, or independently selected stabilisation can interrupt that discrete nesting relation; fixed active topology alone does not verify its numerical preservation. Figure S01(c) reports a step-refinement study of the numerical derivative on four cells: relative to the production step \(10^{-5}\tau_c\), the sensitivity changes by at most \(2.6\times10^{-7}\) at \(10^{-3}\tau_c\), \(2.5\times10^{-9}\) at \(10^{-4}\tau_c\) and \(1.6\times10^{-10}\) at \(10^{-6}\tau_c\); the hundredfold reduction per decade is the second-order truncation of the central difference, so no integration branch changes within \(\pm10^{-3}\tau_c\) at the fixed active set on these cells. On the same cells, the central-difference stiffness derivative agrees with differences of the re-solved compliance to \(4\times10^{-8}\). The element-level check of Supplementary Table ST15 shows that the discrete moments do not preserve the nesting everywhere. The element stiffnesses are positive semidefinite to rounding, but the exact derivative of the discrete moments at fixed clipping topology gives element derivative matrices with negative eigenvalues: in the uncut cells the smallest ratio \(\lambda_{\min}/\max|\lambda|\) is \(-2.3\times10^{-3}\), and in four of the six cut cells between 2 and 11 partially filled elements (of 762 to 6,033) have a uniform-thickening derivative whose most negative eigenvalue exceeds \(10^{-6}\) of its largest in magnitude, some of them negative semidefinite. The production central difference agrees with this exact discrete derivative to \(2\times10^{-8}\). Eq. (H.10), and the bound (H.11) that rests on it, therefore describe exact integration; the discrete model satisfies them only approximately. The reference sensitivities of this study are derivatives of the discrete model and are compared as such.

This positive structure strengthens the sensitivity interpretation without requiring an indefinite derivative. Put \(a_c=u^TK_{,c}u=-s_c\ge0\), \(\zeta_c=d^TK_{,c}d\ge0\). Positive-semidefinite Cauchy–Schwarz gives
\[
|\widetilde s_c-s_c|\le2\sqrt{a_c\zeta_c}+\zeta_c,\qquad
\|\widetilde{\boldsymbol s}-\boldsymbol s\|_2
\le2\sqrt{\sum_c a_c\zeta_c}+\|\boldsymbol\zeta\|_2.
\tag{H.11}
\]
If \(\gamma_c=\|A^{-1/2}K_{II,c}A^{-1/2}\|_2\) (so that \(\|\boldsymbol\gamma\|_2=Q\) of Eq. (H.7)), then \(\zeta_c\le\gamma_c\mathcal E\), giving \(2\sqrt{\sum_c a_c\gamma_c}\sqrt{\mathcal E}+\|\boldsymbol\gamma\|_2\mathcal E\). The quadratic term in the signed sensitivity discrepancy is nonpositive; the cross term can have either sign. The reference sensitivity vector has nonpositive components. A small reference norm reflects small derivative-weighted local strain energy. For a general variable that moves a cut or redistributes material, an indefinite derivative is possible; that generality is unnecessary to explain the present corner-thickening mechanism.

Design-dependent loads and assembly maps add the chain-rule terms stated after Eq. (H.5). These changes are different derivative problems rather than modifications of Eq. (H.9).

## Appendix I. A computable lower bound on the energy error

For any nonzero interior test vector \(z\), Cauchy–Schwarz applied to \(A^{1/2}z\) and \(A^{-1/2}r_I\) gives

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
