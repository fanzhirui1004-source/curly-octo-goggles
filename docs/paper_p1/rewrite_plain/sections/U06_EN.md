## 4. Effect of interior displacement errors on compliance and sensitivities

Because all retained DOFs are kept, a cell is approximated only through its interior displacements. For any linear recovery operator that satisfies \(J_P\widehat\Psi=I_p\) and \(\widehat\Psi R_P=R\), this section traces the error of the operator through the condensed stiffness to the two quantities used in design, the compliance and the sensitivities. By Proposition 1 (Section 3.1), the condensed stiffness exceeds the exact one by the energy of the interior error. It is therefore an upper bound on the exact condensed stiffness, with an error of second order in the interior displacement error that can be computed from the interior residual (Eq. (7)). This error reaches the assembled compliance weighted by the fraction of the strain energy that each cell carries (Section 4.1), and it reaches the thickness sensitivity through the derivative of the stiffness matrix, with a term of first order in the interior displacement error (Section 4.2). An accurate compliance therefore does not imply an accurate local sensitivity.

Proposition 1, the residual form of Eq. (7) (Section 3.1) and the self-adjoint compliance sensitivity [Haftka & Gürdal (1992)](https://doi.org/10.1007/978-94-011-2550-5), [Bendsøe & Sigmund (2004)](https://doi.org/10.1007/978-3-662-05086-6) are classical. For such recovery operators, this paper derives a weighted error bound for the assembled compliance (Proposition 3) and the sensitivity error relations at given retained displacements (Proposition 4).

### 4.1. Compliance

Consider the exact solutions of the reference and the approximate assembled systems of Eq. (3), which use the same cell stiffness matrices, assembly maps and homogeneous supports and the same nonzero load on the retained DOFs. Let \(u_m=\Psi_mB_mU\) and \(\widehat u_m=\widehat\Psi_mB_m\widehat U\) be the displacements of cell \(m\) recovered from the two solutions, and let \(a(v,v)=\sum_m v_m^TK_mv_m\) be the energy form of the assembly. Global equilibrium and the energy orthogonality within each cell then give (Appendix C)

\[
C-\widehat C
=a(\widehat u-u,\widehat u-u)
=\|\widehat U-U\|_{\mathbb K}^{2}
+\sum_m\|H_mB_m\widehat U\|_{A_m}^{2}.
\tag{14}
\]

The compliance is therefore underestimated by the total energy of the error of the recovered displacement field. This energy consists of two orthogonal parts: the first comes from the change of the assembled retained solution, and the second from the departure of the interior displacements from equilibrium at that solution. How much of the error of a cell reaches the compliance depends on the strain energy that the cell carries. At the exact assembled retained displacements \(q_m=B_mU\), the energy fraction of cell \(m\) is \(w_m=q_m^TS_mq_m/C\), and these fractions sum to one. With \(\varepsilon_m\) the energy error of cell \(m\) defined in Eq. (7), the weighted energy error is \(\beta=\sum_mw_m\varepsilon_m(q_m)\); a cell whose response is a rigid-body motion has zero energy and zero interior error, and its contribution to \(\beta\) is taken as zero.

**Proposition 3 (energy-weighted compliance error).** If the linear recovery operators of all cells satisfy \(J_P\widehat\Psi=I_p\) and \(\widehat\Psi R_P=R\), then under the support condition of Section 2.4 the relative compliance error is bounded by the weighted energy error \(\beta\) (Appendix C.1):

\[
\boxed{0\le\frac{C-\widehat C}{C}\le\frac{\beta}{1+\beta}\le\beta.}
\tag{15}
\]

A cell with a small energy fraction can therefore leave the compliance accurate even when its own interior displacements are inaccurate.

Equation (14) also separates the error of the condensed stiffness from the error of an incomplete assembled solve, and the numerical checks of Section 5.8 use this separation. Let \(\bar U\) be an approximate solution of the assembled system, with the recomputed residual \(\rho=f_g-\widehat{\mathbb K}\bar U\) and the recovered cell displacements \(\bar u_m=\widehat\Psi_mB_m\bar U\). Then

\[
C-f_g^T\bar U=a(\bar u-u,\bar u-u)+\bar U^T\rho,
\tag{16}
\]

and the signed residual work \(\bar U^T\rho\) measures the effect of stopping the solve early (Appendix C.2).

### 4.2. Sensitivities

Consider a differentiable design interval on which the active and retained DOFs, the assembly maps and the homogeneous supports are fixed and the load does not depend on the design (complete list in Appendix B.1). On such an interval, the exact compliance sensitivity is \(s_c=-u^TK_{,c}u\), where \(K_{,c}=\partial K/\partial\tau_c\), and for a parameter that affects several substructures their contributions are summed. The design derivative of the exact recovery \(\Psi\) does not appear in \(s_c\), because the exact field is in interior equilibrium [Giles & Pierce (2000)](https://doi.org/10.1023/a:1011430410075), [Haftka & Gürdal (1992)](https://doi.org/10.1007/978-94-011-2550-5). The sensitivity computed from the recovered field, \(\widetilde s_c=-\widehat u^TK_{,c}\widehat u\), uses the same stiffness derivative and ignores the dependence of the recovery operator \(\widehat\Psi\), and hence of the network, on the design. Over the eight corners, \(s_c\) and \(\widetilde s_c\) form the vectors \(\boldsymbol s\) and \(\widetilde{\boldsymbol s}\).

**Proposition 4 (sensitivity error).** On the design interval above, if the linear recovery operator \(\widehat\Psi\) satisfies \(J_P\widehat\Psi=I_p\) and \(\widehat\Psi R_P=R\) and both sensitivities are evaluated at the same retained displacements, then:

(a) the sensitivity \(\widetilde s_c\) differs from the exact sensitivity by a term linear and a term quadratic in the interior displacement error,

\[
\boxed{\widetilde s_c-s_c=-2d^TK_{,c}u-d^TK_{,c}d,
\qquad u=\Psi q,\quad d=\widehat\Psi q-\Psi q;}
\tag{17}
\]

(b) with \(\mathcal E=d^TKd\) the energy of the interior error, \(\|\widetilde{\boldsymbol s}-\boldsymbol s\|_2\le2L\sqrt{\mathcal E}+Q\mathcal E\), where the constants \(L\) and \(Q\) of Eq. (H.4) depend on the cell, the test displacement and the stiffness derivatives. The energy error therefore controls the relative sensitivity error only at the order \(\sqrt\varepsilon\) (Appendix H.1).

Interior equilibrium gives \((Ku)_I=0\), but in general \((K_{,c}u)_I\ne0\), so the linear cross term remains and a decrease of the energy error need not decrease the sensitivity error monotonically. Thickening a corner enlarges the material domain, so that \(K_{,c}\succeq0\) under exact integration (Eq. (H.7)); the quadratic term of Eq. (17) is then nonpositive, whereas the linear term can have either sign (Appendix H.2).

The sensitivity \(\widetilde s_c\) is not the derivative of the approximate compliance \(\widehat C\), because \(\widehat C\) also depends on the design through \(\widehat\Psi\). For a parameter that affects one substructure, eliminating the design derivative of the retained solution through the equilibrium of the approximate assembled system gives

\[
\widehat C_{,c}
=-\widehat u^TK_{,c}\widehat u
 -2\widehat q^T\widehat\Psi_{,c}^TK\widehat u
=\widetilde s_c-2(\widehat\Psi_{I,c}\widehat q)^Tr_I,
\qquad\widehat u=\widehat\Psi\widehat q,
\tag{18}
\]

where \(\widehat q\) is the assembled retained solution obtained with \(\widehat S\), \(r_I=(K\widehat\Psi\widehat q)_I\) and \(\widehat\Psi_{I,c}=J_I\partial \widehat\Psi/\partial\tau_c\). The second term vanishes when the recovered field is in interior equilibrium, but energy accuracy alone does not control it. The linear terms in the error bounds of \(\widetilde s_c\) and Eq. (18) contain \(\|\Psi_{I,c}q\|_A\) and \(\|H_{,c}q\|_A\), respectively (Eq. (H.6)), where \(\Psi_{I,c}=J_I\Psi_{,c}\) and \(H_{,c}\) are the design derivatives of the exact interior recovery and of the interior error of \(\widehat\Psi\) (Appendix H.2). The design problem of Section 2.4 targets the reference compliance; all sensitivities reported and used in the optimisations of this paper are the estimates \(\widetilde s_c=-\widehat u^TK_{,c}\widehat u\) of its sensitivity, which ignore the design dependence of the network. The derivatives in this section hold on intervals on which the discrete choices of Appendix B.1 are fixed. Across a switch of the discrete model the reference compliance itself can jump, but since \(0\le C-\widehat C\le\beta C\) holds at every design by Eq. (15), any jump that the approximate compliance adds is bounded by its own error level (Supplementary Note S4.3). Appendix H specifies the numerical stiffness derivatives.

These relations imply that a condensed stiffness based on network-predicted interior displacements must be checked separately for the compliance and for the sensitivities. By Proposition 3, the relative compliance error is bounded by the energy errors of the cells at the exact assembled retained displacements, weighted by their energy fractions. By Proposition 4, the energy error of a cell controls its relative sensitivity error only at the order \(\sqrt\varepsilon\). Sections 5.6, 5.8 and 5.10 check both quantities in two-cell assemblies, eight-cell lattices and thickness designs, respectively.