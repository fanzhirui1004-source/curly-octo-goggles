## 4. Effect of interior displacement errors on compliance and sensitivities

Because all retained DOFs are kept, a cell is approximated only through its interior displacements. This section follows the error of any recovery operator that satisfies assumption (A2) of Section 2.4 through the condensed stiffness to the two quantities used in design, the compliance and the sensitivities. Proposition 1 in Section 3.1 shows that the condensed stiffness exceeds the exact condensed stiffness by the energy of the interior error. The condensed stiffness is therefore never softer than the exact one, and its error is quadratic in the interior displacement error. This error can be computed from the interior residual (Eq. (7)). Section 4.1 shows that the error reaches the assembled compliance weighted by the fraction of the strain energy that each cell carries. Section 4.2 shows that it reaches the thickness sensitivity through the derivative of the stiffness matrix, with a term linear in the interior displacement error. An accurate compliance therefore does not imply an accurate local sensitivity.

The relations are stated for a single substructure and apply to every cell of an assembly. Proposition 1, the residual form of the energy error in Eq. (7) and the self-adjoint compliance sensitivity are classical results [Fraeijs de Veubeke (1965)](https://doi.org/10.1002/nme.339), [Toselli & Widlund (2005)](https://doi.org/10.1007/b137868), [Becker & Rannacher (2001)](https://doi.org/10.1017/S0962492901000010), [Haftka & Gürdal (1992)](https://doi.org/10.1007/978-94-011-2550-5), [Bendsøe & Sigmund (2004)](https://doi.org/10.1007/978-3-662-05086-6). The bound weighted by the energy fractions and the sensitivity relations are derived in this paper for approximate recovery operators. Propositions 3 and 4 use the assumptions of Section 2.4.

### 4.1. Compliance

Consider the exact solutions of the reference and the approximate assembled systems of Eq. (3). Both systems use the same cell stiffness matrices, assembly maps and homogeneous supports, and the same nonzero load on the retained DOFs. Let \(u_m=E_mB_mU\) and \(\widehat u_m=F_mB_m\widehat U\) be the displacements of cell \(m\) recovered from the two solutions, and let \(a(v,v)=\sum_m v_m^TK_mv_m\) be the energy of the assembly. Global equilibrium and the energy orthogonality within each cell then give (Appendix C)

\[
C-\widehat C
=a(\widehat u-u,\widehat u-u)
=\|\widehat U-U\|_{\mathbb K}^{2}
+\sum_m\|H_mB_m\widehat U\|_{A_m}^{2}.
\tag{14}
\]

The compliance is therefore underestimated by the total energy of the error of the recovered displacement field. This energy is the sum of two orthogonal parts. The first comes from the change of the assembled retained solution, and the second from the departure of the interior displacements from equilibrium at that solution. The part of a cell's error that reaches the compliance depends on the strain energy that the cell carries. At the exact assembled retained displacements \(q_m=B_mU\), the energy fraction of cell \(m\) is \(w_m=q_m^TS_mq_m/C\), and these fractions sum to one. The weighted energy error is \(\beta=\sum_mw_m\varepsilon_m(q_m)\), where \(\varepsilon_m\) is the energy error of cell \(m\) as defined in Eq. (7). A cell whose response is a rigid-body motion has zero energy, and its contribution to \(\beta\) is taken as zero.

**Proposition 3 (energy-weighted compliance error).** Under (A1)–(A3), the relative compliance error is bounded by the weighted energy error \(\beta\) (Appendix C.1):

\[
\boxed{0\le\frac{C-\widehat C}{C}\le\frac{\beta}{1+\beta}\le\beta.}
\tag{15}
\]

A cell with a small energy fraction can therefore leave the compliance accurate even when its own interior displacements are inaccurate.

Eq. (14) also separates the error of the condensed stiffness from the error of an incomplete assembled solve, and the numerical checks of Section 5.8 use this separation. Let \(\bar U\) be an approximate solution of the assembled system, with the recomputed residual \(\rho=f_g-\widehat{\mathbb K}\bar U\) and the recovered cell displacements \(\bar u_m=F_mB_m\bar U\). Then

\[
C-f_g^T\bar U=a(\bar u-u,\bar u-u)+\bar U^T\rho,
\tag{16}
\]

and the signed residual work \(\bar U^T\rho\) measures the effect of stopping the solve early (Appendix C.2).

### 4.2. Sensitivities

Consider a differentiable design interval on which the active and retained DOFs, the assembly maps and the homogeneous supports are fixed, and on which the load does not depend on the design. On such an interval, the exact compliance sensitivity is \(s_c=-u^TK_{,c}u\), where \(K_{,c}=\partial K/\partial\tau_c\). For a parameter that affects several substructures, their contributions are summed. The design derivative of the exact recovery \(E\) does not appear, because the exact field is in interior equilibrium [Giles & Pierce (2000)](https://doi.org/10.1023/a:1011430410075), [Haftka & Gürdal (1992)](https://doi.org/10.1007/978-94-011-2550-5). The sensitivity computed from the recovered field is \(\widetilde s_c=-\widehat u^TK_{,c}\widehat u\), with the same stiffness derivative. It ignores the dependence of the recovery operator \(F\) on the design, and hence the design dependence of the network. Over the eight corners, \(s_c\) and \(\widetilde s_c\) form the vectors \(\boldsymbol s\) and \(\widetilde{\boldsymbol s}\).

**Proposition 4 (sensitivity error).** Under (A1), (A2) and (D), with both sensitivities evaluated at the same retained displacements:

(a) the sensitivity \(\widetilde s_c\) differs from the exact sensitivity by a term linear and a term quadratic in the interior displacement error,

\[
\boxed{\widetilde s_c-s_c=-2d^TK_{,c}u-d^TK_{,c}d,
\qquad u=Eq,\quad d=Fq-Eq;}
\tag{17}
\]

(b) with \(\mathcal E=d^TKd\) the energy of the interior error, \(\|\widetilde{\boldsymbol s}-\boldsymbol s\|_2\le2L\sqrt{\mathcal E}+Q\mathcal E\), where the constants \(L\) and \(Q\) of Eq. (H.4) depend on the cell, the test displacement and the stiffness derivatives. The energy error therefore controls the relative sensitivity error only at the order \(\sqrt\varepsilon\) (Appendix H.1).

Interior equilibrium gives \((Ku)_I=0\), but in general \((K_{,c}u)_I\ne0\). The linear cross term therefore remains, and a decrease of the energy error need not decrease the sensitivity error monotonically. Thickening a corner enlarges the material domain, so that \(K_{,c}\succeq0\) under exact integration (Eq. (H.7)). The quadratic term of Eq. (17) is then nonpositive, whereas the linear term can have either sign (Appendix H.2).

The sensitivity \(\widetilde s_c\) is not the derivative of the approximate compliance \(\widehat C\), because \(\widehat C\) also depends on the design through \(F\). For a parameter that affects one substructure, differentiation with the retained DOFs fixed gives

\[
\widehat C_{,c}
=-\widehat u^TK_{,c}\widehat u
 -2\widehat q^TF_{,c}^TK\widehat u
=\widetilde s_c-2(F_{I,c}\widehat q)^Tr_I,
\qquad\widehat u=F\widehat q,
\tag{18}
\]

where \(\widehat q\) is the assembled retained solution obtained with \(\widehat S\), \(r_I=(KF\widehat q)_I\) and \(F_{I,c}=J_I\partial F/\partial\tau_c\). The second term vanishes when the recovered field is in interior equilibrium, but the energy accuracy alone does not control it. Whether Eq. (18) or \(\widetilde s_c\) gives the more accurate gradient depends on whether \(\|H_{,c}q\|_A\) is small compared with \(\|E_{I,c}q\|_A\) (Eq. (H.6)). The quantities \(H_{,c}\) and \(E_{I,c}=J_IE_{,c}\) are the design derivatives of the interior error of \(F\) and of the exact interior recovery (Appendix H.2). All sensitivities reported in this paper are \(\widetilde s_c=-\widehat u^TK_{,c}\widehat u\). They estimate the exact sensitivity and ignore the design dependence of the network. Both derivatives, \(\widetilde s_c\) and that of Eq. (18), hold on intervals on which the discrete choices of Appendix B.1 are fixed. Across a switch of the discrete model, the reference compliance itself can jump. Any jump that the approximate compliance adds is bounded by its own error level, because \(0\le C-\widehat C\le\beta C\) holds at every design by Eq. (15) (Supplementary Note S4.3). Appendix H specifies the numerical stiffness derivatives.

These relations imply that a learned condensation must be checked separately for the compliance and for the sensitivities. By Proposition 3, a cell with a small energy fraction contributes little to the compliance error. By Proposition 4, the energy error of a cell controls its sensitivity error only at the order \(\sqrt\varepsilon\). An accurate compliance therefore does not show that the cell sensitivities are accurate. Section 5.6 checks both quantities in two-cell assemblies, and Section 5.10 checks them in thickness designs.
