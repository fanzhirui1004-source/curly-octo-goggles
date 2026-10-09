## 3. The NICE method

NICE is defined by one operator, the corrected recovery operator \(F\), which maps the retained displacements of a cell to the displacements of the whole cell. The condensed stiffness \(\widehat S=F^TKF\) follows from the strain energy of the recovered displacement field, and exact static condensation is the case \(F=E\). Section 3.1 computes the condensed stiffness from the strain energy and applies it with the transpose of \(F\). For any recovery operator that reproduces the retained displacements and rigid-body motions, the condensed stiffness is then symmetric positive semidefinite and never softer than the exact one. The interior displacements are predicted by a network that takes the cell geometry as input. The network recovery is exactly linear in the retained displacements and reproduces them by construction (Section 3.2). A fixed two-grid correction \(\mathcal W\) is applied to the predicted displacements before the strain energy is computed, and reduces the interior force imbalance that the network leaves (Sections 3.3 and 3.4). The network is trained with this correction applied (Section 3.5). Section 3.6 describes how the condensed stiffness is applied. Figure 2 shows how these parts define both the action of the condensed stiffness and the displacements recovered after assembly.

![Figure 2](figures/F01_method_overview.png)

**Figure 2. Overview of NICE: condensed stiffness of one cell, assembly and design.** (a) The rigid-body part of the retained displacements is separated before the network predicts the interior displacements, so that the network acts only on the deformation. The rigid-body field is then added back and the prescribed retained displacements are restored, which gives \(\widehat Eq\). The two-grid correction \(\mathcal W\) reduces the interior force imbalance with the retained displacements held fixed, which gives \(\widehat u=Fq\). Applying the cell stiffness \(K\) and then the transpose \(F^T\) gives the corresponding nodal forces on the retained DOFs, \(\widehat Sq\). The ordering \(S\preceq\widehat S\preceq\widehat S_{\rm net}\) in (a), where \(\widehat S_{\rm net}\) is the condensed stiffness without correction, is that of Proposition 2. This ordering concerns the energy error and does not apply to the sensitivity error. (b) The condensed stiffness matrices of the cells are assembled on the shared cell-face DOFs, and the assembled system is solved. The displacements of every cell are recovered from the assembled solution, and the compliance and the thickness sensitivity are computed. A design iteration updates the corner thickness parameters and therefore changes the geometry of every cell. The network parameters stay the same, and the coefficients of \(\mathcal W\) are recomputed.

### 3.1. Condensed stiffness from the strain energy

The condensed stiffness is defined through the strain energy, under the original stiffness \(K\), of a recovered displacement field. Let \(F\) be a linear recovery operator that maps the retained displacements \(q\) to the displacements \(Fq\) of the whole cell. The operator \(F\) must reproduce the retained displacements, \(J_PF=I_p\), and the rigid-body motions, \(FR_P=R\), where \(R\) contains the rigid-body modes of Section 2.2. Together with linearity, these conditions form assumption (A2) of Section 2.4. In NICE, \(F\) consists of the network recovery operator \(\widehat E\) followed by the two-grid correction \(\mathcal W\), both described at the end of this section. The recovery operator and its condensed stiffness are

\[
F=\mathcal W\widehat E,\qquad
\widehat S=F^TKF,\qquad
\widehat{\mathcal U}(q)=\tfrac12q^T\widehat S q,\qquad
\widehat f_P=\widehat S q.
\tag{4}
\]

Here \(\widehat{\mathcal U}(q)\) is the strain energy of the recovered field, and \(\widehat f_P\) are the corresponding nodal forces on the retained DOFs. The definitions of \(\widehat S\), \(\widehat{\mathcal U}\) and \(\widehat f_P\) in Eq. (4) apply to any recovery operator \(F\). Only the relation \(F=\mathcal W\widehat E\) is specific to NICE. For a virtual retained displacement \(\delta q\), the virtual displacement of the whole cell is \(F\delta q\), and the variation of the strain energy is \(\delta\widehat{\mathcal U}=(F\delta q)^TKFq=\delta q^TF^TKFq\). Applying the condensed stiffness therefore requires the transpose \(F^T\) after the stiffness action. In NICE, this transpose includes every step of \(F\), namely the rigid-body reconstruction, the restoration of the retained displacements and the correction. With \(F_I=J_IF\), the action of the condensed stiffness has the block form

\[
\widehat S q=(KFq)_P+F_I^T(KFq)_I.
\tag{5}
\]

The second term transfers the work of the interior residual \((KFq)_I\), the interior force imbalance of the recovered field, to the retained DOFs. It vanishes when the recovered field is in interior equilibrium. Otherwise it is needed for the returned forces to be the derivative of the strain energy \(\widehat{\mathcal U}\). The symmetry and positive semidefiniteness of \(\widehat S\) follow from \(K=K^T\succeq0\). They do not require the gather, scatter and grid-transfer maps of the network to be symmetric (Section 3.2). The null space of \(K\) consists of the rigid-body modes (Section 2.2), and \(FR_P=R\). The only zero-energy modes of \(\widehat S\) are therefore the retained rigid-body modes (Appendix B). Appendix E derives the transpose \(F^T\). Appendix B.2 shows that, with this transpose, the error of the condensed stiffness is quadratic in the interior error.

Under assumption (A1), the exact field \(Eq\) has the smallest strain energy among all displacement fields with retained displacements \(q\), by the principle of minimum potential energy. Any recovery operator \(F\) with \(J_PF=J_PE=I_p\) differs from \(E\) only in the interior DOFs. Let \(H=J_I(F-E)\) denote this interior error. Because the exact field is in interior equilibrium, \(J_IKE=0\), the cross terms between \(E\) and \(H\) vanish from \(\widehat S=F^TKF\) (Appendix B.2).

**Proposition 1 (Ritz identity).** Under (A1) and (A2), the condensed stiffness of \(F\) exceeds the exact condensed stiffness \(S\) by the energy of the interior error:

\[
\boxed{\widehat S-S=H^TAH\succeq0.}
\tag{6}
\]

An approximate recovery operator therefore adds stiffness in proportion to the energy of its interior error. The error has no first-order term, because the exact field is in interior equilibrium. Exact static condensation is the case \(F=E\), \(H=0\). For a given retained displacement \(q\), called a test displacement, the accuracy of \(\widehat S\) is measured by the relative energy error \(\varepsilon(q)=q^T(\widehat S-S)q/(q^TSq)\), which is nonnegative by Eq. (6). Let \(u=Eq\), \(\widehat u=Fq\) and \(d_I=Hq\). The interior residual \(r_I=(K\widehat u)_I\) satisfies \(r_I=Ad_I\), so that the energy error is

\[
\varepsilon(q)=\frac{d_I^TAd_I}{q^TSq}
=\frac{r_I^TA^{-1}r_I}{q^TSq}.
\tag{7}
\]

The residual can be computed without the exact recovery \(E\). Evaluating its \(A^{-1}\)-norm, however, requires one interior solve per test displacement. A bound on the energy error over all retained displacements follows only from how well the sampled test displacements cover them (Appendices B and G.4). Proposition 1 and the residual form of Eq. (7) are classical results [Fraeijs de Veubeke (1965)](https://doi.org/10.1002/nme.339), [Toselli & Widlund (2005)](https://doi.org/10.1007/b137868), [Becker & Rannacher (2001)](https://doi.org/10.1017/S0962492901000010).

In NICE, \(F=\mathcal W\widehat E\). Section 3.2 constructs the network recovery operator \(\widehat E\). It reproduces the retained displacements, \(J_P\widehat E=I_p\), and the rigid-body motions, \(\widehat ER_P=R\), by construction (Eq. (9)). The correction \(\mathcal W\) is one symmetric two-grid cycle applied with the retained displacements held fixed (Sections 3.3 and 3.4). It is linear and leaves the retained displacements unchanged. Taking \(\mathcal W\) as the identity gives the uncorrected recovery \(F=\widehat E\). For each geometry, the steps of the cycle and their coefficients are fixed independently of the retained displacements. A correction driven by the interior residual, such as \(\mathcal W\), leaves a rigid-body field unchanged (Appendix B.2). The corrected operator \(F\) thus also reproduces the retained displacements and the rigid-body motions, and Proposition 1 applies to it. By Eq. (6), the error of the condensed stiffness is the energy of the interior error of \(F\), which Sections 3.3 and 3.4 reduce. After the assembled system is solved, \(F_mB_m\widehat U\) gives the displacements of cell \(m\) for evaluating the response and the sensitivities.
