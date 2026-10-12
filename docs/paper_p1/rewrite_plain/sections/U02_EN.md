## 2. Cut cells, static condensation and the design problem

This section describes the geometry of the cells (Section 2.1) and their cut finite element discretisation (Section 2.2), defines the retained degrees of freedom and static condensation (Section 2.3) and the assembly, compliance and design problem (Section 2.4), and states the assumptions on which Sections 3 and 4 rest (Section 2.5).

### 2.1. Geometry

This paper considers three-dimensional thin-walled cells based on triply periodic minimal surfaces (TPMS). Each cell occupies the reference box \(\mathcal B=[0,1]^3\), whose six faces are called the cell faces. The material domain of a cell is defined by a Schwarz-P-type trigonometric level-set function, a spatially varying band parameter and an optional planar cut:

\[
\begin{aligned}
&\phi(x)=\sum_{a=1}^{3}\cos(2\pi x_a),\qquad
\tau(x)=\sum_{c\in\{0,1\}^{3}}N_c^{Q_1}(x)\tau_c,\\
&\Omega(\eta)=\{x\in\mathcal B:|\phi(x)|\le\tau(x),\ \boldsymbol n\cdot x\le b_{\rm cut}\}.
\end{aligned}
\tag{1}
\]

The eight parameters \(\tau_c\) are interpolated by the trilinear functions \(N_c^{Q_1}\); they set the width of the implicit band, and thereby the thickness and the material distribution of the cell. They are called corner thickness parameters, and the derivatives with respect to them are called thickness sensitivities. When the eight corner values are collected in a vector, such as \(\boldsymbol\tau\) or the sensitivity vector \(\boldsymbol s\), the corners are numbered \(c=1,\ldots,8\). The cut plane \(\boldsymbol n\cdot x=b_{\rm cut}\) has the offset \(b_{\rm cut}\) and the unit normal \(\boldsymbol n\), which points away from the material that remains after the cut; for an uncut cell, the half-space constraint is omitted. The symbol \(\eta\) collects the geometry, material and discretisation parameters.

### 2.2. Cut finite element discretisation

The cells are discretised by the stabilised cut finite element method (CutFEM) [Burman et al. (2015)](https://doi.org/10.1002/nme.4823). The displacements are approximated by continuous tensor-product \(Q_2\) functions on a fixed Cartesian background mesh with \(n\) elements per axis and element width \(h=1/n\). A background element is active if its intersection with \(\Omega(\eta)\) has positive measure, which is certified by interval enclosures of the level-set and band functions over the element (Appendix A.1); the displacement degrees of freedom (DOFs) of the nodes of the active elements are called the active DOFs and form the vector \(u\). The material boundaries, the TPMS surfaces and the cut plane, are not meshed; they only determine which background elements are active and enter the stiffness through integrals over the material part of each element.

The stiffness matrix \(K\) of a cell consists of the elastic bilinear form on the material part and a ghost penalty:

\[
\begin{aligned}
&v^TKu=\int_{\Omega(\eta)}\nabla^{\rm s}v_h:\mathsf C:\nabla^{\rm s}u_h\,\mathrm dx+\gamma\,g_h(v_h,u_h),\\
&g_h(v_h,u_h)=(\lambda_{\rm ref}+2\mu_{\rm ref})\sum_{f\in\mathcal F_g}\sum_{j=1}^{2}h^{2j-1}\int_f\big[\partial_{n_f}^{j}v_h\big]\cdot\big[\partial_{n_f}^{j}u_h\big]\,\mathrm dA.
\end{aligned}
\]

Here \(u_h\) and \(v_h\) are the \(Q_2\) displacement fields of the active DOF vectors \(u\) and \(v\), and \(\mathsf C\) is the isotropic elasticity tensor. The set \(\mathcal F_g\) of ghost faces consists of the interior background faces whose two neighbouring elements are active and not both certified as completely filled with material; \([\partial_{n_f}^{j}\,\cdot\,]\) is the jump of the \(j\)th normal derivative across face \(f\), \(\lambda_{\rm ref}\) and \(\mu_{\rm ref}\) are reference Lamé constants, and \(\gamma\) is the penalty coefficient (\(\gamma=10^{-4}\) for the validation geometries, Table 1). The elastic term is integrated over the material part of each element only. An active element that contains very little material, a small cut, leaves its DOFs almost unconstrained by the elastic term and makes the stiffness matrix ill-conditioned; the ghost penalty penalises the jumps of the displacement derivatives between neighbouring active elements, so that these DOFs are controlled through their neighbours [Burman (2010)](https://doi.org/10.1016/j.crma.2010.10.006), [Burman et al. (2015)](https://doi.org/10.1002/nme.4823). The integrals over the material part are computed by subcell refinement and positive-weight quadrature on clipped tetrahedra (Appendix A.1); the resulting \(K\) is symmetric positive semidefinite and defines the stabilised discrete energy \(\tfrac12u^TKu\).

This discretisation has three consequences for this paper. First, a change of thickness or of the cut requires no remeshing, and the thickness sensitivity follows from the derivatives of the integrals over the material part (Appendix H). Second, all cells lie on the same background mesh, on which the network of Section 3.2 defines its fixed grid hierarchy, but the active elements, the ghost faces and the set of retained DOFs (Section 2.3) change with the geometry. Sections 3 and 4 account for this change, and their relations use only the symmetry and positive semidefiniteness of \(K\) and the positive definiteness of its interior block. Third, the reference solution in this work is the equilibrium of this discrete system, and all errors are measured against this discrete solution, not against the solution of the continuum problem. Figure 1 shows a cut lattice and one of its cut cells, Table 1 (Section 5.1) lists the discretisation settings of the examples, and Appendix A gives the details of integration and assembly.

### 2.3. Retained degrees of freedom and static condensation

For each cell, the retained DOFs form the set \(P\), which has two parts (Appendix A.1): the DOFs of the cell-face nodes of every material patch of positive area on the cell faces, and all displacement DOFs of the active elements that carry a cut-surface patch of positive area. The latter elements form the layer of active elements intersected by the cut plane and are called the cut-plane elements; some of their nodes lie away from the cut plane, but the basis functions of these nodes contribute to the displacement and the virtual work on that plane. The complete set of retained DOFs is used in both the network prediction and the assembly; it includes the DOFs coupled to neighbouring cells and the DOFs on the exterior boundary.

The remaining active DOFs are the interior DOFs and form the set \(I\); the two sets have \(p=|P|\) and \(i=|I|\) DOFs, and \(n_a=p+i\). The selection matrices \(J_P\) and \(J_I\) extract the two sets from the vector \(u\) of active displacements, and \(q=J_Pu\) is the vector of retained displacements. Boundary loads act through \(q\), and interior body loads are zero. With the DOFs ordered as \((P,I)\), static condensation gives the exact displacement recovery \(E\) and the exact condensed stiffness \(S\) (the Schur complement):

\[
\begin{aligned}
&K=\begin{bmatrix}K_{PP}&K_{PI}\\K_{IP}&A\end{bmatrix},\qquad
A=K_{II},\qquad
E=\begin{bmatrix}I_p\\-A^{-1}K_{IP}\end{bmatrix},\\
&S=K_{PP}-K_{PI}A^{-1}K_{IP}=E^TKE.
\end{aligned}
\tag{2}
\]

The stiffness matrix \(K\) is symmetric positive semidefinite; its null space is spanned by the six rigid-body modes, which form the columns of \(R\in\mathbb R^{n_a\times6}\), and their restriction \(R_P=J_PR\) to the retained DOFs has rank six. Two features of the discrete model ensure these properties (Appendix A.1): the geometry generator accepts only cells whose active elements form a single face-connected set, and material that is connected only through partially filled elements is coupled through shared basis functions and the ghost penalty. An interior field with zero energy would be a rigid-body motion that vanishes on the retained DOFs, so the interior block \(A\) is positive definite, and prescribing \(q\) removes every interior motion with zero energy. The exact displacement recovery \(u=Eq\) is the unique minimiser of the discrete energy subject to \(J_Pu=q\), and \(Sq\) is the vector of the corresponding nodal forces on the retained DOFs.

### 2.4. Assembly, compliance and the design problem

For substructure \(m\), the Boolean assembly map \(B_m\) extracts \(q_m=B_mU\) from the vector \(U\) of free global retained displacements. Coincident cell-face DOFs, identified by their position on the background mesh, are shared between neighbouring cells, whereas retained DOFs of the cut-plane elements that do not lie on the cell faces belong only to their own substructure. The thickness field on a shared face depends only on the four corner values of that face, and \(\phi\) is periodic, so the material patches of two neighbouring cells coincide on every face that is not intersected by a cut. Where a cut removes material from one side of a face only, the unmatched face DOFs remain DOFs of the cell that carries them. Fixed homogeneous supports are incorporated into \(B_m\), and the reference and the approximate assembled systems use the same maps \(B_m\):

\[
\mathbb K=\sum_mB_m^TS_mB_m,\qquad
\widehat{\mathbb K}=\sum_mB_m^T\widehat S_mB_m,
\qquad
\mathbb K U=f_g,\quad\widehat{\mathbb K}\widehat U=f_g.
\tag{3}
\]

Here \(\widehat S_m\) is the approximate condensed stiffness of Section 3, and \(f_g\) is the prescribed force on the free retained DOFs. Because the interior DOF sets of different cells are disjoint, condensing each cell before assembly gives the same matrix as condensing the assembled fine-scale system. The supported reference matrix \(\mathbb K\) is assumed to be positive definite.

The compliance is \(C=f_g^TU\) and its approximation \(\widehat C=f_g^T\widehat U\), with the relative error \(e_C=|\widehat C/C-1|\); the local accuracy of the condensed stiffness is measured by the relative energy error of Section 3.1. The thickness sensitivities of a cell form the eight-component vector \(\boldsymbol s\) of Section 4.2, and with \(\widetilde{\boldsymbol s}\) computed from the approximate displacements, their relative error in the Euclidean norm is \(e_s=\|\widetilde{\boldsymbol s}-\boldsymbol s\|_2/\|\boldsymbol s\|_2\).

In design, the corner thickness parameters at the lattice vertices are the design variables; they are collected in \(\boldsymbol\tau_g\) and mapped to the corners of each cell by \(\boldsymbol\tau_m=\boldsymbol\tau_m(\boldsymbol\tau_g)\). The design problem is

\[
\min_{\boldsymbol\tau_g}\;C(\boldsymbol\tau_g)\quad\text{subject to}\quad V(\boldsymbol\tau_g)\le V^*,\qquad \tau_{\min}\le\tau\le\tau_{\max},
\]

where \(V\) is the material volume of the discrete model. In addition, the corner span and the gradient norm of each cell are bounded to keep every cell within the range of geometries used in training (Section 5.10). The compliance gradient collects the cell sensitivities at each vertex, \(\partial C/\partial\boldsymbol\tau_g=\sum_m(\partial\boldsymbol\tau_m/\partial\boldsymbol\tau_g)^T\boldsymbol s_m\), where \(\boldsymbol s_m\) is the eight-component sensitivity vector of cell \(m\) (Section 4.2).

An approximate condensed stiffness \(\widehat S_m\) that replaces \(S_m\) in this design problem must meet three requirements. First, \(\widehat S_m\) must be symmetric positive semidefinite, with a null space consisting only of the rigid-body modes of the retained DOFs, the columns of \(R_P\). Second, the supported assembled matrix \(\widehat{\mathbb K}\) of Eq. (3) must be positive definite, which follows from \(\widehat S_m\succeq S_m\) and \(\mathbb K\succ0\) (Appendix B.2). Third, the error of \(\widehat S_m\) must be traceable to the compliance error \(e_C\) and to the sensitivity error \(e_s\), which weight this error differently. Section 3 constructs a condensed stiffness that meets these requirements: for a new geometry it needs no interior factorisation, even when the cut changes the discrete space, and it can be improved at deployment without retraining. Sections 3.1 and 4 derive what any recovery operator that satisfies the assumptions of Section 2.5 provides towards these requirements.

### 2.5. Assumptions

Sections 3 and 4 consider an approximate displacement recovery operator \(F\), which maps the retained displacements \(q\) of a cell to the displacements \(Fq\) of all its active DOFs. Propositions 1, 3 and 4 are stated under the following assumptions, given in full in Appendix B.1, and Proposition 2 also uses (A1) and (A2).

- (A1) The cell stiffness matrix \(K\) is symmetric positive semidefinite, its null space consists of the rigid-body modes, \(A=K_{II}\succ0\), and interior body loads vanish (Section 2.3).
- (A2) The recovery operator \(F\) is linear, reproduces the retained displacements, \(J_PF=I_p\), and reproduces rigid-body motions, \(FR_P=R\).
- (A3) Cells share only retained DOFs, and the supported assembled stiffness matrix \(\mathbb K\) is positive definite (Section 2.4).
- (D) On a design interval, the discrete choices listed in Appendix B.1 are fixed: the active elements, the ghost-penalty faces, the retained and interior DOF sets, the assembly maps with the supports, and the load.