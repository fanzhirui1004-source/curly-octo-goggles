## Appendix G. Network coefficients and training test displacements

### G.1. Geometry features and displacement maps

The architecture in Figure 3 separates the generation of geometry-dependent coefficients from the propagation of displacements. The displacement features carry 32 feature channels per node on the fine grid (level \(k=0\), 65 positions per axis) and on the coarse grids of levels \(k=1,2,3\) (33, 17 and 9 positions per axis). The network processes several test displacements simultaneously, each with its own displacement features. The geometry embeddings are the 64-component element and node feature vectors of Section 3.2 (Figure 3a); they carry no test-displacement dimension. Element, face, retained-node and grid-transfer incidences are constructed before the network is evaluated. The symbols \(a\), \(b\), \(k\), \(p_i\), \(r_i\), \(\sigma\) and \(\mathcal R\) are local to this appendix and unrelated to the smoothing interval, polynomial and residual notation of the main text and Appendix D.

**Geometry encoding.** Each element has 126 input features: material volume fraction, its logarithm, and the remaining 124 moments normalised by material volume. The 11 node features contain indicators of retained, cell-face, cut-plane-element and weakly connected nodes, a normalised logarithmic stiffness diagonal, and six sine/cosine coordinate entries. The stiffness summary is the Frobenius norm of the node's diagonal \(3\times3\) block. A node is designated weakly connected when this norm is less than 0.01 times its median over active nodes.

The element encoder maps 126 inputs to a 64-component embedding. The mean of the incident element embeddings, concatenated with the 11 node features, forms the 75 inputs of the node encoder. Two residual message-passing rounds (the two rounds of exchange in Section 3.2) then update elements from the mean embeddings of their 27 nodes and update nodes from their incident elements. Each update uses the current embedding and the aggregated neighbouring embedding as a 128-component input. A ghost-face embedding is formed from the two adjacent element embeddings and a three-component indicator of the face axis. All encoders and coefficient heads use two affine layers separated by GELU, with hidden width 64.

**Local linear interactions.** The three retained displacement components are lifted by a shared matrix \(W_{\rm in}\in\mathbb R^{3\times32}\), while interior features are initially zero. Denote this initial feature field by \(X^0(q)\); it depends on \(q\) through \(q_d\). For each element or face stencil \(t\), local node position \(s\) and head \(h\), the interaction first gathers and mixes channels:

\[
Z_{th}=\left(\sum_{s=1}^{27}a_{tsh}(\eta)X_{i(t,s)}\right)W_{\ell h},
\qquad
\Delta X_i=\sum_{(t,s):i(t,s)=i}\sum_{h=1}^{4}
\omega_{ts}\,b_{tsh}(\eta)Z_{th}.
\]

Here \(W_{\ell h}\in\mathbb R^{32\times32}\) is a shared learned channel map for layer \(\ell\) and head \(h\). The scalar gather and scatter coefficients \(a\) and \(b\) are generated separately from the element or face embedding, the incident node embedding and an eight-dimensional position embedding. The position embedding is a learned \(27\times8\) table indexed by the local node position, with one table for element stencils and one for face stencils. The 136-component concatenation of the three embeddings identifies both the local material context and the position within the stencil. The four heads provide distinct gather--mix--scatter contributions on the same fixed incidence. Equation (8) adds their update to the incoming state and restores the prescribed retained features after every local interaction.

The element stencil contains all 27 \(Q_2\) nodes. The learned ghost-face stencil contains 18 owner-side nodes and nine neighbour-side nodes, ordered consistently by face orientation. It defines a learned communication map on a face neighbourhood; the full mechanical ghost-penalty contribution remains in \(K\).

The node scatter accounts for the number and relative stiffness of incident contributions. If \(k_{es}\) is the Frobenius norm of the element's diagonal \(3\times3\) block at local node position \(s\), the element weights are

\[
\pi_{es}=\frac{k_{es}}{\sum_{(e',s')\mapsto i}k_{e's'}},
\qquad
\omega_{es}=\frac{1-\lambda_{\rm mix}}{\deg(i)}
 +\lambda_{\rm mix}\pi_{es},\qquad(e,s)\mapsto i.
\tag{G.1}
\]

The face weights use the corresponding ghost-contribution diagonal blocks and face incidences. When all incident stiffness norms at a node vanish, the stiffness fraction is defined by the inverse incidence count. Each layer has its own learned \(\lambda_{\rm mix}\), initialised at zero (plain incidence averaging). For the additional weak-region layers, a stencil is selected when it contains a weakly connected node; its weights keep the normalisation of the full element or face incidence set.

**Multilevel propagation.** The grid hierarchy enlarges the spatial range of communication, while additive skip connections keep the fine-grid field. Let \(t_{ia}\) be the fixed trilinear incidence weight between a fine node \(i\) and a coarse vertex \(a\). Positive geometry coefficients \(r_i\) and \(p_i\) define restriction and prolongation componentwise:

\[
(\mathcal R X)_a=\frac{\sum_i t_{ia}r_iX_i}{\sum_i t_{ia}r_i},
\qquad
(\mathcal I X_c)_i=p_i\frac{\sum_a t_{ia}X_{c,a}}{\sum_a t_{ia}}.
\]

Only coarse vertices reached by the stored trilinear incidences participate. The denominators depend on geometry alone, so both maps are linear in displacement. Restriction and prolongation are separately parameterised. The network transpose \(\mathcal N_\theta(\eta)^T\) in Eq. (E.1), and hence the action of the condensed stiffness in Eq. (5), is therefore obtained by transposing the complete composition of operations; one transfer map is not identified with the transpose of the other.

On each coarse level, a residual convolution takes the form

\[
X\leftarrow X+g^{(k)}_j(\eta)\odot\operatorname{Conv}^{(k)}_j(X).
\]

Each convolution has a \(3\times3\times3\) kernel, 32 input and output channels, unit padding, no bias and no activation on the displacement features. Participating coarse-grid nodes are inserted into the corresponding background grid, with unused positions set to zero; the convolution output is sampled back at those nodes. The geometry branch supplies one multiplicative coefficient per participating node, channel and convolution. Its coarse-grid embeddings are obtained by fixed trilinear weighted averaging.

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

At the upward step that returns to level \(k\in\{0,1,2\}\), the combination is \(X=Y^{(k)}+\sigma^{(k)}\mathcal I^{(k)}X_c\), where \(X_c\) is the field on level \(k+1\) and \(\sigma^{(k)}\) is a learned scalar. Thus convolution precedes prolongation, the skip connections are additive, and the 9-position grid receives two downward and two upward convolutions. There are twelve coarse convolution updates in total. The fine retained values are restored after the grid hierarchy returns to the active nodes.

**Coefficient bounds and output.** The local heads produce separate gather/scatter values for eight ordinary element layers, twelve face layers and four additional weak-region element layers, each with four heads and 27 local node positions. Raw head outputs are multiplied by 0.2. For a finite positive group bound \(a_{\max}\), the coefficient map is

\[
\mathcal C(z)=
\begin{cases}
z,&|z|\le k_ba_{\max},\\
\operatorname{sgn}(z)\left[k_ba_{\max}+(1-k_b)a_{\max}
\tanh\!\left(\dfrac{|z|-k_ba_{\max}}{(1-k_b)a_{\max}}\right)\right],&|z|>k_ba_{\max}.
\end{cases}
\]

All variants of Table 2 and Supplementary Table ST24 set \(k_b=0.5\). Each \(a_{\max}\) is twice the largest magnitude of its coefficient group recorded in a calibration pass over training geometries, so the map is the identity up to that recorded maximum. Table ST15 (Supplementary Note S5) lists the groups, the initialisation and the parameter count of every block. Positive transfer coefficients apply the corresponding bound to \(\operatorname{softplus}(z)\) and add \(10^{-3}\); convolution coefficients use \(2\operatorname{sigmoid}(z)\).

A bias-free matrix \(W_{\rm out}\in\mathbb R^{32\times3}\) returns the fine features to nodal displacement in the prescribed coordinate order. Reconstruction of the rigid-body motion and restoration of the retained values then give Eq. (9).

**Frame consistency.** Geometry augmentation in training samples the 48 cube symmetries. For an orthogonal cube transformation, the deterministic signed permutation maps satisfy

\[
K'=T_aKT_a^T,\qquad E'=T_aET_P^T,\qquad S'=T_PST_P^T.
\tag{G.2}
\]

Here \(T_a\in\mathbb R^{n_a\times n_a}\) and \(T_P\in\mathbb R^{p\times p}\) are the signed permutation matrices that the cube transformation induces on the active and the retained DOFs of the cell, and primes denote quantities of the transformed cell. The augmented geometry and vector field are passed through the network and the predicted displacement is returned to the original frame for mechanical evaluation.

### G.2. Test-displacement sets

The normalised test displacements are

\[
q_j=\frac{\Pi_Pq_j^{\rm raw}}
{\sqrt{(\Pi_Pq_j^{\rm raw})^TS(\Pi_Pq_j^{\rm raw})}},
\tag{G.3}
\]
where \(q_j^{\rm raw}\) is the test displacement produced by its class before rigid-body removal and energy normalisation.

The load classes used in training have the following mechanical definitions.

| Class | Construction before rigid removal and energy normalisation |
|---|---|
| `force` | Equilibrated nodal point loads from smooth plane waves or localised Gaussian patches, with a subset loading the cut boundary. |
| `force_c` | Consistently integrated, self-equilibrated traction loads on the material cell faces; a subset also loads the cut surface. Equilibrium is imposed in traction-quadrature space. |
| `face` / `face_c` | Equilibrated nodal point loads or consistently integrated traction loads on a single cell face. |
| `support` | Responses with soft spring support on one cell face and equilibrated loads on the other faces. |
| `support_k` | Consistently integrated loads with springs scaled by the local stiffness diagonal and a logarithmically sampled factor in \([0.3,3]\); the cut surface remains unloaded. |
| `macro` | Equal sampling of uniform strain, quadratic, and cubic imposed displacement fields. |
| `grf` | Multiscale displacement fields spanning 0.5–24 spatial cycles across the reference box. |
| `glued` | Displacements imposed by a neighbouring cell that shares the interface degrees of freedom and is clamped or supported by springs on its far face; one quarter of the samples load only the neighbour. |
| `adv` | Test displacements selected by the generalised-energy search described below. |

The class `glued` supplies test displacements shaped by the neighbouring cell, while the energy and sensitivity labels refer to the operator of the isolated test cell; each test displacement is normalised by Eq. (G.3).

### G.3. Search for difficult test displacements

Test displacements with large energy ratios \(q^T\widehat Sq/q^TSq\) are found by a block search on the complement of the rigid-body modes in the space of retained displacements. Whenever a geometry enters the training rotation, the search initialises eight candidate test displacements and performs four block iterations. Applying \(S^{-1}_{\perp}=ZS_*^{-1}Z^T\), the inverse of \(S\) on this complement (with \(Z\) and \(S_*\) as in Eq. (B.4)), uses an equilibrated Neumann solve. Six deterministically selected displacement pins remove the rigid-body motion, and the result is projected onto the complement. The candidate energy ratios are observations for individual test displacements; maximising them does not give an upper bound over all test displacements. The training settings are listed in Supplementary Table ST01.

### G.4. Coverage by the test displacements

With \(Z\) and \(S_*=Z^TSZ\succ0\) as in Eq. (B.4), let
\(T_*=S_*^{-1/2}Z^TH^TAHZ S_*^{-1/2}\).
For an energy-normalised test displacement, \(x_j=S_*^{1/2}Z^Tq_j\) has unit Euclidean norm and, by Eq. (B.5), \(\varepsilon_j=x_j^TT_*x_j\). For \(N_q\) sampled test displacements, define the coverage matrix \(M_{N_q}=N_q^{-1}\sum_{j=1}^{N_q}x_jx_j^T\). Then

\[
\overline\varepsilon=\operatorname{tr}(T_*M_{N_q}),\qquad
M_{N_q}\succeq\alpha I, \alpha>0
\ \Longrightarrow\ 
\varepsilon_*\le\operatorname{tr}(T_*)\le\overline\varepsilon/\alpha.
\tag{G.4}
\]

If the test displacements do not span the complement, choose a unit \(v\) orthogonal to them and \(T_*=\Theta\,vv^T\) with \(\Theta>0\). Every sampled error is then zero, whereas the worst error is \(\Theta\) and can be arbitrarily large. Such a PSD perturbation can be realised as \(H^TAH\) if there is at least one interior degree of freedom. Thus retaining all DOFs in P defines the approximation target, while adequate coverage by the test displacements controls how accurately it is learned. The coverage condition \(M_{N_q}\succeq\alpha I\), \(\alpha>0\), on the \((p-6)\)-dimensional complement requires \(N_q\ge p-6\), i.e. thousands to tens of thousands of test displacements per geometry here. The sampled sets of test displacements do not satisfy it, so Eq. (G.4) is not applied and no upper bound on \(\varepsilon_*\) is claimed.

## Appendix H. Sensitivity identities and design intervals

The derivative of the exact condensed stiffness and the derivative of the energy at fixed displacement are

\[
S_{,c}=E^TK_{,c}E,\qquad
\frac{\partial}{\partial\tau_c}\left(\tfrac12q^TSq\right)
=\tfrac12(Eq)^TK_{,c}(Eq).
\tag{H.1}
\]

On a differentiable interval with fixed active topology and fixed selectors \(J_P\) and \(J_I\), differentiating \(J_PE=I_p\) gives \(J_PE_{,c}=0\). Since \(J_IKE=0\), both terms \(E_{,c}^TKE\) and \(E^TKE_{,c}\) vanish. Differentiating \(S=E^TKE\) therefore proves Eq. (H.1).

For a supported equilibrium, differentiating \(\mathbb K U=f_g\) at a fixed load gives \(U_{,c}=-\mathbb K^{-1}\mathbb K_{,c}U\) and hence \(C_{,c}=-U^T\mathbb K_{,c}U\). With fixed gathers and Eq. (H.1), this becomes the sum of \(-u_m^TK_{m,c}u_m\) over affected cells. The sensitivity labels hold the base-design nodal load fixed, including the nodal loads obtained by consistent integration of traction loads.

Expanding the quadratic sensitivity estimate at \(u+d\) proves Eq. (17).

For the condensed stiffness \(\widehat S=F^TKF\), direct differentiation gives

\[
\widehat S_{,c}=F_{,c}^TKF+F^TK_{,c}F+F^TKF_{,c}.
\tag{H.2}
\]

Apply the compliance derivative at equilibrium to the assembly of learned substructures and use \(J_PF_{,c}=0\). With \(F_{I,c}=J_I\,\partial F/\partial\tau_c\), so that \(F_{,c}=J_I^TF_{I,c}\), the two terms that contain the derivative of the recovery operator become \(2(F_{I,c}\widehat q)^Tr_I\), proving Eq. (18). If the load depends on design, the full compliance derivative also contains \(2f_{g,c}^T\widehat U\), and correspondingly \(2f_{g,c}^TU\) for the exact assembly. Changes of the DOF selectors, assembly maps or support maps contribute their own chain-rule terms, including the derivatives of \(B_m^TS_mB_m\) when \(B_m\) varies.

In the reference computation, moment derivatives are approximated by

\[
M_{e\alpha,c}\approx
\frac{M_{e\alpha}(\boldsymbol\tau+h_ce_c)-M_{e\alpha}(\boldsymbol\tau-h_ce_c)}{2h_c},
\qquad h_c=10^{-5}\tau_c.
\tag{H.3}
\]

The active set is fixed for this calculation, and the elasticity templates and ghost contribution are held constant. The moment quadrature is recomputed at each perturbed thickness, so its clipping and integration branches may change. The resulting matrices represent a numerical design derivative on that prescribed topology. The sensitivities of the NICE lattice analyses and optimisations (Sections 5.9 and 5.10) are instead obtained by reverse-mode differentiation of the moment integrals at the fixed recovered displacement field.

### H.1. Sensitivity error bounds (proof of Proposition 4)

For a fixed retained displacement vector \(q\), set \(u=Eq\), \(d=J_I^THq\), and \(\mathcal E=d^TKd=(Hq)^TA(Hq)\). The derivatives \(K_{,c}\), \(c=1,\ldots,8\), are symmetric. Define the reference vector \(\boldsymbol s\) with components \(s_c=-u^TK_{,c}u\), and assume \(\|\boldsymbol s\|_2>0\). Eq. (17) then reads

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
\tag{H.4}
\]

All eight components are included, and the denominator is the norm of the exact sensitivity vector for that test displacement. The constants separate the coupling through the stiffness derivative, the interior coercivity and the scale of the reference sensitivity. For fixed \(q\) and \(H=tH_0\), the linear coefficient vanishes if and only if \((H_0q)^Tb_c=0\) for every corner. It vanishes for all interior errors if \(J_IK_{,c}E q=0\) for every corner. A useful special case is \(K_{,c}=\alpha_cK\): interior equilibrium removes the linear term, and the relative sensitivity error equals the energy error when the reference vector is nonzero.

### H.2. The complete design derivative and the sign of the thickness derivative

With fixed selectors, differentiating \(J_IKE=0\) gives \(AE_{I,c}=-J_IK_{,c}E\). With \(K_{II,c}\) as in Appendix H.1, differentiating the variational stiffness error gives

\[
\boxed{(\widehat S-S)_{,c}
=H_{,c}^TAH+H^TK_{II,c}H+H^TAH_{,c}.}
\tag{H.5}
\]

Its norm is bounded by \(2\|A\|\|H\|\|H_{,c}\|+\|K_{II,c}\|\|H\|^2\). Thus a value error \(H=O(t)\) by itself does not imply a quadratic derivative error. For example, with \(f=1\), \(K=I_2\), \(E=(1,0)^T\) and \(F_t(\tau)=(1,t+\tau)^T\), the compliance gap at \(\tau=0\) is \(C-\widehat C_t=t^2/(1+t^2)\), whereas the derivative error is \(\partial_\tau(\widehat C_t-C)=-2t/(1+t^2)^2\). With \(H_t(\tau)=t\sin(\tau/t^2+\pi/4)\) the gap is \(O(t^2)\), but the derivative error tends to \(-1\). Neither family is uniformly \(C^1\)-small. A uniformly \(C^1\)-small recovery error, \(H=tH_0(\tau)\) with bounded \(H_0,H_{0,c}\), gives \((\widehat S-S)_{,c}=O(t^2)\). With uniformly bounded exact solutions and inverse stiffnesses, this also yields \(\widehat C_{,c}-C_{,c}=O(t^2)\).

At the same retained displacement vector \(q\), Eqs. (17) and (H.5) give

\[
\widetilde s_c-s_c=2(Hq)^TAE_{I,c}q-(Hq)^TK_{II,c}Hq,\qquad
-q^T\widehat S_{,c}q-s_c=-2(H_{,c}q)^TA\,Hq-(Hq)^TK_{II,c}Hq.
\tag{H.6}
\]

The complete derivative of Eq. (18) evaluates \(-q^T\widehat S_{,c}q\) at the assembled \(\widehat q\), so its comparison with \(s_c\) also contains the change of this quantity between \(q\) and \(\widehat q\). The residual term of Eq. (18) thus removes the term linear in the interior displacement error and replaces it by a term weighted by the design derivative of the recovery error. Which of the two gives the more accurate gradient depends on whether \(\|H_{,c}q\|_A\) is small compared with \(\|E_{I,c}q\|_A\).

The \(O(t)\) term of the sensitivity estimate can cancel against the residual term. At the same retained displacement, \(J_IK_{,c}Eq=-AE_{I,c}q\). For \(H=tH_0\), the linear error of the sensitivity estimate is \(+2t(H_0q)^TAE_{I,c}q\), while the linear contribution of the residual term is \(-2t(E_{I,c}q)^TAH_0q\). A sensitivity estimate with a first-order error and a complete derivative with a second-order error can therefore coexist, even with a positive semidefinite stiffness derivative. For \(K(\tau)=\begin{bmatrix}1+\tau&\tau\\\tau&1+\tau\end{bmatrix}\), \(E=(1,-\tau/(1+\tau))^T\), \(F_t=E+(0,t)^T\) and \(f=1\) at \(\tau=0\), the exact derivative is \(-1\), the sensitivity estimate \(-(1+t)^2/(1+t^2)^2\) and the residual term \(2t/(1+t^2)^2\); their sum \(-1/(1+t^2)\) has error \(t^2/(1+t^2)\).

Under a fixed basis and exact integration, increasing one band parameter \(\tau_c\) with nonnegative \(Q_1\) shape functions enlarges the material domain. With fixed ghost stabilisation, this gives \(K_{,c}\succeq0\), \(S_{,c}\succeq0\), and \(C_{,c}\le0\). The sensitivity estimate is then also nonpositive for any displacement field; the quadratic term in the signed sensitivity discrepancy of Eq. (17) is nonpositive, while the cross term can have either sign. That the variational condensed stiffness dominates the exact one at every design does not by itself make the approximate compliance monotone with respect to the design: the residual term can change the sign of its complete derivative. For a scalar design parameter \(\gamma_1>0\), \(K(\gamma_1)=\gamma_1I_2\), \(E=(1,0)^T\) and \(F=(1,g(\gamma_1))^T\), one has \(\widehat S=\gamma_1(1+g^2)\ge S=\gamma_1\); at \(\gamma_1=1\), \(g=0.1\) and \(g'=-10\), \(\widehat S'=-0.99\), so \(\widehat C'>0\) for \(f=1\) although \(C'=-1\), while the sensitivity estimate remains negative.

More explicitly, for \(h>0\), \(N_c^{Q_1}(x)\ge0\) implies \(\Omega(\tau)\subseteq\Omega(\tau+h e_c)\). For any discrete coefficient vector \(v\), with interpolated displacement field \(v_h\),
\[
v^T[K(\tau+h e_c)-K(\tau)]v
=\int_{\Omega(\tau+h e_c)\setminus\Omega(\tau)}
\nabla^{\rm s}v_h:\mathsf C:\nabla^{\rm s}v_h\,dx\ge0.
\tag{H.7}
\]
Here \(\nabla^{\rm s}\) is the symmetric gradient, the elasticity tensor \(\mathsf C\) and the basis are fixed, and the ghost contribution cancels. Taking the differentiable limit proves that the thickness derivative is positive semidefinite. A centred difference of stiffness matrices assembled from exactly nested domains is also PSD. Changes of the adaptive integration subdivision, approximate moments or independently selected stabilisation can break this discrete nesting relation; a fixed active topology alone does not establish that it holds numerically. A step study of the central difference and an element-level check of the moment derivatives (Supplementary Note S1, Figure S01(c)) show that the numerical derivative is accurate on the fixed active set. They also show that the discrete moments preserve the nesting only approximately: some partially filled elements have indefinite derivative matrices. Eq. (H.7) therefore describes exact integration; the reference sensitivities of this study are derivatives of the discrete model and are compared as such.

## Appendix I. A computable lower bound on the energy error

For any nonzero interior test vector \(z\), Cauchy–Schwarz applied to \(A^{1/2}z\) and \(A^{-1/2}r_I\) gives

\[
\Delta(q):=q^T(\widehat S-S)q=r_I^TA^{-1}r_I
\ge L_z:=\frac{(z^Tr_I)^2}{z^TAz}.
\tag{I.1}
\]

A preconditioned residual, a coarse approximation, or a Krylov iterate can supply \(z\). Evaluating the final one-vector quotient with the original \(A\) makes the inequality independent of how that vector was generated. Let \(q\notin\operatorname{range}R_P\) and \(e_h=\widehat u^TK\widehat u=q^T\widehat Sq\). Then \(e_h-\Delta(q)=q^TSq>0\), so \(e_h-L_z>0\). Moreover \(\varepsilon(q)=\Delta/(e_h-\Delta)\), and the map \(x\mapsto x/(e_h-x)\) is increasing on \([0,e_h)\). Reproduction of rigid-body motions gives \(HR_P=0\) and \(SR_P=0\) (Appendix B.2), so \(\varepsilon(q)=\varepsilon(\Pi_Pq)\le\varepsilon_*\). Therefore

\[
\varepsilon(q)\ge\frac{L_z}{e_h-L_z},\qquad
\mu_*\ge1+\frac{L_z}{e_h-L_z}.
\tag{I.2}
\]

If the exact reference energy is known, \(L_z/(q^TSq)\) gives a sharper lower estimate. A large value identifies a test displacement on which the condensed stiffness is inaccurate; a small value alone does not bound the error from above. These inequalities hold in real arithmetic; their numerical evaluation requires the original quadratic forms stated above.
