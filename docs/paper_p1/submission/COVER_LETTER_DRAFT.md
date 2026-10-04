# Cover letter (draft)

<!-- Draft for the author to rewrite. Bracketed items are for the author. Numbers are those of the Abstract,
Sections 5.9-5.10 and Section 7 (structure of 2026-10-03). -->

[Date]

Editor-in-Chief
*Computer Methods in Applied Mechanics and Engineering*

Dear Editor,

We submit the manuscript "Neural-initialised static condensation with equilibrium correction for the analysis and thickness design of cut thin-walled TPMS lattices" for consideration as a research article in *Computer Methods in Applied Mechanics and Engineering*.

Lattices of geometrically varying components, such as graded or cut thin-walled TPMS lattices, are too large to resolve at every design change, yet lose the scale separation on which homogenisation rests. Static condensation reduces each cell to the degrees of freedom it shares with its surroundings, but it factorises the interior of every geometry, and a cut cell retains tens of thousands of degrees of freedom on its box faces and cut band. The manuscript approximates the condensed stiffness on this complete retained space by neural-initialised static condensation with equilibrium correction (NICE): a geometry-conditioned network, exactly linear in the retained displacement, supplies the interior extension, and a fixed two-grid correction reduces its equilibrium residual inside the energy form.

We believe the work fits the journal for three reasons.

- **A learned substructure with the structure of static condensation.** Because the extension is evaluated in the energy form, the condensed stiffness is symmetric, positive semidefinite, has the rigid-body kernel and is bounded below by the exact Schur complement, with an error quadratic in the interior error; the correction keeps these properties and, under stated conditions, cannot increase the error. One network serves every retained set, and neither the extension nor the condensed matrix is formed.
- **Error analysis for design use.** Following the interior error through the energy share and the stiffness derivative explains, and the experiments confirm, why accurate compliance does not imply accurate local thickness sensitivity.
- **Verification at the lattice level and in design.** On 80 validation geometries not used in training, the mean energy error is 0.074%; assembled compliance and sensitivity errors are below 0.28% and 1.5% in every two-cell configuration and below 0.02% and 0.15% in eight-cell lattices, under face loads, of cells unseen in training and checkpoint selection. An eight-cell analysis with sensitivities takes 79 to 105 s on one GPU against 868 to 1,042 s for the whole-lattice direct solution and 637 to 752 s for whole-lattice conjugate gradients with algebraic multigrid on 16 CPU threads or processes. Thickness optimisation driven by the field-based sensitivities reproduces the design of exact condensation to within 0.0051 per corner parameter; on a 24-cell plate clamped through its cut band, where a homogenised model underestimates the compliance by 27%, the exact compliance of the NICE design is 2.4% lower than that of the homogenisation design.

The limitations are stated in Section 6.4: one cell family and discretisation, one training seed, no certified error bound, field-based sensitivities that omit the design dependence of the extension, and a comparison with whole-lattice iterative solvers on two lattices only.

This manuscript has not been published and is not under consideration elsewhere. All authors have approved the submission. [Competing interests / funding: author.] The code, trained network parameters, geometry parameter files and result records will be archived in a public Zenodo repository upon acceptance, as stated in the manuscript.

[Suggested reviewers, if requested: name, affiliation, e-mail, reason; to be chosen by the author.]

Yours sincerely,

[Corresponding author, affiliation, e-mail]
