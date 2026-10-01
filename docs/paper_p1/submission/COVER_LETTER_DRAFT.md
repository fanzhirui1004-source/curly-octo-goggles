# Cover letter (draft)

<!-- Draft for the author to rewrite. Bracketed items are for the author. Numbers are those of the Abstract and Section 8;
the design-example sentence depends on Section 6.11 and must match its final text. -->

[Date]

Editor-in-Chief
*Computer Methods in Applied Mechanics and Engineering*

Dear Editor,

We submit the manuscript "Neural-initialised static condensation with equilibrium correction for the analysis and thickness design of cut thin-walled TPMS lattices" for consideration as a research article in *Computer Methods in Applied Mechanics and Engineering*.

Thin-walled lattices built from triply periodic minimal surfaces are cut by the part boundary, and every cut cell is a different geometry. Static condensation reduces each cell to its interface, but for a cut cell it needs one interior solve per retained coordinate, tens of thousands of solves for every cell and every design change. The manuscript approximates the condensed operator on the complete retained space, including the cut band, by neural-initialised condensation with equilibrium correction (NICE): a geometry-conditioned network supplies the interior field, and a fixed, geometry-specific multilevel correction reduces its equilibrium residual inside the energy form.

We believe the work fits the journal for three reasons.

- **Structure-preserving learned substructure.** Because the extension is exactly linear in the retained displacement and evaluated in the energy form, the condensed stiffness is symmetric, positive semidefinite, reproduces rigid motion and is bounded below by the exact Schur complement, and the correction cannot increase its error when the smoothing interval contains the spectrum.
- **Error analysis for learned substructures.** Following the interior error through energy participation and the stiffness derivative explains, and the experiments confirm, why accurate compliance does not imply accurate local thickness sensitivity, and what the correction changes.
- **Verified accuracy and cost at the lattice level.** On 80 validation geometries not used in training, the mean directional energy error is 0.074%; in two-cell assemblies and eight-cell lattices the compliance and sensitivity errors are verified against exact condensation, and an eight-cell design iteration takes 81 to 110 s on one GPU against 868 to 1,042 s for the whole-lattice Cholesky solution on 16 CPU threads, excluding the one-off cost of data generation and training. In thickness optimisation driven by the field-based sensitivities, NICE reaches the design of exact condensation on an eight-cell lattice to within 0.0051 per corner parameter and the exact compliances of cut 24-cell plates to within 0.04%, and design iterations were timed on plates of up to 110 cells on one GPU.

The limitations are stated in Section 7.5: one cell family, one training seed, no certified error bound, and no comparison with fine-scale iterative solvers of the whole lattice.

This manuscript has not been published and is not under consideration elsewhere. All authors have approved the submission. [Competing interests / funding: author.] The code, trained weights, geometry parameter files and result records will be archived at [Zenodo] upon acceptance, as stated in the manuscript.

[Suggested reviewers, if requested: name, affiliation, e-mail, reason; to be chosen by the author.]

Yours sincerely,

[Corresponding author, affiliation, e-mail]
