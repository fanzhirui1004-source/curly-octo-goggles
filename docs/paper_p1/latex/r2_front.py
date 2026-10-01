"""Revision 2: replace the abstract (line 5) and the introduction paragraphs (lines 11-25) of MANUSCRIPT_EN.md with the
L2 framing (structure-preserving learned component model; cut thin-walled TPMS cells as the instance). One-off."""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / 'MANUSCRIPT_EN.md'
L = SRC.read_text().split('\n')
assert L[4].startswith('Static condensation of cut thin-walled'), L[4][:60]
assert L[10].startswith('Finite structures built from thin-walled'), L[10][:60]
assert L[24].startswith('Sections 2–4 develop the method'), L[24][:60]

ABSTRACT = (
    "A reduced model of a structural component must do more than reproduce displacements: it must assemble, so it must be "
    "symmetric and positive semidefinite with the rigid-body kernel; its error must be traceable to the assembled response; "
    "it must supply design sensitivities; and it should be improvable without retraining. Learned substructure models seldom "
    "have all of these properties. We show that they follow from where the approximation is placed and how it is read out: "
    "approximate only the interior extension on the complete retained space, evaluate it in the energy form with its complete "
    "transpose, and add a fixed, geometry-specific multilevel equilibrium correction. Learning supplies the trial interior "
    "field, the variational form supplies the structure, and the correction supplies the improvability: the condensed "
    "stiffness is bounded below by the exact Schur complement with an error quadratic in the interior error, and a correction "
    "whose smoothing interval contains the spectrum cannot increase it. Tracing the interior error through energy "
    "participation and the stiffness derivative shows why accurate compliance need not imply accurate thickness sensitivity, "
    "so both are required for design. We realise this construction, neural-initialised condensation with equilibrium "
    "correction (NICE), for cut thin-walled Schwarz-P cells with tens of thousands of retained coordinates, whose cuts change "
    "the discrete space from cell to cell. On 80 validation geometries the mean directional energy error is 0.074%; in "
    "two-cell assemblies the compliance and eight-corner sensitivity errors stay below 0.28% and 1.5%; an eight-cell design "
    "iteration takes 81–110 s on one GPU against 868–1,042 s for the whole-lattice Cholesky solution; and thickness "
    "optimisation reproduces the design of exact condensation to 0.0051 per corner parameter, with a 110-cell plate of "
    "32.7 million degrees of freedom taking 43 min per design iteration.")

P = {}
P[1] = (
    "Structures assembled from repeated thin-walled cells, such as graded triply periodic minimal surface (TPMS) lattices with "
    "supports, external loads and cells cut by the specimen boundary [Yu et al. (2019)](https://doi.org/10.1016/j.matdes.2019.108021), "
    "call for a reusable model of one cell: a component model that returns the forces conjugate to the displacement imposed by "
    "its surroundings, so that the structure can be assembled, and the interior field, from which local design quantities such "
    "as thickness sensitivities are evaluated. Homogenisation provides such a model only under a separation of scales that a "
    "single layer of cut cells does not offer (Section 5.11). Static condensation provides an exact one "
    "[Guyan (1965)](https://doi.org/10.2514/3.2874), [Irons (1965)](https://doi.org/10.2514/3.3027), at the cost of an interior "
    "factorisation for every geometry, which a design iteration repeats for every cell. Machine learning promises an inexpensive "
    "one but, as usually constructed, delivers a map without the structure of a mechanical model. A component model that is to "
    "replace static condensation in analysis and design must be symmetric and positive semidefinite with the rigid-body kernel, "
    "so that it can be assembled; its error must be traceable to the quantities that analysis and design require; it must "
    "deliver sensitivities; and it should be improvable at deployment without retraining. This paper shows that a learned "
    "component model acquires all of these properties from where its approximation is placed and how it is read out, and "
    "quantifies what then determines its error in the assembled compliance and in the local thickness sensitivity.")
P[2] = (
    "For cut thin-walled cells the retained space is large: box-face coefficients carry exterior loads and intercell coupling, "
    "and a band of cut-element coefficients carries virtual work on the cut surface, so each cell here retains tens of thousands "
    "of coordinates. Forming the Schur complement explicitly needs one interior solve per retained coordinate, and applying it "
    "needs an interior factorisation for every geometry. Two approximations reduce this cost. Reducing the retained coordinates, "
    "as in port reduction [Eftang & Patera (2013)](https://doi.org/10.1002/nme.4543), "
    "[Smetana & Patera (2016)](https://doi.org/10.1137/15m1009603) and in learned substructures described by a few boundary "
    "coordinates [Huang et al. (2023)](https://doi.org/10.1016/j.eml.2023.102041), "
    "[Guo et al. (2026a)](https://doi.org/10.1016/j.cma.2026.118955), changes the displacement patterns that substructures can "
    "exchange and leaves a boundary model error independent of the interior. Approximating the interior extension on the "
    "complete retained space, the route of the static-condensation reduced basis element method "
    "[Huynh et al. (2013)](https://doi.org/10.1051/m2an/2012022), [Ballani et al. (2018)](https://doi.org/10.1016/j.cma.2017.09.014) "
    "and of inexact substructuring [Dohrmann (2007)](https://doi.org/10.1002/nla.514), "
    "[Li & Widlund (2007)](https://doi.org/10.1016/j.cma.2006.03.011), [Klawonn & Rheinbach (2007)](https://doi.org/10.1002/nme.1758), "
    "alters only the interior response to each pattern. We follow the second route with a learned, geometry-conditioned interior "
    "extension. The two kinds of error enter the structural response differently. An interior error is weighted by the energy the "
    "cell carries, so a weakly participating cell can leave the compliance accurate while its thickness sensitivity, weighted by "
    "the stiffness derivative, remains inaccurate: the distinction on which goal-oriented error estimation rests "
    "[Becker & Rannacher (2001)](https://doi.org/10.1017/S0962492901000010), "
    "[Oden & Prudhomme (2001)](https://doi.org/10.1016/S0898-1221(00)00317-5), observed for approximate reanalysis and inexact "
    "solves in topology optimisation [Amir et al. (2009)](https://doi.org/10.1002/nme.2536), "
    "[Amir et al. (2010)](https://doi.org/10.1007/s00158-009-0463-4), [Gogu (2015)](https://doi.org/10.1002/nme.4797). We quantify "
    "it for learned substructures and adopt joint accuracy of compliance and local sensitivity as the criterion for design use.")
P[3] = (
    "The construction rests on three elements. Any admissible extension that reproduces rigid motion, evaluated in the energy "
    "form, yields a symmetric positive semidefinite condensed stiffness with the rigid kernel, bounded below by the exact Schur "
    "complement with an excess quadratic in the interior error [Toselli & Widlund (2005)](https://doi.org/10.1007/b137868); this "
    "holds for static condensation itself, for multiscale finite elements [Hou & Wu (1997)](https://doi.org/10.1006/jcph.1997.5682), "
    "for component reduced-basis methods [Huynh et al. (2013)](https://doi.org/10.1051/m2an/2012022) and for learned shape "
    "functions evaluated as \\(N^TKN\\) [Huang et al. (2023)](https://doi.org/10.1016/j.eml.2023.102041). Learning supplies the trial "
    "extension: a geometry-conditioned network, exactly linear in the retained displacement, produces an initial interior field "
    "for each cell. A fixed, geometry-specific multilevel correction, polynomial smoothing and a Galerkin coarse solve with a "
    "prescribed number of steps [Xu (1992)](https://doi.org/10.1137/1034116), then reduces the interior equilibrium residual at "
    "fixed retained displacement. Because the correction is a fixed linear map inside the condensed energy, the corrected operator "
    "keeps every variational property, its error cannot exceed that of the uncorrected one "
    "[Golub & Varga (1961)](https://doi.org/10.1007/BF01386013), [Xu & Zikatanov (2002)](https://doi.org/10.1090/S0894-0347-02-00398-3), "
    "and its complete transpose returns the force conjugate to the stated energy. Unlike hybrid neural solvers, which combine "
    "learned predictions with relaxation, learned inverse actions or coarse spaces to accelerate one global solve "
    "[Zhang et al. (2024)](https://doi.org/10.1038/s42256-024-00910-x), "
    "[Kopaničáková & Karniadakis (2025)](https://doi.org/10.1137/24m162861x), [Xing et al. (2026)](https://doi.org/10.1145/3811333), "
    "the output is a condensed operator per cell that can be assembled, differentiated and improved. We call the construction "
    "neural-initialised condensation with equilibrium correction (NICE).")
P[4] = (
    "Learned substructures have been developed furthest as problem-independent machine learning (PIML), which predicts "
    "substructure shape functions from a few boundary coordinates, imposes rigid-motion constraints and, in its self-consistent "
    "form, evaluates the stiffness as \\(N^TKN\\) [Huang et al. (2022)](https://doi.org/10.1016/j.eml.2022.101887), "
    "[Huang et al. (2023)](https://doi.org/10.1016/j.eml.2023.102041); its variants train without labelled shape functions by "
    "minimum potential energy [Huang et al. (2024)](https://doi.org/10.1016/j.jmps.2024.105893), enrich the boundary with Bézier "
    "functions [Guo et al. (2026a)](https://doi.org/10.1016/j.cma.2026.118955), join oversampled bases by an overlapping partition "
    "of unity [Guo et al. (2026b)](https://arxiv.org/abs/2607.22019v1), impose symmetry equivariance "
    "[Jiang, C. et al. (2026)](https://doi.org/10.1016/j.compstruct.2026.120865), treat isoparametric substructures "
    "[Zhang et al. (2026)](https://doi.org/10.1016/j.ijmecsci.2026.112007) and optimise lattices "
    "[Xu et al. (2025)](https://doi.org/10.1016/j.compstruct.2025.119330). Describing each substructure by a few boundary "
    "coordinates gives coarse models small enough for very large structures, at the price of a boundary model error that this "
    "line of work controls by refining the partition, enriching the interpolation or oversampling. Guo et al. (2026a, Table 1) "
    "also retained every boundary node with a learned interior and set this option aside: for \\(10^3\\)-voxel substructures its "
    "displacement error, 3.46% against 2.22% with the Bézier boundary, grew, which they attribute to the larger network output, "
    "and the denser coupling raised the cost. The present construction retains the complete trace of cut thin-walled cells, "
    "\\(1.7\\)–\\(2.6\\times10^4\\) coordinates per cell, and restores the accuracy of the learned interior with the correction "
    "(directional energy error about 0.07%, assembled compliance errors 0.01–0.27%), at the price of tens of thousands of coarse "
    "coordinates per cell. Related energy-based models include embedded positive-semidefinite learned elements "
    "[Parish et al. (2024)](https://doi.org/10.1007/s00466-024-02481-5) and convex neural energy elements, which learn boundary "
    "energies from Schur-complement targets and recover linear fields by exact interior lifting "
    "[Jiang, H. et al. (2026)](https://arxiv.org/abs/2608.02036v1); here one corrected extension defines both the assembled "
    "operator and the recovered field. Component reduced-basis methods certify an offline interior basis a posteriori "
    "[Huynh et al. (2013)](https://doi.org/10.1051/m2an/2012022), but presuppose a common discrete space across the parameter "
    "family, which a cut removes: the active elements and the retained set change from one geometry to the next, so NICE "
    "conditions one network on the discrete geometry of each cell and provides a computable residual (Eq. (5)) rather than a "
    "certified bound. Learned multigrid prolongations [Greenfeld et al. (2019)](https://proceedings.mlr.press/v97/greenfeld19a.html), "
    "[Luz et al. (2020)](https://proceedings.mlr.press/v119/luz20a.html), adaptive FETI-DP coarse spaces "
    "[Heinlein et al. (2021)](https://doi.org/10.1137/20M1344913), networks trained through the solver "
    "[Um et al. (2020)](https://proceedings.neurips.cc/paper/2020/hash/43e4e6a6f341e00671e123714de019a8-Abstract.html), "
    "geometry-aware neural operators and graph networks [Li et al. (2023a)](https://www.jmlr.org/papers/v24/23-0064.html), "
    "[Li et al. (2023b)](https://doi.org/10.52202/075280-1556), [Pfaff et al. (2021)](https://openreview.net/forum?id=roNqYL0_XP), "
    "multigrid-structured operators "
    "[He et al. (2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/eb3c8135137c8a60425a0320869ad87e-Abstract-Conference.html), "
    "learned Green's functions [Boullé & Townsend (2023)](https://doi.org/10.1007/s10208-022-09556-w) and Ritz-type training "
    "[E & Yu (2018)](https://doi.org/10.1007/s40304-018-0127-z) accelerate a global solve or learn solution maps; here the learned "
    "map is exactly linear in the retained displacement, so that its energy defines an assembled stiffness. Coarse multiscale "
    "bases with fine-scale correction have been used in optimisation without scale separation "
    "[Alexandersen & Lazarov (2015)](https://doi.org/10.1016/j.cma.2015.02.028), "
    "[Efendiev et al. (2013)](https://doi.org/10.1016/j.jcp.2013.04.045), including numerical shape functions on unfitted coarse "
    "elements conditioned against small cuts [Chen & Li (2026)](https://doi.org/10.1016/j.cad.2026.104038), and for unfitted "
    "two-dimensional lattices BDDC has been combined with reduced-order cell stiffnesses "
    "[Bonilla Moreno et al. (2027)](https://doi.org/10.1016/j.cma.2026.119304).")
P[5] = (
    "The contributions of this work are threefold. (i) A construction principle for learned component models: on the complete "
    "retained space of a cell, a learned linear interior extension, a fixed geometry-specific multilevel correction and the "
    "complete transpose are combined in one condensed energy, and the resulting operator is symmetric and positive semidefinite "
    "with the rigid kernel, bounded below by the exact Schur complement, improvable without loss of these properties, "
    "assemblable and differentiable, for any symmetric positive semidefinite discrete model. (ii) An error analysis from the "
    "component to the structure and to the design: the interior error enters the compliance through a load-specific "
    "participation bound and the thickness sensitivity through the stiffness derivative, with a linear term that the energy "
    "error does not control and an amplification of displacement error into energy error in thin walls, which makes joint "
    "accuracy of compliance and local sensitivity the design criterion. (iii) A realisation for cut thin-walled Schwarz-P cells "
    "in a stabilised cut finite element model, with a geometry-conditioned network that handles the cut-dependent discrete "
    "space and the cut band, and its verification on 80 validation geometries, fourteen two-cell configurations of development "
    "cells, eighteen of held-out cells and two eight-cell lattices, together with an ablation of the retained representation, "
    "a cost comparison with the whole-lattice direct solution and conventional exact condensation, and thickness optimisations "
    "driven by the field-based sensitivities: of an eight-cell lattice, matching the design obtained with exact condensation to "
    "within 0.0051 in every corner parameter, and of cut 24-cell plates, compared with a homogenised model, with design "
    "iterations timed on plates of up to 110 cells and 32.7 million degrees of freedom (Section 5.11).")
P[6] = (
    "Section 2 states the discrete cut-cell model, exact condensation and the requirements on an approximate condensed "
    "stiffness; Section 3 derives the error relations that hold for any admissible extension; Section 4 constructs NICE; "
    "Section 5 reports the numerical examples; Section 6 discusses them and states the limitations; Section 7 concludes.")

L[4] = ABSTRACT
intro = []
for k in range(1, 7):
    intro += [P[k], '']
L[10:25] = intro[:-1]
SRC.write_text('\n'.join(L))
print('abstract words:', len(re.findall(r"\S+", ABSTRACT)))
print('intro words:', sum(len(re.findall(r'\S+', re.sub(r'\]\([^)]*\)', ']', P[k]))) for k in P))
