# R7 notes: section-by-section revision from the author's annotations

## Working rules (author, 11 October 2026)

- The author annotates the Chinese text section by section, starting with the abstract. Each revised passage is shown
  in Chinese and English before it enters the paper.
- During this revision the PDFs, the advisor set and the Word files are not rebuilt after each change. Edit the unit
  files in `sections/`, run `python3 assemble_r5.py --write`, then commit and push at once, so that nothing is lost
  when the container is reset. Rebuild the outputs once, when the author asks.
- The full texts of the PIML papers (Huang et al. 2023, 2024; Jiang et al. 2026; Guo et al. 2026a, 2026b; Zhang et al.
  2026) are kept in the private repository `fanzhirui1004-source/piml-papers`. This repository is public, so they are
  never committed here.
- The authors are not in the group that developed the learned substructure (PIML) methods. The abstract does not
  comment on the limits of those methods: such a sentence would read as provocative. The neutral comparison in
  Introduction ¶3 and Section 1.1 stays as it is.

## Abstract (committed; wording settled with the author)

1. Opening: thickness optimisation of cut lattices needs both fine-scale accuracy and a fast analysis at every design
   iteration. Do not open with cut cells carrying supports and loads (in a fine-scale model they are ordinary cells).
2. Fine-scale models: one short sentence on their cost; no degree-of-freedom count.
3. Homogenisation: periodic boundary conditions on a unit cell, inaccurate without scale separation, particularly in
   cut cells, which are not periodic.
4. Static condensation in one sentence: reduces the global system without loss of accuracy, but every interior
   stiffness matrix is refactorised at every design iteration.
5. "本文提出 NICE 方法以避免上述分解", then one sentence on how it works (all face and cut-plane-element degrees of
   freedom retained; network with the cell geometry as input; two-grid cycle on the force imbalance; condensed
   stiffness from the strain energy of the corrected field).
6. Properties: upper bound on the exact condensed stiffness, error of second order in the interior displacement
   error, independent of the network accuracy. The conditions (linear in the retained displacements, reproduces them
   and rigid-body motions) and the guarantee that the correction does not increase the error (Proposition 2) stay in
   the body: the author found the conditional statement unclear in an abstract.
7. Sensitivities: their error is of first order in the interior displacement error, so the accuracy of the compliance
   does not ensure that of the sensitivities; both are trained and checked. No "然而" before this sentence.
8. Formal register throughout (no "这样得到的", "只会", "这两点"). EN at most 250 words; numbers as before.

## Introduction (proposal, awaiting the author)

The introductions of the PIML papers (Huang et al. 2023, 2024; Jiang et al. 2026; Guo et al. 2026a; Zhang et al. 2026)
give their own results only in the "this paper" paragraph and do not expand the theory there. Proposed changes:

1. Move the comparison in ¶1 (homogenisation 27%, NICE 0.011%) to the end of the NICE paragraph.
2. In ¶2–¶3 (static condensation, learned component models) tighten the wording only.
3. Merge ¶4–¶6 (method, properties, compliance and sensitivity; about 1,300 Chinese characters) into one paragraph of
   about 500 characters, and leave the details to the contribution list and Sections 3–4, which repeat them now.
