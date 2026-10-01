"""Post-join fixes of the R3 revision (answers to the part editors' questions)."""
import json, re
from pathlib import Path
D = Path('/home/user/curly-octo-goggles/docs/paper_p1')
M, A, S = (D / f for f in ('MANUSCRIPT_EN.md', 'APPENDICES_EN.md', 'SUPPLEMENTARY_EN.md'))
t = {f: f.read_text() for f in (M, A, S)}
log = []


def rep(f, old, new, n=1, what=''):
    c = t[f].count(old)
    assert c == n, (what or old[:60], c)
    t[f] = t[f].replace(old, new); log.append(f'{f.name}: {what or old[:70]}')


# 1. Introduction: keep the Guo et al. (2026a, Table 1) numbers once (caveat table: Introduction only)
rep(M, 'set it aside because its error grew with the size of the network output.',
    'set it aside because its error grew with the size of the network output (3.46% against 2.22% with the Bézier boundary).',
    what='Guo numbers in the Introduction')
# 2. The correction W is one two-grid cycle: front matter and conclusions
rep(M, 'A fixed, geometry-specific multilevel correction applied inside that energy', 'A fixed, geometry-specific two-grid correction applied inside that energy', what='abstract: two-grid')
rep(M, 'A fixed, geometry-specific multilevel correction, Chebyshev smoothing and a coarse-grid Galerkin correction with a prescribed number of steps',
    'A fixed, geometry-specific two-grid correction, Chebyshev smoothing around a coarse-grid Galerkin correction with a prescribed number of steps',
    what='introduction: two-grid')
rep(M, 'a fixed geometry-specific multilevel correction and the transpose of the complete extension', 'a fixed geometry-specific two-grid correction and the transpose of the complete extension', what='contribution (i): two-grid')
rep(M, 'adding a fixed, geometry-specific multilevel correction', 'adding a fixed, geometry-specific two-grid correction', what='conclusions: two-grid')
# 3. Section 5.1: offline cost and the GPU named at the first cost statement
rep(M, 'Training the base network took 4.6 h and NICE\'s continuation 3.5 h on one GPU, and generating the directions and exact sensitivities of the 691 training and validation geometries took about 42 GPU-hours; the offline cost of NICE, including model selection and every training stage of the base network, is about 60 GPU-hours (Table ST01).',
    'On one NVIDIA GeForce RTX 5090 GPU, training the base network took 4.6 h and NICE\'s continuation 3.5 h, and generating the directions and exact sensitivities of the 691 training and validation geometries took about 42 GPU-hours; data generation, training and model selection together took about 60 GPU-hours (Table ST01).',
    what='5.1 offline cost, GPU named')
rep(M, '(c) NICE, on one NVIDIA GeForce RTX 5090 GPU, prepares', 'Route (c), NICE on the GPU, prepares', what='5.9 route (c)')
rep(M, '\n(a) The whole-lattice direct solution assembles', '\nRoute (a), the whole-lattice direct solution, assembles', what='5.9 route (a)')
rep(M, '\n(b) Conventional exact condensation condenses', '\nRoute (b), conventional exact condensation, condenses', what='5.9 route (b)')
# 4. Figure 10 caption: L1 is not in the figure
rep(M, 'Each point is one load of one variant–configuration combination of Table ST09.',
    'Each point is one load of one variant–configuration combination of Table ST09 other than L1.', what='Figure 10 caption: L1')
# 5. Appendix H.2: the step study is reported in the Figure S01 caption
rep(A, 'Figure S01(c) and Supplementary Note S2 report a step-refinement study', 'Figure S01(c) reports a step-refinement study', what='H.2 pointer')
# 6. 'teacher' (machine-learning jargon) -> exact reference computation
rep(A, 'use a recomputed teacher energy for the supplied direction', 'use a recomputed exact energy for the supplied direction', what='F.1 teacher')
rep(A, 'Recomputed teacher denominators', 'Recomputed exact denominators', what='F.3 teacher')
rep(A, "The teacher's standard entry points initialise", 'The reference integration initialises', what='F.4 teacher')
rep(A, 'The isolated teacher removes the rigid component', 'The exact reference computation for an isolated cell removes the rigid-body component', what='G.2 teacher')
rep(A, 'In the teacher, moment derivatives are approximated by', 'In the reference computation, moment derivatives are approximated by', what='H teacher')
t[A] = t[A].replace('The tetrahedral quadrature helper is called with its order parameter set to four.', 'The tetrahedra are integrated with a rule of order four.')
# 7. Undefined assembled symbol in Note S6.3 (same change as Appendix C.2)
rep(S, r'\widehat{\mathbb S}\succeq\mathbb S', r'\widehat{\mathbb K}\succeq\mathbb K', what='S6.3 symbol')
# 8. Supplement: 3% line is a reference, not a criterion
t[S] = t[S].replace('defining the joint criterion', 'compared with the 3% reference').replace('the joint criterion', 'the 3% reference')
t[S] = re.sub(r'\| Above criterion \|', '| Above reference |', t[S]); t[S] = re.sub(r'\| Pass \|', '| Within reference |', t[S])
log.append('supplement: criterion -> reference')
# 9. Data lines with repository paths -> data archive
for old in ('Data: `evidence/p1_checks_cpu.json` (H2, H1, M2, M1) and `evidence/p1_checks_u2.json` (U2); see Supplementary Note S2.',
            'Data: `evidence/p1_checks_cpu.json`, `evidence/p1_checks_u2.json`.',
            'Data: `evidence/p1_checks_cpu.json`, `evidence/p1_checks_u2.json` and, for U1, `evidence/p1_checks.json`.'):
    rep(S, old, 'Data: records of the data archive.', what='data line')
rep(S, ' (`evidence/meta_p1.json`)', '', what='ST20 data path')
open(D.parent.parent / '..' / 'tmp_postjoin.log', 'w') if False else None
for f in (M, A, S):
    f.write_text(t[f])
print('\n'.join(log))
