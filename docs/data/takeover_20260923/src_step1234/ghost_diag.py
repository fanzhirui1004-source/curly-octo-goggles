"""Why does same_ghost fail when the face count matches? Compare base and perturbed ghost matrices piece by piece."""
import json, sys
from pathlib import Path
import numpy as np
from scipy import sparse
sys.path.insert(0, '/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/xcase_src_11')
from gp_check_body import load_body
from stage_cutfem_gp import assembly

T = Path('/root/autodl-tmp/CLAUDE_TAKEOVER_20260923')
base, pid = sys.argv[1], sys.argv[2]


def ghost(body):
    n = body['contract'].n; nodes = np.asarray(body['nodes']); nb = 3 * len(nodes)
    faces, _ = assembly.select_faces(body)
    order = np.argsort(nodes); sn = nodes[order]
    F = np.asarray(faces)
    cells = np.stack([assembly.cell_index(body, int(o)) for o in F[:, 0]])
    offs = np.stack([assembly.stencil(a)[0] for a in range(3)])
    ids = np.ravel_multi_index((2 * cells[:, None, :] + offs[F[:, 2]]).transpose(2, 0, 1), (2 * n + 1,) * 3)
    local = order[np.searchsorted(sn, ids)]
    dofs = (3 * local[:, :, None] + np.arange(3)).reshape(len(F), -1)
    canonical = np.stack([assembly.face_factor(a, 1 / n) for a in range(3)])
    tk = canonical.transpose(0, 2, 1) @ canonical
    G = sparse.csr_matrix((tk[F[:, 2]].ravel(), (np.repeat(dofs, 135, axis=1).ravel(), np.tile(dofs, (1, 135)).ravel())), shape=(nb, nb))
    return F, nodes, G


def diff(a, b):
    d = abs(a - b)
    return dict(max_abs=float(d.max()) if d.nnz else 0.0, max_rel=float(d.max() / abs(b).max()) if d.nnz else 0.0,
                nnz_a=int(a.nnz), nnz_b=int(b.nnz), pattern_equal=bool((a != 0).nnz == (b != 0).nnz and ((a != 0) != (b != 0)).nnz == 0))


rec = {}
Fb, nb_, Gb = ghost(load_body(T / 'COVER_G' / 'runs' / (base + '_G')))
Fp, np_, Gp = ghost(load_body(T / 'SENS_01' / 'runs' / (pid + '_G')))
G0 = sparse.load_npz(T / 'COVER_INPUTS' / base / 'GHOST.npz').tocsr()
rec['nodes_equal'] = bool(np.array_equal(nb_, np_))
rec['faces_equal'] = bool(Fb.shape == Fp.shape and np.array_equal(Fb, Fp))
rec['faces_base_vs_published'] = bool(np.array_equal(Fb, np.load(Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/packets') / base / 'CONTEXT' / 'GP_FACES.npy')))
rec['vectorised_base_vs_saved'] = diff(Gb, G0)
rec['perturbed_vs_vectorised_base'] = diff(Gp, Gb)
rec['perturbed_vs_saved'] = diff(Gp, G0)
print(json.dumps(rec, indent=1))
