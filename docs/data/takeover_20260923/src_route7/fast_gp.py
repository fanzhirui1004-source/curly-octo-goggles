"""Route 7: the frozen ghost-penalty face selection with its per-cell full-cell test run in a process pool.

The per-cell test (stage_cutfem_graded.fields.full_background_box: exact rational vertices, interval enclosure of
both sheet inequalities, macro plane) is called unchanged on every active cell; only the loop is parallel. The rest
of select_faces is copied verbatim from stage_cutfem_gp.assembly, so the face list must equal the frozen one
exactly (checked against the packets' published GP_FACES). The ghost matrix is then the fixed-template scatter
already used and checked in sens_blocks.py.
"""
import json
import multiprocessing as mp
from collections import deque
import numpy as np
from scipy import sparse
from stage_cutfem_gp import assembly
from stage_cutfem_gp.assembly import cell_index

_BODY = None


def _full(owner):
    from stage_cutfem_graded.fields import full_background_box
    return owner, bool(full_background_box(_BODY['contract'], cell_index(_BODY, owner)))


def select_faces(body, workers=16):
    global _BODY
    c=body['contract'];n=c.n
    if body['record']['component_count']!=1:raise ValueError('GHOST_REQUIRES_CERTIFIED_COMPONENT_PAIRS')
    active=set(map(int,body['active']))
    if hasattr(c, 'thickness') and not c.thickness.legacy_constant:
        _BODY = body
        with mp.get_context('fork').Pool(workers) as pool:
            full = dict(pool.map(_full, sorted(active), chunksize=max(1, len(active) // (8 * workers))))
    else:
        from stage_cutfem_gp.assembly import Domain, classify_box
        scalar = c.constant_legacy_contract() if hasattr(c, 'thickness') else c
        domain=Domain(scalar.case_id,scalar.tau,scalar.normal,scalar.offset)
        full={owner:(bool(body['interval_full'][owner]) if 'interval_full' in body else
            classify_box(domain,cell_index(body,owner)/n,(cell_index(body,owner)+1)/n).startswith('FULL')) for owner in active}
    lookup={tuple(cell_index(body,owner)):owner for owner in active}
    faces=[]
    for owner in sorted(active):
        ijk=cell_index(body,owner)
        for axis in range(3):
            other=ijk.copy();other[axis]+=1;neighbor=lookup.get(tuple(other))
            if neighbor is not None and not (full[owner] and full[neighbor]):faces.append((owner,neighbor,axis))
    fractions={i:float(body['moments'][i,0]*n**3) for i in active}
    roots=sorted(i for i in active if fractions[i]>=.1);distance={i:0 for i in roots}
    graph={i:[] for i in active}
    for a,b,_ in faces:graph[a].append(b);graph[b].append(a)
    queue=deque(roots)
    while queue:
        i=queue.popleft()
        for j in graph[i]:
            if j not in distance:distance[j]=distance[i]+1;queue.append(j)
    uncovered=sorted(i for i in active if i not in distance)
    record=dict(active_cells=len(active),full_certified_cells=sum(full.values()),faces=len(faces),
        root_volume_fraction=.1,root_count=len(roots),uncovered_cells=uncovered,
        maximum_root_path=max(distance.values(),default=None),
        smallest_volume_fraction=min(fractions.values()),cross_module_faces=0,
        component_proof=body['record']['topology_proof'],face_definition='both active, at least one not interval-certified full')
    if uncovered:raise ValueError('GHOST_PATCH_WITHOUT_POSITIVE_VOLUME_ROOT:'+json.dumps(record))
    return faces,record


def ghost_matrix(body, faces):
    """Fixed-template scatter of the ghost faces (same construction as sens_blocks.py v2 / cover_blocks.py)."""
    n = body['contract'].n; nodes = np.asarray(body['nodes']); nb = 3 * len(nodes)
    order = np.argsort(nodes); sn = nodes[order]
    F = np.asarray(faces)
    cells = np.stack([cell_index(body, int(o)) for o in F[:, 0]])
    offs = np.stack([assembly.stencil(a)[0] for a in range(3)])
    ids = np.ravel_multi_index((2 * cells[:, None, :] + offs[F[:, 2]]).transpose(2, 0, 1), (2 * n + 1,) * 3)
    local = order[np.searchsorted(sn, ids)]
    dofs = (3 * local[:, :, None] + np.arange(3)).reshape(len(F), -1)
    canonical = np.stack([assembly.face_factor(a, 1 / n) for a in range(3)])
    tk = canonical.transpose(0, 2, 1) @ canonical
    return sparse.csr_matrix((tk[F[:, 2]].ravel(), (np.repeat(dofs, 135, axis=1).ravel(), np.tile(dofs, (1, 135)).ravel())), shape=(nb, nb))
