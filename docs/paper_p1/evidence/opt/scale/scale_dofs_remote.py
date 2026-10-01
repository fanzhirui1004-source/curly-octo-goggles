import json, numpy as np
def dec(ids, conv):
    a, b, c = ids // 4225, (ids // 65) % 65, ids % 65
    return np.stack([a, b, c] if conv == 0 else [c, b, a], 1)
def count(run, lay, conv, only=None):
    L = json.load(open(f'layouts/{lay}.json')); tot = port = cutn = 0; P = []
    cells = L['cells'] if only is None else [c for c in L['cells'] if c['position'] in only]
    for c in cells:
        d = f'{run}/body/{c["case"]}_o000'
        pr = json.load(open(d + '/PREP.json')); tot += pr['nodes']; port += pr['port_nodes']; cutn += pr['cut_nodes']
        B = np.load(d + '/BOX_NODES.npy'); P.append(dec(B, conv) + 64 * np.array(c['position']))
    P = np.unique(np.concatenate(P), axis=0)
    return dict(cells=len(cells), cell_dofs_summed=3*tot, retained_summed=3*port, cut_nodes=cutn,
                box_assembled=3*len(P), clamped_box=3*int((P[:, 0] == P[:, 0].min()).sum()))
L = json.load(open('layouts/plateS24.json')); print('pos0,1', L['cells'][0]['position'], L['cells'][1]['position'])
for conv in (0, 1): print('conv', conv, count('plateS24r4', 'plateS24', conv, [[0,0,0],[1,0,0]]))
for run, lay in [('plateS24r4','plateS24'),('plateS51r4','plateS51'),('plateS88','plateS88'),('plateS110','plateS110'),('plateS135','plateS135')]:
    for conv in (0, 1): print('ALL', run, conv, json.dumps(count(run, lay, conv)))
