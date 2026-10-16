import sys, torch
sys.path.insert(0, '/root/autodl-tmp/OPL/src')
g = torch.load('/root/autodl-tmp/OPL/S2/slots/%s.pt' % sys.argv[1], map_location='cpu', weights_only=False)
seen, rows = set(), []
def walk(o, path, depth=0):
    if id(o) in seen or depth > 6: return
    seen.add(id(o))
    if torch.is_tensor(o):
        n = o.numel() * o.element_size()
        if o.layout != torch.strided:
            n = sum(t.numel() * t.element_size() for t in (o.crow_indices(), o.col_indices(), o.values())) if o.layout == torch.sparse_csr else 0
        rows.append((n, path, tuple(o.shape), str(o.dtype))); return
    if isinstance(o, dict):
        for k, v in o.items(): walk(v, f'{path}[{k}]', depth + 1)
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o): walk(v, f'{path}[{i}]', depth + 1)
    elif hasattr(o, '__dict__') and not isinstance(o, type):
        for k, v in vars(o).items(): walk(v, f'{path}.{k}', depth + 1)
walk(g, 'geo')
rows.sort(reverse=True)
tot = sum(r[0] for r in rows)
print('total GB %.2f' % (tot / 2**30))
agg = {}
for n, p, s, d in rows:
    key = p.split('[')[0] if '[' in p else p
    agg[key] = agg.get(key, 0) + n
for k, v in sorted(agg.items(), key=lambda x: -x[1])[:25]:
    print('%8.3f GB  %s' % (v / 2**30, k))
