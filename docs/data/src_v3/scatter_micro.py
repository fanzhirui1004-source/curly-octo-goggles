import torch, time, json
dev = torch.device('cuda')
def timed(fn, rep=20):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep): fn()
    torch.cuda.synchronize(); return round(1000 * (time.perf_counter() - t) / rep, 4)
nb = 323424
out = {}
for dtp in (torch.float32, torch.float64):
    for n, uniq in ((303738, True), (7550 * 135, False)):
        idx = torch.randperm(nb, device=dev)[:n] if uniq else torch.randint(0, nb, (n,), device=dev)
        for b in (4, 6, 12):
            y = torch.zeros(nb, b, dtype=dtp, device=dev); v = torch.randn(n, b, dtype=dtp, device=dev)
            ar = torch.arange(b, device=dev)
            flat = (idx[:, None] * b + ar).reshape(-1)
            r = dict(index_add=timed(lambda: y.index_add_(0, idx, v)),
                     flat_index_add=timed(lambda: y.view(-1).index_add_(0, (idx[:, None] * b + ar).reshape(-1), v.reshape(-1))))
            if uniq:
                r.update(put=timed(lambda: y.__setitem__(idx, v)), index_copy=timed(lambda: y.index_copy_(0, idx, v)),
                         flat_put=timed(lambda: y.view(-1).index_copy_(0, (idx[:, None] * b + ar).reshape(-1), v.reshape(-1))))
                a = y.clone().index_copy_(0, idx, v); c = y.clone(); c.view(-1).index_copy_(0, flat, v.reshape(-1)); assert torch.equal(a, c)
            out[f'{str(dtp)[6:]}_{n}_{b}'] = r
print(json.dumps(out))
