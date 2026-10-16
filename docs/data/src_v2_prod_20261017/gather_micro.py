import torch, time, json
dev = torch.device('cuda')
def timed(fn, rep=20):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep): fn()
    torch.cuda.synchronize(); return round(1000 * (time.perf_counter() - t) / rep, 4)
nb = 323424
out = {}
for dtp in (torch.float32, torch.float64):
    for shape in ((7550, 135), (11244, 81), (303738,)):
        idx = torch.randint(0, nb, shape, device=dev)
        fl = idx.reshape(-1)
        for b in (1, 2, 4, 6, 8, 12, 16):
            x = torch.randn(nb, b, dtype=dtp, device=dev)
            a = x[idx]; c = torch.index_select(x, 0, fl).view(*shape, b)
            assert torch.equal(a, c)
            out[f'{str(dtp)[6:]}_{shape[0]}_{b}'] = (timed(lambda: x[idx]), timed(lambda: torch.index_select(x, 0, fl)))
print(json.dumps(out))
