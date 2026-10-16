import torch, time, json
dev = torch.device('cuda')
def timed(fn, rep=20):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep): fn()
    torch.cuda.synchronize(); return round(1000 * (time.perf_counter() - t) / rep, 4)
nb = 323424
out = {}
for dtp in (torch.float32, torch.float64):
    for shape in ((7550, 135), (303738,)):
        idx = torch.randint(0, nb, shape, device=dev); fl = idx.reshape(-1)
        for b in (4, 6, 12):
            x = torch.randn(nb, b, dtype=dtp, device=dev)
            ref = x[fl]
            ar = torch.arange(b, device=dev)
            flat = (fl[:, None] * b + ar).reshape(-1)
            g_idx = fl[:, None].expand(-1, b)
            f1 = lambda: torch.take(x, flat).view(-1, b)
            f2 = lambda: torch.gather(x, 0, g_idx)
            f3 = lambda: x.view(-1)[flat].view(-1, b)
            for f in (f1, f2, f3):
                assert torch.equal(f(), ref)
            out[f'{str(dtp)[6:]}_{shape[0]}_{b}'] = dict(index=timed(lambda: x[fl]), take=timed(f1), gather=timed(f2), flat=timed(f3),
                                                        take_inc_idx=timed(lambda: torch.take(x, (fl[:, None] * b + ar).reshape(-1))))
print(json.dumps(out))
