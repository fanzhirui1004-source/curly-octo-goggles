exec(open('regret_toy.py').read().split("rows = []")[0])
_, _, wref = solve(t0)
def mkfrozen_mask(d, mask, seed=0):
    rr = np.random.default_rng(100+seed); XI = rr.standard_normal((nel, 4)); XI /= np.linalg.norm(XI, axis=1, keepdims=True)
    dl = np.where(mask, d, 0.0)
    return lambda s, w, rng: s + dl[:, None]*np.linalg.norm(s, axis=1, keepdims=True)*XI
lo = wref < 1e-3   # cells with < 0.1% of the energy at the exact optimum
print('cells with share < 0.1%%: %d of %d, carrying %.1f%% of the energy' % (lo.sum(), nel, 100*wref[lo].sum()))
lo2 = wref < 1e-4
print('cells with share < 0.01%%: %d, carrying %.2f%%' % (lo2.sum(), 100*wref[lo2].sum()))
for name, d, mask in [('frozen 3%% on share>=0.1%% only', .03, ~lo), ('frozen 6%% on share<0.1%% only', .06, lo),
                      ('frozen 30%% on share<0.01%% only', .30, lo2), ('frozen 3%% on share>=0.01%% only', .03, ~lo2)]:
    regs = []
    for seed in range(3):
        t, C, w = optimise(mkfrozen_mask(d, mask, seed), seed=seed); regs.append(C/C0-1)
    print('%-34s regret mean %.4f%% max %.4f%%' % (name % (), 100*np.mean(regs), 100*np.max(regs)))
