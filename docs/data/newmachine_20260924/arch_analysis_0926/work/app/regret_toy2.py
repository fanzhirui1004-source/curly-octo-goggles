exec(open('regret_toy.py').read().split("rows = []")[0])
def mkfrozen(d, where=None, seed=0):
    rr = np.random.default_rng(100+seed); XI = rr.standard_normal((nel, 4)); XI /= np.linalg.norm(XI, axis=1, keepdims=True)
    def f(s, w, rng):
        dl = np.full(nel, d)
        if where == 'boundary': dl = np.where((ely == 0) | (ely == ny-1) | (elx == nx-1), 5*d, d)
        return s + dl[:, None]*np.linalg.norm(s, axis=1, keepdims=True)*XI
    return f
for name, dd, wh in [('frozen 3%', .03, None), ('frozen 6%', .06, None), ('frozen 15%', .15, None), ('frozen boundary 5x (15%)', .03, 'boundary')]:
    regs = []
    for seed in range(3):
        t, C, w = optimise(mkfrozen(dd, wh, seed), seed=seed); regs.append(C/C0-1)
    print('%-28s regret mean %.4f%% max %.4f%% (design dist %.3f)' % (name, 100*np.mean(regs), 100*np.max(regs), np.linalg.norm(t-t0)/np.linalg.norm(t0)))
# fraction of nodes at bounds in the exact optimum
print('nodes at lower bound %.2f, upper %.2f' % ((t0 <= tau_lo+1e-6).mean(), (t0 >= tau_hi-1e-6).mean()))
