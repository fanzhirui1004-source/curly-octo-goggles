"""Cross-geometry (amortized) learning toy, 2D plane-stress Q1 CutFEM-like cells (numan/toy.Cell).
A small MGNO-like network, linear in q, geometry-conditioned: fine element-hyperedge layers with coefficients from an
element MLP, a 2-level gated conv U-Net, output = interior displacement.  Loss = per-direction normalised energy excess.
Questions: (1) train-fit vs val gap and data scaling; (2) spectral location of the error (smoothing headroom);
(3) smoother-in-the-loop training (network learns what k Chebyshev sweeps cannot fix) vs post-hoc smoothing."""
import sys, os, json, time, pickle, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'numan'))
import toy
torch.set_num_threads(1)
dt = torch.float64
n = 16; NN = (n + 1) ** 2
CACHE = os.path.join(os.path.dirname(__file__), 'geos_n16.pkl')

def sample_geo(rng):
    tau = rng.uniform(0.22, 0.6, 4)
    u = rng.uniform()
    if u < 0.25: xc, typ = None, 'full'
    elif u < 0.5: xc, typ = rng.uniform(0.62, 0.9), 'light'
    elif u < 0.75: xc, typ = rng.uniform(0.42, 0.62), 'medium'
    else: xc, typ = rng.uniform(0.25, 0.42), 'heavy'
    return tau, xc, typ

def build(tau, xc, typ):
    c = toy.Cell(n, list(tau), xcut=xc)
    if len(c.I) < 8 or len(c.P) < 8: return None
    Xp = c.X[c.port_nodes]; R = toy.rigid(Xp); Qr, _ = np.linalg.qr(R)
    Pn = np.eye(len(c.P)) - Qr @ Qr.T; w, V = np.linalg.eigh(Pn); B = V[:, w > 0.5]
    lam, U = np.linalg.eigh(B.T @ c.S @ B); SU = B @ U
    g = dict(typ=typ, tau=tau, xc=xc, used=c.used, int_nodes=c.int_nodes, port_nodes=c.port_nodes, I=c.I, P=c.P,
             KII=c.KII, KIP=c.KIP, EI=c.EI, S=c.S, D=c.D, lmax=c.lmax, chi=c.chi.reshape(n, n), act=c.act.reshape(n, n),
             Qr=Qr, Sp=SU @ np.diag(1 / lam) @ SU.T, Xp=Xp, Slam=lam)
    # interior generalized eigenpairs K_II v = mu D v (for the spectral split of the error)
    Dm = 1 / np.sqrt(c.D); mu, W = np.linalg.eigh(c.KII * Dm[:, None] * Dm[None, :]); g['mu'] = mu; g['Wd'] = Dm[:, None] * W
    return g

def geos(N, seed):
    rng = np.random.default_rng(seed); out = []
    while len(out) < N:
        g = build(*sample_geo(rng))
        if g is not None: out.append(g)
    return out

def probes(g, m, kind, rng):
    Xp = g['Xp']; nP = len(g['P']); F = np.zeros((nP, m))
    for i in range(m):
        kx, ky = rng.normal(size=(2, 4)); ph = rng.uniform(0, 2 * np.pi, (2, 4)); f = np.zeros(nP)
        for d in range(2):
            f[d::2] = sum(np.cos((j + 1) * np.pi * Xp[:, 0] + ph[d, j]) * kx[j] + np.cos((j + 1) * np.pi * Xp[:, 1] + ph[d, j]) * ky[j] for j in range(4))
        if kind == 'grf': f += 0.3 * rng.normal(size=nP)          # rougher Dirichlet data
        F[:, i] = f
    F -= g['Qr'] @ (g['Qr'].T @ F)
    q = g['Sp'] @ F if kind == 'force' else F
    q -= g['Qr'] @ (g['Qr'].T @ q)
    return q / np.linalg.norm(q, axis=0)

class T:  # torch views of one geometry
    def __init__(s, g):
        s.g = g
        s.KII = torch.tensor(g['KII'], dtype=dt); s.KIP = torch.tensor(g['KIP'], dtype=dt); s.EI = torch.tensor(g['EI'], dtype=dt)
        s.S = torch.tensor(g['S'], dtype=dt); s.D = torch.tensor(g['D'], dtype=dt)
        s.int_grid = torch.tensor(g['used'][g['int_nodes']]); s.port_grid = torch.tensor(g['used'][g['port_nodes']])
        act = torch.tensor(g['act']); s.act = act
        chi = torch.tensor(g['chi'], dtype=dt) * act
        # element features: 3x3 patch of chi and log chi
        pad = torch.nn.functional.pad(chi[None, None], (1, 1, 1, 1))[0, 0]
        patch = torch.stack([pad[i:i + n, j:j + n] for i in range(3) for j in range(3)], -1)
        s.efeat = torch.cat([patch, torch.log(patch + 1e-3) / 7], -1)[act]      # E x 18
        ei, ej = torch.nonzero(act, as_tuple=True)
        nid = lambda i, j: i * (n + 1) + j
        s.en = torch.stack([nid(ei, ej), nid(ei + 1, ej), nid(ei + 1, ej + 1), nid(ei, ej + 1)], 1)  # E x 4 grid nodes
        s.deg = torch.zeros(NN, dtype=dt).index_add_(0, s.en.reshape(-1), torch.ones(s.en.numel(), dtype=dt)).clamp_min(1)
        s.pm = torch.zeros(NN, dtype=dt); s.pm[s.port_grid] = 1
        s.am = torch.zeros(NN, dtype=dt); s.am[s.en.reshape(-1)] = 1
        nch = torch.zeros(NN, dtype=dt).index_add_(0, s.en.reshape(-1), chi[act].repeat_interleave(4)) / s.deg
        s.nfeat = torch.stack([nch, torch.log(nch + 1e-3) / 7, s.pm, s.am], -1)   # NN x 4

def cheb(t, uI, q, k, alpha=30.0):
    if k == 0: return uI
    b_ = t.g['lmax']; a_ = b_ / alpha; th = (b_ + a_) / 2; de = (b_ - a_) / 2; sig = th / de
    f = -t.KIP @ q; r = f - t.KII @ uI; z = r / t.D[:, None]; rho = 1 / sig; d = z / th; x = uI
    for i in range(k):
        x = x + d; r = r - t.KII @ d; z = r / t.D[:, None]; rn = 1 / (2 * sig - rho); d = rn * rho * d + 2 * rn / de * z; rho = rn
    return x

def mlp(i, h, o): return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))

class Net(nn.Module):
    def __init__(s, F=16, H=4, Lpre=3, Lpost=3, Cg=32):
        super().__init__()
        s.F, s.H, s.L = F, H, Lpre + Lpost; s.Lpre = Lpre
        s.ehead = mlp(18, Cg, 2 * 4 * H * s.L)
        s.W = nn.Parameter(torch.randn(s.L, H, F, F, dtype=dt) / np.sqrt(F) * 0.5)
        s.Win = nn.Parameter(torch.randn(2, F, dtype=dt) / np.sqrt(2)); s.Wout = nn.Parameter(torch.randn(F, 2, dtype=dt) / np.sqrt(F) * 0.1)
        s.nhead = mlp(4, Cg, 2 * 2 + 2 * 2 * F)   # restrict/prolong weight per level (2 levels) + gates (2 levels x 2 convs x F)
        s.convs = nn.ParameterList([nn.Parameter(torch.randn(4, F, F, 3, 3, dtype=dt) / np.sqrt(9 * F) * 0.5) for _ in range(2)])
        s.to(dt)
    def forward(s, t, q):
        B = q.shape[1]; E = t.en.shape[0]
        ab = s.ehead(t.efeat).reshape(E, 4, 2, s.L, s.H) * 0.2
        nh = s.nhead(t.nfeat)
        X0 = torch.zeros(NN, B, s.F, dtype=dt)
        X0[t.port_grid] = q.reshape(-1, 2, B).permute(0, 2, 1) @ s.Win
        pm = t.pm[:, None, None]; am = t.am[:, None, None]
        def fine(X, l):
            a, b = ab[:, :, 0, l], ab[:, :, 1, l]
            Z = torch.einsum('eah,eabf->ehbf', a, X[t.en]); Z = torch.einsum('ehbf,hfg->ehbg', Z, s.W[l])
            Y = torch.einsum('eah,ehbg->eabg', b, Z)
            dX = torch.zeros_like(X).index_add_(0, t.en.reshape(-1), Y.reshape(-1, B, s.F))
            X = X + dX / t.deg[:, None, None]
            return (X * (1 - pm) + X0 * pm) * am
        X = X0
        for l in range(s.Lpre): X = fine(X, l)
        # U-Net: node grid (n+1)^2 -> (n/2+1)^2 -> (n/4+1)^2 by weighted injection-averaging (full weighting), gated convs
        grids = [X]; Xl = X; ws = []
        for lev in range(2):
            m = n // (2 ** lev) + 1
            w = torch.nn.functional.softplus(nh[:, lev]) + 1e-3 if lev == 0 else None
            G = Xl.permute(1, 2, 0).reshape(B, s.F, m, m)
            if lev == 0:
                wg = (w * t.am).reshape(1, 1, m, m)
                num = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(G * wg, (1, 1, 1, 1)), 3, 2) ; den = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(wg, (1, 1, 1, 1)), 3, 2) + 1e-9
                s._wc = den
            else:
                wg = (s._wc > 1e-6).to(dt)
                num = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(G * wg, (1, 1, 1, 1)), 3, 2); den = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(wg, (1, 1, 1, 1)), 3, 2) + 1e-9
            Gc = num / den
            mc = Gc.shape[-1]
            gate_src = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(nh[:, 4:].T.reshape(1, -1, n + 1, n + 1), (1, 1, 1, 1)), 3, 2)
            for _ in range(lev): gate_src = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(gate_src, (1, 1, 1, 1)), 3, 2)
            gates = 2 * torch.sigmoid(gate_src.reshape(2, 2, s.F, mc, mc)[lev])      # 2 convs x F x mc x mc
            for j in range(2):
                Gc = Gc + torch.nn.functional.conv2d(Gc, s.convs[lev][j], padding=1) * gates[j][None]
            ws.append((G.shape, Gc)); Xl = Gc.reshape(B, s.F, -1).permute(2, 0, 1)
        up = None
        for lev in reversed(range(2)):
            shp, Gc = ws[lev]
            if up is not None: Gc = Gc + up
            for j in range(2, 4):
                Gc = Gc + torch.nn.functional.conv2d(Gc, s.convs[lev][j], padding=1)
            up = torch.nn.functional.interpolate(Gc, size=shp[-2:], mode='bilinear', align_corners=True)
            if lev == 0:
                w1 = (torch.nn.functional.softplus(nh[:, 2]) + 1e-3).reshape(1, 1, n + 1, n + 1)
                up = up * w1
        X = X + up.reshape(B, s.F, -1).permute(2, 0, 1)
        X = (X * (1 - pm) + X0 * pm) * am
        for l in range(s.Lpre, s.L): X = fine(X, l)
        u = X @ s.Wout                                   # NN x B x 2
        return u[t.int_grid].permute(0, 2, 1).reshape(-1, B)   # interior dofs (node-major, 2 per node) = order of c.I

def excess(t, uI, q):
    e = uI - t.EI @ q
    return torch.einsum('ij,ij->j', e, t.KII @ e) / torch.einsum('ij,ij->j', q, t.S @ q)

def evaluate(net, TT, bank, ks=(0, 1, 4, 8), k_net=0, spec=False):
    res = {}
    with torch.no_grad():
        for i, t in enumerate(TT):
            for kind in ('force', 'grf'):
                q = bank[i][kind]; u = cheb(t, net(t, q), q, k_net)
                for k in ks:
                    res.setdefault((t.g['typ'], kind, k), []).append(float(excess(t, cheb(t, u, q, k), q).mean()))
                if spec:
                    e = (u - t.EI @ q).numpy(); a = t.g['Wd'].T @ (t.g['D'][:, None] * e)   # D-orthonormal coords
                    en = (t.g['mu'][:, None] * a ** 2).sum(1); tot = en.sum(); mu = t.g['mu']
                    for band, lo, hi in (('low', 0, t.g['lmax'] / 100), ('mid', t.g['lmax'] / 100, t.g['lmax'] / 30), ('high', t.g['lmax'] / 30, 1e9)):
                        res.setdefault((t.g['typ'], kind, 'share_' + band), []).append(float(en[(mu >= lo) & (mu < hi)].sum() / tot))
    return {f'{a}|{b}|{c}': float(np.mean(v)) for (a, b, c), v in res.items()}

def run(Ntr, steps, k_train=0, seed=0, lr=2e-3, tag=''):
    allg = pickle.load(open(CACHE, 'rb'))
    tr = [T(g) for g in allg['train'][:Ntr]]; va = [T(g) for g in allg['val']]
    rng = np.random.default_rng(seed + 7)
    bank_tr = [{k: torch.tensor(probes(t.g, 16, k, np.random.default_rng(1000 + i)), dtype=dt) for k in ('force', 'grf')} for i, t in enumerate(tr[:48])]
    bank_va = [{k: torch.tensor(probes(t.g, 16, k, np.random.default_rng(5000 + i)), dtype=dt) for k in ('force', 'grf')} for i, t in enumerate(va)]
    torch.manual_seed(seed); net = Net(H=int(os.environ.get("TOYH", 4)))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    t0 = time.time(); hist = []
    for s_ in range(steps):
        L = 0
        for gi in rng.integers(0, len(tr), 4):
            t = tr[gi]
            q = torch.tensor(np.concatenate([probes(t.g, 4, 'force', rng), probes(t.g, 4, 'grf', rng)], 1), dtype=dt)
            L = L + excess(t, cheb(t, net(t, q), q, k_train), q).mean() / 4
        opt.zero_grad(); L.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
        if s_ % 500 == 0: hist.append((s_, float(L)))
    out = dict(tag=tag, Ntr=Ntr, steps=steps, k_train=k_train, seed=seed, secs=time.time() - t0, hist=hist,
               val=evaluate(net, va, bank_va, k_net=k_train, spec=(k_train == 0)),
               train=evaluate(net, tr[:48], bank_tr, k_net=k_train, spec=False))
    return out

if __name__ == '__main__':
    if sys.argv[1] == 'prep':
        t0 = time.time(); d = dict(train=geos(int(sys.argv[2]), 1), val=geos(48, 2)); pickle.dump(d, open(CACHE, 'wb'))
        print('prep', time.time() - t0, [sum(g['typ'] == k for g in d['train']) for k in ('full', 'light', 'medium', 'heavy')])
        print('val nI', np.mean([len(g['I']) for g in d['val']]), 'kappa', np.mean([g['mu'][-1] / g['mu'][0] for g in d['val']]))
    else:
        Ntr, steps, k = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        o = run(Ntr, steps, k, seed, tag=sys.argv[1])
        json.dump(o, open(f'res_{sys.argv[1]}.json', 'w'), indent=1)
        print(json.dumps({k_: round(v, 5) for k_, v in o['val'].items() if '|0' in k_ or 'share' in k_ or '|4' in k_}), o['secs'])
