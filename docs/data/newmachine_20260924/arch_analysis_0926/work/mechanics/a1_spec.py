import cell, numpy as np, scipy.linalg as sl, time, json, sys
case = sys.argv[1] if len(sys.argv) > 1 else 'fresh_train_0010_cover01_r1'
out = {}
for gamma in [1e-4, 1e-3, 1e-5, 1e-2]:
    C = cell.RealCell(case, gamma=gamma)
    K = C.K.toarray() if C.nb < 20000 else None
    P, I = C.P, C.I
    KII = K[np.ix_(I, I)]; KIP = K[np.ix_(I, P)]; KPP = K[np.ix_(P, P)]
    d = np.diag(KII)
    ev = sl.eigh(KII, eigvals_only=True)
    evj = sl.eigh(KII, np.diag(d), eigvals_only=True)
    X = np.linalg.solve(KII, KIP)
    S = KPP - KIP.T @ X; S = (S + S.T) / 2
    es = np.linalg.eigvalsh(S)
    r = dict(KII_min=ev[0], KII_max=ev[-1], KII_cond=ev[-1] / ev[0], Jac_min=evj[0], Jac_max=evj[-1],
             Jac_q=np.quantile(evj, [.001, .01, .05, .1, .25, .5]).tolist(), frac_Jac_below_lmax30=float((evj < evj[-1] / 30).mean()),
             S_eigs_low=es[:14].tolist(), S_max=es[-1], diagD_q=np.quantile(d, [0, .01, .1, .5, .9, 1]).tolist())
    out[gamma] = r
    print(gamma, json.dumps({k: (np.round(v, 8).tolist() if isinstance(v, (list, np.ndarray)) else float('%.4g' % v)) for k, v in r.items()}), flush=True)
    if gamma == 1e-4:
        np.save('S_%s.npy' % case, S); np.save('evj_%s.npy' % case, evj)
json.dump(out, open('a1_%s.json' % case, 'w'), indent=1, default=float)
