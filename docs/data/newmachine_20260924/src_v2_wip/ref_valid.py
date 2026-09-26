"""Reference-discretisation checks for P1 (host, MKL PARDISO): one cell clamped on its z = 0 box face, a unit uniform
traction (consistent nodal weights over material, lattice3.face_traction_weights, normalised to unit total force) on its
z = 1 face in x, y and z. Responses: compliance C = f^T u and the 8 corner sensitivities s_c = -u^T K_,c u (teacher.sens).
  --ns       background resolution sweep (n = 32 is the stored body; other n use <case>_n<n> packets / bodies built by
             fast_prep4 from the same context with n replaced): relative change against the finest n, observed order
  --gammas   ghost-penalty coefficient sweep at n = 32 (the stored ghost matrix rescaled), plus the ghost share of the
             energy (1 - bulk element energy / total) at every n
  --hs       step of the central-difference moment derivative dM/dtau_c at n = 32; and one direct check
             (C(tau_c + h) - C(tau_c - h)) / 2h (re-assembled, re-solved, h = --h-direct x tau_c) against s_c
  --integ    integration variants s:levels at n = 32 (default 4:1 = the teacher)
Usage: ref_valid.py <out.json> <case>[,...] [--ns 24,32,40,48] [--gammas 1e-5,1e-4,1e-3] [--hs 1e-3,1e-4,1e-5,1e-6]
       [--h-direct 1e-4] [--integ 4:2,6:1] [--body32 /root/autodl-tmp/OPL/S0] [--bodyN /root/autodl-tmp/OPL/S4/body]
Needs OPL_DEV=cpu (diag_sens environment) and OPL_PACKETS_EXTRA including the <case>_n<n> packet root."""
import diag_sens as DS                                                   # first: CPU environment
import sys, json, time, argparse, gc
from pathlib import Path
import numpy as np
import torch
import trainlib as TL
import teacher as TE
import lattice3 as LT
import box_encode as BX

dt = TL.dt
BX.dev = TL.dev


def free_dofs(C):
    M = 2 * C.n + 1
    gz = np.asarray(C.nodes) % M
    fixed = np.repeat(gz == 0, 3)
    return np.flatnonzero(~fixed)


def loads(C):
    w = LT.face_traction_weights(C, 2, 1.0)
    w = w / w.sum()
    F = np.zeros((C.nb, 3))
    for d in range(3):
        F[d::3, d] = w
    return F


def solve(C, F, fr):
    import scipy.sparse as sp, pypardiso
    ru, cu, v = C.ru.long().numpy(), C.cu.long().numpy(), C.vals.numpy()
    U = sp.csr_matrix((v, (ru, cu)), shape=(C.nb, C.nb))
    K = (U + sp.triu(U, 1).T).tocsr()[fr][:, fr]
    ps = pypardiso.PyPardisoSolver()
    uf = ps.solve(K.tocsr(), np.ascontiguousarray(F[fr]))
    ps.free_memory(everything=True)
    u = np.zeros_like(F); u[fr] = uf
    return torch.as_tensor(u, dtype=dt)


def bulk_share(C, u):
    iu = torch.triu_indices(81, 81)
    w = torch.where(iu[0] == iu[1], 1.0, 2.0).to(dt)
    tot = (u * (C.K @ u)).sum(0)
    bulk = torch.zeros(u.shape[1], dtype=dt)
    for lo in range(0, len(C.M), 2048):
        ke = C.M[lo:lo + 2048] @ C.Tm_up
        ue = u[C.dofs[lo:lo + 2048]]
        bulk += torch.einsum('ep,epk->k', ke * w, ue[:, iu[0]] * ue[:, iu[1]])
    return (1 - bulk / tot).numpy()


def respond(C, F, fr, sens=True, rel=1e-5):
    u = solve(C, F, fr)
    comp = (torch.as_tensor(F) * u).sum(0).numpy()
    out = dict(compliance=comp.tolist(), ghost_share=bulk_share(C, u).tolist(), dofs=int(C.nb), free=int(len(fr)),
               elements=int(len(C.cells)))
    if sens:
        C.dmoments(rel)
        out['sens'] = C.sens(u).numpy().tolist()
    return out, u


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('cases')
    ap.add_argument('--ns', default='24,32,40,48'); ap.add_argument('--gammas', default='1e-5,1e-4,1e-3')
    ap.add_argument('--hs', default='1e-3,1e-4,1e-5,1e-6'); ap.add_argument('--h-direct', type=float, default=1e-4)
    ap.add_argument('--integ', default='4:2,6:1')
    ap.add_argument('--body32', default='/root/autodl-tmp/OPL/S0'); ap.add_argument('--bodyN', default='/root/autodl-tmp/OPL/S4/body')
    a = ap.parse_args(argv)
    rec = dict(setup='clamp z=0 box face; unit consistent traction on z=1 (x, y, z); host PARDISO', per_case={})
    if Path(a.out).exists():
        rec = json.loads(Path(a.out).read_text())
    for case in a.cases.split(','):
        r = rec['per_case'].get(case, {})
        t0 = time.perf_counter()
        # resolution sweep
        res = r.get('n', {})
        for n in [int(x) for x in a.ns.split(',')]:
            if str(n) in res:
                continue
            cid, body = (case, a.body32) if n == 32 else (f'{case}_n{n}', a.bodyN)
            if not (Path(body) / cid / 'NODES.npy').exists():
                res[str(n)] = dict(missing=True); continue
            try:
                C = TE.Cell(cid, body, log=lambda s_: None); C.assemble()
                fr = free_dofs(C); F = loads(C)
                res[str(n)], _ = respond(C, F, fr)
                print(json.dumps(dict(case=case, n=n, compliance=res[str(n)]['compliance'], dofs=C.nb,
                                      ghost=np.round(res[str(n)]['ghost_share'], 6).tolist())), flush=True)
            except Exception as e:                                                  # noqa: BLE001  (memory, missing body)
                res[str(n)] = dict(error=repr(e)[:300])
                print(json.dumps(dict(case=case, n=n, error=repr(e)[:200])), flush=True)
            C = None; gc.collect()
            r['n'] = res; rec['per_case'][case] = r; Path(a.out).write_text(json.dumps(rec, indent=1))
        ok = sorted(int(k) for k, v in res.items() if 'compliance' in v)
        if len(ok) >= 2:
            fin = res[str(ok[-1])]
            r['vs_finest'] = {str(n): dict(compliance_rel=[abs(c / f - 1) for c, f in zip(res[str(n)]['compliance'], fin['compliance'])],
                                          sens_rel=[rel(np.asarray(res[str(n)]['sens'])[:, j], np.asarray(fin['sens'])[:, j]) for j in range(3)])
                              for n in ok}
        if len(ok) >= 3:
            n1, n2, n3 = ok[-3:]
            c1, c2, c3 = (np.asarray(res[str(n)]['compliance']) for n in (n1, n2, n3))
            with np.errstate(all='ignore'):
                r['observed_order'] = (np.log(np.abs(c1 - c2) / np.abs(c2 - c3)) / np.log(n2 / n1)).tolist() if n2 / n1 == n3 / n2 else \
                    'unequal ratios: see vs_finest'
        # n = 32 studies
        if 'gamma' not in r or 'fd' not in r or 'integ' not in r:
            C = TE.Cell(case, a.body32, log=lambda s_: None); C.assemble()
            fr = free_dofs(C); F = loads(C)
            base, u = respond(C, F, fr)
            if 'fd' not in r:
                fd = {}
                for h in [float(x) for x in a.hs.split(',')]:
                    C.dmoments(h); fd[str(h)] = C.sens(u).numpy().tolist()
                ref_ = np.asarray(fd[str(1e-5)] if str(1e-5) in fd else base['sens'])
                direct = np.zeros((8, 3))
                taus = list(C.taus0)
                for c in range(8):
                    cc = []
                    for sg in (1, -1):
                        tp = list(taus); tp[c] += sg * a.h_direct * taus[c]
                        C.assemble(tp)
                        cc.append((torch.as_tensor(F) * solve(C, F, fr)).sum(0).numpy())
                    direct[c] = (cc[0] - cc[1]) / (2 * a.h_direct * taus[c])
                C.assemble(taus)
                r['fd'] = dict(sens_by_h=fd, rel_to_h1e_5={h: [rel(np.asarray(v)[:, j], ref_[:, j]) for j in range(3)] for h, v in fd.items()},
                               direct=direct.tolist(), direct_vs_sens=[rel(direct[:, j], ref_[:, j]) for j in range(3)])
                print(json.dumps(dict(case=case, fd=r['fd']['rel_to_h1e_5'], direct=r['fd']['direct_vs_sens'])), flush=True)
            if 'gamma' not in r:
                g0 = C.gamma; gm = {}
                for g in [float(x) for x in a.gammas.split(',')]:
                    C.base *= g / C.gamma; C.gamma = g; C.assemble()
                    o, _ = respond(C, F, fr)
                    gm[str(g)] = dict(o, compliance_rel=[abs(x / y - 1) for x, y in zip(o['compliance'], base['compliance'])],
                                      sens_rel=[rel(np.asarray(o['sens'])[:, j], np.asarray(base['sens'])[:, j]) for j in range(3)])
                C.base *= g0 / C.gamma; C.gamma = g0
                r['gamma'] = gm
                print(json.dumps(dict(case=case, gamma={k: (np.round(v['compliance_rel'], 6).tolist(), np.round(v['ghost_share'], 6).tolist())
                                                        for k, v in gm.items()})), flush=True)
            C = None; gc.collect()
            if 'integ' not in r:
                ig = {}
                for spec in [x for x in a.integ.split(',') if x]:
                    s_, lv = (int(x) for x in spec.split(':'))
                    Ci = TE.Cell(case, a.body32, s=s_, levels=lv, log=lambda s__: None); Ci.assemble()
                    o, _ = respond(Ci, loads(Ci), free_dofs(Ci))
                    ig[spec] = dict(o, compliance_rel=[abs(x / y - 1) for x, y in zip(o['compliance'], base['compliance'])],
                                    sens_rel=[rel(np.asarray(o['sens'])[:, j], np.asarray(base['sens'])[:, j]) for j in range(3)])
                    Ci = None; gc.collect()
                r['integ'] = ig
                print(json.dumps(dict(case=case, integ={k: np.round(v['compliance_rel'], 7).tolist() for k, v in ig.items()})), flush=True)
            r['base32'] = base
        r['seconds'] = r.get('seconds', 0) + time.perf_counter() - t0
        rec['per_case'][case] = r
        Path(a.out).write_text(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
