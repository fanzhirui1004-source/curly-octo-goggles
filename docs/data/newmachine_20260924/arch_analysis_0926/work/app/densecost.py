import sys, time, numpy as np, scipy.linalg as sl
sys.path.insert(0, '/tmp/claude-0/-home-user-curly-octo-goggles/76f53c89-e2a9-52bb-8817-9460fd780b76/scratchpad/pack/work/mechanics')
import cell
C = cell.RealCell('fresh_train_0010_cover01_r1'); K = C.K.tocsr(); P, I = C.P, C.I
KII = K[I][:, I].toarray(); KIP = K[I][:, P]; KPP = K[P][:, P]
t0 = time.perf_counter(); cf = sl.cho_factor(KII, lower=True); t_f = time.perf_counter() - t0
q = np.random.default_rng(0).standard_normal((len(P), 16))
t0 = time.perf_counter()
for _ in range(10): y = KPP @ q[:, :1] - KIP.T @ sl.cho_solve(cf, KIP @ q[:, :1])
t_q1 = (time.perf_counter() - t0) / 10
t0 = time.perf_counter()
for _ in range(10): y = KPP @ q - KIP.T @ sl.cho_solve(cf, KIP @ q)
t_q16 = (time.perf_counter() - t0) / 10
t0 = time.perf_counter(); X = sl.cho_solve(cf, KIP.toarray()); t_X = time.perf_counter() - t0
print('interior', len(I), 'dense chol %.3f s | S q B1 %.4f s  B16 %.4f s | E = K_II^-1 K_IP (dense, %d cols) %.2f s' % (t_f, t_q1, t_q16, len(P), t_X))
# how many interior dofs are actually coupled to ports / K_IP rank structure: nonzero columns of K_IP
nzc = np.unique(KIP.nonzero()[1]); print('ports coupled to interior:', len(nzc), 'of', len(P))
