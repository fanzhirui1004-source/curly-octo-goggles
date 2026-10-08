"""New machine reproduction: fp64 Schur box operators of 0013 cut + 0013 FULL, 2-cell lattice (config x), compared
with the old machine's stored T64 compliance and cell energies (R7_16 SOFT_CORRECTION.json)."""
import json, sys, time
sys.path.insert(0, '/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/xcase_src_36')
import numpy as np
import precision_study as PS
body = sys.argv[1]
ref_c = [304.6858523427121, 413.0620238567342, 394.9271084712375, 93.78672640951045, 104.18518284665292, 102.9245201672372]
ref_e = [[218.81134641789538, 205.77213488554582, 172.23906380406805, 0.07327675447627202, 0.08465777731494886, 0.02059817800703053],
         [85.87450592481483, 207.28988897120112, 222.68804466716998, 93.7134496550338, 104.10052506933832, 102.90392198922875]]
t0 = time.perf_counter()
Tc, idc, n, stc = PS.schur_T('fresh_train_0013_cover01_r1', body, 'fp64', host=False)
Tf, idf, _, stf = PS.schur_T('fresh_train_0013_full', body, 'fp64', host=True)
print(json.dumps(dict(cut_box=int(Tc.shape[0]), full_box=int(Tf.shape[0]), schur_cut=stc, schur_full=stf)), flush=True)
res = PS.lattice_check([('cut', (0, 0, 0), idc, {'T64': Tc}), ('full', (-1, 0, 0), idf, {'T64': Tf})], log=print)
c = np.asarray(res['T64']['compliance']); e = np.asarray(res['T64']['energy'])
out = dict(compliance=c.tolist(), rel_diff_compliance=float(np.abs(c / ref_c - 1).max()),
           rel_diff_energy=float(np.abs(e / np.asarray(ref_e) - 1).max()), seconds=time.perf_counter() - t0)
print(json.dumps(out), flush=True)
