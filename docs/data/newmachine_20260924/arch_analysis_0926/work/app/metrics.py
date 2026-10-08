import json, glob, numpy as np
KR = 0.93/0.58   # ratio range of kappa = ||s|| tau/(2W) measured on a real cell (sensratio.json)
print('%-22s %-3s | %7s %7s | %7s %7s | %7s %7s | %6s' % ('cell', 'cfg', 'curMax', 'cmpMax', 'thr5%', 'wRMS', 'Glat', 'GlatUB', 'pass?'))
for f in sorted(glob.glob('gate_cont_v2L1_fresh_val_*.json')):
    d = json.load(open(f))
    for r in d['results']:
        t = r['test']; g = np.array(r['gate']); w = np.array(r['energy_share'][0]); wn = np.array(r['energy_share'][1])
        dt = np.array(t['sens_vec_rel_err'][0]); dn = np.array(t['sens_vec_rel_err'][1]); ce = np.array(t['compliance_rel_err'])
        cur = dt[g].max(); cmax = ce[g].max()
        thr = dt[g & (w >= 0.05)].max()
        wrms = np.sqrt((w[g]*dt[g]**2).sum()/w[g].sum())
        # lattice-gradient (assembled over shared corners) relative error, ||s_c|| ~ W_c; all s <= 0 => ||G|| >= max ||s_c||
        Wt, Wn = w, wn
        glat = (dt*Wt + dn*Wn)/np.maximum(Wt, Wn)                 # central estimate (kappa equal)
        glub = (dt*np.minimum(1, KR*Wt/np.maximum(Wt, Wn)) + dn*np.minimum(1, KR*Wn/np.maximum(Wt, Wn)))
        name = f.split('fresh_val_')[1].replace('.json', '')
        print('%-22s %-3s | %6.2f%% %6.2f%% | %6.2f%% %6.2f%% | %6.2f%% %6.2f%% | %s/%s' % (name, r['config'], 100*cur, 100*cmax, 100*thr, 100*wrms,
              100*glat[g].max(), 100*glub[g].max(), 'P' if cur <= .03 and cmax <= .03 else 'F', 'P' if max(thr, cmax) <= .03 else 'F'))
        if (~g).any():
            print('%28s cut loads (not gated): sens %s  share %s  comp %s' % ('', np.round(100*dt[~g], 1), np.round(w[~g], 2), np.round(100*ce[~g], 1)))
