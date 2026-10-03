"""Table 5 learned route re-timed on the final route (R1/FINAL3, coarse pivot rule) against the original timed records
(S1/V2: learned_hlat221a/b.json, d5_off.json, learned_hlat331.json).  Usage: compare_old_new.py DIR with old_<L>.json and new_<L>.json."""
import json, sys
S = sys.argv[1]
for L in ['hlat221a', 'hlat221b', 'hlat222', 'hlat331']:
    out = []
    for tag in ['old', 'new']:
        try:
            d = json.load(open(f'{S}/{tag}_{L}.json'))
        except Exception as e:
            out.append(f'{tag}: missing'); continue
        it = d['iterations'][0]; p = it['phases']
        out.append(f"{tag}: fe {p['front_end']:.2f} lat+kpp {p['lattice_setup']+p['kpp_assemble']:.2f} prec {p['prec_setup']:.2f} solve {p['solve']:.2f} "
                   f"pcg {it['solve']['iterations']} sens {p['sens_total']:.2f} total {p['iteration_total']:.2f} peak {it['peak_device_GB']*1e9/2**30:.2f} "
                   f"rss {it['host_rss_GB_after_front_end']*1e9/2**30:.2f} stream {it['stream_GB_total']*1e9/2**30:.2f} C {[round(c, 4) for c in it['compliance']]}")
    print(L); [print('  ', o) for o in out]
