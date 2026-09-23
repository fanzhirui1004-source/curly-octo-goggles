import json, sys
import numpy as np
r = json.load(open(sys.argv[1]))
d = r['directions']
rc = np.array([d[k]['compliance_ratio'] for k in d if k.startswith('random_')])
rs = np.array([d[k]['stiffness_ratio'] for k in d if k.startswith('random_')])
print('random r_c min %.4g median %.4g max %.4g' % (rc.min(), np.median(rc), rc.max()))
print('random r_s min %.4g median %.4g max %.4g' % (rs.min(), np.median(rs), rs.max()))
print('random r_c*r_s median %.4g' % np.median(rc * rs))
for k, v in d.items():
    if not k.startswith('random_'):
        print('%-34s r_c %.4g r_s %.4g product %.4g  r_c pct among random %.2f  r_s pct %.2f' % (
            k, v['compliance_ratio'], v['stiffness_ratio'], v['compliance_ratio'] * v['stiffness_ratio'],
            float(np.mean(rc < v['compliance_ratio'])), float(np.mean(rs < v['stiffness_ratio']))))
