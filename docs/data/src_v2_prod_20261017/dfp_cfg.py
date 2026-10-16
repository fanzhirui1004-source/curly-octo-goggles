"""Configs of the data-free pilot (2026-09-28): three arms from v2L1.json, random init, 10k steps, label-free selection.
Usage: dfp_cfg.py [--smoke]   (writes $O/DFP_DF.json, DFP_CTRL.json, DFP_DFW.json; --smoke: DFP_smoke_DFW.json)"""
import json, sys
O = '/root/autodl-tmp/OPL/S1/V2'
base = json.load(open(f'{O}/v2L1.json'))
base.pop('init', None)                                               # random initialisation
ma = {k: v for k, v in base['model_args'].items() if k not in ('bounded', 'bounds', 'bound_knee')}   # bounds_B1 came from a trained ckpt
base.update(model_args=ma, split='/root/autodl-tmp/OPL/S2/SPLIT_ARMS.json', val_max=40, eval_views=[0, 17], seed=0,
            steps=10000, eval_every=2500, select_min_step=2500, select_by='logE', select_classes=['macro', 'grf'],
            probe_max=0, cert_every=4, resume=True)
DF = dict(mix={'macro': 0.1, 'grf': 0.125}, sens_w=0.0, adv_on_load=False)
W = dict(smooth_k=8, smooth_alpha=30.0, coarse_space='Q1_17')
arms = {'DFP_DF': DF, 'DFP_CTRL': {}, 'DFP_DFW': dict(DF, model_args=dict(ma, **W))}
if '--smoke' in sys.argv:
    arms = {'DFP_smoke_DFW': dict(arms['DFP_DFW'], val_max=4, eval_every=200, stop_after=300, log_every=25, resume=False)}
for name, ov in arms.items():
    c = dict(base, out=f'{O}/{name}', **ov)
    json.dump(c, open(f'{O}/{name}.json', 'w'), indent=1)
    print(name, json.dumps({k: c[k] for k in ('mix', 'sens_w', 'adv_on_load', 'model_args', 'steps', 'select_by')}))
