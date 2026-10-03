"""Bank-level evaluation of a model (checkpoint or zero-parameter): validation/test energy errors per class and the
worst direction (largest eigenvalue of S^-1 S_hat by block power iteration).
Usage: evalbank.py <out.json> <model_name> <model_args_json|-> <checkpoint|-> <body> <data> <case> [<case> ...]"""
import json, sys, time
import torch
import trainlib as TL
import models as MD

out, name, margs, ck, body, data = sys.argv[1:7]
cases = sys.argv[7:]
margs = {} if margs == '-' else json.loads(margs)
rec = {}
for case in cases:
    t0 = time.perf_counter()
    geo = TL.Geo(case, body, data, log=lambda s_: None)
    model = MD.build(name, [geo], **margs).cuda()
    if ck != '-':
        model.load_state_dict(torch.load(ck, map_location='cuda:0', weights_only=False)['model'], strict=False)
    model.eval()
    r = dict(val=geo.evaluate(model, 'val'), test=geo.evaluate(model, 'test'))
    X, ritz = geo.adversarial(model, k=8, iters=12, gen=torch.Generator(device='cuda:0').manual_seed(7))
    r['worst_ratio_top'] = ritz[:4].tolist()
    r['seconds'] = time.perf_counter() - t0
    rec[case] = r
    print(json.dumps({case: r}), flush=True)
    for gl in getattr(model, 'gl', {}).values():
        gl.sol.free()
    geo.C._free()
    del geo, model, X; import gc; gc.collect(); torch.cuda.empty_cache()
open(out, 'w').write(json.dumps(rec, indent=1))
