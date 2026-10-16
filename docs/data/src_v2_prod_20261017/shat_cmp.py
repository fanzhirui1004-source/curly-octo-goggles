import sys, json, torch
ref = torch.load(sys.argv[1])
for p in sys.argv[2:]:
    d = torch.load(p)
    print(json.dumps({'file': p.split('/')[-1], **{k: float((d[k] - ref[k]).norm() / ref[k].norm()) for k in ref}}))
