"""Build the GP_UPPER cache for each case (one Cell at a time) and check the fp32+refinement extension against fp64."""
import sys, json, gc
import torch
import teacher as TE
body = sys.argv[1]
for case in sys.argv[2:]:
    C = TE.Cell(case, body, log=lambda s_: None); C.assemble()
    q = torch.randn((C.np_, 8), dtype=TE.dt, device=TE.dev, generator=torch.Generator(device=TE.dev).manual_seed(0))
    C.factor(neumann=False); u64 = C.extend(q)
    C.factor(neumann=False, fp32=True); u32 = C.extend(q)
    print(json.dumps(dict(case=case, fp32_refined_vs_fp64=float((u32 - u64).norm() / u64.norm()),
                          peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)), flush=True)
    C._free(); del C, u64, u32; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
