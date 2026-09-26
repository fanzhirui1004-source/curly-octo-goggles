import json
b=json.load(open('../../bench_deploy.json'))
print("case  Kprod_B16ms  learnedB16  exact64B16 exact32B16  f64s f32s | +tail k=2/4/8 (2k Kprods: fwd+adjoint) ms | break-even B16 queries vs exact fp64 factor: none/k2/k4/k8")
for c in b['cells']:
    kp=(c['cert_m8_B16_s']-c['cert_m0_B16_s'])/9
    L=c['learned_fused_Sq_B16_s']; X=c['exact_fp64_Sq_B16_s']; X32=c['exact_fp32_Sq_B16_s']
    t={k:L+2*k*kp for k in (2,4,8)}
    be=lambda Lq: c['factor_fp64_s']/(Lq-X) if Lq>X else float('inf')
    be32=lambda Lq: c['factor_fp32_s']/(Lq-X32) if Lq>X32 else float('inf')
    print(c['case'][-14:], f"{kp*1e3:6.1f} {L*1e3:7.1f} {X*1e3:7.1f} {X32*1e3:7.1f} {c['factor_fp64_s']:5.2f} {c['factor_fp32_s']:5.2f} |",
          " ".join(f"{t[k]*1e3:6.0f}" for k in t), "|", f"{be(L):5.0f}", " ".join(f"{be(t[k]):5.0f}" for k in t),
          "| vs fp32:", f"{be32(L):5.0f}", " ".join(f"{be32(t[k]):5.0f}" for k in t))
# 3x3x3 lattice, V2 doc sec 10: learned 60 ms/cell/iter (B6, unfused), exact 26 ms, 138 / 122 iterations, 27 cells
for k,kp in ((0,0),(2,.009),(4,.009),(8,.009)):
    # assume B6 K product ~ 9 ms (between r2 B16 8.9 and FULL 15; FULL B6 guess)
    lt=138*27*(0.060+2*k*kp)
    print(f"3x3x3 learned+tail{k}: PCG {lt:6.0f} s   (exact: 27*8.6 factor=232 s + 122*27*0.026=86 s = 318 s; fp32 factor 27*3.45=93 s -> 179 s)")
