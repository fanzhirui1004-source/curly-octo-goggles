source /root/autodl-tmp/OPL/S1/env_gpu.sh
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S3/packets OPL_GP_CACHE=0
cd /root/autodl-tmp/OPL/src_v2
O=/root/autodl-tmp/OPL/S1/V2
run() { n=$1; shift; echo "$(date +%T) START $n" >> $O/chain_arms.status; "$@" > $O/$n.log 2>&1; echo "$(date +%T) END $n rc=$?" >> $O/chain_arms.status; }
# first: FastNet vs Geo.field consistency with the tail on one cell (the gate uses FastNet)
OPL_CONV_FP32=1 FUSED_HYPER=1 run tailcheck $PY -u - <<'PY'
import torch, json, models as MD, trainlib as TL, diag_cert as DC, lat_full as LF, lattice3 as LT
ck = DC.load_ckpt('/root/autodl-tmp/OPL/S1/V2/A2_tail8/best.pt'); case = 'fresh_val_2006_d0_v1'; body = ck['cfg']['body']
C, _ = LT.prepared(case, body)
op, miss = LF.fast_op(ck, case, body, ck['cfg']['data'], C)
geo = op.fast.geo; model = op.fast.model
q = torch.randn(len(geo.P), 4, dtype=TL.dt, device=TL.dev)
u1 = op.fast.field(q).to(TL.dt); u2 = geo.field(model, q).to(TL.dt)
s1 = op.apply(q); s2 = geo.s_hat_apply(model, q)
print(json.dumps(dict(smooth_k=model.smooth_k, missing=miss, field_rel=float((u1-u2).norm()/u2.norm()), shat_rel=float((s1-s2).norm()/s2.norm()),
      sym=float(abs((q[:, :1]*op.apply(q[:, 1:2])).sum() - (q[:, 1:2]*op.apply(q[:, :1])).sum()) / abs((q[:, :1]*op.apply(q[:, :1])).sum())))))
PY
for c in fresh_val_2000_full fresh_val_2001_full fresh_val_2005_d1_v0 fresh_val_2006_d0_v1 fresh_val_2003_d1_v1 fresh_val_2004_d0_v2 fresh_val_2010_d0_v0 fresh_val_2002_d0_v0; do
  for conf in x y; do
    OPL_CONV_FP32=1 FUSED_HYPER=1 run gate_A2_tail8_${c}_$conf $PY -u lat_full.py $O/gate_A2_tail8_${c}_$conf.json --pairs '' --custom "$O/A2_tail8/best.pt:$c:$O/A2_tail8/best.pt:" --nb-mode explicit --sets test --configs $conf
  done
done
for c in fresh_val_2000_full fresh_val_2003_d1_v1 fresh_val_2006_d0_v1; do
  for conf in x y; do
    OPL_CONV_FP32=1 FUSED_HYPER=1 run decomp_A2_tail8_${c}_$conf $PY -u gate_decomp.py $O/decomp_A2_tail8_${c}_$conf.json $O/A2_tail8/best.pt $c --configs $conf
  done
done
echo CHAIN_TAILGATE_DONE >> $O/chain_arms.status
