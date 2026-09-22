"""J1 step 3 for one development case: bind the native qualification probes into the asset (same
ASSET_SUPPLEMENT_01 convention Codex used for the moderate case), then write Codex's REFERENCE_STRUCTURE
(frozen run_mechanics.reference_structure, unchanged) and the 16 validation directions (rng 92218).
No model, no training, no GPU."""
import argparse, json, shutil, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse

ROOT = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921')
MN = ROOT / 'diagnostics' / 'MECHANICS_NETWORK_20260922_01'
sys.path.insert(0, str(MN / 'source_v1'))
import run_mechanics as rm  # noqa: E402  frozen, read-only


def main(a):
    asset, qual, out = Path(a.asset), Path(a.qualification), Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    result = json.loads((asset / 'RESULT.json').read_text())
    if result['status'] != 'PASS':
        raise ValueError('ASSETS_NOT_QUALIFIED')
    probes = asset / 'compiled' / 'QUALIFICATION_PROBES.npz'
    if not probes.exists():
        shutil.copyfile(qual / 'QUALIFICATION_PROBES.npz', probes)
        rm.write(asset / 'ASSET_SUPPLEMENT_01.json', dict(
            reason='native PARDISO qualification probes placed beside the compiled blocks, as for the moderate case',
            preserved_original_result_sha256=rm.sha(asset / 'RESULT.json'),
            output_sha256={'compiled/QUALIFICATION_PROBES.npz': rm.sha(probes)},
            source=str(qual / 'QUALIFICATION_PROBES.npz')))
    compiled = asset / 'compiled'
    for name in ['A.npz', 'C.npz', 'D.npz', 'Q_RIGID.npy', 'INTERIOR_RIGID.npy', 'INTERIOR_POINTS.npy']:
        if rm.sha(compiled / name) != result['output_sha256']['compiled/' + name]:
            raise ValueError('ASSET_SHA:' + name)
    A = sparse.load_npz(compiled / 'A.npz').tocsr()
    points = np.load(compiled / 'INTERIOR_POINTS.npy')
    local, levels, scale, reference = rm.reference_structure(A, points)
    reference.update(step_scale=scale, setup_seconds=time.monotonic() - start, source_sha256=rm.sha(rm.__file__))
    rm.write(out / 'REFERENCE_STRUCTURE.json', reference)
    np.savez(out / 'REFERENCE_STRUCTURE.npz', local=local, step_scale=scale,
             **{f'level{i}_{k}': v for i, l in enumerate(levels) for k, v in l.items()})
    target = ROOT / 'targets' / (a.case + '_v1')
    manifest = json.loads((target / 'MANIFEST.json').read_text())
    for name in ['REFERENCE_RQ.npy', 'RIGID_Q.npy']:
        if rm.sha(target / name) != manifest[name]:
            raise ValueError('TARGET_SHA:' + name)
    d = np.load(target / 'REFERENCE_RQ.npy', mmap_mode='r').shape[0]
    evalrng = np.random.default_rng(92218)
    zval = evalrng.normal(size=(d, 16)); zval /= np.linalg.norm(zval, axis=0)
    np.save(out / 'VALIDATION_Z.npy', zval)
    rm.write(out / 'RESULT.json', dict(status='READY', case=a.case, asset=str(asset), physical_dimension=int(d),
             interior_nodes=int(len(points)), reference=reference, seconds=time.monotonic() - start,
             output_sha256={p.name: rm.sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(event='REFERENCE_READY', case=a.case, d=int(d), seconds=time.monotonic() - start)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--case', required=True); p.add_argument('--asset', required=True)
    p.add_argument('--qualification', required=True); p.add_argument('--output', required=True)
    main(p.parse_args())
