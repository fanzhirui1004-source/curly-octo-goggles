"""src_v3 P16 (OPL_V3_HCUDA): fastnet's hyperedge gather and scatter (fused_hyper.gather / scatter) as CUDA kernels
(hyper_kernel.cu, built with nvcc on first use and cached like kmf's library): one warp per hyperedge and column slice
of 128, lanes over the columns, the 27 slots' nodes and weights staged once per hyperedge, all slot rows loaded (or
atomically added) at once. For 27 slots and H = 4 heads; otherwise, or without nvcc, fused_hyper's Triton kernels. The
gather is bitwise the Triton kernels' (slot order, fused multiply-adds, the same division instruction for w / deg); the
scatter's atomic row adds are in data-dependent order either way (rounding level; HS: the head-sum order per slot)."""
import os
import ctypes
import hashlib
import shutil
import subprocess
import torch

ON = os.environ.get('OPL_V3_HCUDA', '1') == '1'
MAXBF = int(os.environ.get('OPL_V3_HCUDA_MAXBF', '0') or 0)          # widest B F for the CUDA kernels (0: any)
HS = int(os.environ.get('OPL_V3_HCUDA_HS', '0') or 0)                # scatter head sum: 0 fused, 1 left to right, 2 pairwise,
                                                                        # 3 butterfly
GMAXBF = int(os.environ.get('OPL_V3_HCUDA_GMAXBF', '128') or 0)      # gather: widest B F (beyond: Triton's, faster there)
_LIB = None
_ERR = None


def _lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hyper_kernel.cu')
    code = open(src, 'rb').read()
    maj, mnr = torch.cuda.get_device_capability()
    arch = f'sm_{maj}{mnr}'
    tag = hashlib.sha1(code + arch.encode()).hexdigest()[:16]
    d = os.environ.get('OPL_V3_KMF_DIR') or os.path.join(os.path.expanduser('~'), '.cache', 'opl_v3_kmf')
    os.makedirs(d, exist_ok=True)
    so = os.path.join(d, f'libhyperk_{tag}.so')
    if not os.path.exists(so):
        nvcc = os.environ.get('OPL_V3_NVCC') or shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
        tmp = f'{so}.{os.getpid()}.tmp'
        r = subprocess.run([nvcc, '-O3', '-std=c++17', f'-arch={arch}', '-Xcompiler', '-fPIC', '-shared', '-o', tmp, src],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError('NVCC:' + (r.stderr or r.stdout)[-1500:])
        os.replace(tmp, so)
    lib = ctypes.CDLL(so)
    lib.hk_gather.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    lib.hk_scatter.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    lib.hk_layer.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
    lib.sd_head.argtypes = ([ctypes.c_void_p] * 2 + [ctypes.c_int] + [ctypes.c_void_p] * 6 + [ctypes.c_int]
                            + [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_float] + [ctypes.c_int] * 3
                            + [ctypes.c_void_p])
    rc = lib.hk_init()
    if rc != 0:
        raise RuntimeError(f'HK_INIT:{rc}')
    _LIB = lib
    return lib


def usable(H, S, BF, w, hn, on=None, maxbf=None):
    """The CUDA kernels apply: switch on, library built (a failure is remembered), 27 slots, 4 heads, B F <= maxbf
    (MAXBF; when set), int64 or int32 (src_v3 item 1c) node indices, 16-byte aligned contiguous float32 weights."""
    global _ERR
    lim = MAXBF if maxbf is None else maxbf
    if not (ON if on is None else on) or _ERR is not None or H != 4 or S != 27 or (lim and BF > lim):
        return False
    if hn.dtype not in (torch.int64, torch.int32) or w.dtype != torch.float32 or not w.is_contiguous() or w.data_ptr() % 16:
        return False
    try:
        _lib()
        return True
    except Exception as ex:                                                 # noqa: BLE001 (no nvcc: Triton kernels)
        _ERR = repr(ex)[:1600]
        return False


def gather(src, hn, w, deg, out, use_deg):
    """out (H x Eh x BF) = sum_s w[e, s, h] src[hn[e, s]] (w / deg[hn] when use_deg); src (N x BF) contiguous."""
    Eh, BF = hn.shape[0], src.shape[1]
    rc = _lib().hk_gather(src.data_ptr(), hn.data_ptr(), w.data_ptr(), deg.data_ptr(), out.data_ptr(), Eh, BF,
                          int(bool(use_deg)), int(hn.dtype == torch.int32), torch.cuda.current_stream(src.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f'HK_GATHER:{rc}')
    return out


def scatter(src, hn, w, deg, out, rmask, use_deg, hs=None):
    """out (N x BF) += sum_h w[e, s, h] src[h, e] at row hn[e, s] (rows with rmask != 0 skipped)."""
    Eh, BF = hn.shape[0], src.shape[2]
    rc = _lib().hk_scatter(src.data_ptr(), hn.data_ptr(), w.data_ptr(), deg.data_ptr(), out.data_ptr(),
                           rmask.data_ptr() if rmask is not None else None, Eh, BF, int(bool(use_deg)),
                           int(HS if hs is None else hs), int(hn.dtype == torch.int32),
                           torch.cuda.current_stream(src.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f'HK_SCATTER:{rc}')
    return out


_SMS = {}


def layer(X, hn, ga, sb, deg, W, out, gdeg, sdeg, rmask=None, blocks=None):
    """src_v3 P18: out (N x B F) += one hyperedge layer applied to X (hyperfused.layer's contract; F = 32, 4 heads)."""
    dev = X.device
    if blocks is None:
        if dev not in _SMS:
            _SMS[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
        blocks = 8 * _SMS[dev]
    rc = _lib().hk_layer(X.data_ptr(), hn.data_ptr(), ga.data_ptr(), sb.data_ptr(), deg.data_ptr(), W.data_ptr(),
                         out.data_ptr(), rmask.data_ptr() if rmask is not None else None, hn.shape[0], X.shape[1],
                         int(bool(gdeg)), int(bool(sdeg)), int(blocks), int(hn.dtype == torch.int32),
                         torch.cuda.current_stream(dev).cuda_stream)
    if rc != 0:
        raise RuntimeError(f'HK_LAYER:{rc}')
    return out
