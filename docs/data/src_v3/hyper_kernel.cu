// src_v3 P16: fastnet's hyperedge gather and scatter (fused_hyper._gather_t / _scatter_t) as CUDA kernels, for 27
// slots, H = 4 heads and B F <= 128 columns. One warp per hyperedge, lanes over the columns (NC per lane): lanes 0-26
// load the hyperedge's slot nodes and weights once (weights divided by deg of the node when DEG, as the Triton kernels
// do) into shared memory, then every lane issues all 27 row loads (gather) or atomic row adds (scatter) of its columns
// at once. Gather: out[h, e, c] = sum_s w[e, s, h] src[hn[e, s], c], the sum in slot order with fused multiply-adds as
// the Triton kernel; scatter: out[hn[e, s], c] += sum_h w[e, s, h] src[h, e, c] (rows with rmask != 0 skipped).
#include <cuda_runtime.h>
#include <stdint.h>

#define HK_WARPS 8

template <int NC, bool DEG>
__global__ void __launch_bounds__(32 * HK_WARPS)
hg_kernel(const float* __restrict__ src, const int64_t* __restrict__ hn, const float* __restrict__ wgt,
          const float* __restrict__ deg, float* __restrict__ out, int Eh, int BF)
{
    __shared__ int64_t sN[HK_WARPS][27];
    __shared__ float4 sW[HK_WARPS][27];
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = blockIdx.x * HK_WARPS + wid;
    if (e >= Eh) return;                                          // (whole warps)
    if (lane < 27) {
        const int64_t n = hn[(int64_t)e * 27 + lane];
        float4 w = reinterpret_cast<const float4*>(wgt)[(int64_t)e * 27 + lane];
        if (DEG) {
            const float d = deg[n];
            w.x = w.x / d; w.y = w.y / d; w.z = w.z / d; w.w = w.w / d;
        }
        sN[wid][lane] = n; sW[wid][lane] = w;
    }
    __syncwarp();
    float acc[4][NC];
#pragma unroll
    for (int h = 0; h < 4; h++)
#pragma unroll
        for (int k = 0; k < NC; k++) acc[h][k] = 0.f;
#pragma unroll
    for (int s = 0; s < 27; s++) {
        const int64_t n = sN[wid][s];
        const float4 w = sW[wid][s];
#pragma unroll
        for (int k = 0; k < NC; k++) {
            const int c = lane + 32 * k;
            const float x = c < BF ? src[n * BF + c] : 0.f;
            acc[0][k] = fmaf(w.x, x, acc[0][k]); acc[1][k] = fmaf(w.y, x, acc[1][k]);
            acc[2][k] = fmaf(w.z, x, acc[2][k]); acc[3][k] = fmaf(w.w, x, acc[3][k]);
        }
    }
#pragma unroll
    for (int h = 0; h < 4; h++)
#pragma unroll
        for (int k = 0; k < NC; k++) {
            const int c = lane + 32 * k;
            if (c < BF) out[((int64_t)h * Eh + e) * BF + c] = acc[h][k];
        }
}

template <int NC, bool DEG, bool RMASK>
__global__ void __launch_bounds__(32 * HK_WARPS)
hs_kernel(const float* __restrict__ src, const int64_t* __restrict__ hn, const float* __restrict__ wgt,
          const float* __restrict__ deg, float* __restrict__ out, const uint8_t* __restrict__ rmask, int Eh, int BF)
{
    __shared__ int64_t sN[HK_WARPS][27];
    __shared__ float4 sW[HK_WARPS][27];
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = blockIdx.x * HK_WARPS + wid;
    if (e >= Eh) return;
    if (lane < 27) {
        const int64_t n = hn[(int64_t)e * 27 + lane];
        float4 w = reinterpret_cast<const float4*>(wgt)[(int64_t)e * 27 + lane];
        if (DEG) {
            const float d = deg[n];
            w.x = w.x / d; w.y = w.y / d; w.z = w.z / d; w.w = w.w / d;
        }
        sN[wid][lane] = (RMASK && rmask[n] != 0) ? -1 : n;
        sW[wid][lane] = w;
    }
    float z[4][NC];
#pragma unroll
    for (int h = 0; h < 4; h++)
#pragma unroll
        for (int k = 0; k < NC; k++) {
            const int c = lane + 32 * k;
            z[h][k] = c < BF ? src[((int64_t)h * Eh + e) * BF + c] : 0.f;
        }
    __syncwarp();
#pragma unroll
    for (int s = 0; s < 27; s++) {
        const int64_t n = sN[wid][s];
        if (RMASK && n < 0) continue;                             // (uniform in the warp)
        const float4 w = sW[wid][s];
#pragma unroll
        for (int k = 0; k < NC; k++) {
            const int c = lane + 32 * k;
            if (c < BF) atomicAdd(out + n * BF + c, w.x * z[0][k] + w.y * z[1][k] + w.z * z[2][k] + w.w * z[3][k]);
        }
    }
}

template <int NC>
static void hg_go(const float* src, const int64_t* hn, const float* w, const float* deg, float* out, int Eh, int BF,
                  int use_deg, cudaStream_t s)
{
    const int gx = (Eh + HK_WARPS - 1) / HK_WARPS;
    if (use_deg) hg_kernel<NC, true><<<gx, 32 * HK_WARPS, 0, s>>>(src, hn, w, deg, out, Eh, BF);
    else hg_kernel<NC, false><<<gx, 32 * HK_WARPS, 0, s>>>(src, hn, w, deg, out, Eh, BF);
}

template <int NC>
static void hs_go(const float* src, const int64_t* hn, const float* w, const float* deg, float* out,
                  const uint8_t* rmask, int Eh, int BF, int use_deg, cudaStream_t s)
{
    const int gx = (Eh + HK_WARPS - 1) / HK_WARPS;
    if (use_deg) {
        if (rmask) hs_kernel<NC, true, true><<<gx, 32 * HK_WARPS, 0, s>>>(src, hn, w, deg, out, rmask, Eh, BF);
        else hs_kernel<NC, true, false><<<gx, 32 * HK_WARPS, 0, s>>>(src, hn, w, deg, out, rmask, Eh, BF);
    } else {
        if (rmask) hs_kernel<NC, false, true><<<gx, 32 * HK_WARPS, 0, s>>>(src, hn, w, deg, out, rmask, Eh, BF);
        else hs_kernel<NC, false, false><<<gx, 32 * HK_WARPS, 0, s>>>(src, hn, w, deg, out, rmask, Eh, BF);
    }
}

extern "C" {

int hk_init()
{
    cudaFuncAttributes at;                                        // load the kernels now (not during a graph capture)
    cudaFuncGetAttributes(&at, hg_kernel<1, false>); cudaFuncGetAttributes(&at, hg_kernel<1, true>);
    cudaFuncGetAttributes(&at, hg_kernel<2, false>); cudaFuncGetAttributes(&at, hg_kernel<2, true>);
    cudaFuncGetAttributes(&at, hg_kernel<3, false>); cudaFuncGetAttributes(&at, hg_kernel<3, true>);
    cudaFuncGetAttributes(&at, hg_kernel<4, false>); cudaFuncGetAttributes(&at, hg_kernel<4, true>);
    cudaFuncGetAttributes(&at, hs_kernel<1, false, false>); cudaFuncGetAttributes(&at, hs_kernel<1, true, false>);
    cudaFuncGetAttributes(&at, hs_kernel<1, false, true>); cudaFuncGetAttributes(&at, hs_kernel<1, true, true>);
    cudaFuncGetAttributes(&at, hs_kernel<2, false, false>); cudaFuncGetAttributes(&at, hs_kernel<2, true, false>);
    cudaFuncGetAttributes(&at, hs_kernel<2, false, true>); cudaFuncGetAttributes(&at, hs_kernel<2, true, true>);
    cudaFuncGetAttributes(&at, hs_kernel<3, false, false>); cudaFuncGetAttributes(&at, hs_kernel<3, true, false>);
    cudaFuncGetAttributes(&at, hs_kernel<3, false, true>); cudaFuncGetAttributes(&at, hs_kernel<3, true, true>);
    cudaFuncGetAttributes(&at, hs_kernel<4, false, false>); cudaFuncGetAttributes(&at, hs_kernel<4, true, false>);
    cudaFuncGetAttributes(&at, hs_kernel<4, false, true>); cudaFuncGetAttributes(&at, hs_kernel<4, true, true>);
    return (int)cudaDeviceSynchronize();
}

// src (N x BF) -> out (H x Eh x BF), H = 4, 27 slots, BF <= 128
int hk_gather(const float* src, const int64_t* hn, const float* w, const float* deg, float* out, int Eh, int BF,
              int use_deg, void* stream)
{
    if (Eh <= 0 || BF <= 0) return 0;
    if (BF > 128) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    const int nc = (BF + 31) / 32;
    if (nc == 1) hg_go<1>(src, hn, w, deg, out, Eh, BF, use_deg, s);
    else if (nc == 2) hg_go<2>(src, hn, w, deg, out, Eh, BF, use_deg, s);
    else if (nc == 3) hg_go<3>(src, hn, w, deg, out, Eh, BF, use_deg, s);
    else hg_go<4>(src, hn, w, deg, out, Eh, BF, use_deg, s);
    return (int)cudaGetLastError();
}

// src (H x Eh x BF) -> out (N x BF) += ..., rows with rmask != 0 skipped (rmask may be null)
int hk_scatter(const float* src, const int64_t* hn, const float* w, const float* deg, float* out, const uint8_t* rmask,
               int Eh, int BF, int use_deg, void* stream)
{
    if (Eh <= 0 || BF <= 0) return 0;
    if (BF > 128) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    const int nc = (BF + 31) / 32;
    if (nc == 1) hs_go<1>(src, hn, w, deg, out, rmask, Eh, BF, use_deg, s);
    else if (nc == 2) hs_go<2>(src, hn, w, deg, out, rmask, Eh, BF, use_deg, s);
    else if (nc == 3) hs_go<3>(src, hn, w, deg, out, rmask, Eh, BF, use_deg, s);
    else hs_go<4>(src, hn, w, deg, out, rmask, Eh, BF, use_deg, s);
    return (int)cudaGetLastError();
}

}
