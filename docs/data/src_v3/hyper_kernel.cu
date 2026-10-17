// src_v3 P16: fastnet's hyperedge gather and scatter (fused_hyper._gather_t / _scatter_t) as CUDA kernels, for 27
// slots and H = 4 heads. One warp per hyperedge and column slice of up to 128 (blockIdx.y), lanes over the slice's
// columns (NC per lane): lanes 0-26 load the hyperedge's slot nodes and weights once (weights divided by deg of the
// node when DEG, with the division instruction of the Triton kernels, div.full.f32) into shared memory, then every lane
// issues all 27 row loads (gather) or atomic row adds (scatter) of its columns at once. Gather: out[h, e, c] =
// sum_s w[e, s, h] src[hn[e, s], c], in slot order with fused multiply-adds as the Triton kernel (bitwise the same);
// scatter: out[hn[e, s], c] += sum_h w[e, s, h] src[h, e, c] (rows with rmask != 0 skipped; HS: the head-sum order).
#include <cuda_runtime.h>
#include <stdint.h>

#define HK_WARPS 8

__device__ __forceinline__ float div_full(float a, float b)
{
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

// head sum of a slot's row: 0 fused chain, 1 products then left to right, 2 products then pairwise
template <int HS>
__device__ __forceinline__ float hsum(const float4 w, float z0, float z1, float z2, float z3)
{
    if (HS == 1) {
        const float p0 = __fmul_rn(w.x, z0), p1 = __fmul_rn(w.y, z1), p2 = __fmul_rn(w.z, z2), p3 = __fmul_rn(w.w, z3);
        return __fadd_rn(__fadd_rn(__fadd_rn(p0, p1), p2), p3);
    } else if (HS == 2) {
        const float p0 = __fmul_rn(w.x, z0), p1 = __fmul_rn(w.y, z1), p2 = __fmul_rn(w.z, z2), p3 = __fmul_rn(w.w, z3);
        return __fadd_rn(__fadd_rn(p0, p1), __fadd_rn(p2, p3));
    }
    return fmaf(w.w, z3, fmaf(w.z, z2, fmaf(w.y, z1, w.x * z0)));
}

template <int NC, bool DEG>
__global__ void __launch_bounds__(32 * HK_WARPS)
hg_kernel(const float* __restrict__ src, const int64_t* __restrict__ hn, const float* __restrict__ wgt,
          const float* __restrict__ deg, float* __restrict__ out, int Eh, int BF)
{
    __shared__ int64_t sN[HK_WARPS][27];
    __shared__ float4 sW[HK_WARPS][27];
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = blockIdx.x * HK_WARPS + wid;
    const int col0 = blockIdx.y * 32 * NC;
    if (e >= Eh) return;                                          // (whole warps)
    if (lane < 27) {
        const int64_t n = hn[(int64_t)e * 27 + lane];
        float4 w = reinterpret_cast<const float4*>(wgt)[(int64_t)e * 27 + lane];
        if (DEG) {
            const float d = deg[n];
            w.x = div_full(w.x, d); w.y = div_full(w.y, d); w.z = div_full(w.z, d); w.w = div_full(w.w, d);
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
            const int c = col0 + lane + 32 * k;
            const float x = c < BF ? src[n * BF + c] : 0.f;
            acc[0][k] = fmaf(w.x, x, acc[0][k]); acc[1][k] = fmaf(w.y, x, acc[1][k]);
            acc[2][k] = fmaf(w.z, x, acc[2][k]); acc[3][k] = fmaf(w.w, x, acc[3][k]);
        }
    }
#pragma unroll
    for (int h = 0; h < 4; h++)
#pragma unroll
        for (int k = 0; k < NC; k++) {
            const int c = col0 + lane + 32 * k;
            if (c < BF) out[((int64_t)h * Eh + e) * BF + c] = acc[h][k];
        }
}

template <int NC, bool DEG, bool RMASK, int HS>
__global__ void __launch_bounds__(32 * HK_WARPS)
hs_kernel(const float* __restrict__ src, const int64_t* __restrict__ hn, const float* __restrict__ wgt,
          const float* __restrict__ deg, float* __restrict__ out, const uint8_t* __restrict__ rmask, int Eh, int BF)
{
    __shared__ int64_t sN[HK_WARPS][27];
    __shared__ float4 sW[HK_WARPS][27];
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = blockIdx.x * HK_WARPS + wid;
    const int col0 = blockIdx.y * 32 * NC;
    if (e >= Eh) return;
    if (lane < 27) {
        const int64_t n = hn[(int64_t)e * 27 + lane];
        float4 w = reinterpret_cast<const float4*>(wgt)[(int64_t)e * 27 + lane];
        if (DEG) {
            const float d = deg[n];
            w.x = div_full(w.x, d); w.y = div_full(w.y, d); w.z = div_full(w.z, d); w.w = div_full(w.w, d);
        }
        sN[wid][lane] = (RMASK && rmask[n] != 0) ? -1 : n;
        sW[wid][lane] = w;
    }
    float z[4][NC];
#pragma unroll
    for (int h = 0; h < 4; h++)
#pragma unroll
        for (int k = 0; k < NC; k++) {
            const int c = col0 + lane + 32 * k;
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
            const int c = col0 + lane + 32 * k;
            if (c < BF) atomicAdd(out + n * BF + c, hsum<HS>(w, z[0][k], z[1][k], z[2][k], z[3][k]));
        }
    }
}

template <int NC>
static void hg_go(const float* src, const int64_t* hn, const float* w, const float* deg, float* out, int Eh, int BF,
                  int use_deg, cudaStream_t s)
{
    const dim3 grid((Eh + HK_WARPS - 1) / HK_WARPS, (BF + 32 * NC - 1) / (32 * NC));
    if (use_deg) hg_kernel<NC, true><<<grid, 32 * HK_WARPS, 0, s>>>(src, hn, w, deg, out, Eh, BF);
    else hg_kernel<NC, false><<<grid, 32 * HK_WARPS, 0, s>>>(src, hn, w, deg, out, Eh, BF);
}

template <int NC, int HS>
static void hs_go(const float* src, const int64_t* hn, const float* w, const float* deg, float* out,
                  const uint8_t* rmask, int Eh, int BF, int use_deg, cudaStream_t s)
{
    const dim3 grid((Eh + HK_WARPS - 1) / HK_WARPS, (BF + 32 * NC - 1) / (32 * NC));
    if (use_deg) {
        if (rmask) hs_kernel<NC, true, true, HS><<<grid, 32 * HK_WARPS, 0, s>>>(src, hn, w, deg, out, rmask, Eh, BF);
        else hs_kernel<NC, true, false, HS><<<grid, 32 * HK_WARPS, 0, s>>>(src, hn, w, deg, out, rmask, Eh, BF);
    } else {
        if (rmask) hs_kernel<NC, false, true, HS><<<grid, 32 * HK_WARPS, 0, s>>>(src, hn, w, deg, out, rmask, Eh, BF);
        else hs_kernel<NC, false, false, HS><<<grid, 32 * HK_WARPS, 0, s>>>(src, hn, w, deg, out, rmask, Eh, BF);
    }
}

template <int HS>
static void hs_nc(const float* src, const int64_t* hn, const float* w, const float* deg, float* out,
                  const uint8_t* rmask, int Eh, int BF, int use_deg, cudaStream_t s)
{
    const int nc = BF >= 128 ? 4 : (BF + 31) / 32;
    if (nc == 1) hs_go<1, HS>(src, hn, w, deg, out, rmask, Eh, BF, use_deg, s);
    else if (nc == 2) hs_go<2, HS>(src, hn, w, deg, out, rmask, Eh, BF, use_deg, s);
    else if (nc == 3) hs_go<3, HS>(src, hn, w, deg, out, rmask, Eh, BF, use_deg, s);
    else hs_go<4, HS>(src, hn, w, deg, out, rmask, Eh, BF, use_deg, s);
}

extern "C" {

int hk_init()
{
    cudaFuncAttributes at;                                        // load the kernels now (not during a graph capture)
    cudaFuncGetAttributes(&at, hg_kernel<1, false>); cudaFuncGetAttributes(&at, hg_kernel<1, true>);
    cudaFuncGetAttributes(&at, hg_kernel<2, false>); cudaFuncGetAttributes(&at, hg_kernel<2, true>);
    cudaFuncGetAttributes(&at, hg_kernel<3, false>); cudaFuncGetAttributes(&at, hg_kernel<3, true>);
    cudaFuncGetAttributes(&at, hg_kernel<4, false>); cudaFuncGetAttributes(&at, hg_kernel<4, true>);
    cudaFuncGetAttributes(&at, hs_kernel<1, false, false, 0>); cudaFuncGetAttributes(&at, hs_kernel<1, true, false, 0>);
    cudaFuncGetAttributes(&at, hs_kernel<1, false, true, 0>); cudaFuncGetAttributes(&at, hs_kernel<1, true, true, 0>);
    cudaFuncGetAttributes(&at, hs_kernel<1, false, false, 1>); cudaFuncGetAttributes(&at, hs_kernel<1, true, false, 1>);
    cudaFuncGetAttributes(&at, hs_kernel<1, false, true, 1>); cudaFuncGetAttributes(&at, hs_kernel<1, true, true, 1>);
    cudaFuncGetAttributes(&at, hs_kernel<1, false, false, 2>); cudaFuncGetAttributes(&at, hs_kernel<1, true, false, 2>);
    cudaFuncGetAttributes(&at, hs_kernel<1, false, true, 2>); cudaFuncGetAttributes(&at, hs_kernel<1, true, true, 2>);
    cudaFuncGetAttributes(&at, hs_kernel<2, false, false, 0>); cudaFuncGetAttributes(&at, hs_kernel<2, true, false, 0>);
    cudaFuncGetAttributes(&at, hs_kernel<2, false, true, 0>); cudaFuncGetAttributes(&at, hs_kernel<2, true, true, 0>);
    cudaFuncGetAttributes(&at, hs_kernel<2, false, false, 1>); cudaFuncGetAttributes(&at, hs_kernel<2, true, false, 1>);
    cudaFuncGetAttributes(&at, hs_kernel<2, false, true, 1>); cudaFuncGetAttributes(&at, hs_kernel<2, true, true, 1>);
    cudaFuncGetAttributes(&at, hs_kernel<2, false, false, 2>); cudaFuncGetAttributes(&at, hs_kernel<2, true, false, 2>);
    cudaFuncGetAttributes(&at, hs_kernel<2, false, true, 2>); cudaFuncGetAttributes(&at, hs_kernel<2, true, true, 2>);
    cudaFuncGetAttributes(&at, hs_kernel<3, false, false, 0>); cudaFuncGetAttributes(&at, hs_kernel<3, true, false, 0>);
    cudaFuncGetAttributes(&at, hs_kernel<3, false, true, 0>); cudaFuncGetAttributes(&at, hs_kernel<3, true, true, 0>);
    cudaFuncGetAttributes(&at, hs_kernel<3, false, false, 1>); cudaFuncGetAttributes(&at, hs_kernel<3, true, false, 1>);
    cudaFuncGetAttributes(&at, hs_kernel<3, false, true, 1>); cudaFuncGetAttributes(&at, hs_kernel<3, true, true, 1>);
    cudaFuncGetAttributes(&at, hs_kernel<3, false, false, 2>); cudaFuncGetAttributes(&at, hs_kernel<3, true, false, 2>);
    cudaFuncGetAttributes(&at, hs_kernel<3, false, true, 2>); cudaFuncGetAttributes(&at, hs_kernel<3, true, true, 2>);
    cudaFuncGetAttributes(&at, hs_kernel<4, false, false, 0>); cudaFuncGetAttributes(&at, hs_kernel<4, true, false, 0>);
    cudaFuncGetAttributes(&at, hs_kernel<4, false, true, 0>); cudaFuncGetAttributes(&at, hs_kernel<4, true, true, 0>);
    cudaFuncGetAttributes(&at, hs_kernel<4, false, false, 1>); cudaFuncGetAttributes(&at, hs_kernel<4, true, false, 1>);
    cudaFuncGetAttributes(&at, hs_kernel<4, false, true, 1>); cudaFuncGetAttributes(&at, hs_kernel<4, true, true, 1>);
    cudaFuncGetAttributes(&at, hs_kernel<4, false, false, 2>); cudaFuncGetAttributes(&at, hs_kernel<4, true, false, 2>);
    cudaFuncGetAttributes(&at, hs_kernel<4, false, true, 2>); cudaFuncGetAttributes(&at, hs_kernel<4, true, true, 2>);
    return (int)cudaDeviceSynchronize();
}

// src (N x BF) -> out (H x Eh x BF), H = 4, 27 slots (column slices of 128)
int hk_gather(const float* src, const int64_t* hn, const float* w, const float* deg, float* out, int Eh, int BF,
              int use_deg, void* stream)
{
    if (Eh <= 0 || BF <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    const int nc = BF >= 128 ? 4 : (BF + 31) / 32;
    if (nc == 1) hg_go<1>(src, hn, w, deg, out, Eh, BF, use_deg, s);
    else if (nc == 2) hg_go<2>(src, hn, w, deg, out, Eh, BF, use_deg, s);
    else if (nc == 3) hg_go<3>(src, hn, w, deg, out, Eh, BF, use_deg, s);
    else hg_go<4>(src, hn, w, deg, out, Eh, BF, use_deg, s);
    return (int)cudaGetLastError();
}

// src (H x Eh x BF) -> out (N x BF) += ..., rows with rmask != 0 skipped (rmask may be null); hs: head-sum order
int hk_scatter(const float* src, const int64_t* hn, const float* w, const float* deg, float* out, const uint8_t* rmask,
               int Eh, int BF, int use_deg, int hs, void* stream)
{
    if (Eh <= 0 || BF <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    if (hs == 1) hs_nc<1>(src, hn, w, deg, out, rmask, Eh, BF, use_deg, s);
    else if (hs == 2) hs_nc<2>(src, hn, w, deg, out, rmask, Eh, BF, use_deg, s);
    else hs_nc<0>(src, hn, w, deg, out, rmask, Eh, BF, use_deg, s);
    return (int)cudaGetLastError();
}

}
