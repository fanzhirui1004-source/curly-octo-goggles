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

// head sum of a slot's row: 0 fused chain, 1 products then left to right, 2 products then pairwise, 3 products then
// butterfly over lanes (h with h ^ 2 first: (p0 + p2) + (p1 + p3))
template <int HS>
__device__ __forceinline__ float hsum(const float4 w, float z0, float z1, float z2, float z3)
{
    if (HS == 3) {
        const float p0 = __fmul_rn(w.x, z0), p1 = __fmul_rn(w.y, z1), p2 = __fmul_rn(w.z, z2), p3 = __fmul_rn(w.w, z3);
        return __fadd_rn(__fadd_rn(p0, p2), __fadd_rn(p1, p3));
    } else if (HS == 1) {
        const float p0 = __fmul_rn(w.x, z0), p1 = __fmul_rn(w.y, z1), p2 = __fmul_rn(w.z, z2), p3 = __fmul_rn(w.w, z3);
        return __fadd_rn(__fadd_rn(__fadd_rn(p0, p1), p2), p3);
    } else if (HS == 2) {
        const float p0 = __fmul_rn(w.x, z0), p1 = __fmul_rn(w.y, z1), p2 = __fmul_rn(w.z, z2), p3 = __fmul_rn(w.w, z3);
        return __fadd_rn(__fadd_rn(p0, p1), __fadd_rn(p2, p3));
    }
    return fmaf(w.w, z3, fmaf(w.z, z2, fmaf(w.y, z1, w.x * z0)));
}

template <int NC, bool DEG, typename IT>
__global__ void __launch_bounds__(32 * HK_WARPS)
hg_kernel(const float* __restrict__ src, const IT* __restrict__ hn, const float* __restrict__ wgt,
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

template <int NC, bool DEG, bool RMASK, int HS, typename IT>
__global__ void __launch_bounds__(32 * HK_WARPS)
hs_kernel(const float* __restrict__ src, const IT* __restrict__ hn, const float* __restrict__ wgt,
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

template <int NC, typename IT>
static void hg_go(const float* src, const IT* hn, const float* w, const float* deg, float* out, int Eh, int BF,
                  int use_deg, cudaStream_t s)
{
    const dim3 grid((Eh + HK_WARPS - 1) / HK_WARPS, (BF + 32 * NC - 1) / (32 * NC));
    if (use_deg) hg_kernel<NC, true, IT><<<grid, 32 * HK_WARPS, 0, s>>>(src, hn, w, deg, out, Eh, BF);
    else hg_kernel<NC, false, IT><<<grid, 32 * HK_WARPS, 0, s>>>(src, hn, w, deg, out, Eh, BF);
}

template <int NC, int HS, typename IT>
static void hs_go(const float* src, const IT* hn, const float* w, const float* deg, float* out,
                  const uint8_t* rmask, int Eh, int BF, int use_deg, cudaStream_t s)
{
    const dim3 grid((Eh + HK_WARPS - 1) / HK_WARPS, (BF + 32 * NC - 1) / (32 * NC));
    if (use_deg) {
        if (rmask) hs_kernel<NC, true, true, HS, IT><<<grid, 32 * HK_WARPS, 0, s>>>(src, hn, w, deg, out, rmask, Eh, BF);
        else hs_kernel<NC, true, false, HS, IT><<<grid, 32 * HK_WARPS, 0, s>>>(src, hn, w, deg, out, rmask, Eh, BF);
    } else {
        if (rmask) hs_kernel<NC, false, true, HS, IT><<<grid, 32 * HK_WARPS, 0, s>>>(src, hn, w, deg, out, rmask, Eh, BF);
        else hs_kernel<NC, false, false, HS, IT><<<grid, 32 * HK_WARPS, 0, s>>>(src, hn, w, deg, out, rmask, Eh, BF);
    }
}

template <int HS, typename IT>
static void hs_nc(const float* src, const IT* hn, const float* w, const float* deg, float* out,
                  const uint8_t* rmask, int Eh, int BF, int use_deg, cudaStream_t s)
{
    const int nc = BF >= 128 ? 4 : (BF + 31) / 32;
    if (nc == 1) hs_go<1, HS, IT>(src, hn, w, deg, out, rmask, Eh, BF, use_deg, s);
    else if (nc == 2) hs_go<2, HS, IT>(src, hn, w, deg, out, rmask, Eh, BF, use_deg, s);
    else if (nc == 3) hs_go<3, HS, IT>(src, hn, w, deg, out, rmask, Eh, BF, use_deg, s);
    else hs_go<4, HS, IT>(src, hn, w, deg, out, rmask, Eh, BF, use_deg, s);
}

// src_v3 P18: one hyperedge layer in one kernel (hyperfused.layer, fastnet with OPL_V3_HFUSE): for hyperedge e and right-
// hand side b (F = 32 channels, lane f): Z[h] = sum_s ga[e, s, h] X[hn[e, s], b, f] (ga / deg with GDEG; slot order,
// fused multiply-adds: hg_kernel's Z bitwise), the channel mix Z'[h, g] = sum_f Z[h, f] W[h, f, g] (lane g; Z[h, f] by
// warp shuffles, W in shared memory; f in order: cuBLAS sums in another order, rounding level), then out[hn[e, s], b, g] +=
// sum_h sb[e, s, h] Z'[h, g] (sb / deg with SDEG; rows with rmask != 0 skipped; hs_kernel's fused head sum). Z and Z'
// stay in registers. Warps loop over hyperedges (W staged once per block).
template <bool GDEG, bool SDEG, bool RMASK, typename IT>
__global__ void __launch_bounds__(32 * HK_WARPS)
hl_kernel(const float* __restrict__ X, const IT* __restrict__ hn, const float* __restrict__ ga,
          const float* __restrict__ sb, const float* __restrict__ deg, const float* __restrict__ W, float* __restrict__ out,
          const uint8_t* __restrict__ rmask, int Eh, int BF)
{
    __shared__ float sW[4 * 32 * 32];
    __shared__ int64_t sN[HK_WARPS][27];
    __shared__ float4 sA[HK_WARPS][27], sB[HK_WARPS][27];
    __shared__ unsigned int sK[HK_WARPS];                         // skip bits (rmask) of the 27 slots
    for (int i = threadIdx.x; i < 4 * 32 * 32; i += 32 * HK_WARPS) sW[i] = W[i];
    __syncthreads();
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int c0 = blockIdx.y * 32;
    for (int e = blockIdx.x * HK_WARPS + wid; e < Eh; e += gridDim.x * HK_WARPS) {   // (whole warps)
        bool skip = false;
        if (lane < 27) {
            const int64_t n = hn[(int64_t)e * 27 + lane];
            float4 a = reinterpret_cast<const float4*>(ga)[(int64_t)e * 27 + lane];
            float4 b = reinterpret_cast<const float4*>(sb)[(int64_t)e * 27 + lane];
            if (GDEG || SDEG) {
                const float d = deg[n];
                if (GDEG) { a.x = div_full(a.x, d); a.y = div_full(a.y, d); a.z = div_full(a.z, d); a.w = div_full(a.w, d); }
                if (SDEG) { b.x = div_full(b.x, d); b.y = div_full(b.y, d); b.z = div_full(b.z, d); b.w = div_full(b.w, d); }
            }
            if (RMASK) skip = rmask[n] != 0;
            sN[wid][lane] = n; sA[wid][lane] = a; sB[wid][lane] = b;
        }
        const unsigned int kb = __ballot_sync(0xffffffffu, skip);
        if (lane == 0) sK[wid] = kb;
        __syncwarp();
        float z0 = 0.f, z1 = 0.f, z2 = 0.f, z3 = 0.f;
#pragma unroll
        for (int s = 0; s < 27; s++) {
            const float x = X[sN[wid][s] * BF + c0 + lane];
            const float4 a = sA[wid][s];
            z0 = fmaf(a.x, x, z0); z1 = fmaf(a.y, x, z1); z2 = fmaf(a.z, x, z2); z3 = fmaf(a.w, x, z3);
        }
        float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
#pragma unroll 8
        for (int f = 0; f < 32; f++) {
            const float y0 = __shfl_sync(0xffffffffu, z0, f), y1 = __shfl_sync(0xffffffffu, z1, f);
            const float y2 = __shfl_sync(0xffffffffu, z2, f), y3 = __shfl_sync(0xffffffffu, z3, f);
            p0 = fmaf(y0, sW[f * 32 + lane], p0);
            p1 = fmaf(y1, sW[1024 + f * 32 + lane], p1);
            p2 = fmaf(y2, sW[2048 + f * 32 + lane], p2);
            p3 = fmaf(y3, sW[3072 + f * 32 + lane], p3);
        }
        const unsigned int sk = sK[wid];
#pragma unroll
        for (int s = 0; s < 27; s++) {
            if (RMASK && ((sk >> s) & 1u)) continue;              // (uniform in the warp)
            const float4 b = sB[wid][s];
            atomicAdd(out + sN[wid][s] * BF + c0 + lane, hsum<0>(b, p0, p1, p2, p3));
        }
        __syncwarp();                                             // (the slot tables are reused by the next hyperedge)
    }
}

// src_v3 P17: one slot head of fastnet's slot weights (slotdec): for rows r = 27 e' + s of a hyperedge set (e = emap[e']
// when given), u = (P[e] + Q[idx[27 e + s]]) + R[s] (64 channels), h = GELU(u) (erf), then for the OS selected output
// columns o = cols[j] of the head ((2, LH, 4) layout: o = (ab LH + l) 4 + hh): y = 0.2 (h . W2[o] + b2[o]), soft clip
// (bound Abnd[o / 4], knee), b (ab = 1) times 1 + lam[l] fac[27 e + s] (B3), stored to Ao / Bo[l - l0][r][hh]. 64 rows per
// block of 128 threads: h staged transposed in shared memory, W2's selected rows transposed beside it, each thread
// 4 rows x OS / 8 columns of float32 multiply-adds (no tensor cores).
#define SD_ROWS 64
#define SD_CH 64
#define SD_HP 68                                                  // h row stride (16-byte aligned float4 reads)

template <int OS, typename IT>
__global__ void __launch_bounds__(128)
sd_head_kernel(const float* __restrict__ P, const float* __restrict__ Q, int q_ld, const float* __restrict__ R,
               const IT* __restrict__ idx, const IT* __restrict__ emap, const float* __restrict__ W2,
               const float* __restrict__ b2, const int* __restrict__ cols, const float* __restrict__ Abnd,
               const float* __restrict__ lam, const float* __restrict__ fac, float* __restrict__ Ao,
               float* __restrict__ Bo, int n_rows, int l0, int LH, float knee, int clip, int b3)
{
    constexpr int CPT = OS / 8;                                   // columns per thread (8 or 4)
    __shared__ __align__(16) float sH[SD_CH][SD_HP];              // h transposed: [channel][row]
    __shared__ __align__(16) float sW[SD_CH][OS + 4];             // W2 selected rows transposed: [channel][col]
    __shared__ int64_t sE[SD_ROWS], sN[SD_ROWS], sF[SD_ROWS];
    __shared__ int sS[SD_ROWS];
    const int t = threadIdx.x;
    const int r0 = blockIdx.x * SD_ROWS;
    if (t < SD_ROWS) {
        const int r = r0 + t;
        int64_t e = 0, rf = 0, n = 0; int s = 0;
        if (r < n_rows) {
            const int es = r / 27;
            s = r - 27 * es;
            e = emap ? emap[es] : (int64_t)es;
            rf = e * 27 + s;
            n = idx[rf];
        }
        sE[t] = e; sN[t] = n; sF[t] = rf; sS[t] = s;
    }
    for (int k = t; k < SD_CH * OS; k += 128) {                   // W2[cols[j]][c] -> sW[c][j]
        const int j = k / SD_CH, c = k - SD_CH * j;
        sW[c][j] = W2[(int64_t)cols[j] * SD_CH + c];
    }
    __syncthreads();
    for (int i = 0; i < SD_ROWS * SD_CH / 128; i++) {             // u, GELU: consecutive threads, consecutive channels
        const int flat = i * 128 + t, row = flat / SD_CH, c = flat - SD_CH * row;
        float h = 0.f;
        if (r0 + row < n_rows) {
            const float u = (P[sE[row] * SD_CH + c] + Q[sN[row] * q_ld + c]) + R[sS[row] * SD_CH + c];
            h = 0.5f * u * (1.0f + erff(u * 0.7071067811865476f));
        }
        sH[c][row] = h;
    }
    __syncthreads();
    const int ty = t >> 3, tx = t & 7;                            // rows 4 ty .. 4 ty + 3, columns CPT tx ..
    float acc[4][CPT];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < CPT; j++) acc[i][j] = 0.f;
#pragma unroll 8
    for (int c = 0; c < SD_CH; c++) {
        const float4 hv = *reinterpret_cast<const float4*>(&sH[c][4 * ty]);
        float wv[CPT];
#pragma unroll
        for (int j = 0; j < CPT; j++) wv[j] = sW[c][CPT * tx + j];
#pragma unroll
        for (int j = 0; j < CPT; j++) {
            acc[0][j] = fmaf(hv.x, wv[j], acc[0][j]); acc[1][j] = fmaf(hv.y, wv[j], acc[1][j]);
            acc[2][j] = fmaf(hv.z, wv[j], acc[2][j]); acc[3][j] = fmaf(hv.w, wv[j], acc[3][j]);
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int row = 4 * ty + i, r = r0 + row;
        if (r >= n_rows) continue;
        const float fr = b3 ? fac[sF[row]] : 0.f;
#pragma unroll
        for (int q = 0; q < CPT / 4; q++) {                       // groups of 4 heads (one (ab, l) each)
            const int o0 = cols[CPT * tx + 4 * q];
            const int g = o0 >> 2, ab = g / LH, l = g - LH * ab;
            float v[4];
#pragma unroll
            for (int hh = 0; hh < 4; hh++) {
                float x = (acc[i][4 * q + hh] + b2[o0 + hh]) * 0.2f;
                if (clip) {
                    const float a = Abnd[g], tk = a * knee, ax = fabsf(x), den = a - tk;
                    const float th = tanhf((ax - tk) / den);
                    const float w = tk + den * th;
                    x = ax > tk ? (x > 0.f ? w : (x < 0.f ? -w : 0.f)) : x;
                }
                if (ab == 1 && b3) x = x * (1.0f + lam[l] * fr);
                v[hh] = x;
            }
            float* dst = (ab == 0 ? Ao : Bo) + (((int64_t)(l - l0) * n_rows + r) << 2);
            *reinterpret_cast<float4*>(dst) = make_float4(v[0], v[1], v[2], v[3]);
        }
    }
}

// every kernel instance loaded at hk_init (not during a graph capture)
template <int NC, typename IT>
static void touch_nc(cudaFuncAttributes* at)
{
    cudaFuncGetAttributes(at, hg_kernel<NC, false, IT>); cudaFuncGetAttributes(at, hg_kernel<NC, true, IT>);
#define HK_T(HS_) \
    cudaFuncGetAttributes(at, hs_kernel<NC, false, false, HS_, IT>); cudaFuncGetAttributes(at, hs_kernel<NC, true, false, HS_, IT>); \
    cudaFuncGetAttributes(at, hs_kernel<NC, false, true, HS_, IT>); cudaFuncGetAttributes(at, hs_kernel<NC, true, true, HS_, IT>);
    HK_T(0) HK_T(1) HK_T(2) HK_T(3)
#undef HK_T
}

template <typename IT>
static void touch(cudaFuncAttributes* at)
{
    cudaFuncGetAttributes(at, sd_head_kernel<32, IT>); cudaFuncGetAttributes(at, sd_head_kernel<64, IT>);
    cudaFuncGetAttributes(at, hl_kernel<false, true, false, IT>); cudaFuncGetAttributes(at, hl_kernel<false, true, true, IT>);
    cudaFuncGetAttributes(at, hl_kernel<true, false, false, IT>); cudaFuncGetAttributes(at, hl_kernel<true, false, true, IT>);
    cudaFuncGetAttributes(at, hl_kernel<false, false, false, IT>); cudaFuncGetAttributes(at, hl_kernel<true, true, false, IT>);
    touch_nc<1, IT>(at); touch_nc<2, IT>(at); touch_nc<3, IT>(at); touch_nc<4, IT>(at);
}

template <typename IT>
static void gather_go(const float* src, const IT* hn, const float* w, const float* deg, float* out, int Eh, int BF,
                      int use_deg, cudaStream_t s)
{
    const int nc = BF >= 128 ? 4 : (BF + 31) / 32;
    if (nc == 1) hg_go<1, IT>(src, hn, w, deg, out, Eh, BF, use_deg, s);
    else if (nc == 2) hg_go<2, IT>(src, hn, w, deg, out, Eh, BF, use_deg, s);
    else if (nc == 3) hg_go<3, IT>(src, hn, w, deg, out, Eh, BF, use_deg, s);
    else hg_go<4, IT>(src, hn, w, deg, out, Eh, BF, use_deg, s);
}

template <typename IT>
static void scatter_go(const float* src, const IT* hn, const float* w, const float* deg, float* out, const uint8_t* rmask,
                       int Eh, int BF, int use_deg, int hs, cudaStream_t s)
{
    if (hs == 3) hs_nc<3, IT>(src, hn, w, deg, out, rmask, Eh, BF, use_deg, s);
    else if (hs == 1) hs_nc<1, IT>(src, hn, w, deg, out, rmask, Eh, BF, use_deg, s);
    else if (hs == 2) hs_nc<2, IT>(src, hn, w, deg, out, rmask, Eh, BF, use_deg, s);
    else hs_nc<0, IT>(src, hn, w, deg, out, rmask, Eh, BF, use_deg, s);
}

template <typename IT>
static int head_go(const float* P, const float* Q, int q_ld, const float* R, const IT* idx, const IT* emap,
                   const float* W2, const float* b2, const int* cols, int OS, const float* Abnd, const float* lam,
                   const float* fac, float* Ao, float* Bo, int n_rows, int l0, int LH, float knee, int clip, int b3,
                   cudaStream_t s)
{
    const int gx = (n_rows + SD_ROWS - 1) / SD_ROWS;
    if (OS == 64)
        sd_head_kernel<64, IT><<<gx, 128, 0, s>>>(P, Q, q_ld, R, idx, emap, W2, b2, cols, Abnd, lam, fac, Ao, Bo, n_rows, l0,
                                                  LH, knee, clip, b3);
    else if (OS == 32)
        sd_head_kernel<32, IT><<<gx, 128, 0, s>>>(P, Q, q_ld, R, idx, emap, W2, b2, cols, Abnd, lam, fac, Ao, Bo, n_rows, l0,
                                                  LH, knee, clip, b3);
    else
        return -1;
    return 0;
}

template <typename IT>
static void layer_go(const float* X, const IT* hn, const float* ga, const float* sb, const float* deg, const float* W,
                     float* out, const uint8_t* rmask, int Eh, int BF, int gdeg, int sdeg, const dim3 grid, cudaStream_t s)
{
    if (gdeg && !sdeg) {
        if (rmask) hl_kernel<true, false, true, IT><<<grid, 32 * HK_WARPS, 0, s>>>(X, hn, ga, sb, deg, W, out, rmask, Eh, BF);
        else hl_kernel<true, false, false, IT><<<grid, 32 * HK_WARPS, 0, s>>>(X, hn, ga, sb, deg, W, out, rmask, Eh, BF);
    } else if (!gdeg && sdeg) {
        if (rmask) hl_kernel<false, true, true, IT><<<grid, 32 * HK_WARPS, 0, s>>>(X, hn, ga, sb, deg, W, out, rmask, Eh, BF);
        else hl_kernel<false, true, false, IT><<<grid, 32 * HK_WARPS, 0, s>>>(X, hn, ga, sb, deg, W, out, rmask, Eh, BF);
    } else if (gdeg && sdeg) {
        hl_kernel<true, true, false, IT><<<grid, 32 * HK_WARPS, 0, s>>>(X, hn, ga, sb, deg, W, out, rmask, Eh, BF);
    } else {
        hl_kernel<false, false, false, IT><<<grid, 32 * HK_WARPS, 0, s>>>(X, hn, ga, sb, deg, W, out, rmask, Eh, BF);
    }
}

// The node index arrays (hn, idx, emap) are int64 (i32 = 0) or int32 (i32 = 1, src_v3 item 1c: fastnet's OPL_V3_HN32);
// the values loaded are the same either way.
extern "C" {

int hk_init()
{
    cudaFuncAttributes at;
    touch<int64_t>(&at); touch<int>(&at);
    return (int)cudaDeviceSynchronize();
}

// src (N x BF) -> out (H x Eh x BF), H = 4, 27 slots (column slices of 128)
int hk_gather(const float* src, const void* hn, const float* w, const float* deg, float* out, int Eh, int BF,
              int use_deg, int i32, void* stream)
{
    if (Eh <= 0 || BF <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    if (i32) gather_go<int>(src, (const int*)hn, w, deg, out, Eh, BF, use_deg, s);
    else gather_go<int64_t>(src, (const int64_t*)hn, w, deg, out, Eh, BF, use_deg, s);
    return (int)cudaGetLastError();
}

// src (H x Eh x BF) -> out (N x BF) += ..., rows with rmask != 0 skipped (rmask may be null); hs: head-sum order
int hk_scatter(const float* src, const void* hn, const float* w, const float* deg, float* out, const uint8_t* rmask,
               int Eh, int BF, int use_deg, int hs, int i32, void* stream)
{
    if (Eh <= 0 || BF <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    if (i32) scatter_go<int>(src, (const int*)hn, w, deg, out, rmask, Eh, BF, use_deg, hs, s);
    else scatter_go<int64_t>(src, (const int64_t*)hn, w, deg, out, rmask, Eh, BF, use_deg, hs, s);
    return (int)cudaGetLastError();
}

// P17: one slot head over n_rows rows (OS = 32 or 64 selected columns); emap may be null
int sd_head(const float* P, const float* Q, int q_ld, const float* R, const void* idx, const void* emap,
            const float* W2, const float* b2, const int* cols, int OS, const float* Abnd, const float* lam,
            const float* fac, float* Ao, float* Bo, int n_rows, int l0, int LH, float knee, int clip, int b3, int i32,
            void* stream)
{
    if (n_rows <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    const int rc = i32 ? head_go<int>(P, Q, q_ld, R, (const int*)idx, (const int*)emap, W2, b2, cols, OS, Abnd, lam, fac,
                                      Ao, Bo, n_rows, l0, LH, knee, clip, b3, s)
                       : head_go<int64_t>(P, Q, q_ld, R, (const int64_t*)idx, (const int64_t*)emap, W2, b2, cols, OS, Abnd,
                                          lam, fac, Ao, Bo, n_rows, l0, LH, knee, clip, b3, s);
    if (rc != 0) return rc;
    return (int)cudaGetLastError();
}

// P18: out (N x BF) += one hyperedge layer applied to X (N x BF), BF = 32 B; W (4, 32, 32); rmask may be null
int hk_layer(const float* X, const void* hn, const float* ga, const float* sb, const float* deg, const float* W,
             float* out, const uint8_t* rmask, int Eh, int BF, int gdeg, int sdeg, int blocks, int i32, void* stream)
{
    if (Eh <= 0 || BF <= 0) return 0;
    if (BF % 32) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    int gx = (Eh + HK_WARPS - 1) / HK_WARPS;
    if (blocks > 0 && gx > blocks) gx = blocks;
    const dim3 grid(gx, BF / 32);
    if (i32) layer_go<int>(X, (const int*)hn, ga, sb, deg, W, out, rmask, Eh, BF, gdeg, sdeg, grid, s);
    else layer_go<int64_t>(X, (const int64_t*)hn, ga, sb, deg, W, out, rmask, Eh, BF, gdeg, sdeg, grid, s);
    return (int)cudaGetLastError();
}

}
