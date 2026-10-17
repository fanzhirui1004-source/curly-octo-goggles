// src_v3 P13 (kmf.py): matrix-free float32 product y += sum_e P_e^T K_e P_e x of moment-fitted Q2 elements,
// K_e = sum_q w_e[q] B_q^T C B_q at the 5 x 5 x 5 tensor Gauss points, by sum factorisation (per element and column
// ~12.4k multiply-adds instead of the 81 x 81 matrix's 6561 reads). 25 threads per element, one per point pair
// (q0, q1), each looping over q2; the contractions along axes 0 and 1 (and their transposes) go through shared memory.
// Up to NB columns per block (one after another, sharing the element's indices and weights; gathers and atomic
// scatters of a node's 3 x ncol values adjacent).
// Element data: nd (E x 27) node indices in tensor order t = 9 i0 + 3 i1 + i2 (i_d = local coordinate + 1),
// w (E x 125) weights at the points p = 25 q0 + 5 q1 + q2, scaled by (2n)^2 so that the reference-coordinate
// derivative tables serve; x, y (nb x LD) float32 row-major. Node-major DOFs: dof = 3 node + component.
#include <cuda_runtime.h>
#include <stdint.h>

__constant__ float cL[15];          // cL[3 q + i] = L_i(g_q)        (L_0, L_1, L_2: Q2 Lagrange at xi = -1, 0, 1)
__constant__ float cD[15];          // cD[3 q + i] = dL_i/dxi (g_q)

template <int EPB, int NB>
__global__ void __launch_bounds__(25 * EPB)
kmf_kernel(const float* __restrict__ x, float* __restrict__ y, const int* __restrict__ nd,
           const float* __restrict__ w, int E, int LD, float lam, float mu)
{
    __shared__ float sX[EPB][NB][81]; // X[col][c][t] (27 c + t); after column col is done, its result Y[col][c][t]
    __shared__ float sA[EPB][270];    // A_L, A_D [c][q0][i1 i2]  (45 c + 9 q0 + 3 i1 + i2; A_D at +135); reused for Rb
    __shared__ float sR[EPB][675];    // R_DL, R_LD, R_LL [c][q0 q1][i2]   ((3 v + c) 75 + 3 (5 q0 + q1) + i2)
    const int le = threadIdx.x / 25, qq = threadIdx.x - 25 * (threadIdx.x / 25);
    const int e = blockIdx.x * EPB + le;
    const int col0 = blockIdx.y * NB;
    const int ncol = min(NB, LD - col0);
    const bool ok = (le < EPB) && (e < E);
    if (ok) {
        for (int k = qq; k < 81 * ncol; k += 25) {                // gather: node t, component c, column j (adjacent)
            const int tc = k / ncol, j = k - ncol * tc, t = tc / 3, c = tc - 3 * t;
            const int64_t node = nd[(int64_t)e * 27 + t];
            sX[le][j][27 * c + t] = x[(3 * node + c) * LD + col0 + j];
        }
    }
    const int q0 = qq / 5, q1 = qq - 5 * (qq / 5);
    float wq[5];
#pragma unroll
    for (int q2 = 0; q2 < 5; q2++) wq[q2] = ok ? w[(int64_t)e * 125 + 25 * q0 + 5 * q1 + q2] : 0.f;
    for (int j = 0; j < ncol; j++) {
        __syncthreads();                                          // sX[j] loaded; sA free (previous column done)
        if (ok) {                                                 // axis 0: A_V[c][q0][m] = sum_i0 V[q0][i0] X[c][i0][m]
            for (int k = qq; k < 135; k += 25) {
                const int c = k / 45, r = k - 45 * c, p0 = r / 9, m = r - 9 * p0;
                const float x0 = sX[le][j][27 * c + m], x1 = sX[le][j][27 * c + 9 + m], x2 = sX[le][j][27 * c + 18 + m];
                sA[le][k] = cL[3 * p0] * x0 + cL[3 * p0 + 1] * x1 + cL[3 * p0 + 2] * x2;
                sA[le][135 + k] = cD[3 * p0] * x0 + cD[3 * p0 + 1] * x1 + cD[3 * p0 + 2] * x2;
            }
        }
        __syncthreads();
        if (ok) {
            float BLL[3][3], BDL[3][3], BLD[3][3];                // axis 1, this thread's (q0, q1): [c][i2]
            float RDL[3][3], RLD[3][3], RLL[3][3];
#pragma unroll
            for (int c = 0; c < 3; c++)
#pragma unroll
                for (int i2 = 0; i2 < 3; i2++) {
                    float sll = 0.f, sdl = 0.f, sld = 0.f;
#pragma unroll
                    for (int i1 = 0; i1 < 3; i1++) {
                        const float aL = sA[le][45 * c + 9 * q0 + 3 * i1 + i2];
                        const float aD = sA[le][135 + 45 * c + 9 * q0 + 3 * i1 + i2];
                        sll += cL[3 * q1 + i1] * aL;
                        sdl += cL[3 * q1 + i1] * aD;
                        sld += cD[3 * q1 + i1] * aL;
                    }
                    BLL[c][i2] = sll; BDL[c][i2] = sdl; BLD[c][i2] = sld;
                    RDL[c][i2] = RLD[c][i2] = RLL[c][i2] = 0.f;
                }
#pragma unroll
            for (int q2 = 0; q2 < 5; q2++) {                      // axis 2, the stress, and back along axis 2
                float g0[3], g1[3], g2[3];                        // g_d[c] = d u_c / d xi_d
#pragma unroll
                for (int c = 0; c < 3; c++) {
                    g0[c] = cL[3 * q2] * BDL[c][0] + cL[3 * q2 + 1] * BDL[c][1] + cL[3 * q2 + 2] * BDL[c][2];
                    g1[c] = cL[3 * q2] * BLD[c][0] + cL[3 * q2 + 1] * BLD[c][1] + cL[3 * q2 + 2] * BLD[c][2];
                    g2[c] = cD[3 * q2] * BLL[c][0] + cD[3 * q2 + 1] * BLL[c][1] + cD[3 * q2 + 2] * BLL[c][2];
                }
                const float ld = lam * (g0[0] + g1[1] + g2[2]);
                const float s00 = wq[q2] * (ld + 2.f * mu * g0[0]);
                const float s11 = wq[q2] * (ld + 2.f * mu * g1[1]);
                const float s22 = wq[q2] * (ld + 2.f * mu * g2[2]);
                const float s01 = wq[q2] * (mu * (g1[0] + g0[1]));
                const float s02 = wq[q2] * (mu * (g2[0] + g0[2]));
                const float s12 = wq[q2] * (mu * (g2[1] + g1[2]));
                const float sc[3][3] = {{s00, s01, s02}, {s01, s11, s12}, {s02, s12, s22}};      // sigma_cd
#pragma unroll
                for (int c = 0; c < 3; c++)
#pragma unroll
                    for (int i2 = 0; i2 < 3; i2++) {
                        RDL[c][i2] += cL[3 * q2 + i2] * sc[c][0];
                        RLD[c][i2] += cL[3 * q2 + i2] * sc[c][1];
                        RLL[c][i2] += cD[3 * q2 + i2] * sc[c][2];
                    }
            }
#pragma unroll
            for (int c = 0; c < 3; c++)
#pragma unroll
                for (int i2 = 0; i2 < 3; i2++) {
                    sR[le][(0 + c) * 75 + 3 * qq + i2] = RDL[c][i2];
                    sR[le][(3 + c) * 75 + 3 * qq + i2] = RLD[c][i2];
                    sR[le][(6 + c) * 75 + 3 * qq + i2] = RLL[c][i2];
                }
        }
        __syncthreads();
        if (ok) {                                                 // back along axis 1 (into sA: Rb_D, Rb_L)
            for (int k = qq; k < 135; k += 25) {
                const int c = k / 45, r = k - 45 * c, p0 = r / 9, m = r - 9 * p0, i1 = m / 3, i2 = m - 3 * i1;
                float bD = 0.f, bL = 0.f;
#pragma unroll
                for (int p1 = 0; p1 < 5; p1++) {
                    const int idx = 3 * (5 * p0 + p1) + i2;
                    bD += cL[3 * p1 + i1] * sR[le][c * 75 + idx];
                    bL += cD[3 * p1 + i1] * sR[le][(3 + c) * 75 + idx] + cL[3 * p1 + i1] * sR[le][(6 + c) * 75 + idx];
                }
                sA[le][k] = bD;
                sA[le][135 + k] = bL;
            }
        }
        __syncthreads();
        if (ok) {                                                 // back along axis 0, into sX[j] (its X is used up)
            for (int k = qq; k < 81; k += 25) {
                const int c = k / 27, t = k - 27 * c, i0 = t / 9, m = t - 9 * i0;
                float acc = 0.f;
#pragma unroll
                for (int p0 = 0; p0 < 5; p0++)
                    acc += cD[3 * p0 + i0] * sA[le][45 * c + 9 * p0 + m] + cL[3 * p0 + i0] * sA[le][135 + 45 * c + 9 * p0 + m];
                sX[le][j][27 * c + t] = acc;
            }
        }
    }
    __syncthreads();
    if (ok) {                                                     // scatter: node t, component c, column j (adjacent)
        for (int k = qq; k < 81 * ncol; k += 25) {
            const int tc = k / ncol, j = k - ncol * tc, t = tc / 3, c = tc - 3 * t;
            const int64_t node = nd[(int64_t)e * 27 + t];
            atomicAdd(y + (3 * node + c) * LD + col0 + j, sX[le][j][27 * c + t]);
        }
    }
}

// Ghost-penalty faces (teacher.GhostFaces: y += gamma sum_f P_f^T (b_d (x) I_3)^T (b_d (x) I_3) P_f x): face f joins
// element fe[f] and its neighbour fn[f] one cell further along axis d (faces grouped by d: [0, c1), [c1, c2), [c2, nf));
// its 45 nodes are the two elements' 27 (tensor order, nd) through fmap[45 d + j] = 32 * (neighbour) + t, and the
// product per component is the 45 x 45 Gram matrix G_d = gamma b_d^T b_d (float32, G: 3 x 45 x 45). 45 threads per
// face (one per face node: gather, row of G_d times the face vector, atomic adds); up to NB columns per block; each
// block takes a contiguous range of faces (so it loads the Gram matrices of one or two directions only) in groups of
// FPB faces.
template <int FPB, int NB>
__global__ void __launch_bounds__(45 * FPB)
gf_kernel(const float* __restrict__ x, float* __restrict__ y, const int* __restrict__ nd, const int* __restrict__ fe,
          const int* __restrict__ fn, const int* __restrict__ fmap, const float* __restrict__ G, int nf, int c1, int c2,
          int LD)
{
    __shared__ float sG[3 * 2025];
    __shared__ float sX[FPB][NB][135];                            // face vectors [col][c][j] (45 c + j)
    const int per = (nf + gridDim.x - 1) / gridDim.x;
    const int lo = blockIdx.x * per, hi = min(nf, lo + per);
    if (lo >= hi) return;                                         // (uniform in the block)
    const int d0 = (lo >= c1) + (lo >= c2), d1 = (hi - 1 >= c1) + (hi - 1 >= c2);
    for (int i = threadIdx.x; i < (d1 - d0 + 1) * 2025; i += blockDim.x) sG[2025 * d0 + i] = G[2025 * d0 + i];
    const int lf = threadIdx.x / 45, j = threadIdx.x - 45 * (threadIdx.x / 45);
    const int col0 = blockIdx.y * NB;
    const int ncol = min(NB, LD - col0);
    for (int f0 = lo; f0 < hi; f0 += FPB) {
        const int f = f0 + lf;
        const bool ok = (lf < FPB) && (f < hi);
        int d = 0;
        int64_t node = 0;
        __syncthreads();                                          // sG loaded / previous group's sX consumed
        if (ok) {
            d = (f >= c1) + (f >= c2);
            const int m = fmap[45 * d + j];
            const int el = (m >= 32) ? fn[f] : fe[f];
            node = nd[(int64_t)el * 27 + (m & 31)];
            for (int c = 0; c < 3; c++)
                for (int k = 0; k < ncol; k++)
                    sX[lf][k][45 * c + j] = x[(3 * node + c) * LD + col0 + k];
        }
        __syncthreads();
        if (ok) {
            const float* g = sG + 2025 * d + 45 * j;
            float a[3][NB];
#pragma unroll
            for (int c = 0; c < 3; c++)
#pragma unroll
                for (int k = 0; k < NB; k++) a[c][k] = 0.f;
#pragma unroll 5
            for (int jj = 0; jj < 45; jj++) {
                const float gv = g[jj];
#pragma unroll
                for (int k = 0; k < NB; k++) {
                    if (k < ncol) {
                        a[0][k] += gv * sX[lf][k][jj];
                        a[1][k] += gv * sX[lf][k][45 + jj];
                        a[2][k] += gv * sX[lf][k][90 + jj];
                    }
                }
            }
#pragma unroll
            for (int c = 0; c < 3; c++)
#pragma unroll
                for (int k = 0; k < NB; k++)
                    if (k < ncol) atomicAdd(y + (3 * node + c) * LD + col0 + k, a[c][k]);
        }
    }
}

extern "C" {

int kmf_init(const float* tabL, const float* tabD)
{
    cudaError_t r = cudaMemcpyToSymbol(cL, tabL, 15 * sizeof(float));
    if (r != cudaSuccess) return (int)r;
    r = cudaMemcpyToSymbol(cD, tabD, 15 * sizeof(float));
    if (r != cudaSuccess) return (int)r;
    cudaFuncAttributes at;                                        // load the kernels now (not during a graph capture)
    cudaFuncGetAttributes(&at, kmf_kernel<2, 1>); cudaFuncGetAttributes(&at, kmf_kernel<2, 2>);
    cudaFuncGetAttributes(&at, kmf_kernel<2, 4>);
    cudaFuncGetAttributes(&at, kmf_kernel<4, 1>); cudaFuncGetAttributes(&at, kmf_kernel<4, 2>);
    cudaFuncGetAttributes(&at, kmf_kernel<4, 4>);
    cudaFuncGetAttributes(&at, kmf_kernel<8, 1>); cudaFuncGetAttributes(&at, kmf_kernel<8, 2>);
    cudaFuncGetAttributes(&at, kmf_kernel<8, 4>);
    cudaFuncGetAttributes(&at, gf_kernel<4, 1>); cudaFuncGetAttributes(&at, gf_kernel<4, 2>);
    cudaFuncGetAttributes(&at, gf_kernel<4, 4>);
    cudaFuncGetAttributes(&at, gf_kernel<8, 1>); cudaFuncGetAttributes(&at, gf_kernel<8, 2>);
    cudaFuncGetAttributes(&at, gf_kernel<8, 4>);
    return (int)cudaDeviceSynchronize();
}

template <int EPB>
static void kmf_go(const float* x, float* y, const int* nd, const float* w, int E, int B, float lam, float mu, int nb,
                   cudaStream_t s)
{
    const int gx = (E + EPB - 1) / EPB;
    if (nb == 1) kmf_kernel<EPB, 1><<<dim3(gx, B), 25 * EPB, 0, s>>>(x, y, nd, w, E, B, lam, mu);
    else if (nb == 2) kmf_kernel<EPB, 2><<<dim3(gx, (B + 1) / 2), 25 * EPB, 0, s>>>(x, y, nd, w, E, B, lam, mu);
    else kmf_kernel<EPB, 4><<<dim3(gx, (B + 3) / 4), 25 * EPB, 0, s>>>(x, y, nd, w, E, B, lam, mu);
}

// nb: columns per block (1, 2 or 4; 0: 1 for one column, 2 for two, else 4)
int kmf_launch(const float* x, float* y, const int* nd, const float* w, int E, int B, float lam, float mu, int epb,
               int nb, void* stream)
{
    if (E <= 0 || B <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    if (nb <= 0) nb = B == 1 ? 1 : (B == 2 ? 2 : 4);
    if (epb == 2) kmf_go<2>(x, y, nd, w, E, B, lam, mu, nb, s);
    else if (epb == 8) kmf_go<8>(x, y, nd, w, E, B, lam, mu, nb, s);
    else kmf_go<4>(x, y, nd, w, E, B, lam, mu, nb, s);
    return (int)cudaGetLastError();
}

template <int FPB>
static void gf_go(const float* x, float* y, const int* nd, const int* fe, const int* fn, const int* fmap,
                  const float* G, int nf, int c1, int c2, int B, int nb, int blocks, cudaStream_t s)
{
    int gx = (nf + FPB - 1) / FPB;
    if (blocks > 0 && gx > blocks) gx = blocks;
    if (nb == 1) gf_kernel<FPB, 1><<<dim3(gx, B), 45 * FPB, 0, s>>>(x, y, nd, fe, fn, fmap, G, nf, c1, c2, B);
    else if (nb == 2) gf_kernel<FPB, 2><<<dim3(gx, (B + 1) / 2), 45 * FPB, 0, s>>>(x, y, nd, fe, fn, fmap, G, nf, c1, c2, B);
    else gf_kernel<FPB, 4><<<dim3(gx, (B + 3) / 4), 45 * FPB, 0, s>>>(x, y, nd, fe, fn, fmap, G, nf, c1, c2, B);
}

int gf_launch(const float* x, float* y, const int* nd, const int* fe, const int* fn, const int* fmap, const float* G,
              int nf, int c1, int c2, int B, int fpb, int nb, int blocks, void* stream)
{
    if (nf <= 0 || B <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    if (nb <= 0) nb = B == 1 ? 1 : (B == 2 ? 2 : 4);
    if (fpb == 4) gf_go<4>(x, y, nd, fe, fn, fmap, G, nf, c1, c2, B, nb, blocks, s);
    else gf_go<8>(x, y, nd, fe, fn, fmap, G, nf, c1, c2, B, nb, blocks, s);
    return (int)cudaGetLastError();
}

}
