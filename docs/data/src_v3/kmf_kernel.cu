// src_v3 P13 / P14 (kmf.py): matrix-free float32 / float64 product y += sum_e P_e^T K_e P_e x of moment-fitted Q2 elements,
// K_e = sum_q w_e[q] B_q^T C B_q at the 5 x 5 x 5 tensor Gauss points, by sum factorisation (per element and column
// ~12.4k multiply-adds instead of the 81 x 81 matrix's 6561 reads). 25 threads per element, one per point pair
// (q0, q1), each looping over q2; the contractions along axes 0 and 1 (and their transposes) go through shared memory.
// Up to NB columns per block (one after another, sharing the element's indices and weights; gathers and atomic
// scatters of a node's 3 x ncol values adjacent).
// Element data: nd (E x 27) node indices in tensor order t = 9 i0 + 3 i1 + i2 (i_d = local coordinate + 1),
// weights at the points p = 25 q0 + 5 q1 + q2 (float64 rows w, scaled by (2n)^2 so that the reference-coordinate
// derivative tables serve; one store per cell for both precisions, kmf.wstore: the partial elements' rows, the full
// elements' (one shared row where bitwise equal)), times lam and mu formed in float64 per element and rounded to the product's precision (points:
// the 5-point Gauss rule, or equispaced -1, -1/2, 0, 1/2, 1 whose Q2 tables are exact in float32); x, y (nb x LD)
// row-major (float32, or float64 for P14). Node-major DOFs: dof = 3 node + component.
#include <cuda_runtime.h>
#include <stdint.h>

__constant__ float cL32[15];        // cL(3 q + i) = L_i(g_q)        (L_0, L_1, L_2: Q2 Lagrange at xi = -1, 0, 1)
__constant__ float cD32[15];        // cD(3 q + i) = dL_i/dxi (g_q)
__constant__ double cL64[15];       // the same tables in float64 (src_v3 P14: the float64 K product)
__constant__ double cD64[15];

template <typename T> struct Tab;
template <> struct Tab<float> {
    static __device__ __forceinline__ float L(int i) { return cL32[i]; }
    static __device__ __forceinline__ float D(int i) { return cD32[i]; }
};
template <> struct Tab<double> {
    static __device__ __forceinline__ double L(int i) { return cL64[i]; }
    static __device__ __forceinline__ double D(int i) { return cD64[i]; }
};
#define cL(i) Tab<T>::L(i)
#define cD(i) Tab<T>::D(i)

// Weights (src_v3, shared by the float32 and float64 products): rows of 125 float64 (times (2n)^2, no Lame constants),
// w (nw rows) then wx; element e's row widx[e] (null: e itself; -1: wfull, the full elements' row); lam w and mu w are formed
// here in float64 and rounded to T, exactly as the stored lam w / mu w of before (bitwise the same weights).
template <typename T, int EPB, int NB>
__global__ void __launch_bounds__(25 * EPB)
kmf_kernel(const T* __restrict__ x, T* __restrict__ y, const int* __restrict__ nd, const double* __restrict__ w,
           const double* __restrict__ wx, int nw, const int* __restrict__ widx, const double* __restrict__ wfull, double lam,
           double mu, int E, int LD)
{
    __shared__ T sX[EPB][NB][81];     // X[col][c][t] (27 c + t); after column col is done, its result Y[col][c][t]
    __shared__ T sA[EPB][270];        // A_L, A_D [c][q0][i1 i2]  (45 c + 9 q0 + 3 i1 + i2; A_D at +135); reused for Rb
    __shared__ T sR[EPB][675];        // R_DL, R_LD, R_LL [c][q0 q1][i2]   ((3 v + c) 75 + 3 (5 q0 + q1) + i2)
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
    T wlq[5], wmq[5];                                             // lam w, mu w at this thread's 5 points
    {
        const int64_t row = ok ? (widx ? (int64_t)widx[e] : (int64_t)e) : 0;
        const double* wr = row < 0 ? wfull : (row < nw ? w + row * 125 : wx + (row - nw) * 125);
#pragma unroll
        for (int q2 = 0; q2 < 5; q2++) {
            const double wv = ok ? wr[25 * q0 + 5 * q1 + q2] : 0.0;
            wlq[q2] = T(lam * wv);
            wmq[q2] = T(mu * wv);
        }
    }
    for (int j = 0; j < ncol; j++) {
        __syncthreads();                                          // sX[j] loaded; sA free (previous column done)
        if (ok) {                                                 // axis 0: A_V[c][q0][m] = sum_i0 V[q0][i0] X[c][i0][m]
            for (int k = qq; k < 135; k += 25) {
                const int c = k / 45, r = k - 45 * c, p0 = r / 9, m = r - 9 * p0;
                const T x0 = sX[le][j][27 * c + m], x1 = sX[le][j][27 * c + 9 + m], x2 = sX[le][j][27 * c + 18 + m];
                sA[le][k] = cL(3 * p0) * x0 + cL(3 * p0 + 1) * x1 + cL(3 * p0 + 2) * x2;
                sA[le][135 + k] = cD(3 * p0) * x0 + cD(3 * p0 + 1) * x1 + cD(3 * p0 + 2) * x2;
            }
        }
        __syncthreads();
        if (ok) {
            T BLL[3][3], BDL[3][3], BLD[3][3];                    // axis 1, this thread's (q0, q1): [c][i2]
            T RDL[3][3], RLD[3][3], RLL[3][3];
#pragma unroll
            for (int c = 0; c < 3; c++)
#pragma unroll
                for (int i2 = 0; i2 < 3; i2++) {
                    T sll = 0, sdl = 0, sld = 0;
#pragma unroll
                    for (int i1 = 0; i1 < 3; i1++) {
                        const T aL = sA[le][45 * c + 9 * q0 + 3 * i1 + i2];
                        const T aD = sA[le][135 + 45 * c + 9 * q0 + 3 * i1 + i2];
                        sll += cL(3 * q1 + i1) * aL;
                        sdl += cL(3 * q1 + i1) * aD;
                        sld += cD(3 * q1 + i1) * aL;
                    }
                    BLL[c][i2] = sll; BDL[c][i2] = sdl; BLD[c][i2] = sld;
                    RDL[c][i2] = RLD[c][i2] = RLL[c][i2] = 0;
                }
#pragma unroll
            for (int q2 = 0; q2 < 5; q2++) {                      // axis 2, the stress, and back along axis 2
                T g0[3], g1[3], g2[3];                            // g_d[c] = d u_c / d xi_d
#pragma unroll
                for (int c = 0; c < 3; c++) {
                    g0[c] = cL(3 * q2) * BDL[c][0] + cL(3 * q2 + 1) * BDL[c][1] + cL(3 * q2 + 2) * BDL[c][2];
                    g1[c] = cL(3 * q2) * BLD[c][0] + cL(3 * q2 + 1) * BLD[c][1] + cL(3 * q2 + 2) * BLD[c][2];
                    g2[c] = cD(3 * q2) * BLL[c][0] + cD(3 * q2 + 1) * BLL[c][1] + cD(3 * q2 + 2) * BLL[c][2];
                }
                const T ld = wlq[q2] * (g0[0] + g1[1] + g2[2]);
                const T m2 = T(2) * wmq[q2];
                const T s00 = ld + m2 * g0[0];
                const T s11 = ld + m2 * g1[1];
                const T s22 = ld + m2 * g2[2];
                const T s01 = wmq[q2] * (g1[0] + g0[1]);
                const T s02 = wmq[q2] * (g2[0] + g0[2]);
                const T s12 = wmq[q2] * (g2[1] + g1[2]);
                const T sc[3][3] = {{s00, s01, s02}, {s01, s11, s12}, {s02, s12, s22}};      // sigma_cd
#pragma unroll
                for (int c = 0; c < 3; c++)
#pragma unroll
                    for (int i2 = 0; i2 < 3; i2++) {
                        RDL[c][i2] += cL(3 * q2 + i2) * sc[c][0];
                        RLD[c][i2] += cL(3 * q2 + i2) * sc[c][1];
                        RLL[c][i2] += cD(3 * q2 + i2) * sc[c][2];
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
                T bD = 0, bL = 0;
#pragma unroll
                for (int p1 = 0; p1 < 5; p1++) {
                    const int idx = 3 * (5 * p0 + p1) + i2;
                    bD += cL(3 * p1 + i1) * sR[le][c * 75 + idx];
                    bL += cD(3 * p1 + i1) * sR[le][(3 + c) * 75 + idx] + cL(3 * p1 + i1) * sR[le][(6 + c) * 75 + idx];
                }
                sA[le][k] = bD;
                sA[le][135 + k] = bL;
            }
        }
        __syncthreads();
        if (ok) {                                                 // back along axis 0, into sX[j] (its X is used up)
            for (int k = qq; k < 81; k += 25) {
                const int c = k / 27, t = k - 27 * c, i0 = t / 9, m = t - 9 * i0;
                T acc = 0;
#pragma unroll
                for (int p0 = 0; p0 < 5; p0++)
                    acc += cD(3 * p0 + i0) * sA[le][45 * c + 9 * p0 + m] + cL(3 * p0 + i0) * sA[le][135 + 45 * c + 9 * p0 + m];
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
// product per component is the 45 x 45 Gram matrix G_d = gamma b_d^T b_d (G: 3 x 45 x 45). 45 threads per face (one per
// face node: gather, row of G_d times the face vector, atomic adds); up to NB columns per block; each block takes a
// contiguous range of at most min_d (faces of direction d) faces, so it meets at most two directions and keeps their
// Gram matrices in shared memory, in groups of FPB faces.
template <typename T, int FPB, int NB>
__global__ void __launch_bounds__(45 * FPB)
gf_kernel(const T* __restrict__ x, T* __restrict__ y, const int* __restrict__ nd, const int* __restrict__ fe,
          const int* __restrict__ fn, const int* __restrict__ fmap, const T* __restrict__ G, int nf, int c1, int c2,
          int LD)
{
    __shared__ T sG[2 * 2025];
    __shared__ T sX[FPB][NB][135];                                // face vectors [col][c][j] (45 c + j)
    const int per = (nf + gridDim.x - 1) / gridDim.x;
    const int lo = blockIdx.x * per, hi = min(nf, lo + per);
    if (lo >= hi) return;                                         // (uniform in the block)
    const int d0 = (lo >= c1) + (lo >= c2), d1 = (hi - 1 >= c1) + (hi - 1 >= c2);
    for (int i = threadIdx.x; i < (d1 - d0 + 1) * 2025; i += blockDim.x) sG[i] = G[2025 * d0 + i];
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
            const T* g = sG + 2025 * (d - d0) + 45 * j;
            T a[3][NB];
#pragma unroll
            for (int c = 0; c < 3; c++)
#pragma unroll
                for (int k = 0; k < NB; k++) a[c][k] = 0;
#pragma unroll 5
            for (int jj = 0; jj < 45; jj++) {
                const T gv = g[jj];
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

// Ghost faces, DOF-parallel form (gf2): 135 threads per face, thread (j, c) = face node j, component c, so a warp's
// gathers and atomic adds cover consecutive DOFs (the three components of a node, and node runs along the fastest grid
// axis); the face's 45 node indices are precomputed (fnode: nf x 45, int32), so no index chain through the elements.
template <typename T, int FPB, int NB>
__global__ void __launch_bounds__(135 * FPB)
gf2_kernel(const T* __restrict__ x, T* __restrict__ y, const int* __restrict__ fnode, const T* __restrict__ G, int nf,
           int c1, int c2, int LD)
{
    __shared__ T sG[2 * 2025];
    __shared__ T sX[FPB][NB][135];                                // face vectors [col][3 j + c]
    const int per = (nf + gridDim.x - 1) / gridDim.x;
    const int lo = blockIdx.x * per, hi = min(nf, lo + per);
    if (lo >= hi) return;                                         // (uniform in the block)
    const int d0 = (lo >= c1) + (lo >= c2), d1 = (hi - 1 >= c1) + (hi - 1 >= c2);
    for (int i = threadIdx.x; i < (d1 - d0 + 1) * 2025; i += blockDim.x) sG[i] = G[2025 * d0 + i];
    const int lf = threadIdx.x / 135, k = threadIdx.x - 135 * (threadIdx.x / 135), j = k / 3, c = k - 3 * j;
    const int col0 = blockIdx.y * NB;
    const int ncol = min(NB, LD - col0);
    for (int f0 = lo; f0 < hi; f0 += FPB) {
        const int f = f0 + lf;
        const bool ok = (lf < FPB) && (f < hi);
        int d = 0;
        int64_t dof = 0;
        __syncthreads();                                          // sG loaded / previous group's sX consumed
        if (ok) {
            d = (f >= c1) + (f >= c2);
            dof = 3 * (int64_t)fnode[(int64_t)f * 45 + j] + c;
            for (int q = 0; q < ncol; q++) sX[lf][q][k] = x[dof * LD + col0 + q];
        }
        __syncthreads();
        if (ok) {
            const T* g = sG + 2025 * (d - d0) + 45 * j;
            T a[NB];
#pragma unroll
            for (int q = 0; q < NB; q++) a[q] = 0;
#pragma unroll 5
            for (int jj = 0; jj < 45; jj++) {
                const T gv = g[jj];
#pragma unroll
                for (int q = 0; q < NB; q++)
                    if (q < ncol) a[q] += gv * sX[lf][q][3 * jj + c];
            }
#pragma unroll
            for (int q = 0; q < NB; q++)
                if (q < ncol) atomicAdd(y + dof * LD + col0 + q, a[q]);
        }
    }
}

// Ghost faces, Kronecker form (src_v3 P15, gk): every face's Gram matrix is gamma A (x) m (x) m in the face's (s, u, v)
// node order (s: the five node layers along the face normal, i_d = 0, 1, 2 of fe and i_d = 1, 2 of fn; u, v: the other
// two axes in increasing order; A 5 x 5 with gamma, m the 3 x 3 tangential factor), checked against the templates when
// the state is built (kmf._gk_factors). 495 instead of 2025 multiply-adds per face and component, 34 table values
// instead of three 45 x 45 matrices. 15 threads per face, thread (s, c) with c the component; two faces per warp
// (lanes 0-14 and 16-30), warp-synchronous: the thread gathers the 3 x 3 plane of layer s, applies m along u and v in
// registers and writes the plane to shared memory; then row s of A over the face's five planes and 9 atomic adds.
// fsuv: nf x 45 node indices in (s, u, v) order; tab: A (25) then m (9).
template <typename T, int NB>
__global__ void __launch_bounds__(128)
gk_kernel(const T* __restrict__ x, T* __restrict__ y, const int* __restrict__ fsuv, const T* __restrict__ tab, int nf,
          int LD)
{
    __shared__ T sZ[8][NB][135];                                  // [face of the block][col][27 s + 9 c + 3 u + v]
    const int lane = threadIdx.x & 31, wf = 2 * (threadIdx.x >> 5) + (lane >> 4), l = lane & 15;
    const bool act = l < 15;
    const int s = act ? l / 3 : 0, c = act ? l - 3 * (l / 3) : 0;
    T a[5], m[9];
#pragma unroll
    for (int i = 0; i < 5; i++) a[i] = tab[5 * s + i];
#pragma unroll
    for (int i = 0; i < 9; i++) m[i] = tab[25 + i];
    const int col0 = blockIdx.y * NB;
    const int ncol = min(NB, LD - col0);
    for (int fb = 8 * blockIdx.x; fb < nf; fb += 8 * gridDim.x) {  // (uniform in the block)
        const int f = fb + wf;
        const bool ok = act && f < nf;
        int nodes[9];
        if (ok) {
#pragma unroll
            for (int q = 0; q < 9; q++) nodes[q] = fsuv[(int64_t)f * 45 + 9 * s + q];
            T p[NB][9];
#pragma unroll
            for (int q = 0; q < 9; q++)
#pragma unroll
                for (int k = 0; k < NB; k++) p[k][q] = k < ncol ? x[(3 * (int64_t)nodes[q] + c) * LD + col0 + k] : T(0);
#pragma unroll
            for (int k = 0; k < NB; k++) {
                T w[9];
#pragma unroll
                for (int u = 0; u < 3; u++)
#pragma unroll
                    for (int v = 0; v < 3; v++)
                        w[3 * u + v] = m[3 * u] * p[k][v] + m[3 * u + 1] * p[k][3 + v] + m[3 * u + 2] * p[k][6 + v];
#pragma unroll
                for (int u = 0; u < 3; u++)
#pragma unroll
                    for (int v = 0; v < 3; v++)
                        sZ[wf][k][27 * s + 9 * c + 3 * u + v] = m[3 * v] * w[3 * u] + m[3 * v + 1] * w[3 * u + 1] + m[3 * v + 2] * w[3 * u + 2];
            }
        }
        __syncwarp();
        if (ok) {
#pragma unroll
            for (int k = 0; k < NB; k++) {
                if (k < ncol) {
#pragma unroll
                    for (int q = 0; q < 9; q++) {
                        const T* z = &sZ[wf][k][9 * c + q];
                        const T acc = a[0] * z[0] + a[1] * z[27] + a[2] * z[54] + a[3] * z[81] + a[4] * z[108];
                        atomicAdd(y + (3 * (int64_t)nodes[q] + c) * LD + col0 + k, acc);
                    }
                }
            }
        }
        __syncwarp();                                             // (sZ reused by the next pair)
    }
}

// Ghost faces, Kronecker form for wide products (src_v3 P15b, gkw): lanes over the columns. A group of GW lanes (GW = B
// rounded up to a power of two, at most 32) takes one (face, component c) and GW consecutive columns: each lane gathers
// its column's 45 face values (coalesced across the group), applies m along u and v and A along s in registers (the
// same 495 multiply-adds per face, component and column as gk_kernel, in the same order), and adds the 45 results
// atomically. Float32; for B >= 8 (gk_kernel's column blocks re-read the face for every 4 columns).
template <int GW>
__global__ void __launch_bounds__(128)
gkw_kernel(const float* __restrict__ x, float* __restrict__ y, const int* __restrict__ fsuv, const float* __restrict__ tab,
           int nf, int LD)
{
    const int lane = threadIdx.x & 31, g = lane / GW, l = lane - GW * g;
    const int task = (blockIdx.x * 4 + (threadIdx.x >> 5)) * (32 / GW) + g;   // (face, component)
    if (task >= 3 * nf) return;
    const int f = task / 3, c = task - 3 * f;
    float A[25], m[9];
#pragma unroll
    for (int i = 0; i < 25; i++) A[i] = __ldg(tab + i);
#pragma unroll
    for (int i = 0; i < 9; i++) m[i] = __ldg(tab + 25 + i);
    const int* nf45 = fsuv + (int64_t)f * 45;
    for (int j0 = 0; j0 < LD; j0 += GW) {
        const int col = j0 + l;
        const bool ok = col < LD;
        float v[45];
#pragma unroll
        for (int q = 0; q < 45; q++) v[q] = ok ? x[(3 * (int64_t)__ldg(nf45 + q) + c) * LD + col] : 0.f;
        float z[45];
#pragma unroll
        for (int s = 0; s < 5; s++) {                             // tangential: m along u, then along v (gk_kernel's order)
            float w[9];
#pragma unroll
            for (int u = 0; u < 3; u++)
#pragma unroll
                for (int vv = 0; vv < 3; vv++)
                    w[3 * u + vv] = m[3 * u] * v[9 * s + vv] + m[3 * u + 1] * v[9 * s + 3 + vv] + m[3 * u + 2] * v[9 * s + 6 + vv];
#pragma unroll
            for (int u = 0; u < 3; u++)
#pragma unroll
                for (int vv = 0; vv < 3; vv++)
                    z[9 * s + 3 * u + vv] = m[3 * vv] * w[3 * u] + m[3 * vv + 1] * w[3 * u + 1] + m[3 * vv + 2] * w[3 * u + 2];
        }
        if (ok) {
#pragma unroll
            for (int s = 0; s < 5; s++)
#pragma unroll
                for (int q = 0; q < 9; q++) {
                    const float acc = A[5 * s] * z[q] + A[5 * s + 1] * z[9 + q] + A[5 * s + 2] * z[18 + q] + A[5 * s + 3] * z[27 + q]
                                      + A[5 * s + 4] * z[36 + q];
                    atomicAdd(y + (3 * (int64_t)__ldg(nf45 + 9 * s + q) + c) * LD + col, acc);
                }
        }
    }
}

template <typename T>
static void gk_go(const T* x, T* y, const int* fsuv, const T* tab, int nf, int B, int nb, int blocks, cudaStream_t s)
{
    int gx = (nf + 7) / 8;
    if (blocks > 0 && gx > blocks) gx = blocks;
    if (nb == 1) gk_kernel<T, 1><<<dim3(gx, B), 128, 0, s>>>(x, y, fsuv, tab, nf, B);
    else if (nb == 2) gk_kernel<T, 2><<<dim3(gx, (B + 1) / 2), 128, 0, s>>>(x, y, fsuv, tab, nf, B);
    else gk_kernel<T, 4><<<dim3(gx, (B + 3) / 4), 128, 0, s>>>(x, y, fsuv, tab, nf, B);
}

template <typename T, int FPB, int NB>
static void gf2_go(const T* x, T* y, const int* fnode, const T* G, int nf, int c1, int c2, int B, int blocks,
                   cudaStream_t s)
{
    int gx = (nf + FPB - 1) / FPB;
    if (blocks > 0 && gx > blocks) gx = blocks;
    int minc = nf;
    const int cnt[3] = {c1, c2 - c1, nf - c2};
    for (int d = 0; d < 3; d++) if (cnt[d] > 0 && cnt[d] < minc) minc = cnt[d];
    const int need = (nf + minc - 1) / minc;
    if (gx < need) gx = need;
    gf2_kernel<T, FPB, NB><<<dim3(gx, (B + NB - 1) / NB), 135 * FPB, 0, s>>>(x, y, fnode, G, nf, c1, c2, B);
}

template <typename T, int EPB>
static void kmf_go(const T* x, T* y, const int* nd, const double* w, const double* wx, int nw, const int* widx,
                   const double* wfull, double lam, double mu, int E, int B, int nb, cudaStream_t s)
{
    const int gx = (E + EPB - 1) / EPB;
    if (nb == 1) kmf_kernel<T, EPB, 1><<<dim3(gx, B), 25 * EPB, 0, s>>>(x, y, nd, w, wx, nw, widx, wfull, lam, mu, E, B);
    else if (nb == 2)
        kmf_kernel<T, EPB, 2><<<dim3(gx, (B + 1) / 2), 25 * EPB, 0, s>>>(x, y, nd, w, wx, nw, widx, wfull, lam, mu, E, B);
    else kmf_kernel<T, EPB, 4><<<dim3(gx, (B + 3) / 4), 25 * EPB, 0, s>>>(x, y, nd, w, wx, nw, widx, wfull, lam, mu, E, B);
}

template <typename T, int FPB, int NB>
static void gf_go(const T* x, T* y, const int* nd, const int* fe, const int* fn, const int* fmap, const T* G, int nf,
                  int c1, int c2, int B, int blocks, cudaStream_t s)
{
    int gx = (nf + FPB - 1) / FPB;
    if (blocks > 0 && gx > blocks) gx = blocks;
    int minc = nf;                                                // a block's range at most the smallest direction
    const int cnt[3] = {c1, c2 - c1, nf - c2};
    for (int d = 0; d < 3; d++) if (cnt[d] > 0 && cnt[d] < minc) minc = cnt[d];
    const int need = (nf + minc - 1) / minc;
    if (gx < need) gx = need;
    gf_kernel<T, FPB, NB><<<dim3(gx, (B + NB - 1) / NB), 45 * FPB, 0, s>>>(x, y, nd, fe, fn, fmap, G, nf, c1, c2, B);
}

static int auto_nb(int B) { return B == 1 ? 1 : (B == 2 ? 2 : 4); }
#ifndef GKW_MIN
#define GKW_MIN 8                                                 // gk_launch: gkw_kernel from this many columns
#endif

extern "C" {

int kmf_init(const float* tabL, const float* tabD, const double* tabL64, const double* tabD64)
{
    cudaError_t r = cudaMemcpyToSymbol(cL32, tabL, 15 * sizeof(float));
    if (r == cudaSuccess) r = cudaMemcpyToSymbol(cD32, tabD, 15 * sizeof(float));
    if (r == cudaSuccess) r = cudaMemcpyToSymbol(cL64, tabL64, 15 * sizeof(double));
    if (r == cudaSuccess) r = cudaMemcpyToSymbol(cD64, tabD64, 15 * sizeof(double));
    if (r != cudaSuccess) return (int)r;
    cudaFuncAttributes at;                                        // load the kernels now (not during a graph capture)
    cudaFuncGetAttributes(&at, kmf_kernel<float, 2, 1>); cudaFuncGetAttributes(&at, kmf_kernel<float, 2, 2>);
    cudaFuncGetAttributes(&at, kmf_kernel<float, 2, 4>); cudaFuncGetAttributes(&at, kmf_kernel<float, 4, 1>);
    cudaFuncGetAttributes(&at, kmf_kernel<float, 4, 2>); cudaFuncGetAttributes(&at, kmf_kernel<float, 4, 4>);
    cudaFuncGetAttributes(&at, kmf_kernel<float, 8, 1>); cudaFuncGetAttributes(&at, kmf_kernel<float, 8, 2>);
    cudaFuncGetAttributes(&at, kmf_kernel<float, 8, 4>);
    cudaFuncGetAttributes(&at, kmf_kernel<double, 2, 1>); cudaFuncGetAttributes(&at, kmf_kernel<double, 2, 2>);
    cudaFuncGetAttributes(&at, kmf_kernel<double, 2, 4>); cudaFuncGetAttributes(&at, kmf_kernel<double, 4, 1>);
    cudaFuncGetAttributes(&at, kmf_kernel<double, 4, 2>); cudaFuncGetAttributes(&at, kmf_kernel<double, 4, 4>);
    cudaFuncGetAttributes(&at, gf_kernel<float, 4, 1>); cudaFuncGetAttributes(&at, gf_kernel<float, 4, 2>);
    cudaFuncGetAttributes(&at, gf_kernel<float, 4, 4>); cudaFuncGetAttributes(&at, gf_kernel<float, 8, 1>);
    cudaFuncGetAttributes(&at, gf_kernel<float, 8, 2>); cudaFuncGetAttributes(&at, gf_kernel<float, 8, 4>);
    cudaFuncGetAttributes(&at, gf_kernel<double, 4, 1>); cudaFuncGetAttributes(&at, gf_kernel<double, 4, 2>);
    cudaFuncGetAttributes(&at, gf_kernel<double, 2, 4>);
    cudaFuncGetAttributes(&at, gf2_kernel<float, 2, 1>); cudaFuncGetAttributes(&at, gf2_kernel<float, 2, 2>);
    cudaFuncGetAttributes(&at, gf2_kernel<float, 2, 4>); cudaFuncGetAttributes(&at, gf2_kernel<float, 4, 1>);
    cudaFuncGetAttributes(&at, gf2_kernel<float, 4, 2>); cudaFuncGetAttributes(&at, gf2_kernel<float, 4, 4>);
    cudaFuncGetAttributes(&at, gf2_kernel<double, 2, 1>); cudaFuncGetAttributes(&at, gf2_kernel<double, 2, 2>);
    cudaFuncGetAttributes(&at, gf2_kernel<double, 2, 4>);
    cudaFuncGetAttributes(&at, gk_kernel<float, 1>); cudaFuncGetAttributes(&at, gk_kernel<float, 2>);
    cudaFuncGetAttributes(&at, gk_kernel<float, 4>); cudaFuncGetAttributes(&at, gk_kernel<double, 1>);
    cudaFuncGetAttributes(&at, gk_kernel<double, 2>); cudaFuncGetAttributes(&at, gk_kernel<double, 4>);
    cudaFuncGetAttributes(&at, gkw_kernel<8>); cudaFuncGetAttributes(&at, gkw_kernel<16>); cudaFuncGetAttributes(&at, gkw_kernel<32>);
    return (int)cudaDeviceSynchronize();
}

// nb: columns per block (1, 2 or 4; 0: 1 for one column, 2 for two, else 4)
int kmf_launch(const float* x, float* y, const int* nd, const double* w, const double* wx, int nw, const int* widx,
               const double* wfull, double lam, double mu, int E, int B, int epb, int nb, void* stream)
{
    if (E <= 0 || B <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    if (nb <= 0) nb = auto_nb(B);
    if (epb == 2) kmf_go<float, 2>(x, y, nd, w, wx, nw, widx, wfull, lam, mu, E, B, nb, s);
    else if (epb == 8) kmf_go<float, 8>(x, y, nd, w, wx, nw, widx, wfull, lam, mu, E, B, nb, s);
    else kmf_go<float, 4>(x, y, nd, w, wx, nw, widx, wfull, lam, mu, E, B, nb, s);
    return (int)cudaGetLastError();
}

int kmf_launch64(const double* x, double* y, const int* nd, const double* w, const int* widx, const double* wfull,
                 double lam, double mu, int E, int B, int epb, int nb, void* stream)
{
    if (E <= 0 || B <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    if (nb <= 0) nb = auto_nb(B);
    if (epb == 2) kmf_go<double, 2>(x, y, nd, w, nullptr, E, widx, wfull, lam, mu, E, B, nb, s);
    else kmf_go<double, 4>(x, y, nd, w, nullptr, E, widx, wfull, lam, mu, E, B, nb, s);
    return (int)cudaGetLastError();
}

int gf_launch(const float* x, float* y, const int* nd, const int* fe, const int* fn, const int* fmap, const float* G,
              int nf, int c1, int c2, int B, int fpb, int nb, int blocks, void* stream)
{
    if (nf <= 0 || B <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    if (nb <= 0) nb = auto_nb(B);
    if (fpb == 4) {
        if (nb == 1) gf_go<float, 4, 1>(x, y, nd, fe, fn, fmap, G, nf, c1, c2, B, blocks, s);
        else if (nb == 2) gf_go<float, 4, 2>(x, y, nd, fe, fn, fmap, G, nf, c1, c2, B, blocks, s);
        else gf_go<float, 4, 4>(x, y, nd, fe, fn, fmap, G, nf, c1, c2, B, blocks, s);
    } else {
        if (nb == 1) gf_go<float, 8, 1>(x, y, nd, fe, fn, fmap, G, nf, c1, c2, B, blocks, s);
        else if (nb == 2) gf_go<float, 8, 2>(x, y, nd, fe, fn, fmap, G, nf, c1, c2, B, blocks, s);
        else gf_go<float, 8, 4>(x, y, nd, fe, fn, fmap, G, nf, c1, c2, B, blocks, s);
    }
    return (int)cudaGetLastError();
}

int gf2_launch(const float* x, float* y, const int* fnode, const float* G, int nf, int c1, int c2, int B, int fpb, int nb,
               int blocks, void* stream)
{
    if (nf <= 0 || B <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    if (nb <= 0) nb = auto_nb(B);
    if (fpb == 2) {
        if (nb == 1) gf2_go<float, 2, 1>(x, y, fnode, G, nf, c1, c2, B, blocks, s);
        else if (nb == 2) gf2_go<float, 2, 2>(x, y, fnode, G, nf, c1, c2, B, blocks, s);
        else gf2_go<float, 2, 4>(x, y, fnode, G, nf, c1, c2, B, blocks, s);
    } else {
        if (nb == 1) gf2_go<float, 4, 1>(x, y, fnode, G, nf, c1, c2, B, blocks, s);
        else if (nb == 2) gf2_go<float, 4, 2>(x, y, fnode, G, nf, c1, c2, B, blocks, s);
        else gf2_go<float, 4, 4>(x, y, fnode, G, nf, c1, c2, B, blocks, s);
    }
    return (int)cudaGetLastError();
}

int gf2_launch64(const double* x, double* y, const int* fnode, const double* G, int nf, int c1, int c2, int B, int nb,
                 int blocks, void* stream)
{
    if (nf <= 0 || B <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    if (nb <= 0) nb = auto_nb(B);
    if (nb == 1) gf2_go<double, 2, 1>(x, y, fnode, G, nf, c1, c2, B, blocks, s);
    else if (nb == 2) gf2_go<double, 2, 2>(x, y, fnode, G, nf, c1, c2, B, blocks, s);
    else gf2_go<double, 2, 4>(x, y, fnode, G, nf, c1, c2, B, blocks, s);
    return (int)cudaGetLastError();
}

int gf_launch64(const double* x, double* y, const int* nd, const int* fe, const int* fn, const int* fmap,
                const double* G, int nf, int c1, int c2, int B, int nb, int blocks, void* stream)
{
    if (nf <= 0 || B <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    if (nb <= 0) nb = auto_nb(B);
    if (nb == 1) gf_go<double, 4, 1>(x, y, nd, fe, fn, fmap, G, nf, c1, c2, B, blocks, s);
    else if (nb == 2) gf_go<double, 4, 2>(x, y, nd, fe, fn, fmap, G, nf, c1, c2, B, blocks, s);
    else gf_go<double, 2, 4>(x, y, nd, fe, fn, fmap, G, nf, c1, c2, B, blocks, s);       // (48 KB static limit)
    return (int)cudaGetLastError();
}

int gk_launch(const float* x, float* y, const int* fsuv, const float* tab, int nf, int B, int nb, int blocks, void* stream)
{
    if (nf <= 0 || B <= 0) return 0;
    if (B >= GKW_MIN && nb <= 0) {                                // P15b: lanes over the columns
        cudaStream_t s = (cudaStream_t)stream;
        const int gw = B >= 32 ? 32 : (B > 8 ? 16 : 8);
        const int tasks = 3 * nf, per_block = 4 * (32 / gw);
        const int gx = (tasks + per_block - 1) / per_block;
        if (gw == 32) gkw_kernel<32><<<gx, 128, 0, s>>>(x, y, fsuv, tab, nf, B);
        else if (gw == 16) gkw_kernel<16><<<gx, 128, 0, s>>>(x, y, fsuv, tab, nf, B);
        else gkw_kernel<8><<<gx, 128, 0, s>>>(x, y, fsuv, tab, nf, B);
        return (int)cudaGetLastError();
    }
    if (nb <= 0) nb = auto_nb(B);
    gk_go<float>(x, y, fsuv, tab, nf, B, nb, blocks, (cudaStream_t)stream);
    return (int)cudaGetLastError();
}

int gk_launch64(const double* x, double* y, const int* fsuv, const double* tab, int nf, int B, int nb, int blocks,
                void* stream)
{
    if (nf <= 0 || B <= 0) return 0;
    if (nb <= 0) nb = auto_nb(B);
    gk_go<double>(x, y, fsuv, tab, nf, B, nb, blocks, (cudaStream_t)stream);
    return (int)cudaGetLastError();
}

}
