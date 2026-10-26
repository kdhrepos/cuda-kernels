// decode harness: GEMV + KV cache attention (batch 1, one token)
// build: nvcc -O3 -arch=sm_80 -o decoder decoder.cu
// run:   ./decoder [context length T, default 4096] [v2 splits, default 16]
//
// To fill in: the kernels and launch_* functions in the "your implementation" section,
// grid and block included. Allocation, init, CPU reference, timing and checking are done.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../../common/cuda_check.cuh"
#include "../../common/kernel_report.cuh"

// Measured iterations, after one discarded warm-up. Override at build time with
// -DREPS=n.
#ifndef REPS
#define REPS 10
#endif

constexpr int NUM_HEADS  = 32;  // Llama-2-7B sizing
constexpr int HEAD_DIM   = 128;
constexpr int MAX_SPLITS = 64;  // cap on v2 splits, used to size the workspace

// Memory layout (all row-major and contiguous):
//   W    [rows][cols]                 __half   y[r] = sum_c W[r][c] * x[c]
//   x    [cols]                       __half
//   y    [rows]                       float
//   q    [NUM_HEADS][HEAD_DIM]        __half
//   K, V [NUM_HEADS][T][HEAD_DIM]     __half   element (h, t, d) sits at (h * T + t) * HEAD_DIM + d
//   out  [NUM_HEADS][HEAD_DIM]        float
// Per head h: s_t = scale * (q_h . K_h[t]),  p_t = softmax(s)_t,  out_h = sum_t p_t * V_h[t]

// ===================== your implementation =====================

__global__ void gemv_fp16(const __half* W, const __half* x, float* y, int rows, int cols) {
    // TODO
    __half x_reg[128];

    int tx  = threadIdx.x;
    int bid = blockIdx.x;

    int warp_id = tx / 32;
    int lane_id = tx % 32;

    for (int i = 0; i < 16; i++) {
        for (int k = 0; k < 8; k++) {
            int c            = i * 256 + lane_id * 8 + k;
            x_reg[i * 8 + k] = (c < cols) ? x[c] : __float2half(0.0f);
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < 16; i++) {
        int idx        = (bid * 2 + warp_id) * cols + i * 256 + lane_id * 8;
        const float4 v = *reinterpret_cast<const float4*>(W + idx);

        __half2 v1 = *reinterpret_cast<const __half2*>(&v.x);
        __half2 v2 = *reinterpret_cast<const __half2*>(&v.y);
        __half2 v3 = *reinterpret_cast<const __half2*>(&v.z);
        __half2 v4 = *reinterpret_cast<const __half2*>(&v.w);

        sum += __half2float(x_reg[i * 8]) * __half2float(v1.x);
        sum += __half2float(x_reg[i * 8 + 1]) * __half2float(v1.y);
        sum += __half2float(x_reg[i * 8 + 2]) * __half2float(v2.x);
        sum += __half2float(x_reg[i * 8 + 3]) * __half2float(v2.y);
        sum += __half2float(x_reg[i * 8 + 4]) * __half2float(v3.x);
        sum += __half2float(x_reg[i * 8 + 5]) * __half2float(v3.y);
        sum += __half2float(x_reg[i * 8 + 6]) * __half2float(v4.x);
        sum += __half2float(x_reg[i * 8 + 7]) * __half2float(v4.y);
    }

    // reduction
    for (int s = 16; s > 0; s /= 2) sum += __shfl_down_sync(0xFFFFFFFF, sum, s);

    // write result
    if (lane_id == 0) y[bid * 2 + warp_id] = sum;
}

void launch_gemv(const __half* W, const __half* x, float* y, int rows, int cols) {
    // TODO: pick grid and block, then launch gemv_fp16
    dim3 threadsPerBlock(64);
    dim3 blocksPerGrid(2048);

    gemv_fp16<<<blocksPerGrid, threadsPerBlock>>>(W, x, y, rows, cols);
}

__global__ void attn_decode_v1(const __half* q, const __half* K, const __half* V, float* out, int T,
                               float scale) {
    // TODO: one block per head
    int tid     = threadIdx.x;
    int bid     = blockIdx.x;
    int bsize   = blockDim.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;

    // load q
    __half q_reg[128];
    for (int k = 0; k < 4; k++) {
        int c    = lane_id * 4 + k;
        q_reg[c] = (c < HEAD_DIM) ? q[(bid * HEAD_DIM + c)] : __float2half(0.0f);
    }

    // Q*K
    float sum;
    extern __shared__ float scores[];
    for (int i = 0; i < 512; i++) {
        sum            = 0.0f;
        int idx        = (bid * T + i * 8 + warp_id) * HEAD_DIM + lane_id * 4;
        const float2 v = *reinterpret_cast<const float2*>(K + idx);

        __half2 v1 = *reinterpret_cast<const __half2*>(&v.x);
        __half2 v2 = *reinterpret_cast<const __half2*>(&v.y);

        sum += __half2float(q_reg[lane_id * 4]) * __half2float(v1.x);
        sum += __half2float(q_reg[lane_id * 4 + 1]) * __half2float(v1.y);
        sum += __half2float(q_reg[lane_id * 4 + 2]) * __half2float(v2.x);
        sum += __half2float(q_reg[lane_id * 4 + 3]) * __half2float(v2.y);

        for (int s = 16; s > 0; s /= 2) sum += __shfl_down_sync(0xFFFFFFFF, sum, s);

        if (lane_id == 0) scores[i * 8 + warp_id] = sum;  // i * (num_warp) + warp_id
    }
    __syncthreads();

    // softmax
    __shared__ float red[32];  // one slot per warp (max 1024 threads = 32 warps)
    __shared__ float row_max, row_sum;
    int num_warps = bsize / 32;

    float m = -INFINITY;
    for (int t = tid; t < T; t += bsize) {
        float s   = scores[t] * scale;
        scores[t] = s;
        m         = fmaxf(m, s);
    }
    for (int s = 16; s > 0; s /= 2) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, s));
    if (lane_id == 0) red[warp_id] = m;
    __syncthreads();

    if (warp_id == 0) {
        m = (lane_id < num_warps) ? red[lane_id] : -INFINITY;
        for (int s = 16; s > 0; s /= 2) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, s));
        if (lane_id == 0) row_max = m;
    }
    __syncthreads();

    m       = row_max;
    float l = 0.0f;
    for (int t = tid; t < T; t += bsize) {
        float e   = expf(scores[t] - m);
        scores[t] = e;
        l += e;
    }

    for (int s = 16; s > 0; s /= 2) l += __shfl_xor_sync(0xFFFFFFFF, l, s);

    if (lane_id == 0) red[warp_id] = l;
    __syncthreads();

    if (warp_id == 0) {
        l = (lane_id < num_warps) ? red[lane_id] : 0.0f;
        for (int s = 16; s > 0; s /= 2) l += __shfl_xor_sync(0xFFFFFFFF, l, s);
        if (lane_id == 0) row_sum = l;
    }
    __syncthreads();

    float inv = 1.0f / row_sum;
    for (int t = tid; t < T; t += bsize) scores[t] *= inv;
    __syncthreads();

    // weighted sum with V
    float sum1 = 0.0f, sum2 = 0.0f;
    float sum3 = 0.0f, sum4 = 0.0f;
    for (int i = 0; i < 512; i++) {
        int idx        = (bid * T + i * 8 + warp_id) * HEAD_DIM + lane_id * 4;
        const float2 v = *reinterpret_cast<const float2*>(V + idx);

        __half2 v1 = *reinterpret_cast<const __half2*>(&v.x);
        __half2 v2 = *reinterpret_cast<const __half2*>(&v.y);

        float score = scores[i * 8 + warp_id];
        sum1 += score * __half2float(v1.x);
        sum2 += score * __half2float(v1.y);
        sum3 += score * __half2float(v2.x);
        sum4 += score * __half2float(v2.y);
    }

    __shared__ float out_s[8][HEAD_DIM];  // [num warps][HEAD_DIM];
    out_s[warp_id][lane_id * 4]     = sum1;
    out_s[warp_id][lane_id * 4 + 1] = sum2;
    out_s[warp_id][lane_id * 4 + 2] = sum3;
    out_s[warp_id][lane_id * 4 + 3] = sum4;
    __syncthreads();

    if (warp_id == 0) {
        float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
        for (int i = 0; i < 8; i++) {
            o0 += out_s[i][lane_id * 4];
            o1 += out_s[i][lane_id * 4 + 1];
            o2 += out_s[i][lane_id * 4 + 2];
            o3 += out_s[i][lane_id * 4 + 3];
        }
        *reinterpret_cast<float4*>(out + bid * HEAD_DIM + lane_id * 4) =
            make_float4(o0, o1, o2, o3);
    }
}

void launch_attn_v1(const __half* q, const __half* K, const __half* V, float* out, int T,
                    float scale) {
    // TODO
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(32);

    size_t smemSize = T * sizeof(float);
    attn_decode_v1<<<blocksPerGrid, threadsPerBlock, smemSize>>>(q, K, V, out, T, scale);
}

__global__ void attn_decode_v2_split(const __half* q, const __half* K, const __half* V,
                                     float* workspace, int T, int num_splits, float scale) {
    // TODO: one block per head
    int tid        = threadIdx.x;
    int bid        = blockIdx.x;
    int hid        = bid / num_splits;  // head id
    int bsize      = blockDim.x;
    int split_size = T / num_splits;
    int n_warp     = bsize / 32;  // number of warps
    int lane_id    = tid % 32;
    int warp_id    = tid / 32;

    // load q
    __half q_reg[HEAD_DIM];
    for (int k = 0; k < 4; k++) {
        int c    = lane_id * 4 + k;
        q_reg[c] = (c < HEAD_DIM) ? q[(hid * HEAD_DIM + c)] : __float2half(0.0f);
    }

    // Q*K
    float sum;
    extern __shared__ float scores[];
    for (int i = 0; i < (split_size / n_warp); i++) {  // split_size: 256, num_warp: 8
        sum     = 0.0f;
        int idx = (hid * T + (bid % num_splits) * split_size + i * n_warp + warp_id) * HEAD_DIM +
                  lane_id * 4;
        const float2 v = *reinterpret_cast<const float2*>(K + idx);

        __half2 v1 = *reinterpret_cast<const __half2*>(&v.x);
        __half2 v2 = *reinterpret_cast<const __half2*>(&v.y);

        sum += __half2float(q_reg[lane_id * 4]) * __half2float(v1.x);
        sum += __half2float(q_reg[lane_id * 4 + 1]) * __half2float(v1.y);
        sum += __half2float(q_reg[lane_id * 4 + 2]) * __half2float(v2.x);
        sum += __half2float(q_reg[lane_id * 4 + 3]) * __half2float(v2.y);

        for (int s = 16; s > 0; s /= 2) sum += __shfl_down_sync(0xFFFFFFFF, sum, s);

        if (lane_id == 0) scores[i * n_warp + warp_id] = sum * scale;  // i * (num_warp) + warp_id
    }
    __syncthreads();

    // softmax
    __shared__ float smem[8];  // number of warps
    __shared__ float row_max, row_sum;

    float m = scores[warp_id * 32 + lane_id];
    float l = scores[warp_id * 32 + lane_id];
    __syncthreads();

    // max inside the warp
    for (int s = 16; s > 0; s /= 2) m = fmaxf(m, __shfl_down_sync(0xFFFFFFFF, m, s));

    // max per warp
    if (lane_id == 0) smem[warp_id] = m;
    __syncthreads();

    // total max
    if (warp_id == 0) {
        m = (lane_id < n_warp) ? smem[lane_id] : -INFINITY;

        for (int s = 16; s > 0; s /= 2) m = fmaxf(m, __shfl_down_sync(0xFFFFFFFF, m, s));
        if (lane_id == 0) row_max = m;
    }
    __syncthreads();

    // sum inside the warp
    l                              = expf(l - row_max);
    scores[warp_id * 32 + lane_id] = l;
    for (int s = 16; s > 0; s /= 2) l += __shfl_down_sync(0xFFFFFFFF, l, s);

    // sum per warp
    if (lane_id == 0) smem[warp_id] = l;
    __syncthreads();

    // total sum
    if (warp_id == 0) {
        l = (lane_id < n_warp) ? smem[lane_id] : 0.0f;
        for (int s = 16; s > 0; s /= 2) l += __shfl_down_sync(0xFFFFFFFF, l, s);
        if (lane_id == 0) row_sum = l;
    }
    __syncthreads();

    // score * V
    float sum1 = 0.0f, sum2 = 0.0f;
    float sum3 = 0.0f, sum4 = 0.0f;
    for (int i = 0; i < (split_size / n_warp); i++) {
        int idx = (hid * T + (bid % num_splits) * split_size + i * n_warp + warp_id) * HEAD_DIM +
                  lane_id * 4;
        const float2 v = *reinterpret_cast<const float2*>(V + idx);

        __half2 v1 = *reinterpret_cast<const __half2*>(&v.x);
        __half2 v2 = *reinterpret_cast<const __half2*>(&v.y);

        float score = scores[i * n_warp + warp_id];
        sum1 += score * __half2float(v1.x);
        sum2 += score * __half2float(v1.y);
        sum3 += score * __half2float(v2.x);
        sum4 += score * __half2float(v2.y);
    }

    __shared__ float out_s[8][HEAD_DIM];  // [num warps][HEAD_DIM];
    out_s[warp_id][lane_id * 4]     = sum1;
    out_s[warp_id][lane_id * 4 + 1] = sum2;
    out_s[warp_id][lane_id * 4 + 2] = sum3;
    out_s[warp_id][lane_id * 4 + 3] = sum4;
    __syncthreads();

    if (warp_id == 0) {
        float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
        for (int i = 0; i < n_warp; i++) {
            o0 += out_s[i][lane_id * 4];
            o1 += out_s[i][lane_id * 4 + 1];
            o2 += out_s[i][lane_id * 4 + 2];
            o3 += out_s[i][lane_id * 4 + 3];
        }
        workspace[bid * (HEAD_DIM + 2) + lane_id * 4]     = o0;
        workspace[bid * (HEAD_DIM + 2) + lane_id * 4 + 1] = o1;
        workspace[bid * (HEAD_DIM + 2) + lane_id * 4 + 2] = o2;
        workspace[bid * (HEAD_DIM + 2) + lane_id * 4 + 3] = o3;

        if (lane_id == 0) {
            workspace[bid * (HEAD_DIM + 2) + HEAD_DIM]     = row_max;
            workspace[bid * (HEAD_DIM + 2) + HEAD_DIM + 1] = row_sum;
        }
    }
}

__global__ void attn_decode_v2_merge(const __half* q, const __half* K, const __half* V, float* out,
                                     float* workspace, int T, int num_splits) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    float max   = -INFINITY;
    float acc   = 0.0f;
    float denom = 0.0f;

    __shared__ float partial_max_s[MAX_SPLITS];
    __shared__ float sum_s[MAX_SPLITS];
    for (int i = 0; i < num_splits; i++) {
        partial_max_s[i] = workspace[(bid * num_splits + i) * (HEAD_DIM + 2) + HEAD_DIM];
        sum_s[i]         = workspace[(bid * num_splits + i) * (HEAD_DIM + 2) + HEAD_DIM + 1];
    }
    __syncthreads();

    for (int i = 0; i < num_splits; i++) max = fmaxf(max, partial_max_s[i]);
    __syncthreads();

    for (int i = 0; i < num_splits; i++)
        partial_max_s[i] =
            expf(workspace[(bid * num_splits + i) * (HEAD_DIM + 2) + HEAD_DIM] - max);
    __syncthreads();

    for (int i = 0; i < num_splits; i++) denom += partial_max_s[i] * sum_s[i];

    for (int i = 0; i < num_splits; i++) {
        acc += workspace[(bid * num_splits + i) * (HEAD_DIM + 2) + tid] * partial_max_s[i];
    }

    out[bid * HEAD_DIM + tid] = acc / denom;
}

// v2: pass 1 writes a partial result per (head, split) into the workspace; pass 2 combines
// the splits per head. Workspace: float[NUM_HEADS * MAX_SPLITS * (HEAD_DIM + 2)], layout is yours.
void launch_attn_v2(const __half* q, const __half* K, const __half* V, float* out, float* workspace,
                    int T, int num_splits, float scale) {
    // TODO: define the two kernels above and launch them in order here
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(num_splits * NUM_HEADS);

    size_t smemSize = (T / num_splits) * sizeof(float);
    attn_decode_v2_split<<<blocksPerGrid, threadsPerBlock, smemSize>>>(q, K, V, workspace, T,
                                                                       num_splits, scale);

    threadsPerBlock.x = 128;
    blocksPerGrid.x   = NUM_HEADS;
    attn_decode_v2_merge<<<blocksPerGrid, threadsPerBlock>>>(q, K, V, out, workspace, T,
                                                             num_splits);
}

// ======================================================

static float frand() {
    return 2.0f * (float)rand() / (float)RAND_MAX - 1.0f;
}  // [-1, 1]

// Mean ms over iters back-to-back runs, after one warm-up
template <typename F>
static float time_ms(F run, int iters) {
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    run();
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) run();
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaGetLastError());
    float ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    return ms / iters;
}

static double max_abs_err(const std::vector<float>& got, const std::vector<double>& ref) {
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        double e = std::fabs((double)got[i] - ref[i]);
        if (!(e <= worst)) worst = e;  // written so NaN is caught too
    }
    return worst;
}

// One row of a KernelReport: the timing, the traffic it implies, and how far
// the result drifted from the CPU reference.
static void addRow(KernelReport& rep, int v, int colErr, float ms, double bytes, double err,
                   double tol) {
    rep.setBytes(v, bytes);
    rep.addSample(v, ms);
    rep.setCell(v, colErr, err);
    rep.setCheck(v, err <= tol ? 0 : 1);
}

int main(int argc, char** argv) {
    bool csv       = bench::takeFlag(argc, argv, "--csv");
    const int rows = 4096, cols = 4096;
    const int T          = (argc > 1) ? atoi(argv[1]) : 4096;
    const int num_splits = (argc > 2) ? atoi(argv[2]) : 16;
    const float scale    = 1.0f / sqrtf((float)HEAD_DIM);
    if (T <= 0 || num_splits <= 0 || num_splits > MAX_SPLITS) {
        fprintf(stderr, "T > 0, 1 <= num_splits <= %d\n", MAX_SPLITS);
        return 1;
    }
    srand(0);

    // ---------------- GEMV ----------------
    const size_t nW = (size_t)rows * cols;
    std::vector<__half> hW(nW), hx(cols);
    for (size_t i = 0; i < nW; ++i) hW[i] = __float2half(frand());
    for (int i = 0; i < cols; ++i) hx[i] = __float2half(frand());

    std::vector<double> ref_y(rows);
    for (int r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (int c = 0; c < cols; ++c)
            acc += (double)__half2float(hW[(size_t)r * cols + c]) * (double)__half2float(hx[c]);
        ref_y[r] = acc;
    }

    __half *dW, *dx;
    float* dy;
    CHECK_CUDA_ERROR(cudaMalloc(&dW, nW * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&dx, cols * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&dy, rows * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(dW, hW.data(), nW * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dx, hx.data(), cols * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(dy, 0, rows * sizeof(float)));

    float ms_gemv = time_ms([&] { launch_gemv(dW, dx, dy, rows, cols); }, REPS);
    std::vector<float> hy(rows);
    CHECK_CUDA_ERROR(cudaMemcpy(hy.data(), dy, rows * sizeof(float), cudaMemcpyDeviceToHost));
    double bytes_gemv = (double)nW * 2 + (double)cols * 2 + (double)rows * 4;

    // ---------------- attention ----------------
    const size_t nKV = (size_t)NUM_HEADS * T * HEAD_DIM;
    const int nQ     = NUM_HEADS * HEAD_DIM;
    std::vector<__half> hq(nQ), hK(nKV), hV(nKV);
    for (int i = 0; i < nQ; ++i) hq[i] = __float2half(frand());
    for (size_t i = 0; i < nKV; ++i) hK[i] = __float2half(frand());
    for (size_t i = 0; i < nKV; ++i) hV[i] = __float2half(frand());

    std::vector<double> ref_out(nQ, 0.0), s(T);
    for (int h = 0; h < NUM_HEADS; ++h) {
        double m = -INFINITY;
        for (int t = 0; t < T; ++t) {
            double dot = 0.0;
            for (int d = 0; d < HEAD_DIM; ++d)
                dot += (double)__half2float(hq[h * HEAD_DIM + d]) *
                       (double)__half2float(hK[((size_t)h * T + t) * HEAD_DIM + d]);
            s[t] = dot * scale;
            if (s[t] > m) m = s[t];
        }
        double l = 0.0;
        for (int t = 0; t < T; ++t) {
            s[t] = std::exp(s[t] - m);
            l += s[t];
        }
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < HEAD_DIM; ++d)
                ref_out[h * HEAD_DIM + d] +=
                    s[t] * (double)__half2float(hV[((size_t)h * T + t) * HEAD_DIM + d]);
        for (int d = 0; d < HEAD_DIM; ++d) ref_out[h * HEAD_DIM + d] /= l;
    }

    __half *dq, *dK, *dV;
    float *dout, *dws;
    CHECK_CUDA_ERROR(cudaMalloc(&dq, nQ * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&dK, nKV * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&dV, nKV * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&dout, nQ * sizeof(float)));
    CHECK_CUDA_ERROR(
        cudaMalloc(&dws, (size_t)NUM_HEADS * MAX_SPLITS * (HEAD_DIM + 2) * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(dq, hq.data(), nQ * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dK, hK.data(), nKV * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dV, hV.data(), nKV * sizeof(__half), cudaMemcpyHostToDevice));

    double bytes_attn = 2.0 * nKV * 2 + (double)nQ * 2 + (double)nQ * 4;  // K + V + q + out
    std::vector<float> hout(nQ);

    CHECK_CUDA_ERROR(cudaMemset(dout, 0, nQ * sizeof(float)));
    float ms_v1 = time_ms([&] { launch_attn_v1(dq, dK, dV, dout, T, scale); }, REPS);
    CHECK_CUDA_ERROR(cudaMemcpy(hout.data(), dout, nQ * sizeof(float), cudaMemcpyDeviceToHost));
    double err_v1 = max_abs_err(hout, ref_out);

    CHECK_CUDA_ERROR(cudaMemset(dout, 0, nQ * sizeof(float)));
    float ms_v2 =
        time_ms([&] { launch_attn_v2(dq, dK, dV, dout, dws, T, num_splits, scale); }, REPS);
    CHECK_CUDA_ERROR(cudaMemcpy(hout.data(), dout, nQ * sizeof(float), cudaMemcpyDeviceToHost));
    double err_v2 = max_abs_err(hout, ref_out);

    // ---------------- report ----------------
    KernelReport gemvRep("decoder gemv", REPS, 8);
    gemvRep.addConfig("shape", std::to_string(rows) + " x " + std::to_string(cols) +
                                   " (FP16 weights " + bench::humanBytes(nW * 2.0) + ")");
    gemvRep.addConfig("iterations", std::to_string(REPS) + " reps (+1 warm-up), mean");
    gemvRep.addConfig("verified", "vs CPU in double, abs tol 1e-2");
    int gemvErr = gemvRep.addColumn("max err", 10, 2, true);
    int vGemv   = gemvRep.addVariant("gemv");
    addRow(gemvRep, vGemv, gemvErr, ms_gemv, bytes_gemv, max_abs_err(hy, ref_y), 1e-2);
    gemvRep.print(csv);
    if (!csv) std::printf("\n");

    KernelReport attnRep("decoder attention", REPS, 8);
    attnRep.addConfig("heads", std::to_string(NUM_HEADS) + " x head_dim " +
                                   std::to_string(HEAD_DIM) + ", T " + std::to_string(T));
    attnRep.addConfig("kv cache", bench::humanBytes(2.0 * nKV * 2));
    attnRep.addConfig("v2 splits", std::to_string(num_splits));
    attnRep.addConfig("iterations", std::to_string(REPS) + " reps (+1 warm-up), mean");
    attnRep.addConfig("verified", "vs CPU in double, abs tol 1e-3");
    int attnErr = attnRep.addColumn("max err", 10, 2, true);
    int vV1     = attnRep.addVariant("v1");
    int vV2     = attnRep.addVariant("v2 split");
    addRow(attnRep, vV1, attnErr, ms_v1, bytes_attn, err_v1, 1e-3);
    addRow(attnRep, vV2, attnErr, ms_v2, bytes_attn, err_v2, 1e-3);
    attnRep.setBaseline(vV1);
    attnRep.print(csv);

    CHECK_CUDA_ERROR(cudaFree(dW));
    CHECK_CUDA_ERROR(cudaFree(dx));
    CHECK_CUDA_ERROR(cudaFree(dy));
    CHECK_CUDA_ERROR(cudaFree(dq));
    CHECK_CUDA_ERROR(cudaFree(dK));
    CHECK_CUDA_ERROR(cudaFree(dV));
    CHECK_CUDA_ERROR(cudaFree(dout));
    CHECK_CUDA_ERROR(cudaFree(dws));
    return (gemvRep.passed() && attnRep.passed()) ? 0 : 1;
}
