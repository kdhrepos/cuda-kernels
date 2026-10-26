// softmax harness -- row-wise softmax: x[M][N] (FP32) -> y[M][N] (FP32)
//   y[r][j] = exp(x[r][j] - m_r) / sum_k exp(x[r][k] - m_r),   m_r = max_k x[r][k]
// build and run:
//   nvcc -O3 -arch=sm_86 -o softmax softmax.cu && ./softmax
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#include "../../common/cuda_check.cuh"
#include "../../common/kernel_report.cuh"

// Measured iterations, after one discarded warm-up. Override at build time with
// -DREPS=n.
#ifndef REPS
#define REPS 10
#endif

constexpr int M      = 2048;   // rows
constexpr int N      = 32768;  // row length (about an LLM vocab) -> 128 KB per row
constexpr int BLOCK  = 256;    // one block per row, so grid = M
constexpr double TOL = 1e-4;   // relative error tolerance
static_assert(N % (4 * BLOCK) == 0, "N must be a multiple of 4*BLOCK so float4 divides evenly");

// ===================== your implementation below =====================
// M, N and BLOCK are usable inside the kernels: constexpr scalars reach device code.

__device__ float max_reduction(float m_val) {
    int tx      = threadIdx.x;
    int warp_id = tx / 32;
    int lane_id = tx % 32;

    __shared__ float partial[8];
    for (int s = 16; s > 0; s /= 2) {
        m_val = fmaxf(m_val, __shfl_down_sync(0xFFFFFFFF, m_val, s));
    }

    if (lane_id == 0) partial[warp_id] = m_val;
    __syncthreads();

    if (warp_id == 0) {
        m_val = (tx < 8) ? partial[tx] : -INFINITY;
        for (int s = 4; s > 0; s /= 2) {
            m_val = fmaxf(m_val, __shfl_down_sync(0xFFFFFFFF, m_val, s));
        }
    }

    __shared__ float result;
    if (tx == 0) result = m_val;
    __syncthreads();

    return result;
}

__device__ float sum_reduction(float sum) {
    int tx      = threadIdx.x;
    int warp_id = tx / 32;
    int lane_id = tx % 32;

    __shared__ float partial[8];
    for (int s = 16; s > 0; s /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, s);
    }

    if (lane_id == 0) partial[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (tx < 8) ? partial[tx] : 0.0f;
        for (int s = 4; s > 0; s /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, s);
        }
    }

    __shared__ float result;
    if (tx == 0) result = sum;
    __syncthreads();

    return result;
}

// v0 unfused: three kernels
__global__ void k_rowmax(const float* x, float* rowmax) {
    // TODO: rowmax[row] = max_j x[row][j]
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    float m_val = -INFINITY;

    // step 1: each thread finds the max over its 128 elements
    for (int i = 0; i < 32; i++) {
        int idx             = bx * N + tx * 4 + i * 1024;
        const float4* elem4 = reinterpret_cast<const float4*>(x + idx);
        m_val = fmaxf(m_val, fmaxf(fmaxf(elem4->x, elem4->y), fmaxf(elem4->w, elem4->z)));
    }

    m_val = max_reduction(m_val);

    if (tx == 0) rowmax[bx] = m_val;
}

__global__ void k_rowsumexp(const float* x, const float* rowmax, float* rowsum) {
    // TODO: rowsum[row] = sum_j exp(x[row][j] - rowmax[row])
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    float t_sum = 0.0f;
    float m_val = rowmax[bx];

    for (int i = 0; i < 32; i++) {
        int idx             = bx * N + tx * 4 + i * 1024;
        const float4* elem4 = reinterpret_cast<const float4*>(x + idx);
        t_sum += expf(elem4->x - m_val);
        t_sum += expf(elem4->y - m_val);
        t_sum += expf(elem4->w - m_val);
        t_sum += expf(elem4->z - m_val);
    }

    t_sum = sum_reduction(t_sum);

    if (tx == 0) rowsum[bx] = t_sum;
}

__global__ void k_normalize(const float* x, const float* rowmax, const float* rowsum, float* y) {
    // TODO: y = exp(x - rowmax[row]) / rowsum[row]
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    float m_val = rowmax[bx];
    float sum   = 1.0f / rowsum[bx];

    for (int i = 0; i < 32; i++) {
        int idx    = bx * N + tx * 4 + i * 1024;
        y[idx]     = expf(x[idx] - m_val) * sum;
        y[idx + 1] = expf(x[idx + 1] - m_val) * sum;
        y[idx + 2] = expf(x[idx + 2] - m_val) * sum;
        y[idx + 3] = expf(x[idx + 3] - m_val) * sum;
    }
}

// v1 fused safe softmax: one kernel, reads the row from global 3x (max -> sum -> normalize and
// store)
__global__ void k_softmax_3pass(const float* x, float* y) {
    // TODO
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    float m_val = -INFINITY;
    float t_sum = 0.0f;

    for (int i = 0; i < 32; i++) {
        int idx             = bx * N + tx * 4 + i * 1024;
        const float4* elem4 = reinterpret_cast<const float4*>(x + idx);
        m_val = fmaxf(m_val, fmaxf(fmaxf(elem4->x, elem4->y), fmaxf(elem4->w, elem4->z)));
    }
    m_val = max_reduction(m_val);

    for (int i = 0; i < 32; i++) {
        int idx            = bx * N + tx * 4 + i * 1024;
        const float4 elem4 = *reinterpret_cast<const float4*>(x + idx);
        t_sum += expf(elem4.x - m_val);
        t_sum += expf(elem4.y - m_val);
        t_sum += expf(elem4.z - m_val);
        t_sum += expf(elem4.w - m_val);
    }
    t_sum = sum_reduction(t_sum);

    float inv = 1.0f / t_sum;
    for (int i = 0; i < 32; i++) {
        int idx            = bx * N + tx * 4 + i * 1024;
        const float4 elem4 = *reinterpret_cast<const float4*>(x + idx);
        float4 out;
        out.x                               = expf(elem4.x - m_val) * inv;
        out.y                               = expf(elem4.y - m_val) * inv;
        out.z                               = expf(elem4.z - m_val) * inv;
        out.w                               = expf(elem4.w - m_val) * inv;
        *reinterpret_cast<float4*>(y + idx) = out;
    }
}

struct MaxSum {
    float max_value;  // the max this partial result is referenced to
    float sum;        // sum_j exp(elem_j - max_value)
};

__device__ __forceinline__ MaxSum merge_max_sum(MaxSum a, MaxSum b) {
    float merged_max = fmaxf(a.max_value, b.max_value);
    if (merged_max == -INFINITY) {  // two empty partials: avoids exp(-inf - (-inf)) = NaN
        MaxSum empty = {-INFINITY, 0.0f};
        return empty;
    }
    MaxSum merged;
    merged.max_value = merged_max;
    merged.sum = a.sum * expf(a.max_value - merged_max) + b.sum * expf(b.max_value - merged_max);
    return merged;
}

__device__ MaxSum max_sum_reduction(MaxSum part) {
    int tx      = threadIdx.x;
    int warp_id = tx / 32;
    int lane_id = tx % 32;

    __shared__ float partial_max[8];
    __shared__ float partial_sum[8];
    __shared__ MaxSum result;

    // A. within a warp: pull the peer lane's max and sum separately, then merge
    for (int s = 16; s > 0; s /= 2) {
        MaxSum other;
        other.max_value = __shfl_down_sync(0xFFFFFFFF, part.max_value, s);
        other.sum       = __shfl_down_sync(0xFFFFFFFF, part.sum, s);
        part            = merge_max_sum(part, other);
    }

    if (lane_id == 0) {
        partial_max[warp_id] = part.max_value;
        partial_sum[warp_id] = part.sum;
    }
    __syncthreads();

    // B. warp 0 merges the 8 partials; lanes 8..31 hold the empty partial (-inf, 0)
    if (warp_id == 0) {
        part.max_value = (tx < 8) ? partial_max[tx] : -INFINITY;
        part.sum       = (tx < 8) ? partial_sum[tx] : 0.0f;
        for (int s = 4; s > 0; s /= 2) {
            MaxSum other;
            other.max_value = __shfl_down_sync(0xFFFFFFFF, part.max_value, s);
            other.sum       = __shfl_down_sync(0xFFFFFFFF, part.sum, s);
            part            = merge_max_sum(part, other);
        }
        if (tx == 0) result = part;
    }
    __syncthreads();

    return result;  // every thread in the block gets the same value
}

// v2 fused online softmax: one kernel, reads the row 2x ((max, sum) in one pass -> normalize and
// store)
__global__ void k_softmax_online(const float* x, float* y) {
    // TODO
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    float sum       = 0.0f;
    float max_value = -INFINITY;

    constexpr int ITERS  = N / (4 * BLOCK);  // float4s per thread (32)
    constexpr int STRIDE = 4 * BLOCK;        // elements skipped per iteration (1024)

    MaxSum part = {-INFINITY, 0.0f};
    for (int i = 0; i < ITERS; i++) {
        int idx            = bx * N + tx * 4 + i * STRIDE;
        const float4 elem4 = *reinterpret_cast<const float4*>(x + idx);
        float new_max =
            fmaxf(part.max_value, fmaxf(fmaxf(elem4.x, elem4.y), fmaxf(elem4.z, elem4.w)));

        part.sum = part.sum * expf(part.max_value - new_max) + expf(elem4.x - new_max) +
                   expf(elem4.y - new_max) + expf(elem4.z - new_max) + expf(elem4.w - new_max);
        part.max_value = new_max;
    }

    part      = max_sum_reduction(part);
    max_value = part.max_value;
    sum       = part.sum;

    float inv = 1.0f / sum;
    for (int i = 0; i < ITERS; i++) {
        int idx            = bx * N + tx * 4 + i * STRIDE;
        const float4 elem4 = *reinterpret_cast<const float4*>(x + idx);
        float4 out;
        out.x                               = expf(elem4.x - max_value) * inv;
        out.y                               = expf(elem4.y - max_value) * inv;
        out.z                               = expf(elem4.z - max_value) * inv;
        out.w                               = expf(elem4.w - max_value) * inv;
        *reinterpret_cast<float4*>(y + idx) = out;
    }
}
// ===================== end of your implementation =====================

// ---- harness below; no need to touch ----
template <class F>
static std::vector<float> time_ms(F launch, int warm = 3, int reps = REPS) {
    cudaEvent_t a, b;
    CHECK_CUDA_ERROR(cudaEventCreate(&a));
    CHECK_CUDA_ERROR(cudaEventCreate(&b));
    for (int i = 0; i < warm; ++i) launch();
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    std::vector<float> t(reps);
    for (int i = 0; i < reps; ++i) {
        CHECK_CUDA_ERROR(cudaEventRecord(a));
        launch();
        CHECK_CUDA_ERROR(cudaEventRecord(b));
        CHECK_CUDA_ERROR(cudaEventSynchronize(b));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&t[i], a, b));
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventDestroy(a));
    CHECK_CUDA_ERROR(cudaEventDestroy(b));
    return t;
}

int main(int argc, char** argv) {
    bool csv           = bench::takeFlag(argc, argv, "--csv");
    const size_t elems = (size_t)M * N, bytes = elems * sizeof(float);

    // input: uniform [-100, 100]. Without subtracting the max, exp(x) overflows FP32 (exp(89) >
    // FLT_MAX)
    std::vector<float> hx(elems), hy(elems), ref(elems);
    srand(42);
    for (auto& v : hx) v = rand() / (float)RAND_MAX * 200.f - 100.f;

    // CPU reference, accumulated in double
    for (int r = 0; r < M; ++r) {
        const float* xr = &hx[(size_t)r * N];
        double m = -INFINITY, s = 0;
        for (int j = 0; j < N; ++j) m = std::max(m, (double)xr[j]);
        for (int j = 0; j < N; ++j) s += exp((double)xr[j] - m);
        for (int j = 0; j < N; ++j) ref[(size_t)r * N + j] = (float)(exp((double)xr[j] - m) / s);
    }

    float *dx, *dy, *dmax, *dsum;
    CHECK_CUDA_ERROR(cudaMalloc(&dx, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&dy, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&dmax, M * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&dsum, M * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));

    KernelReport rep("softmax", REPS, 8);
    rep.addConfig("shape", "M=" + std::to_string(M) + " N=" + std::to_string(N) + " (" +
                               bench::humanBytes((double)bytes) + " per tensor)");
    rep.addConfig("traffic", bench::humanBytes(2.0 * bytes) + " (x read once + y written once)");
    rep.addConfig("iterations", std::to_string(REPS) + " reps (+3 warm-up, discarded)");
    {
        char tol[48];
        snprintf(tol, sizeof(tol), "every element vs CPU in double, rel tol %.0e", TOL);
        rep.addConfig("verified", tol);
    }
    int colErr = rep.addColumn("max rel err", 13, 1, true);
    rep.setShowMedian(true);  // the median this harness used to headline
    bool firstVariant = true;
    auto run          = [&](const char* name, auto launch) {
        CHECK_CUDA_ERROR(
            cudaMemset(dy, 0xFF, bytes));  // fill with NaN so a stale result cannot pass
        launch();
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaMemcpy(hy.data(), dy, bytes, cudaMemcpyDeviceToHost));
        double maxerr = 0;
        size_t bad = 0, first = 0;
        for (size_t i = 0; i < elems; ++i) {
            double e = fabs((double)hy[i] - ref[i]) / std::max(fabs((double)ref[i]), 1e-6);
            if (!(e <= TOL)) {
                if (!bad) first = i;
                ++bad;
            }  // NaN lands here too
            else
                maxerr = std::max(maxerr, e);
        }
        int v = rep.addVariant(name);
        rep.setBytes(v, 2.0 * bytes);
        rep.setCell(v, colErr, maxerr);
        rep.setCheck(v, (long long)bad);
        if (firstVariant) {
            rep.setBaseline(v);
            firstVariant = false;
        }

        std::vector<float> samples = time_ms(launch);
        for (size_t i = 0; i < samples.size(); ++i) rep.addSample(v, samples[i]);

        if (bad)
            fprintf(stderr, "[%s] %zu wrong elements, first at row %zu col %zu: got %g, ref %g\n",
                    name, bad, first / N, first % N, hy[first], ref[first]);
    };

    run("unfused", [&] {
        k_rowmax<<<M, BLOCK>>>(dx, dmax);
        k_rowsumexp<<<M, BLOCK>>>(dx, dmax, dsum);
        k_normalize<<<M, BLOCK>>>(dx, dmax, dsum, dy);
    });
    run("fused3", [&] { k_softmax_3pass<<<M, BLOCK>>>(dx, dy); });
    run("online", [&] { k_softmax_online<<<M, BLOCK>>>(dx, dy); });

    rep.print(csv);

    CHECK_CUDA_ERROR(cudaFree(dx));
    CHECK_CUDA_ERROR(cudaFree(dy));
    CHECK_CUDA_ERROR(cudaFree(dmax));
    CHECK_CUDA_ERROR(cudaFree(dsum));
    return rep.passed() ? 0 : 1;
}
