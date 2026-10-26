// RMSNorm harness -- row-wise RMSNorm: x[M][N] (FP32), gamma[N] -> y[M][N] (FP32)
//   y[r][j] = x[r][j] * gamma[j] / sqrt(mean_j(x[r][j]^2) + EPS)
// build and run:
//   nvcc -O3 -arch=sm_86 -o rms_norm rms_norm.cu && ./rms_norm
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

constexpr int    M     = 16384;   // rows (tokens)
constexpr int    N     = 4096;    // row length (hidden size, Llama-7B) -> 16 KB per row
constexpr int    BLOCK = 256;     // one block per row, so grid = M
constexpr float  EPS   = 1e-6f;
constexpr double TOL   = 1e-4;    // relative error tolerance
static_assert(N % (4 * BLOCK) == 0, "N must be a multiple of 4*BLOCK so float4 divides evenly");

// ===================== your implementation below =====================
// M, N, BLOCK and EPS are usable inside the kernels.

__device__ float sum_reduction(
    float sum
) {
    int tx = threadIdx.x;
    int warp_id = tx / 32;
    int lane_id = tx % 32;

    __shared__ float partial[8];
    for (int s=16; s>0; s/=2){
        sum += __shfl_down_sync(0xFFFFFFFF, sum, s);
    }

    if (lane_id == 0)
        partial[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (tx < 8) ? partial[tx] : 0.0f;
        for (int s=4; s>0; s/=2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, s);
        }
    }

    __shared__ float result;
    if (tx == 0) 
        result = sum;
    __syncthreads();

    return result;
}

// v0 unfused: two kernels
__global__ void k_inv_rms(
    const float* x, 
    float* inv_rms
) {
    // TODO: inv_rms[row] = 1 / sqrt(mean_j x[row][j]^2 + EPS)
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    
    float sum = 0.0f;
    
    for (int i=0; i<4; i++) {
        int idx = bx*N + tx*4 + i*1024;
        const float4 elem4 = *reinterpret_cast<const float4*>(x + idx);
        sum += elem4.x * elem4.x;
        sum += elem4.y * elem4.y;
        sum += elem4.w * elem4.w;
        sum += elem4.z * elem4.z;
    }
    sum = sum_reduction(sum);
    float mean_sq_eps = sum / N + EPS;

    if (tx == 0)
        inv_rms[bx] = rsqrtf(mean_sq_eps);
}

__global__ void k_scale(
    const float* x, 
    const float* inv_rms, 
    const float* gamma, 
    float* y
) {
    // TODO: y = x * inv_rms[row] * gamma
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    
    float rms = inv_rms[bx];

    for (int i=0; i<4; i++) {
        int idx = bx*N + tx*4 + i*1024;
        int gmm_idx = tx*4 + i*1024;
        float4 out;
        const float4 x_val = *reinterpret_cast<const float4*>(x + idx);
        const float4 gmm_val = *reinterpret_cast<const float4*>(gamma + gmm_idx);
        out.x   = x_val.x * rms * gmm_val.x;
        out.y   = x_val.y * rms * gmm_val.y;
        out.w   = x_val.w * rms * gmm_val.w;
        out.z   = x_val.z * rms * gmm_val.z;
        *reinterpret_cast<float4*>(y + idx) = out;
    }
}

// v1 fused: one kernel, reads the row from global once and keeps it in registers to normalize
__global__ void k_rmsnorm_fused(const float* x, const float* gamma, float* y) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    
    float sum = 0.0f;
    float elem[16] = {0.0f, };
    
    for (int i=0; i<4; i++) {
        int idx = bx*N + tx*4 + i*1024;
        const float4 elem4 = *reinterpret_cast<const float4*>(x + idx);
        elem[i*4]   = elem4.x;
        elem[i*4+1] = elem4.y;
        elem[i*4+2] = elem4.w;
        elem[i*4+3] = elem4.z;

        sum += elem[i*4] * elem[i*4];
        sum += elem[i*4+1] * elem[i*4+1];
        sum += elem[i*4+2] * elem[i*4+2];
        sum += elem[i*4+3] * elem[i*4+3];
    }
    sum = sum_reduction(sum);
    float mean_sq_eps = sum / N + EPS;
    float inv = rsqrtf(mean_sq_eps);

    for (int i=0; i<4; i++) {
        int idx = bx*N + tx*4 + i*1024;
        int gmm_idx = tx*4 + i*1024;
        const float4 gmm_val = *reinterpret_cast<const float4*>(gamma + gmm_idx);
        float4 out;
        out.x   = elem[i*4] * inv * gmm_val.x;
        out.y   = elem[i*4+1] * inv * gmm_val.y;
        out.w   = elem[i*4+2] * inv * gmm_val.w;
        out.z   = elem[i*4+3] * inv * gmm_val.z;
        *reinterpret_cast<float4*>(y + idx) = out;
    }
}
// ===================== end of your implementation =====================


// ---- harness below; no need to touch ----
template <class F>
static std::vector<float> time_ms(F launch, int warm = 3, int reps = REPS) {
    cudaEvent_t a, b;
    CHECK_CUDA_ERROR(cudaEventCreate(&a)); CHECK_CUDA_ERROR(cudaEventCreate(&b));
    for (int i = 0; i < warm; ++i) launch();
    CHECK_CUDA_ERROR(cudaGetLastError()); CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    std::vector<float> t(reps);
    for (int i = 0; i < reps; ++i) {
        CHECK_CUDA_ERROR(cudaEventRecord(a)); launch(); CHECK_CUDA_ERROR(cudaEventRecord(b));
        CHECK_CUDA_ERROR(cudaEventSynchronize(b)); CHECK_CUDA_ERROR(cudaEventElapsedTime(&t[i], a, b));
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventDestroy(a)); CHECK_CUDA_ERROR(cudaEventDestroy(b));
    return t;
}

int main(int argc, char** argv) {
    bool csv = bench::takeFlag(argc, argv, "--csv");
    const size_t elems = (size_t)M * N, bytes = elems * sizeof(float);

    std::vector<float> hx(elems), hg(N), hy(elems), ref(elems);
    srand(42);
    for (auto& v : hx) v = rand() / (float)RAND_MAX * 4.f - 2.f;   // [-2, 2]
    for (auto& v : hg) v = rand() / (float)RAND_MAX + 0.5f;         // [0.5, 1.5]

    // CPU reference, accumulated in double
    for (int r = 0; r < M; ++r) {
        const float* xr = &hx[(size_t)r * N];
        double ss = 0;
        for (int j = 0; j < N; ++j) ss += (double)xr[j] * xr[j];
        double inv = 1.0 / sqrt(ss / N + EPS);
        for (int j = 0; j < N; ++j) ref[(size_t)r * N + j] = (float)(xr[j] * inv * hg[j]);
    }

    float *dx, *dg, *dy, *dinv;
    CHECK_CUDA_ERROR(cudaMalloc(&dx, bytes)); 
    CHECK_CUDA_ERROR(cudaMalloc(&dy, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&dg, N * sizeof(float))); 
    CHECK_CUDA_ERROR(cudaMalloc(&dinv, M * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dg, hg.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    KernelReport rep("rmsnorm", REPS, 8);
    rep.addConfig("shape", "M=" + std::to_string(M) + " N=" + std::to_string(N)
                           + " (" + bench::humanBytes((double)bytes) + " per tensor)");
    rep.addConfig("traffic", bench::humanBytes(2.0 * bytes)
                             + " (x read once + y written once, gamma excluded)");
    rep.addConfig("iterations", std::to_string(REPS)
                            + " reps (+3 warm-up, discarded)");
    {
        char tol[48];
        snprintf(tol, sizeof(tol), "every element vs CPU in double, rel tol %.0e", TOL);
        rep.addConfig("verified", tol);
    }
    int colErr = rep.addColumn("max rel err", 13, 1, true);
    rep.setShowMedian(true);   // the median this harness used to headline
    bool firstVariant = true;
    auto run = [&](const char* name, auto launch) {
        CHECK_CUDA_ERROR(cudaMemset(dy, 0xFF, bytes));     // fill with NaN so a stale result cannot pass
        launch();
        CHECK_CUDA_ERROR(cudaGetLastError()); 
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaMemcpy(hy.data(), dy, bytes, cudaMemcpyDeviceToHost));
        double maxerr = 0; size_t bad = 0, first = 0;
        for (size_t i = 0; i < elems; ++i) {
            double e = fabs((double)hy[i] - ref[i]) / std::max(fabs((double)ref[i]), 1e-6);
            if (!(e <= TOL)) { if (!bad) first = i; ++bad; }   // NaN lands here too
            else maxerr = std::max(maxerr, e);
        }
        int v = rep.addVariant(name);
        rep.setBytes(v, 2.0 * bytes);
        rep.setCell(v, colErr, maxerr);
        rep.setCheck(v, (long long)bad);
        if (firstVariant) { rep.setBaseline(v); firstVariant = false; }

        std::vector<float> samples = time_ms(launch);
        for (size_t i = 0; i < samples.size(); ++i) rep.addSample(v, samples[i]);

        if (bad) fprintf(stderr, "[%s] %zu wrong elements, first at row %zu col %zu: got %g, ref %g\n",
                         name, bad, first / N, first % N, hy[first], ref[first]);
    };

    run("unfused", [&] {
        k_inv_rms<<<M, BLOCK>>>(dx, dinv);
        k_scale<<<M, BLOCK>>>(dx, dinv, dg, dy);
    });
    run("fused", [&] { k_rmsnorm_fused<<<M, BLOCK>>>(dx, dg, dy); });

    rep.print(csv);

    CHECK_CUDA_ERROR(cudaFree(dx)); CHECK_CUDA_ERROR(cudaFree(dy)); CHECK_CUDA_ERROR(cudaFree(dg)); CHECK_CUDA_ERROR(cudaFree(dinv));
    return rep.passed() ? 0 : 1;
}
