// weight-only INT8 GEMV harness
// The two kernels and their launch config (marked TODO) are yours to fill in; data generation,
// quantization, CPU reference and timing are done. Build: nvcc -O3 -arch=sm_80 -o w8_gemv w8_gemv.cu
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_fp16.h>

#include "../../common/cuda_check.cuh"
#include "../../common/kernel_report.cuh"

// Measured iterations, after one discarded warm-up. Override at build time with
// -DREPS=n.
#ifndef REPS
#define REPS 10
#endif

#define CHECK(call) CHECK_CUDA_ERROR(call)

constexpr int ROWS   = 4096;
constexpr int COLS   = 4096;
constexpr int WARMUP = 1;
constexpr int NBUF   = 4;        // W copies. Each rep reads a different one so L2 (6 MB) cannot serve it
constexpr double PEAK_GBS = 826.0; // measured sequential-read bandwidth

// ============================================================================
// TODO 1: paste in the FP16 GEMV kernel
//   W [rows x cols] FP16 row-major, x [cols] FP16 -> y [rows] FP32
// ============================================================================
__global__ void gemv_fp16(const __half* W, const __half* x, float* y, int rows, int cols)
{
    __half x_reg[128];

    int tx = threadIdx.x;
    int bid = blockIdx.x;

    int warp_id = tx / 32;
    int lane_id = tx % 32;

    for (int i = 0; i < 16; i++) {
        for (int k = 0; k < 8; k++) {
            int c = i*256 + lane_id*8 + k;
            x_reg[i*8 + k] = (c < cols) ? x[c] : __float2half(0.0f);
        }
    }

    float sum = 0.0f;
    for (int i=0; i<16; i++) {
        int idx = (bid*2 + warp_id)*cols + i*256 + lane_id*8;
        const float4 v = *reinterpret_cast<const float4*>(W + idx);

        __half2 v1 = *reinterpret_cast<const __half2*>(&v.x);
        __half2 v2 = *reinterpret_cast<const __half2*>(&v.y);
        __half2 v3 = *reinterpret_cast<const __half2*>(&v.z);
        __half2 v4 = *reinterpret_cast<const __half2*>(&v.w);
        
        sum += __half2float(x_reg[i*8])     * __half2float(v1.x); 
        sum += __half2float(x_reg[i*8 + 1]) * __half2float(v1.y); 
        sum += __half2float(x_reg[i*8 + 2]) * __half2float(v2.x);
        sum += __half2float(x_reg[i*8 + 3]) * __half2float(v2.y); 
        sum += __half2float(x_reg[i*8 + 4]) * __half2float(v3.x); 
        sum += __half2float(x_reg[i*8 + 5]) * __half2float(v3.y); 
        sum += __half2float(x_reg[i*8 + 6]) * __half2float(v4.x); 
        sum += __half2float(x_reg[i*8 + 7]) * __half2float(v4.y); 
    }
 
    // reduction
    for (int s=16; s>0; s/=2)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, s);

    // write result
    if (lane_id == 0)
        y[bid*2 + warp_id] = sum;
}

// ============================================================================
// TODO 2: INT8 weights with a per-row scale, dequantization fused into the GEMV
//   Wq    : int8  [rows x cols], row-major (one row = cols bytes)
//   scale : float [rows]         original weight ~= Wq[row][col] * scale[row]
//   x     : half  [cols]
//   y     : float [rows]         y[row] = sum_col Wq[row][col] * scale[row] * x[col]
// ============================================================================
__global__ void gemv_int8(const int8_t* __restrict__ Wq, const float* __restrict__ scale,
                          const __half* __restrict__ x, float* __restrict__ y, 
                          int rows, int cols)
{
    __half x_reg[128];

    int tx = threadIdx.x;
    int bid = blockIdx.x;

    int warp_id = tx / 32;
    int lane_id = tx % 32;

    for (int i = 0; i < 8; i++) {
        int c0 = i*512 + lane_id*16;
        int c1 = i*512 + lane_id*16 + 8;
        const float4 v0 = *reinterpret_cast<const float4*>(x + c0);
        const float4 v1 = *reinterpret_cast<const float4*>(x + c1);

        __half2 v00 = *reinterpret_cast<const __half2*>(&v0.x);
        __half2 v01 = *reinterpret_cast<const __half2*>(&v0.y);
        __half2 v02 = *reinterpret_cast<const __half2*>(&v0.z);
        __half2 v03 = *reinterpret_cast<const __half2*>(&v0.w);

        __half2 v10 = *reinterpret_cast<const __half2*>(&v1.x);
        __half2 v11 = *reinterpret_cast<const __half2*>(&v1.y);
        __half2 v12 = *reinterpret_cast<const __half2*>(&v1.z);
        __half2 v13 = *reinterpret_cast<const __half2*>(&v1.w);

        x_reg[i*16]     = v00.x; x_reg[i*16 + 1] = v00.y;
        x_reg[i*16 + 2] = v01.x; x_reg[i*16 + 3] = v01.y;
        x_reg[i*16 + 4] = v02.x; x_reg[i*16 + 5] = v02.y;
        x_reg[i*16 + 6] = v03.x; x_reg[i*16 + 7] = v03.y;

        x_reg[i*16 + 8]  = v10.x; x_reg[i*16 + 9]  = v10.y;
        x_reg[i*16 + 10] = v11.x; x_reg[i*16 + 11] = v11.y;
        x_reg[i*16 + 12] = v12.x; x_reg[i*16 + 13] = v12.y;
        x_reg[i*16 + 14] = v13.x; x_reg[i*16 + 15] = v13.y;
    }

    float acc = 0.0f;
    for (int i=0; i<8; i++) {
        int idx = (bid*2 + warp_id)*cols + i*512 + lane_id*16;
        
        const int4 v = *reinterpret_cast<const int4*>(&Wq[idx]);
        const char4* c = reinterpret_cast<const char4*>(&v);

        acc = fmaf(__half2float(x_reg[i*16]),      static_cast<float>(c[0].x), acc);
        acc = fmaf(__half2float(x_reg[i*16 + 1]),  static_cast<float>(c[0].y), acc);
        acc = fmaf(__half2float(x_reg[i*16 + 2]),  static_cast<float>(c[0].z), acc);
        acc = fmaf(__half2float(x_reg[i*16 + 3]),  static_cast<float>(c[0].w), acc);
        acc = fmaf(__half2float(x_reg[i*16 + 4]),  static_cast<float>(c[1].x), acc);
        acc = fmaf(__half2float(x_reg[i*16 + 5]),  static_cast<float>(c[1].y), acc);
        acc = fmaf(__half2float(x_reg[i*16 + 6]),  static_cast<float>(c[1].z), acc);
        acc = fmaf(__half2float(x_reg[i*16 + 7]),  static_cast<float>(c[1].w), acc);
        acc = fmaf(__half2float(x_reg[i*16 + 8]),  static_cast<float>(c[2].x), acc);
        acc = fmaf(__half2float(x_reg[i*16 + 9]),  static_cast<float>(c[2].y), acc);
        acc = fmaf(__half2float(x_reg[i*16 + 10]), static_cast<float>(c[2].z), acc);
        acc = fmaf(__half2float(x_reg[i*16 + 11]), static_cast<float>(c[2].w), acc);
        acc = fmaf(__half2float(x_reg[i*16 + 12]), static_cast<float>(c[3].x), acc);
        acc = fmaf(__half2float(x_reg[i*16 + 13]), static_cast<float>(c[3].y), acc);
        acc = fmaf(__half2float(x_reg[i*16 + 14]), static_cast<float>(c[3].z), acc);
        acc = fmaf(__half2float(x_reg[i*16 + 15]), static_cast<float>(c[3].w), acc);
    }
 
    // reduction
    for (int s=16; s>0; s/=2)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, s);

    // write result
    if (lane_id == 0)
        y[bid*2 + warp_id] = acc * scale[bid*2 + warp_id];
}

// ============================================================================
// TODO 3: launch config -- pick grid and block, then uncomment
// ============================================================================
void launch_fp16(const __half* W, const __half* x, float* y)
{
    dim3 threadsPerBlock(64);
    dim3 blocksPerGrid(2048);

    gemv_fp16<<<blocksPerGrid, threadsPerBlock>>>(W, x, y, ROWS, COLS);
}

void launch_int8(const int8_t* Wq, const float* scale, const __half* x, float* y)
{
    dim3 threadsPerBlock(64);
    dim3 blocksPerGrid(2048);

    gemv_int8<<<blocksPerGrid, threadsPerBlock>>>(Wq, scale, x, y, ROWS, COLS);
}

// ----------------------------------------------------------------------------
// Nothing below needs changing
// ----------------------------------------------------------------------------

template <typename F>
float time_ms(F run)
{
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < WARMUP; ++i) run(i);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < REPS; ++i) run(i);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaGetLastError());
    float ms = 0.f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return ms / REPS;
}

struct ErrorStats {
    double max_abs;     // max |out − ref|
    double max_ref;     // max |ref|
    double rel_l2;      // ||out − ref||₂ ÷ ||ref||₂
    bool   finite;      // no NaN or inf (an unwritten row stays 0xFF, which reads as NaN)
};

template <typename OutT>
ErrorStats compare(const std::vector<OutT>& out, const std::vector<double>& ref)
{
    ErrorStats s{0.0, 0.0, 0.0, true};
    double diff_sq = 0.0, ref_sq = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double o = static_cast<double>(out[i]);
        if (!std::isfinite(o)) { s.finite = false; continue; }
        double d = o - ref[i];
        s.max_abs = std::max(s.max_abs, std::fabs(d));
        s.max_ref = std::max(s.max_ref, std::fabs(ref[i]));
        diff_sq += d * d;
        ref_sq  += ref[i] * ref[i];
    }
    s.rel_l2 = std::sqrt(diff_sq / ref_sq);
    return s;
}

// Kernel correctness: did it compute the same expression? Only FP32 accumulation order may differ
bool is_match(const ErrorStats& s) { return s.finite && s.max_abs <= 1e-3 * std::max(1.0, s.max_ref); }

int main(int argc, char** argv)
{
    bool csv = bench::takeFlag(argc, argv, "--csv");
    const size_t n = static_cast<size_t>(ROWS) * COLS;

    // ---- 1. build the original FP16 weights and input x: normal, weight stddev 0.02 ----
    std::mt19937 rng(42);
    std::normal_distribution<float> weight_dist(0.f, 0.02f), x_dist(0.f, 1.f);
    std::vector<__half> hW(n), hx(COLS);
    for (size_t i = 0; i < n; ++i) hW[i] = __float2half(weight_dist(rng));
    for (int c = 0; c < COLS; ++c)  hx[c] = __float2half(x_dist(rng));

    // ---- 2. per-row symmetric quantization: scale = max|row| / 127, q = round(w / scale), -127..127 ----
    std::vector<int8_t> hWq(n);
    std::vector<float>  hscale(ROWS);
    for (int r = 0; r < ROWS; ++r) {
        float absmax = 0.f;
        for (int c = 0; c < COLS; ++c)
            absmax = std::max(absmax, std::fabs(__half2float(hW[(size_t)r * COLS + c])));
        float s = absmax > 0.f ? absmax / 127.f : 1.f;
        hscale[r] = s;
        for (int c = 0; c < COLS; ++c) {
            long q = lrintf(__half2float(hW[(size_t)r * COLS + c]) / s);
            hWq[(size_t)r * COLS + c] = static_cast<int8_t>(std::min(127L, std::max(-127L, q)));
        }
    }

    // ---- 3. CPU reference, accumulated in double ----
    //   y_fp16_ref : from the original FP16 weights -> baseline for FP16 correctness and INT8 quant error
    //   y_deq_ref  : scale * sum(q * x)               -> baseline for INT8 kernel correctness
    std::vector<double> y_fp16_ref(ROWS), y_deq_ref(ROWS);
    std::vector<float> xf(COLS);
    for (int c = 0; c < COLS; ++c) xf[c] = __half2float(hx[c]);
    for (int r = 0; r < ROWS; ++r) {
        double acc_fp16 = 0.0, acc_int = 0.0;
        for (int c = 0; c < COLS; ++c) {
            size_t idx = (size_t)r * COLS + c;
            acc_fp16 += (double)__half2float(hW[idx]) * xf[c];
            acc_int  += (double)hWq[idx] * xf[c];
        }
        y_fp16_ref[r] = acc_fp16;
        y_deq_ref[r]  = (double)hscale[r] * acc_int;
    }

    // ---- 4. GPU memory (NBUF copies each of W and Wq) ----
    __half *dW, *dx; int8_t* dWq; float *dscale, *dy;
    CHECK(cudaMalloc(&dW,  NBUF * n * sizeof(__half)));
    CHECK(cudaMalloc(&dWq, NBUF * n * sizeof(int8_t)));
    CHECK(cudaMalloc(&dscale, ROWS * sizeof(float)));
    CHECK(cudaMalloc(&dx, COLS * sizeof(__half)));
    CHECK(cudaMalloc(&dy, ROWS * sizeof(float)));
    for (int b = 0; b < NBUF; ++b) {
        CHECK(cudaMemcpy(dW  + b * n, hW.data(),  n * sizeof(__half), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dWq + b * n, hWq.data(), n * sizeof(int8_t), cudaMemcpyHostToDevice));
    }
    CHECK(cudaMemcpy(dscale, hscale.data(), ROWS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dx, hx.data(), COLS * sizeof(__half), cudaMemcpyHostToDevice));

    std::vector<float> hy(ROWS);

    // ---- 5. correctness: fill y with 0xFF (NaN) and run once, so an unwritten row shows up ----
    CHECK(cudaMemset(dy, 0xFF, ROWS * sizeof(float)));
    launch_fp16(dW, dx, dy);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hy.data(), dy, ROWS * sizeof(float), cudaMemcpyDeviceToHost));
    ErrorStats e_fp16 = compare(hy, y_fp16_ref);

    CHECK(cudaMemset(dy, 0xFF, ROWS * sizeof(float)));
    launch_int8(dWq, dscale, dx, dy);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hy.data(), dy, ROWS * sizeof(float), cudaMemcpyDeviceToHost));
    ErrorStats e_int8 = compare(hy, y_deq_ref);   // did the kernel compute the same expression?
    ErrorStats e_quant = compare(hy, y_fp16_ref); // output error introduced by quantization
    ErrorStats e_quant_cpu = compare(y_deq_ref, y_fp16_ref); // the same error seen on the CPU; should match

    // ---- 6. timing (rep i reads copy i % NBUF) ----
    float ms_fp16 = time_ms([&](int i) { launch_fp16(dW + (i % NBUF) * n, dx, dy); });
    float ms_int8 = time_ms([&](int i) { launch_int8(dWq + (i % NBUF) * n, dscale, dx, dy); });

    // Minimum DRAM bytes: weights + (scales) + x + y
    double bytes_fp16 = n * 2.0 + COLS * 2.0 + ROWS * 4.0;
    double bytes_int8 = n * 1.0 + ROWS * 4.0 + COLS * 2.0 + ROWS * 4.0;
    double gbs_fp16 = bytes_fp16 / (ms_fp16 * 1e-3) / 1e9;
    double gbs_int8 = bytes_int8 / (ms_int8 * 1e-3) / 1e9;

    KernelReport rep("w8 gemv", REPS, 6);
    rep.addConfig("shape", std::to_string(ROWS) + " x " + std::to_string(COLS)
                           + ", " + std::to_string(NBUF) + " W copies rotated");
    rep.addConfig("iterations", std::to_string(REPS) + " reps (+"
                                + std::to_string(WARMUP) + " warm-up), mean of the batch");
    {
        char note[96];
        snprintf(note, sizeof(note), "%.0f GB/s measured sequential read", PEAK_GBS);
        rep.addConfig("peak", note);
        snprintf(note, sizeof(note), "x%.2f, the byte-ratio ceiling on the speedup",
                 bytes_fp16 / bytes_int8);
        rep.addConfig("byte ratio", note);
    }
    rep.addConfig("verified", "vs CPU reference; INT8 against the dequantized weights");

    int colPeak = rep.addColumn("% peak", 9, 1);
    int colErr  = rep.addColumn("max err", 10, 1, true);

    int vFp16 = rep.addVariant("fp16");
    int vInt8 = rep.addVariant("int8");
    rep.setBytes(vFp16, bytes_fp16);
    rep.setBytes(vInt8, bytes_int8);
    rep.addSample(vFp16, ms_fp16);
    rep.addSample(vInt8, ms_int8);
    rep.setCell(vFp16, colPeak, 100.0 * gbs_fp16 / PEAK_GBS);
    rep.setCell(vInt8, colPeak, 100.0 * gbs_int8 / PEAK_GBS);
    rep.setCell(vFp16, colErr, e_fp16.max_abs);
    rep.setCell(vInt8, colErr, e_int8.max_abs);
    rep.setCheck(vFp16, is_match(e_fp16) ? 0 : 1);
    rep.setCheck(vInt8, is_match(e_int8) ? 0 : 1);
    rep.setBaseline(vFp16);
    rep.print(csv);

    // Quantization error is about accuracy, not speed, so it stays its own block
    // rather than being squeezed into the timing table.
    if (!csv) {
        printf("\n 양자화 출력 오차 (FP16 가중치 결과 기준)\n");
        printf("   INT8 커널 : 상대 L2 %.3e, max abs %.3e (max |y| %.3f 대비 %.3e)\n",
               e_quant.rel_l2, e_quant.max_abs, e_quant.max_ref, e_quant.max_abs / e_quant.max_ref);
        printf("   CPU 계산  : 상대 L2 %.3e, max abs %.3e\n", e_quant_cpu.rel_l2, e_quant_cpu.max_abs);
    }

    CHECK(cudaFree(dW)); CHECK(cudaFree(dWq)); CHECK(cudaFree(dscale));
    CHECK(cudaFree(dx)); CHECK(cudaFree(dy));
    return rep.passed() ? 0 : 1;
}
