#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#include "../../common/cuda_check.cuh"
#include "../../common/kernel_report.cuh"

#define CHECK(call) CHECK_CUDA_ERROR(call)

constexpr int STRIDE_BYTES = 128;
constexpr int TIMED_STEPS  = 1 << 18;

void build_chain(std::vector<int>& h, int num_elems, int stride_elems) {
    for (int i = 0; i < num_elems; i++)
        if (i % stride_elems == 0) h[i] = (i + stride_elems) % num_elems;
}

__global__ void chase(const int* arr, int warm_steps, int timed_steps, long long* cycles,
                      int* sink) {
    long long t0, t1;
    int j = 0;
    for (int i = 0; i < warm_steps; i++) j = arr[j];

    t0 = clock64();
    for (int i = 0; i < timed_steps; i++) j = arr[j];
    t1 = clock64();

    *cycles = t1 - t0;
    *sink   = j;
}

int main(int argc, char** argv) {
    bool csv      = bench::takeFlag(argc, argv, "--csv");
    int clock_khz = 0;
    CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    const double clock_ghz = clock_khz * 1e-6;
    const int stride_elems = STRIDE_BYTES / (int)sizeof(int);

    // A dependent pointer chase measures latency, not throughput: there is no
    // wall-clock time per variant to average, only cycles per load read from
    // clock64() inside the kernel. So this table has no time columns.
    KernelReport rep("pointer chase", 1, 8);
    rep.setShowTimes(false);
    rep.setVariantHeader("size");
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "%.3f GHz", clock_ghz);
        rep.addConfig("SM clock", buf);
    }
    rep.addConfig("stride", std::to_string(STRIDE_BYTES) + " B (one cache line apart)");
    rep.addConfig("timed loads", std::to_string(TIMED_STEPS));
    int colLines  = rep.addColumn("lines", 10, 0);
    int colCycles = rep.addColumn("cycles/load", 14, 1);
    int colNs     = rep.addColumn("ns/load", 10, 1);

    long long* d_cycles;
    int* d_sink;
    CHECK(cudaMalloc(&d_cycles, sizeof(long long)));
    CHECK(cudaMalloc(&d_sink, sizeof(int)));

    for (size_t bytes = size_t(4) << 10; bytes <= (size_t(64) << 20); bytes *= 2) {
        const int num_elems = (int)(bytes / sizeof(int));
        const int lines     = num_elems / stride_elems;  // accesses in one trip round the chain

        std::vector<int> h(num_elems, 0);
        build_chain(h, num_elems, stride_elems);

        int* d_arr;
        CHECK(cudaMalloc(&d_arr, bytes));
        CHECK(cudaMemcpy(d_arr, h.data(), bytes, cudaMemcpyHostToDevice));

        chase<<<1, 1>>>(d_arr, lines, TIMED_STEPS, d_cycles, d_sink);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        long long cycles = 0;
        CHECK(cudaMemcpy(&cycles, d_cycles, sizeof(cycles), cudaMemcpyDeviceToHost));
        const double per_load = (double)cycles / TIMED_STEPS;

        int v = rep.addVariant(bench::humanBytes((double)bytes));
        rep.setCell(v, colLines, (double)lines);
        rep.setCell(v, colCycles, per_load);
        rep.setCell(v, colNs, per_load / clock_ghz);

        CHECK(cudaFree(d_arr));
    }

    rep.print(csv);

    CHECK(cudaFree(d_cycles));
    CHECK(cudaFree(d_sink));
    return 0;
}
