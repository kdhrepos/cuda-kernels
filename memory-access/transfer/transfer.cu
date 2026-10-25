#include "../../common/cuda_check.cuh"
#include "../../common/kernel_report.cuh"

#include <string>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <chrono>

#include <cuda_runtime.h>

// Measured iterations, after one discarded warm-up. Override at build time with
// -DREPS=n.
#ifndef REPS
#define REPS 10
#endif




__global__ void vecadd(
    int N,
    int* d_a,
    int* d_b,
    int* d_c
) {
    int tid = threadIdx.x + blockDim.x*blockIdx.x;

    if (tid < N)
        d_c[tid] = d_a[tid] + d_b[tid];
}

long long verify(int N, const int* ref_a, const int* ref_b, const int* c)
{
    long long bad = 0;
    for (int i = 0; i < N; i++)
        if (c[i] != ref_a[i] + ref_b[i]) bad++;
    return bad;
}

// Each strategy gets its own stage table, and reports its end-to-end time into
// a shared summary table so the five can be ranked against each other. In --csv
// mode only the summary is printed, since that is the comparison worth sweeping.
struct Strategy {
    KernelReport& summary;
    int sv;
    bool csv;
};

static void configureStages(KernelReport& rep, const char* tag, int N, size_t bytes)
{
    rep.setVariantHeader("stage");
    rep.addConfig("strategy", tag);
    rep.addConfig("N", std::to_string(N) + " elements ("
                       + bench::humanBytes((double)bytes) + "/vector)");
    rep.addConfig("iterations", std::to_string(REPS) + " (+1 warm-up, discarded)");
    rep.addConfig("verified", "every element vs CPU");
}

void explicit_copy_vecadd(
    const char* tag, int N,
    const int* h_a, const int* h_b, int* h_c,
    const int* ref_a, const int* ref_b,
    Strategy st
) {
    size_t bytes = sizeof(int) * (size_t)N;

    int *d_a, *d_b, *d_c;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, bytes));

    cudaEvent_t e0, e1, e2, e3;
    CHECK_CUDA_ERROR(cudaEventCreate(&e0));
    CHECK_CUDA_ERROR(cudaEventCreate(&e1));
    CHECK_CUDA_ERROR(cudaEventCreate(&e2));
    CHECK_CUDA_ERROR(cudaEventCreate(&e3));

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((N + 255) / 256);

    memset(h_c, 0, bytes);

    KernelReport rep(tag, REPS, 7);
    configureStages(rep, tag, N, bytes);
    int vH2D    = rep.addVariant("H2D");
    int vKernel = rep.addVariant("kernel");
    int vD2H    = rep.addVariant("D2H");
    rep.setBytes(vH2D, 2.0 * bytes);
    rep.setBytes(vKernel, 3.0 * bytes);   // a, b read; c written
    rep.setBytes(vD2H, (double)bytes);
    rep.setTotalRow("total");

    for (int i = 0; i <= REPS; i++) {
        CHECK_CUDA_ERROR(cudaEventRecord(e0));
        CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaEventRecord(e1));
        vecadd<<<blocksPerGrid, threadsPerBlock>>>(N, d_a, d_b, d_c);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaEventRecord(e2));
        CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaEventRecord(e3));
        CHECK_CUDA_ERROR(cudaEventSynchronize(e3));

        float h2d = 0.0f, kernel = 0.0f, d2h = 0.0f;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&h2d, e0, e1));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernel, e1, e2));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&d2h, e2, e3));

        if (i != 0) {
            rep.addSample(vH2D, h2d);
            rep.addSample(vKernel, kernel);
            rep.addSample(vD2H, d2h);
            st.summary.addSample(st.sv, h2d + kernel + d2h);
        }
    }

    long long bad = verify(N, ref_a, ref_b, h_c);
    rep.setTotalCheck(bad);
    st.summary.setCheck(st.sv, bad);
    if (!st.csv) { rep.print(false); std::cout << "\n"; }

    CHECK_CUDA_ERROR(cudaEventDestroy(e0));
    CHECK_CUDA_ERROR(cudaEventDestroy(e1));
    CHECK_CUDA_ERROR(cudaEventDestroy(e2));
    CHECK_CUDA_ERROR(cudaEventDestroy(e3));

    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));
}

void pageable_vecadd(int N, const int* ref_a, const int* ref_b, Strategy st)
{
    size_t bytes = sizeof(int) * (size_t)N;
    int* h_a = (int*)malloc(bytes);
    int* h_b = (int*)malloc(bytes);
    int* h_c = (int*)malloc(bytes);
    memcpy(h_a, ref_a, bytes);
    memcpy(h_b, ref_b, bytes);

    explicit_copy_vecadd("Pageable", N, h_a, h_b, h_c, ref_a, ref_b, st);

    free(h_a);
    free(h_b);
    free(h_c);
}

void pinned_vecadd(int N, const int* ref_a, const int* ref_b, Strategy st)
{
    size_t bytes = sizeof(int) * (size_t)N;
    int *h_a, *h_b, *h_c;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_a, bytes));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_b, bytes));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_c, bytes));
    memcpy(h_a, ref_a, bytes);
    memcpy(h_b, ref_b, bytes);

    explicit_copy_vecadd("Pinned", N, h_a, h_b, h_c, ref_a, ref_b, st);

    CHECK_CUDA_ERROR(cudaFreeHost(h_a));
    CHECK_CUDA_ERROR(cudaFreeHost(h_b));
    CHECK_CUDA_ERROR(cudaFreeHost(h_c));
}

void managed_vecadd(int N, const int* ref_a, const int* ref_b, Strategy st)
{
    const char* tag = "Managed";
    size_t bytes = sizeof(int) * (size_t)N;

    int deviceId;
    CHECK_CUDA_ERROR(cudaGetDevice(&deviceId));

    int *a, *b, *c;
    CHECK_CUDA_ERROR(cudaMallocManaged(&a, bytes));
    CHECK_CUDA_ERROR(cudaMallocManaged(&b, bytes));
    CHECK_CUDA_ERROR(cudaMallocManaged(&c, bytes));

    memcpy(a, ref_a, bytes);
    memcpy(b, ref_b, bytes);

    cudaEvent_t e0, e1;
    CHECK_CUDA_ERROR(cudaEventCreate(&e0));
    CHECK_CUDA_ERROR(cudaEventCreate(&e1));

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((N + 255) / 256);

    KernelReport rep(tag, REPS, 11);
    configureStages(rep, tag, N, bytes);
    // No explicit copies here: the pages fault in while the kernel runs, so the
    // kernel time IS the transfer time.
    int vKernel = rep.addVariant("kernel+fault");
    rep.setBytes(vKernel, 2.0 * bytes);

    for (int i = 0; i <= REPS; i++) {
        CHECK_CUDA_ERROR(cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId));
        CHECK_CUDA_ERROR(cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId));
        CHECK_CUDA_ERROR(cudaMemPrefetchAsync(c, bytes, deviceId));
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        CHECK_CUDA_ERROR(cudaEventRecord(e0));
        vecadd<<<blocksPerGrid, threadsPerBlock>>>(N, a, b, c);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaEventRecord(e1));
        CHECK_CUDA_ERROR(cudaEventSynchronize(e1));

        float kernel = 0.0f;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernel, e0, e1));
        if (i != 0) {
            rep.addSample(vKernel, kernel);
            st.summary.addSample(st.sv, kernel);
        }
    }

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    long long bad = verify(N, ref_a, ref_b, c);
    rep.setCheck(vKernel, bad);
    st.summary.setCheck(st.sv, bad);
    if (!st.csv) { rep.print(false); std::cout << "\n"; }

    CHECK_CUDA_ERROR(cudaEventDestroy(e0));
    CHECK_CUDA_ERROR(cudaEventDestroy(e1));

    CHECK_CUDA_ERROR(cudaFree(a));
    CHECK_CUDA_ERROR(cudaFree(b));
    CHECK_CUDA_ERROR(cudaFree(c));
}

void prefetch_vecadd(int N, const int* ref_a, const int* ref_b, Strategy st)
{
    const char* tag = "Prefetch";
    size_t bytes = sizeof(int) * (size_t)N;

    int deviceId;
    CHECK_CUDA_ERROR(cudaGetDevice(&deviceId));

    int *a, *b, *c;
    CHECK_CUDA_ERROR(cudaMallocManaged(&a, bytes));
    CHECK_CUDA_ERROR(cudaMallocManaged(&b, bytes));
    CHECK_CUDA_ERROR(cudaMallocManaged(&c, bytes));
    memcpy(a, ref_a, bytes);
    memcpy(b, ref_b, bytes);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((N + 255) / 256);

    cudaStream_t s;
    CHECK_CUDA_ERROR(cudaStreamCreate(&s));

    cudaEvent_t e0, e1, e2, e3;
    CHECK_CUDA_ERROR(cudaEventCreate(&e0));
    CHECK_CUDA_ERROR(cudaEventCreate(&e1));
    CHECK_CUDA_ERROR(cudaEventCreate(&e2));
    CHECK_CUDA_ERROR(cudaEventCreate(&e3));

    KernelReport rep(tag, REPS, 9);
    configureStages(rep, tag, N, bytes);
    int vToDevice = rep.addVariant("pf->GPU");
    int vKernel   = rep.addVariant("kernel");
    int vToHost   = rep.addVariant("pf->CPU");
    rep.setBytes(vToDevice, 2.0 * bytes);
    rep.setBytes(vKernel, 3.0 * bytes);
    rep.setBytes(vToHost, (double)bytes);
    rep.setTotalRow("total");

    for (int i = 0; i <= REPS; i++) {
        CHECK_CUDA_ERROR(cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId, s));
        CHECK_CUDA_ERROR(cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId, s));
        CHECK_CUDA_ERROR(cudaMemPrefetchAsync(c, bytes, deviceId, s));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(s));

        CHECK_CUDA_ERROR(cudaEventRecord(e0, s));
        CHECK_CUDA_ERROR(cudaMemPrefetchAsync(a, bytes, deviceId, s));
        CHECK_CUDA_ERROR(cudaMemPrefetchAsync(b, bytes, deviceId, s));
        CHECK_CUDA_ERROR(cudaEventRecord(e1, s));
        vecadd<<<blocksPerGrid, threadsPerBlock, 0, s>>>(N, a, b, c);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaEventRecord(e2, s));
        CHECK_CUDA_ERROR(cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId, s));
        CHECK_CUDA_ERROR(cudaEventRecord(e3, s));
        CHECK_CUDA_ERROR(cudaEventSynchronize(e3));

        float toDevice = 0.0f, kernel = 0.0f, toHost = 0.0f;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&toDevice, e0, e1));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernel, e1, e2));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&toHost, e2, e3));

        if (i != 0) {
            rep.addSample(vToDevice, toDevice);
            rep.addSample(vKernel, kernel);
            rep.addSample(vToHost, toHost);
            st.summary.addSample(st.sv, toDevice + kernel + toHost);
        }
    }

    long long bad = verify(N, ref_a, ref_b, c);
    rep.setTotalCheck(bad);
    st.summary.setCheck(st.sv, bad);
    if (!st.csv) { rep.print(false); std::cout << "\n"; }

    CHECK_CUDA_ERROR(cudaEventDestroy(e0));
    CHECK_CUDA_ERROR(cudaEventDestroy(e1));
    CHECK_CUDA_ERROR(cudaEventDestroy(e2));
    CHECK_CUDA_ERROR(cudaEventDestroy(e3));

    CHECK_CUDA_ERROR(cudaStreamDestroy(s));

    CHECK_CUDA_ERROR(cudaFree(a));
    CHECK_CUDA_ERROR(cudaFree(b));
    CHECK_CUDA_ERROR(cudaFree(c));
}

void multi_stream_pinned_vecadd(int N, const int* ref_a, const int* ref_b, Strategy st) {
    const char* tag = "Pinned + Streams";
    size_t bytes = sizeof(int) * (size_t)N;
    int chunk = N / 8;
    
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_a, sizeof(int) * N));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_b, sizeof(int) * N));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_c, sizeof(int) * N));
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, sizeof(int) * N));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, sizeof(int) * N));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, sizeof(int) * N));
    memcpy(h_a, ref_a, bytes);   // same input as every other strategy
    memcpy(h_b, ref_b, bytes);
    memset(h_c, 0, bytes);       // so verification cannot be fooled by a stale result

    cudaEvent_t e0, e1, e2;
    CHECK_CUDA_ERROR(cudaEventCreate(&e0));
    CHECK_CUDA_ERROR(cudaEventCreate(&e1));
    CHECK_CUDA_ERROR(cudaEventCreate(&e2));

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(((255+chunk)/256));

    cudaStream_t s[8];
    for (int j=0; j<8; j++) CHECK_CUDA_ERROR(cudaStreamCreate(&s[j]));

    KernelReport rep(tag, REPS, 11);
    configureStages(rep, tag, N, bytes);
    // Copies and kernels overlap across 8 streams, so only the wall time of the
    // whole pipeline is meaningful; there are no separable stages to time.
    int vTotal = rep.addVariant("overlapped");
    rep.setBytes(vTotal, 3.0 * bytes);

    for (int i=0; i<=REPS; i++) {
        cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();

        for (int j=0; j<8; j++) {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a + chunk*j, h_a + chunk*j, sizeof(int) * chunk, cudaMemcpyHostToDevice, s[j]));
            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_b + chunk*j, h_b + chunk*j, sizeof(int) * chunk, cudaMemcpyHostToDevice, s[j]));

            vecadd<<<blocksPerGrid, threadsPerBlock, 0, s[j]>>>(chunk, d_a + chunk*j, d_b + chunk*j, d_c + chunk*j);

            CHECK_CUDA_ERROR(cudaMemcpyAsync(h_c + chunk*j, d_c + chunk*j, sizeof(int) * chunk, cudaMemcpyDeviceToHost, s[j]));
        }

        cudaDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();

        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        if (i != 0) {
            rep.addSample(vTotal, ms);
            st.summary.addSample(st.sv, ms);
        }
    }

    long long bad = verify(N, ref_a, ref_b, h_c);
    rep.setCheck(vTotal, bad);
    st.summary.setCheck(st.sv, bad);
    if (!st.csv) { rep.print(false); std::cout << "\n"; }

    CHECK_CUDA_ERROR(cudaEventDestroy(e0));
    CHECK_CUDA_ERROR(cudaEventDestroy(e1));
    CHECK_CUDA_ERROR(cudaEventDestroy(e2));

    for (int j=0; j<8; j++) CHECK_CUDA_ERROR(cudaStreamDestroy(s[j]));

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main(int argc, char** argv) {
    bool csv = bench::takeFlag(argc, argv, "--csv");

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <power> [--csv]  (N = 2^power)\n";
        return 1;
    }
    int power = std::stoi(argv[1]);
    int N = 1 << power;

    size_t bytes = sizeof(int) * (size_t)N;

    KernelReport summary("transfer (end-to-end)", REPS, 10);
    summary.setVariantHeader("strategy");
    summary.addConfig("N", "2^" + std::to_string(power) + " = " + std::to_string(N)
                           + " elements (" + bench::humanBytes((double)bytes) + "/vector)");
    summary.addConfig("traffic", bench::humanBytes(3.0 * bytes)
                                 + " host<->device per iteration");
    summary.addIterations();
    summary.addConfig("verified", "every element vs CPU");
    int svPageable  = summary.addVariant("pageable");
    int svPinned    = summary.addVariant("pinned");
    int svManaged   = summary.addVariant("managed");
    int svPrefetch  = summary.addVariant("prefetch");
    int svStreams   = summary.addVariant("streams");
    int svs[5] = {svPageable, svPinned, svManaged, svPrefetch, svStreams};
    for (int i = 0; i < 5; i++) summary.setBytes(svs[i], 3.0 * bytes);
    summary.setBaseline(svPageable);

    int* ref_a = (int*)malloc(bytes);
    int* ref_b = (int*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        ref_a[i] = rand() % 5;
        ref_b[i] = rand() % 5;
    }

    Strategy st = {summary, svPageable, csv};
    st.sv = svPageable;  pageable_vecadd(N, ref_a, ref_b, st);
    st.sv = svPinned;    pinned_vecadd(N, ref_a, ref_b, st);
    st.sv = svManaged;   managed_vecadd(N, ref_a, ref_b, st);
    st.sv = svPrefetch;  prefetch_vecadd(N, ref_a, ref_b, st);
    st.sv = svStreams;   multi_stream_pinned_vecadd(N, ref_a, ref_b, st);

    summary.print(csv);

    free(ref_a);
    free(ref_b);
    return summary.passed() ? 0 : 1;
}
