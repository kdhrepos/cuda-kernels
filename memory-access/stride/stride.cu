#include "../../common/cuda_check.cuh"
#include "../../common/kernel_report.cuh"

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <cmath>

// Measured iterations, after one discarded warm-up. Override at build time with
// -DREPS=n.
#ifndef REPS
#define REPS 10
#endif



double gbps(int N, float ms) {
    double bytes = 2.0 * N * sizeof(float);
    return bytes / (ms * 1e-3) / 1e9;   // = bytes / ms / 1e6
}

__global__ void readStridedAccess(
    int N,
    int stride,
    float *in,
    float *out
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int R   = N / stride;
    int row = tid % R;
    int col = tid / R;
    size_t idx = (size_t)row * stride + col;

    if (tid < N)
        out[tid] = in[idx];
}

__global__ void writeStridedAccess(
    int N,
    int stride,
    float *in,
    float *out
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int R   = N / stride;
    int row = tid % R;
    int col = tid / R;
    size_t idx = (size_t)row * stride + col;

    if (tid < N)
        out[idx] = in[tid];
}

int main(int argc, char** argv) {
    bool csv = bench::takeFlag(argc, argv, "--csv");
    int N;

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <N> [--csv]\n";
        return 1;
    } else {
        N = std::stoi(argv[1]);
    }

    cudaEvent_t kernelStart, kernelEnd;
    CHECK_CUDA_ERROR(cudaEventCreate(&kernelStart));
    CHECK_CUDA_ERROR(cudaEventCreate(&kernelEnd));

    float kernelTime=0.0;

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((256+N-1)/256);

    // One table per direction: a stride sweep is a sweep, not a set of rival
    // kernels, so the rows are stride values and stride 1 is the baseline.
    const int kStrides = 11;
    KernelReport readRep("stride (read)", REPS, 7);
    KernelReport writeRep("stride (write)", REPS, 7);
    int readVar[kStrides], writeVar[kStrides];
    double traffic = 2.0 * N * sizeof(float);   // one read + one write per element
    for (int r = 0; r < 2; r++) {
        KernelReport& rep = (r == 0) ? readRep : writeRep;
        rep.setVariantHeader("stride");
        rep.addConfig("N", std::to_string(N) + " floats ("
                           + bench::humanBytes(sizeof(float)*(double)N) + ")");
        rep.addConfig("launch", std::to_string(blocksPerGrid.x) + " blocks x "
                                + std::to_string(threadsPerBlock.x) + " threads");
        rep.addIterations();
        for (int p = 0; p < kStrides; p++) {
            int v = rep.addVariant(std::to_string(1 << p));
            rep.setBytes(v, traffic);
            if (r == 0) readVar[p] = v; else writeVar[p] = v;
        }
        rep.setBaseline(r == 0 ? readVar[0] : writeVar[0]);
    }

    float *h_in, *d_in, *d_out;

    size_t size=sizeof(float)*N;

    h_in=(float *)malloc(size);
    CHECK_CUDA_ERROR(cudaMalloc(&d_in, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, size));

    CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    for (int p=0; p<11; p++) {
        int stride = pow(2, p);

        for (int i=0; i<=REPS; i++) {
            CHECK_CUDA_ERROR(cudaMemset(d_out, 0xFF, size));
            CHECK_CUDA_ERROR(cudaEventRecord(kernelStart));
            readStridedAccess<<<blocksPerGrid, threadsPerBlock>>>(N, stride, d_in, d_out);
            CHECK_LAST_CUDA_ERROR();
            CHECK_CUDA_ERROR(cudaEventRecord(kernelEnd));
            CHECK_CUDA_ERROR(cudaEventSynchronize(kernelEnd));
            CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, kernelStart, kernelEnd));

            if (i != 0) readRep.addSample(readVar[p], kernelTime);
        }

        for (int i=0; i<=REPS; i++) {
            CHECK_CUDA_ERROR(cudaMemset(d_out, 0xFF, size));
            CHECK_CUDA_ERROR(cudaEventRecord(kernelStart));
            writeStridedAccess<<<blocksPerGrid, threadsPerBlock>>>(N, stride, d_in, d_out);
            CHECK_LAST_CUDA_ERROR();
            CHECK_CUDA_ERROR(cudaEventRecord(kernelEnd));
            CHECK_CUDA_ERROR(cudaEventSynchronize(kernelEnd));
            CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, kernelStart, kernelEnd));

            if (i != 0) writeRep.addSample(writeVar[p], kernelTime);
        }
    }

    readRep.print(csv);
    if (!csv) std::cout << "\n";
    writeRep.print(csv);

    CHECK_CUDA_ERROR(cudaEventDestroy(kernelStart));
    CHECK_CUDA_ERROR(cudaEventDestroy(kernelEnd));

    free(h_in);

    CHECK_CUDA_ERROR(cudaFree(d_in));
    CHECK_CUDA_ERROR(cudaFree(d_out));
}