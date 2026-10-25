#include "../../common/cuda_check.cuh"
#include "../../common/kernel_report.cuh"

#include <iostream>
#include <string>
#include <cmath>
#include <chrono>

#include <cuda_runtime.h>

// Measured iterations, after one discarded warm-up. Override at build time with
// -DREPS=n.
#ifndef REPS
#define REPS 10
#endif



__global__ void kernel(
    int N,
    float *arr
) {
    int tid = blockDim.x*blockIdx.x + threadIdx.x;

    if (tid < N) 
        arr[tid]++;
}

int main(int argc, char** argv) {
    bool csv = bench::takeFlag(argc, argv, "--csv");
    int N=1024;
    int n_launches=100;

    float *h_arr = (float *)malloc(sizeof(float) * N);
    for (int i=0; i<N; i++) h_arr[i] = 0;

    float *d_arr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_arr, sizeof(float) * N));
    CHECK_CUDA_ERROR(cudaMemcpy(d_arr, h_arr, sizeof(float) * N, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(4);

    cudaStream_t s;
    CHECK_CUDA_ERROR(cudaStreamCreate(&s));

    // Two measures per launch style, so two tables: how long the launch call
    // itself takes to return, and how long until the work has actually run.
    // Times are per single kernel launch, chrono microseconds scaled to ms.
    KernelReport dispatchRep("graph-launch (dispatch)", REPS, 7);
    KernelReport totalRep("graph-launch (end-to-end)", REPS, 7);
    int dispatchNormal = dispatchRep.addVariant("normal");
    int dispatchGraph  = dispatchRep.addVariant("graph");
    int totalNormal    = totalRep.addVariant("normal");
    int totalGraph     = totalRep.addVariant("graph");
    for (int r = 0; r < 2; r++) {
        KernelReport& rep = (r == 0) ? dispatchRep : totalRep;
        rep.setVariantHeader("launch");
        rep.addConfig("N", std::to_string(N) + " elements, "
                           + std::to_string(n_launches) + " kernels per launch batch");
        rep.addConfig("launch", std::to_string(blocksPerGrid.x) + " blocks x "
                                + std::to_string(threadsPerBlock.x) + " threads");
        rep.addConfig("iterations", std::to_string(REPS) + " (warm-up done separately)");
        rep.setBaseline(r == 0 ? dispatchNormal : totalNormal);
    }
    const double perLaunchMs = 1.0 / (1000.0 * n_launches);   // us total -> ms each

    /* warm up for normal launch */
    for (int j=0; j<n_launches; j++) 
        kernel<<<blocksPerGrid, threadsPerBlock, 0, s>>>(N, d_arr);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(s));

    /* normal launch */
    for (int i=0; i<REPS; i++) {
        const auto start = std::chrono::steady_clock::now();

        for (int j=0; j<n_launches; j++) 
            kernel<<<blocksPerGrid, threadsPerBlock, 0, s>>>(N, d_arr);
        
        const auto dispatch = std::chrono::steady_clock::now();
        dispatchRep.addSample(dispatchNormal,
            std::chrono::duration<double, std::micro>(dispatch - start).count() * perLaunchMs);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(s));

        const auto end = std::chrono::steady_clock::now();
        totalRep.addSample(totalNormal,
            std::chrono::duration<double, std::micro>(end - start).count() * perLaunchMs);
    }

    /* stream setup for graph launch */
    cudaGraph_t g;
    cudaGraphExec_t ge;

    CHECK_CUDA_ERROR(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int i=0; i<n_launches; i++)
        kernel<<<blocksPerGrid, threadsPerBlock, 0, s>>>(N, d_arr);
    CHECK_CUDA_ERROR(cudaStreamEndCapture(s, &g));

    CHECK_CUDA_ERROR(cudaGraphInstantiate(&ge, g, 0, 0, 0));

    /* warm up for graph launch */
    for (int i=0; i<REPS; i++) {
        CHECK_CUDA_ERROR(cudaGraphLaunch(ge, s));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(s));
    }

    /* graph launch */
    for (int i=0; i<REPS; i++) {
        const auto start = std::chrono::steady_clock::now();
        CHECK_CUDA_ERROR(cudaGraphLaunch(ge, s));
        const auto dispatch = std::chrono::steady_clock::now();

        CHECK_CUDA_ERROR(cudaStreamSynchronize(s));
        const auto end = std::chrono::steady_clock::now();

        dispatchRep.addSample(dispatchGraph,
            std::chrono::duration<double, std::micro>(dispatch - start).count() * perLaunchMs);
        totalRep.addSample(totalGraph,
            std::chrono::duration<double, std::micro>(end - start).count() * perLaunchMs);
    }

    dispatchRep.print(csv);
    if (!csv) std::cout << "\n";
    totalRep.print(csv);

    /* free */
    CHECK_CUDA_ERROR(cudaGraphExecDestroy(ge));
    CHECK_CUDA_ERROR(cudaGraphDestroy(g));
    CHECK_CUDA_ERROR(cudaStreamDestroy(s));

    free(h_arr);

    CHECK_CUDA_ERROR(cudaFree(d_arr));
}