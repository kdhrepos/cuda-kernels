#include "../../common/cuda_check.cuh"

#include <cuda_runtime.h>
#include <iostream>

int main(void) {
    struct cudaDeviceProp prop;
    int device = 0;

    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, device));

    std::cout << "Name: " << prop.name << "\n";
    std::cout << "Number of SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "Warp size: " << prop.warpSize << "\n";
    std::cout << "Max threads per SM: " << prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << "Max blocks per SM: " << prop.maxBlocksPerMultiProcessor << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Registers per SM: " << prop.regsPerMultiprocessor << "\n";
    std::cout << "Shared memory per SM (B): " << prop.sharedMemPerMultiprocessor << "\n";
    std::cout << "Shared memory per block (B): " << prop.sharedMemPerBlock << "\n";
    std::cout << "L2 cache size (B): " << prop.l2CacheSize << "\n";
    std::cout << "Memory bus width (bit): " << prop.memoryBusWidth << "\n";
    std::cout << "Memory clock rate (kHz): " << prop.memoryClockRate << "\n";
    std::cout << "Memory bandwidth " << double(prop.memoryClockRate) * prop.memoryBusWidth << "\n";
    std::cout << "Async Engine Count " << prop.asyncEngineCount << "\n";
}