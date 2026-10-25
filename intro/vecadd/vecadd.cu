#include "../../common/bench_format.cuh"
#include "../../common/cuda_check.cuh"

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>

// Default measured iterations, after one discarded warm-up. This benchmark also
// takes the count on the command line. Override the default with -DREPS=n.
#ifndef REPS
#define REPS 10
#endif

using namespace std;

__global__ void vec_add(int n,
                        int* d_a,
                        int* d_b,
                        int* d_c) {
    int tid = threadIdx.x + blockDim.x*blockIdx.x;

    if (tid < n) {
        d_c[tid] = d_a[tid] + d_b[tid];
    }
}

// The shared 43 time columns, plus the bytes this stage moves and the bandwidth
// that implies.
static const int kMovedWidth = 10;
static const int kBandwidthWidth = 8;
static const int kWidth = bench::kTimeCols + kMovedWidth + kBandwidthWidth;

void printRow(const char* stage, const bench::Stat& s, const bench::Unit& unit, double bytes)
{
    bench::nameCell(stage);
    bench::timeCells(s, unit);
    cout << setw(kMovedWidth) << bench::humanBytes(bytes)
         << setw(kBandwidthWidth) << fixed << setprecision(2)
         << bench::gbPerSec(bytes, s.avg) << "\n";
}

int main(int argc, char** argv) {
    int n = 6400;
    int reps = REPS;
    bool csv = bench::takeFlag(argc, argv, "--csv");

    int positional = 0;
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            cerr << "Usage: " << argv[0] << " [n] [reps] [--csv]\n";
            return 0;
        } else if (positional == 0) {
            n = stoi(arg);
            positional++;
        } else if (positional == 1) {
            reps = stoi(arg);
            positional++;
        } else {
            cerr << "Usage: " << argv[0] << " [n] [reps] [--csv]\n";
            return 1;
        }
    }

    if (n <= 0 || reps <= 0) {
        cerr << "n and reps must be positive (got n=" << n << ", reps=" << reps << ")\n";
        return 1;
    }

    int* a=(int*)malloc(sizeof(int)*n);
    int* b=(int*)malloc(sizeof(int)*n);
    int* c=(int*)malloc(sizeof(int)*n);

    for (int i=0; i<n; i++) {
        a[i]=rand() % 5;
        b[i]=rand() % 5;
    }

    int* d_a;
    int* d_b;
    int* d_c;

    CHECK_CUDA_ERROR(cudaMalloc(&d_a, n*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, n*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, n*sizeof(int)));

    cudaEvent_t kernelTimeStart, kernelTimeEnd;
    cudaEvent_t toDeviceStart, toDeviceEnd;
    cudaEvent_t toHostStart, toHostEnd;
    cudaEvent_t totalTimeStart, totalTimeEnd;

    int threadPerBlock=256;
    int blocksPerGrid=(n + threadPerBlock - 1)/threadPerBlock;

    vector<float> toDeviceSamples, kernelSamples, toHostSamples, totalSamples;
    toDeviceSamples.reserve(reps);
    kernelSamples.reserve(reps);
    toHostSamples.reserve(reps);
    totalSamples.reserve(reps);

    long long mismatches = 0;
    int firstBadIndex = -1;

    for (int i=0; i<reps+1; i++) {
        CHECK_CUDA_ERROR(cudaEventCreate(&kernelTimeStart));
        CHECK_CUDA_ERROR(cudaEventCreate(&kernelTimeEnd));
        CHECK_CUDA_ERROR(cudaEventCreate(&toDeviceStart));
        CHECK_CUDA_ERROR(cudaEventCreate(&toDeviceEnd));
        CHECK_CUDA_ERROR(cudaEventCreate(&toHostStart));
        CHECK_CUDA_ERROR(cudaEventCreate(&toHostEnd));
        CHECK_CUDA_ERROR(cudaEventCreate(&totalTimeStart));
        CHECK_CUDA_ERROR(cudaEventCreate(&totalTimeEnd));

        CHECK_CUDA_ERROR(cudaEventRecord(totalTimeStart));

        CHECK_CUDA_ERROR(cudaEventRecord(toDeviceStart));
        CHECK_CUDA_ERROR(cudaMemcpy(d_a, a, n*sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_b, b, n*sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaEventRecord(toDeviceEnd));

        CHECK_CUDA_ERROR(cudaEventRecord(kernelTimeStart));
        vec_add<<<blocksPerGrid, threadPerBlock>>>(n, d_a, d_b, d_c);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaEventRecord(kernelTimeEnd));

        CHECK_CUDA_ERROR(cudaEventRecord(toHostStart));
        CHECK_CUDA_ERROR(cudaMemcpy(c, d_c, n*sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaEventRecord(toHostEnd));

        CHECK_CUDA_ERROR(cudaEventRecord(totalTimeEnd));
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        for (int j=0; j<n; j++) {
            if (a[j]+b[j] != c[j]) {
                if (firstBadIndex < 0) {
                    firstBadIndex = j;
                    cerr << "[vecadd] Mismatch at " << j << ": " << c[j]
                         << " != " << a[j]+b[j] << "\n";
                }
                mismatches++;
            }
        }

        float kernelElapsedTime, toDeviceElapsedTime, toHostElapsedTime, totalTime;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelElapsedTime, kernelTimeStart, kernelTimeEnd));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&toDeviceElapsedTime, toDeviceStart, toDeviceEnd));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&toHostElapsedTime, toHostStart, toHostEnd));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&totalTime, totalTimeStart, totalTimeEnd));

        // i == 0 is the warm-up: timed like the rest, then thrown away.
        if (i!=0) {
            toDeviceSamples.push_back(toDeviceElapsedTime);
            kernelSamples.push_back(kernelElapsedTime);
            toHostSamples.push_back(toHostElapsedTime);
            totalSamples.push_back(totalTime);
        }

        CHECK_CUDA_ERROR(cudaEventDestroy(kernelTimeStart));
        CHECK_CUDA_ERROR(cudaEventDestroy(kernelTimeEnd));
        CHECK_CUDA_ERROR(cudaEventDestroy(toDeviceStart));
        CHECK_CUDA_ERROR(cudaEventDestroy(toDeviceEnd));
        CHECK_CUDA_ERROR(cudaEventDestroy(toHostStart));
        CHECK_CUDA_ERROR(cudaEventDestroy(toHostEnd));
        CHECK_CUDA_ERROR(cudaEventDestroy(totalTimeStart));
        CHECK_CUDA_ERROR(cudaEventDestroy(totalTimeEnd));
    }

    bench::Stat toDevice = bench::summarize(toDeviceSamples);
    bench::Stat kernel   = bench::summarize(kernelSamples);
    bench::Stat toHost   = bench::summarize(toHostSamples);
    bench::Stat total    = bench::summarize(totalSamples);

    double vectorBytes   = (double)sizeof(int) * n;
    double toDeviceBytes = vectorBytes * 2;   // a and b in
    double kernelBytes   = vectorBytes * 3;   // a and b read, c written
    double toHostBytes   = vectorBytes;       // c out

    bool pass = (mismatches == 0);

    if (csv) {
        cout << "name,n,reps,h2d_us,kernel_us,d2h_us,total_us,"
             << "h2d_gbs,kernel_gbs,d2h_gbs,status\n";
        cout << "vecadd," << n << "," << reps << fixed << setprecision(2)
             << "," << toDevice.avg*1000 << "," << kernel.avg*1000
             << "," << toHost.avg*1000   << "," << total.avg*1000
             << "," << bench::gbPerSec(toDeviceBytes, toDevice.avg)
             << "," << bench::gbPerSec(kernelBytes, kernel.avg)
             << "," << bench::gbPerSec(toHostBytes, toHost.avg)
             << "," << (pass ? "pass" : "fail") << "\n";
    } else {
        // One unit for the whole table, picked from the fastest stage so no row
        // collapses to "0.03 ms".
        double fastest = toDevice.avg;
        if (kernel.avg < fastest) fastest = kernel.avg;
        if (toHost.avg < fastest) fastest = toHost.avg;
        bench::Unit unit = bench::pickUnit(fastest);

        bench::banner("vecadd", kWidth);
        bench::configLine("N", to_string(n) + " elements ("
                               + bench::humanBytes(vectorBytes) + "/vector)");
        bench::configLine("launch", to_string(blocksPerGrid) + " blocks x "
                                    + to_string(threadPerBlock) + " threads");
        bench::configLine("iterations", to_string(reps) + " (+1 warm-up, discarded)");
        if (pass) {
            bench::configLine("check", "PASS");
        } else {
            bench::configLine("check", "FAIL (" + to_string(mismatches)
                                       + " mismatches, first at "
                                       + to_string(firstBadIndex) + ")");
        }

        cout << "\n";
        cout << bench::timeHeader("stage")
             << bench::rightLabel("moved", kMovedWidth)
             << bench::rightLabel("GB/s", kBandwidthWidth) << "\n";
        bench::rule(kWidth);
        printRow("H2D",    toDevice, unit, toDeviceBytes);
        printRow("kernel", kernel,   unit, kernelBytes);
        printRow("D2H",    toHost,   unit, toHostBytes);
        bench::rule(kWidth);

        double stageSum    = toDevice.avg + kernel.avg + toHost.avg;
        double unaccounted = total.avg - stageSum;
        double transferPct = (total.avg > 0.0)
                           ? (toDevice.avg + toHost.avg) / total.avg * 100.0
                           : 0.0;

        bench::nameCell("total");
        cout << fixed << setprecision(2)
             << setw(bench::kTimeWidth) << total.avg*unit.scale << " " << unit.name
             << "   transfer " << setprecision(1) << transferPct << "%, "
             << setprecision(2) << unaccounted*unit.scale << " " << unit.name
             << " unaccounted\n";
        bench::bannerEnd(kWidth);
    }

    free(a);
    free(b);
    free(c);

    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));

    return pass ? 0 : 1;
}
