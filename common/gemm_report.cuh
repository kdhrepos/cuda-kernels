#pragma once

// Report for the GEMM benchmarks: one row per timed variant, so a file that
// races its own kernel against cuBLAS simply registers two variants.
//
//   #include "../../common/gemm_report.cuh"
//
//   GemmReport rep("tiled_gemm", M, N, K, iter);
//   rep.setLaunch(blocksPerGrid, threadsPerBlock);
//   int tiled = rep.addVariant("tiled");
//   for (...) rep.addSample(tiled, kernelTime);
//   rep.setCheck(tiled, mismatches, rowsChecked, colsChecked);
//   rep.print(csv);
//
// print() writes the human table; print(true) writes one CSV row per variant,
// for sweeping over shapes from a script.

#include "bench_format.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// How deep into C the CPU reference checks, in each dimension. Capped because the
// O(M*N*K) reference is far slower than the kernel under test, and overridable so
// a shape sweep can trade verification depth for wall time.
#ifndef GEMM_CHECK_CORNER
#define GEMM_CHECK_CORNER 1024
#endif

// Reference check of the top-left corner of C against a CPU recomputation.
// `limit` defaults to GEMM_CHECK_CORNER so every GEMM here verifies the same
// square, which keeps one benchmark's PASS as strong a claim as another's.
struct GemmCheck {
    long long mismatches;
    int rows;
    int cols;
};

inline GemmCheck gemmCheckCpu(int m, int n, int k, const float* h_a, const float* h_b,
                              const float* h_c, int limit = GEMM_CHECK_CORNER, float tol = 1e-7f) {
    GemmCheck r;
    r.rows       = std::min(m, limit);
    r.cols       = std::min(n, limit);
    r.mismatches = 0;

    for (int i = 0; i < r.rows; i++) {
        for (int j = 0; j < r.cols; j++) {
            float elem = 0.0f;
            for (int l = 0; l < k; l++) elem += h_a[i * k + l] * h_b[l * n + j];

            if (fabs(elem - h_c[i * n + j]) > tol) r.mismatches++;
        }
    }
    return r;
}

class GemmReport {
   public:
    GemmReport(const char* name, int m, int n, int k, int iter)
        : name_(name),
          m_(m),
          n_(n),
          k_(k),
          iter_(iter),
          abBytes_(sizeof(float)),
          cBytes_(sizeof(float)),
          hasLaunch_(false) {}

    // Half-precision inputs with a float accumulator: setElementBytes(2, 4).
    void setElementBytes(size_t ab, size_t c) {
        abBytes_ = ab;
        cBytes_  = c;
    }

    void setLaunch(dim3 grid, dim3 block) {
        grid_      = grid;
        block_     = block;
        hasLaunch_ = true;
    }

    int addVariant(const char* label) {
        Variant v;
        v.label      = label;
        v.mismatches = 0;
        v.rows       = 0;
        v.cols       = 0;
        v.checked    = false;
        variants_.push_back(v);
        return (int)variants_.size() - 1;
    }

    void addSample(int variant, float ms) { variants_[variant].samples.push_back(ms); }

    void setCheck(int variant, const GemmCheck& c) {
        setCheck(variant, c.mismatches, c.rows, c.cols);
    }

    void setCheck(int variant, long long mismatches, int rows, int cols) {
        Variant& v   = variants_[variant];
        v.mismatches = mismatches;
        v.rows       = rows;
        v.cols       = cols;
        v.checked    = true;
    }

    bool passed() const {
        for (size_t i = 0; i < variants_.size(); i++)
            if (variants_[i].checked && variants_[i].mismatches != 0) return false;
        return true;
    }

    void print(bool csv = false) const {
        if (csv)
            printCsv();
        else
            printTable();
    }

   private:
    struct Variant {
        std::string label;
        std::vector<float> samples;
        long long mismatches;
        int rows;
        int cols;
        bool checked;
    };

    double flop() const { return 2.0 * m_ * n_ * k_; }

    static std::string dims(dim3 d) {
        std::string s = std::to_string(d.x) + "x" + std::to_string(d.y);
        if (d.z > 1) s += "x" + std::to_string(d.z);
        return s;
    }

    void printTable() const {
        using namespace bench;

        // Unit comes from the fastest variant so the quick ones keep their digits.
        double fastest = 0.0;
        for (size_t i = 0; i < variants_.size(); i++) {
            Stat s = summarize(variants_[i].samples);
            if (s.avg > 0.0 && (fastest == 0.0 || s.avg < fastest)) fastest = s.avg;
        }
        Unit unit = pickUnit(fastest);

        banner(name_, kWidth);
        configLine("problem", std::to_string(m_) + " x " + std::to_string(n_) + " x " +
                                  std::to_string(k_) + " (M x N x K)");
        if (hasLaunch_) configLine("launch", dims(grid_) + " blocks, " + dims(block_) + " threads");
        configLine("iterations", std::to_string(iter_) + " (+1 warm-up, discarded)");

        char work[96];
        snprintf(work, sizeof(work), "%.2f GFLOP/call", flop() / 1e9);
        configLine("work", std::string(work) + ", A " + humanBytes((double)abBytes_ * m_ * k_) +
                               " + B " + humanBytes((double)abBytes_ * k_ * n_) + " + C " +
                               humanBytes((double)cBytes_ * m_ * n_));

        const Variant* checked = 0;
        for (size_t i = 0; i < variants_.size(); i++)
            if (variants_[i].checked) {
                checked = &variants_[i];
                break;
            }
        if (checked)
            configLine("verified", std::to_string(checked->rows) + "x" +
                                       std::to_string(checked->cols) + " corner of C vs CPU");

        std::cout << "\n";
        std::cout << timeHeader("variant") << rightLabel("GFLOP/s", kGflopWidth)
                  << rightLabel("check", kCheckWidth) << "\n";
        rule(kWidth);
        for (size_t i = 0; i < variants_.size(); i++) {
            const Variant& v = variants_[i];
            Stat s           = summarize(v.samples);

            nameCell(v.label);
            timeCells(s, unit);
            std::cout << std::setw(kGflopWidth) << std::fixed << std::setprecision(2)
                      << gflopPerSec(flop(), s.avg);
            std::cout << std::setw(kCheckWidth)
                      << (!v.checked ? "-" : (v.mismatches == 0 ? "PASS" : "FAIL")) << "\n";
        }
        rule(kWidth);

        for (size_t i = 0; i < variants_.size(); i++) {
            const Variant& v = variants_[i];
            if (v.checked && v.mismatches != 0)
                std::cout << " " << v.label << ": " << v.mismatches
                          << " mismatching elements in the verified corner\n";
        }
        bannerEnd(kWidth);
    }

    void printCsv() const {
        using namespace bench;

        std::cout << "name,variant,m,n,k,iter,avg_ms,min_ms,max_ms,gflops,status\n";
        for (size_t i = 0; i < variants_.size(); i++) {
            const Variant& v = variants_[i];
            Stat s           = summarize(v.samples);

            std::cout << name_ << "," << v.label << "," << m_ << "," << n_ << "," << k_ << ","
                      << iter_ << std::fixed << std::setprecision(4) << "," << s.avg << "," << s.min
                      << "," << s.max << std::setprecision(2) << "," << gflopPerSec(flop(), s.avg)
                      << "," << (!v.checked ? "unchecked" : (v.mismatches == 0 ? "pass" : "fail"))
                      << "\n";
        }
    }

    static const int kGflopWidth = 12;
    static const int kCheckWidth = 8;
    static const int kWidth      = bench::kTimeCols + kGflopWidth + kCheckWidth;  // 63

    std::string name_;
    int m_, n_, k_, iter_;
    size_t abBytes_, cBytes_;
    bool hasLaunch_;
    dim3 grid_, block_;
    std::vector<Variant> variants_;
};
