#pragma once

// Shared pieces of the benchmark report format, so every benchmark in this repo
// prints one recognisable table.
//
// Table geometry: the left 43 columns are the same everywhere --
//
//      " " + name(9) + 3 x [ time(8) + " " + unit(2) ]
//
// and each benchmark appends the metric columns it cares about (GB/s for the
// bandwidth-bound ones, GFLOP/s for GEMM). Build the header with timeHeader()
// plus rightLabel() so the labels land on the same column edges as the data.

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>
#include <cstdio>

namespace bench {

const int kNameWidth = 9;   // left label column, minus its leading space
const int kTimeWidth = 8;   // one time figure, excluding " us" / " ms"
const int kTimeCols  = 1 + kNameWidth + 3 * (kTimeWidth + 3);   // 43

// Timings stay in ms, the unit cudaEventElapsedTime hands us; scaling to us
// happens at print time.
struct Stat {
    double avg = 0.0;
    double min = 0.0;
    double max = 0.0;
    double med = 0.0;      // some harnesses headline the median, being outlier-proof
    size_t n   = 0;
};

inline Stat summarize(const std::vector<float>& samples)
{
    Stat s;
    if (samples.empty()) return s;

    double sum = 0.0;
    s.min = std::numeric_limits<double>::max();
    for (size_t i = 0; i < samples.size(); i++) {
        double x = samples[i];
        sum += x;
        if (x < s.min) s.min = x;
        if (x > s.max) s.max = x;
    }
    s.avg = sum / samples.size();
    s.n = samples.size();

    std::vector<float> sorted(samples);
    std::sort(sorted.begin(), sorted.end());
    s.med = sorted[sorted.size() / 2];
    return s;
}

// One unit for the whole table so the column stays comparable, picked from the
// fastest row so that no row collapses to "0.03 ms". The slower rows just use
// more digits.
struct Unit {
    const char* name;
    double scale;
};

inline Unit pickUnit(double fastestMs)
{
    Unit u;
    if (fastestMs >= 1.0) {
        u.name  = "ms";
        u.scale = 1.0;
    } else {
        u.name  = "us";
        u.scale = 1000.0;
    }
    return u;
}

inline double gbPerSec(double bytes, double ms)
{
    if (ms <= 0.0) return 0.0;
    return bytes / (ms * 1e-3) / 1e9;
}

inline double gflopPerSec(double flop, double ms)
{
    if (ms <= 0.0) return 0.0;
    return flop / (ms * 1e-3) / 1e9;
}

inline std::string humanBytes(double bytes)
{
    char buf[32];
    if (bytes >= 1024.0*1024.0*1024.0)
        snprintf(buf, sizeof(buf), "%.1f GiB", bytes/(1024.0*1024.0*1024.0));
    else if (bytes >= 1024.0*1024.0)
        snprintf(buf, sizeof(buf), "%.1f MiB", bytes/(1024.0*1024.0));
    else if (bytes >= 1024.0)
        snprintf(buf, sizeof(buf), "%.1f KiB", bytes/1024.0);
    else
        snprintf(buf, sizeof(buf), "%.0f B", bytes);
    return std::string(buf);
}

// Removes `flag` from argv when present, so the positional-argument checks that
// follow can stay written as they were. Returns true when the flag was there.
inline bool takeFlag(int& argc, char** argv, const char* flag)
{
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == flag) {
            for (int j = i; j + 1 < argc; j++) argv[j] = argv[j+1];
            argc--;
            return true;
        }
    }
    return false;
}

// --- table chrome ---------------------------------------------------------

inline void banner(const std::string& name, int width)
{
    std::string head = "== " + name + " ";
    if ((int)head.size() < width) head += std::string(width - head.size(), '=');
    std::cout << head << "\n";
}

inline void bannerEnd(int width)
{
    std::cout << std::string(width, '=') << "\n";
}

inline void rule(int width)
{
    std::cout << " " << std::string(width - 1, '-') << "\n";
}

inline void configLine(const char* label, const std::string& value)
{
    std::cout << " " << std::left << std::setw(13) << label << value << "\n";
}

inline std::string rightLabel(const char* label, int width)
{
    std::string s(label);
    if ((int)s.size() >= width) return s;
    return std::string(width - s.size(), ' ') + s;
}

// " name     " + right-aligned avg/min/max. Pass the width timeWidthFor() gave
// you so the labels land on the same edges as the widened data cells.
inline std::string timeHeader(const char* firstCol, int w = kTimeWidth,
                              int nameW = kNameWidth)
{
    std::string h = " ";
    h += firstCol;
    if ((int)h.size() < 1 + nameW) h += std::string(1 + nameW - h.size(), ' ');

    const char* labels[3] = {"avg", "min", "max"};
    for (int i = 0; i < 3; i++) h += rightLabel(labels[i], w + 3);
    return h;
}

// Columns taken by the name plus the three time groups.
inline int timeCols(int w = kTimeWidth, int nameW = kNameWidth)
{
    return 1 + nameW + 3 * (w + 3);
}

inline void nameCell(const std::string& name)
{
    std::cout << " " << std::left << std::setw(kNameWidth) << name << std::right;
}

// Width the time cells need to hold `maxScaled` with two decimals and still keep
// a space in front of it. A table spanning two orders of magnitude -- a 50x
// speedup, say -- would otherwise run its columns together.
inline int timeWidthFor(double maxScaled)
{
    char buf[64];
    int n = snprintf(buf, sizeof(buf), "%.2f", maxScaled);
    return (n + 1 > kTimeWidth) ? n + 1 : kTimeWidth;
}

inline void timeCells(const Stat& s, const Unit& u, int w = kTimeWidth)
{
    std::cout << std::fixed << std::setprecision(2)
              << std::setw(w) << s.avg*u.scale << " " << u.name
              << std::setw(w) << s.min*u.scale << " " << u.name
              << std::setw(w) << s.max*u.scale << " " << u.name;
}

}  // namespace bench
