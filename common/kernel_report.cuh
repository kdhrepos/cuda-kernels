#pragma once

// Report for the kernel-comparison benchmarks: one row per variant, which is
// what almost every benchmark here is doing -- time N kernels on the same input
// and compare them.
//
//   #include "../common/kernel_report.cuh"
//
//   KernelReport rep("reduction", iter);
//   rep.addConfig("N", std::to_string(N) + " elements");
//   rep.addIterations();
//   int interleaved = rep.addVariant("interleaved");
//   int shuffled    = rep.addVariant("warp-shuffle");
//   rep.setBytes(interleaved, 4.0*N);      // -> GB/s column
//   rep.setBytes(shuffled,    4.0*N);
//   rep.setBaseline(interleaved);          // -> "x base" column
//   ...
//   rep.addSample(interleaved, kernelTime);
//   rep.setCheck(interleaved, mismatches); // -> "check" column
//   rep.print(csv);
//
// Columns appear only when they carry something, so a single-variant benchmark
// gets a plain table and a sweep gets a wide one. Anything the report does not
// model -- max relative error, % of peak, cycles per load -- goes in a column of
// its own via addColumn()/setCell(). print(true) writes one CSV row per variant.

#include "bench_format.cuh"

#include <string>
#include <utility>
#include <vector>

class KernelReport {
public:
    KernelReport(const char* name, int iter, int nameWidth = 9)
        : name_(name), iter_(iter), nameWidth_(nameWidth),
          metric_(kNoMetric), baseline_(-1),
          totalMismatches_(0), totalChecked_(false),
          variantHeader_("variant"), showTimes_(true), showMedian_(false) {}

    void addConfig(const char* label, const std::string& value)
    {
        config_.push_back(std::make_pair(std::string(label), value));
    }

    // Shorthand for the line every benchmark here wants.
    void addIterations()
    {
        addConfig("iterations", std::to_string(iter_) + " (+1 warm-up, discarded)");
    }

    // The first column's header. "variant" fits a kernel comparison; a sweep
    // wants "stride" or "size", a pipeline wants "stage".
    void setVariantHeader(const char* h) { variantHeader_ = h; }

    // Drop the time columns for a benchmark that does not measure wall time at
    // all -- a pointer chase reporting cycles per load, say. Its numbers then
    // live entirely in addColumn() columns.
    void setShowTimes(bool on) { showTimes_ = on; }

    // Add a median column. Worth it when a harness headlines the median because
    // it is outlier-proof, and you want that number kept alongside avg/min/max.
    void setShowMedian(bool on) { showMedian_ = on; }

    int addVariant(const std::string& label) { return addVariant(label.c_str()); }

    int addVariant(const char* label)
    {
        Variant v;
        v.label = label;
        v.amount = 0.0;
        v.mismatches = 0;
        v.checked = false;
        v.cells.resize(columns_.size());
        variants_.push_back(v);
        if ((int)v.label.size() > nameWidth_) nameWidth_ = (int)v.label.size();
        return (int)variants_.size() - 1;
    }

    // Bytes this variant moves -> GB/s. Flops it does -> GFLOP/s. One or the
    // other for a whole report, whichever the benchmark is bound by.
    void setBytes(int v, double bytes) { variants_[v].amount = bytes; metric_ = kBandwidth; }
    void setFlop(int v, double flop)   { variants_[v].amount = flop;  metric_ = kThroughput; }

    // A column the report does not model. `scientific` prints 1.2e-07 rather
    // than a fixed-point number, which is what error figures need.
    int addColumn(const char* header, int width, int precision = 2, bool scientific = false)
    {
        Column c;
        c.header = header;
        c.width = width;
        c.precision = precision;
        c.scientific = scientific;
        columns_.push_back(c);
        for (size_t i = 0; i < variants_.size(); i++) variants_[i].cells.resize(columns_.size());
        return (int)columns_.size() - 1;
    }

    void setCell(int v, int col, double value)
    {
        Cell& c = variants_[v].cells[col];
        c.set = true;
        c.isText = false;
        c.value = value;
    }

    void setCell(int v, int col, const std::string& text)
    {
        Cell& c = variants_[v].cells[col];
        c.set = true;
        c.isText = true;
        c.text = text;
    }

    void setCheck(int v, long long mismatches)
    {
        variants_[v].mismatches = mismatches;
        variants_[v].checked = true;
    }

    // The variant every other row is compared against in the "x base" column.
    void setBaseline(int v) { baseline_ = v; }

    // Adds a summed row under the rule. Use it when the variants are stages of
    // one pipeline rather than competing implementations: the times add up, and
    // the metric becomes the whole pipeline's traffic over its whole time.
    void setTotalRow(const char* label) { totalLabel_ = label; }
    void setTotalCheck(long long mismatches)
    {
        totalMismatches_ = mismatches;
        totalChecked_ = true;
    }

    void addSample(int v, float ms) { variants_[v].samples.push_back(ms); }

    bool passed() const
    {
        if (totalChecked_ && totalMismatches_ != 0) return false;
        for (size_t i = 0; i < variants_.size(); i++)
            if (variants_[i].checked && variants_[i].mismatches != 0) return false;
        return true;
    }

    void print(bool csv = false) const
    {
        if (csv) printCsv();
        else     printTable();
    }

private:
    enum Metric { kNoMetric, kBandwidth, kThroughput };

    struct Cell {
        Cell() : set(false), isText(false), value(0.0) {}
        bool set;
        bool isText;
        double value;
        std::string text;
    };

    struct Column {
        std::string header;
        int width;
        int precision;
        bool scientific;
    };

    struct Variant {
        std::string label;
        std::vector<float> samples;
        double amount;        // bytes or flops, by metric_
        long long mismatches;
        bool checked;
        std::vector<Cell> cells;
    };

    const char* metricLabel() const
    {
        return metric_ == kBandwidth ? "GB/s" : "GFLOP/s";
    }

    double metricValue(const Variant& v, double avgMs) const
    {
        if (metric_ == kBandwidth)  return bench::gbPerSec(v.amount, avgMs);
        if (metric_ == kThroughput) return bench::gflopPerSec(v.amount, avgMs);
        return 0.0;
    }

    bool anyChecked() const
    {
        if (totalChecked_) return true;
        for (size_t i = 0; i < variants_.size(); i++)
            if (variants_[i].checked) return true;
        return false;
    }

    // With one sample per variant there is no spread to show, so the table
    // collapses avg/min/max into a single "time" column.
    bool singleSample() const
    {
        for (size_t i = 0; i < variants_.size(); i++)
            if (variants_[i].samples.size() > 1) return false;
        return true;
    }

    int timeGroups() const
    {
        if (!showTimes_) return 0;
        int n = singleSample() ? 1 : 3;
        if (showMedian_ && !singleSample()) n++;
        return n;
    }

    int width(int timeW) const
    {
        int w = 1 + nameWidth_ + timeGroups() * (timeW + 3);
        if (metric_ != kNoMetric) w += kMetricWidth;
        for (size_t i = 0; i < columns_.size(); i++) w += columns_[i].width;
        if (baseline_ >= 0)       w += kSpeedupWidth;
        if (anyChecked())         w += kCheckWidth;
        return w;
    }

    void printCell(const Column& col, const Cell& cell) const
    {
        if (!cell.set) {
            std::cout << std::setw(col.width) << "-";
            return;
        }
        if (cell.isText) {
            std::cout << std::setw(col.width) << cell.text;
            return;
        }
        std::cout << std::setw(col.width) << std::setprecision(col.precision);
        if (col.scientific) std::cout << std::scientific << cell.value << std::fixed;
        else                std::cout << std::fixed << cell.value;
    }

    void printTable() const
    {
        using namespace bench;

        double fastest = 0.0, baseAvg = 0.0, slowest = 0.0;
        for (size_t i = 0; i < variants_.size(); i++) {
            Stat s = summarize(variants_[i].samples);
            if (s.avg > 0.0 && (fastest == 0.0 || s.avg < fastest)) fastest = s.avg;
            if (s.max > slowest) slowest = s.max;
            if ((int)i == baseline_) baseAvg = s.avg;
        }
        // Unit from the fastest row so it keeps its digits; column width from the
        // slowest so the two still line up.
        Unit unit = pickUnit(fastest);
        int timeW = timeWidthFor(slowest * unit.scale);
        int w = width(timeW);
        bool single = singleSample();

        banner(name_, w);
        for (size_t i = 0; i < config_.size(); i++)
            configLine(config_[i].first.c_str(), config_[i].second);

        std::cout << "\n";
        std::string head = " " + variantHeader_;
        if ((int)head.size() < 1 + nameWidth_)
            head += std::string(1 + nameWidth_ - head.size(), ' ');
        if (showTimes_) {
            if (single) {
                head += rightLabel("time", timeW + 3);
            } else {
                const char* labels[3] = {"avg", "min", "max"};
                for (int i = 0; i < 3; i++) head += rightLabel(labels[i], timeW + 3);
                if (showMedian_) head += rightLabel("med", timeW + 3);
            }
        }
        if (metric_ != kNoMetric) head += rightLabel(metricLabel(), kMetricWidth);
        for (size_t i = 0; i < columns_.size(); i++)
            head += rightLabel(columns_[i].header.c_str(), columns_[i].width);
        if (baseline_ >= 0) head += rightLabel("x base", kSpeedupWidth);
        if (anyChecked())   head += rightLabel("check", kCheckWidth);
        std::cout << head << "\n";
        rule(w);

        for (size_t i = 0; i < variants_.size(); i++) {
            const Variant& v = variants_[i];
            Stat s = summarize(v.samples);

            std::cout << " " << std::left << std::setw(nameWidth_) << v.label << std::right;
            if (showTimes_) {
                if (single) {
                    std::cout << std::fixed << std::setprecision(2)
                              << std::setw(timeW) << s.avg*unit.scale << " " << unit.name;
                } else {
                    timeCells(s, unit, timeW);
                    if (showMedian_)
                        std::cout << std::setw(timeW) << s.med*unit.scale << " " << unit.name;
                }
            }
            if (metric_ != kNoMetric)
                std::cout << std::setw(kMetricWidth) << std::fixed << std::setprecision(2)
                          << metricValue(v, s.avg);
            for (size_t c = 0; c < columns_.size(); c++) printCell(columns_[c], v.cells[c]);
            if (baseline_ >= 0)
                std::cout << std::setw(kSpeedupWidth - 1) << std::fixed << std::setprecision(2)
                          << ((s.avg > 0.0 && baseAvg > 0.0) ? baseAvg / s.avg : 0.0) << "x";
            if (anyChecked())
                std::cout << std::setw(kCheckWidth)
                          << (!v.checked ? "-" : (v.mismatches == 0 ? "PASS" : "FAIL"));
            std::cout << "\n";
        }
        rule(w);

        if (!totalLabel_.empty()) {
            Stat t = totalStat();
            double amount = totalAmount();
            std::cout << " " << std::left << std::setw(nameWidth_) << totalLabel_ << std::right;
            if (showTimes_) {
                if (single) {
                    std::cout << std::fixed << std::setprecision(2)
                              << std::setw(timeW) << t.avg*unit.scale << " " << unit.name;
                } else {
                    timeCells(t, unit, timeW);
                    if (showMedian_) std::cout << std::setw(timeW + 3) << "";
                }
            }
            if (metric_ != kNoMetric) {
                double mv = (metric_ == kBandwidth) ? gbPerSec(amount, t.avg)
                                                    : gflopPerSec(amount, t.avg);
                std::cout << std::setw(kMetricWidth) << std::fixed << std::setprecision(2) << mv;
            }
            for (size_t c = 0; c < columns_.size(); c++)
                std::cout << std::setw(columns_[c].width) << "";
            if (baseline_ >= 0) std::cout << std::setw(kSpeedupWidth) << "";
            if (anyChecked())
                std::cout << std::setw(kCheckWidth)
                          << (!totalChecked_ ? "-" : (totalMismatches_ == 0 ? "PASS" : "FAIL"));
            std::cout << "\n";
        }

        for (size_t i = 0; i < variants_.size(); i++) {
            const Variant& v = variants_[i];
            if (v.checked && v.mismatches != 0)
                std::cout << " " << v.label << ": " << v.mismatches << " mismatches\n";
        }
        if (totalChecked_ && totalMismatches_ != 0)
            std::cout << " " << totalLabel_ << ": " << totalMismatches_ << " mismatches\n";
        bannerEnd(w);
    }

    bench::Stat totalStat() const
    {
        bench::Stat t;
        for (size_t i = 0; i < variants_.size(); i++) {
            bench::Stat s = bench::summarize(variants_[i].samples);
            t.avg += s.avg;
            t.min += s.min;
            t.max += s.max;
        }
        return t;
    }

    double totalAmount() const
    {
        double amount = 0.0;
        for (size_t i = 0; i < variants_.size(); i++) amount += variants_[i].amount;
        return amount;
    }

    // "% peak" -> pct_peak, "cycles/load" -> cycles_load: a column header is for
    // people, a CSV field name has to survive a script.
    static std::string csvName(const std::string& header)
    {
        std::string out;
        for (size_t i = 0; i < header.size(); i++) {
            char ch = header[i];
            if (ch == '%') out += "pct";
            else if (ch == ' ' || ch == '/' || ch == '-') out += '_';
            else out += ch;
        }
        std::string squeezed;
        for (size_t i = 0; i < out.size(); i++) {
            if (out[i] == '_' && !squeezed.empty() && squeezed[squeezed.size()-1] == '_') continue;
            squeezed += out[i];
        }
        while (!squeezed.empty() && squeezed[0] == '_') squeezed.erase(0, 1);
        while (!squeezed.empty() && squeezed[squeezed.size()-1] == '_') squeezed.erase(squeezed.size()-1);
        return squeezed;
    }

    std::string csvHeader() const
    {
        std::string h = "name,variant,iter";
        if (showTimes_) {
            h += singleSample() ? ",ms" : ",avg_ms,min_ms,max_ms";
            if (showMedian_ && !singleSample()) h += ",med_ms";
        }
        if (metric_ != kNoMetric) h += (metric_ == kBandwidth) ? ",gbs" : ",gflops";
        for (size_t i = 0; i < columns_.size(); i++) h += "," + csvName(columns_[i].header);
        if (baseline_ >= 0) h += ",speedup";
        if (anyChecked())   h += ",status";
        return h;
    }

    void printCsvRow(const std::string& label, const bench::Stat& s, double amount,
                     const std::vector<Cell>* cells, double baseAvg,
                     bool checked, long long mismatches) const
    {
        using namespace bench;
        std::cout << name_ << "," << label << "," << iter_;
        if (showTimes_) {
            std::cout << std::fixed << std::setprecision(4) << "," << s.avg;
            if (!singleSample()) {
                std::cout << "," << s.min << "," << s.max;
                if (showMedian_) std::cout << "," << s.med;
            }
        }
        if (metric_ != kNoMetric) {
            double mv = (metric_ == kBandwidth) ? gbPerSec(amount, s.avg)
                                                : gflopPerSec(amount, s.avg);
            std::cout << std::setprecision(2) << "," << mv;
        }
        for (size_t c = 0; c < columns_.size(); c++) {
            std::cout << ",";
            if (!cells || !(*cells)[c].set) continue;
            const Cell& cell = (*cells)[c];
            if (cell.isText) { std::cout << cell.text; continue; }
            std::cout << std::setprecision(columns_[c].precision);
            if (columns_[c].scientific) std::cout << std::scientific << cell.value << std::fixed;
            else                        std::cout << std::fixed << cell.value;
        }
        if (baseline_ >= 0)
            std::cout << std::setprecision(3) << ","
                      << ((s.avg > 0.0 && baseAvg > 0.0) ? baseAvg / s.avg : 0.0);
        if (anyChecked())
            std::cout << "," << (!checked ? "unchecked" : (mismatches == 0 ? "pass" : "fail"));
        std::cout << "\n";
    }

    void printCsv() const
    {
        using namespace bench;

        double baseAvg = 0.0;
        if (baseline_ >= 0) baseAvg = summarize(variants_[baseline_].samples).avg;

        std::cout << csvHeader() << "\n";
        for (size_t i = 0; i < variants_.size(); i++) {
            const Variant& v = variants_[i];
            printCsvRow(v.label, summarize(v.samples), v.amount, &v.cells, baseAvg,
                        v.checked, v.mismatches);
        }
        // The total row carries the pipeline's verification, so the CSV needs it
        // too or a sweep would only ever see "unchecked" stages.
        if (!totalLabel_.empty())
            printCsvRow(totalLabel_, totalStat(), totalAmount(), 0, 0.0,
                        totalChecked_, totalMismatches_);
    }

    static const int kMetricWidth = 10;
    static const int kSpeedupWidth = 9;
    static const int kCheckWidth = 8;

    std::string name_;
    int iter_;
    int nameWidth_;
    Metric metric_;
    int baseline_;
    std::string variantHeader_;
    std::string totalLabel_;
    long long totalMismatches_;
    bool totalChecked_;
    bool showTimes_;
    bool showMedian_;
    std::vector<std::pair<std::string, std::string> > config_;
    std::vector<Column> columns_;
    std::vector<Variant> variants_;
};
