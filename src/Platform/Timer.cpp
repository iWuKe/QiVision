/**
 * @file Timer.cpp
 * @brief Timer implementation
 */

#include <QiVision/Platform/Timer.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <thread>
#include <vector>

namespace Qi::Vision::Platform {

// ============================================================================
// Timer Implementation
// ============================================================================

Timer::Timer(bool autoStart) {
    if (autoStart) {
        Start();
    }
}

void Timer::Start() {
    if (!running_) {
        startTime_ = Clock::now();
        running_ = true;
    }
}

void Timer::Stop() {
    if (running_) {
        stopTime_ = Clock::now();
        accumulated_ += std::chrono::duration_cast<Duration>(stopTime_ - startTime_);
        running_ = false;
    }
}

void Timer::Reset() {
    accumulated_ = Duration{0};
    running_ = false;
}

Timer::Duration Timer::Elapsed() const {
    if (running_) {
        auto now = Clock::now();
        return accumulated_ + std::chrono::duration_cast<Duration>(now - startTime_);
    }
    return accumulated_;
}

double Timer::ElapsedSeconds() const {
    return Elapsed().count();
}

double Timer::ElapsedMs() const {
    return ElapsedSeconds() * 1000.0;
}

double Timer::ElapsedUs() const {
    return ElapsedSeconds() * 1000000.0;
}

int64_t Timer::ElapsedNs() const {
    auto elapsed = Elapsed();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
}

double Timer::Lap() {
    double elapsed = ElapsedMs();
    Reset();
    Start();
    return elapsed;
}

// ============================================================================
// ScopedTimer Implementation
// ============================================================================

ScopedTimer::ScopedTimer(const std::string& name, bool printOnDestruct)
    : name_(name)
    , timer_(true)  // Auto-start
    , printOnDestruct_(printOnDestruct) {
}

ScopedTimer::~ScopedTimer() {
    if (printOnDestruct_) {
        std::cout << name_ << ": " << std::fixed << std::setprecision(3)
                  << timer_.ElapsedMs() << " ms" << std::endl;
    }
}

void ScopedTimer::Checkpoint(const std::string& label) {
    std::string checkpointName = name_;
    if (!label.empty()) {
        checkpointName += " [" + label + "]";
    }
    std::cout << checkpointName << ": " << std::fixed << std::setprecision(3)
              << timer_.ElapsedMs() << " ms" << std::endl;
}

// ============================================================================
// Benchmarking Functions
// ============================================================================

double Benchmark(const std::function<void()>& func,
                 size_t iterations,
                 size_t warmup) {
    // Warmup
    for (size_t i = 0; i < warmup; ++i) {
        func();
    }

    // Actual timing
    Timer timer(true);
    for (size_t i = 0; i < iterations; ++i) {
        func();
    }
    timer.Stop();

    return timer.ElapsedMs() / static_cast<double>(iterations);
}

BenchmarkResult BenchmarkDetailed(const std::function<void()>& func,
                                   size_t iterations,
                                   size_t warmup) {
    // Warmup
    for (size_t i = 0; i < warmup; ++i) {
        func();
    }

    // Collect individual timings
    std::vector<double> times(iterations);
    for (size_t i = 0; i < iterations; ++i) {
        Timer timer(true);
        func();
        timer.Stop();
        times[i] = timer.ElapsedMs();
    }

    // Calculate statistics
    BenchmarkResult result;
    result.iterations = iterations;

    // Min/Max
    result.minMs = *std::min_element(times.begin(), times.end());
    result.maxMs = *std::max_element(times.begin(), times.end());

    // Average
    double sum = 0.0;
    for (double t : times) {
        sum += t;
    }
    result.avgMs = sum / static_cast<double>(iterations);

    // Median
    std::sort(times.begin(), times.end());
    if (iterations % 2 == 0) {
        result.medianMs = (times[iterations / 2 - 1] + times[iterations / 2]) / 2.0;
    } else {
        result.medianMs = times[iterations / 2];
    }

    // Standard deviation
    double sumSqDiff = 0.0;
    for (double t : times) {
        double diff = t - result.avgMs;
        sumSqDiff += diff * diff;
    }
    result.stddevMs = std::sqrt(sumSqDiff / static_cast<double>(iterations));

    return result;
}

void PrintBenchmarkResult(const std::string& name, const BenchmarkResult& result) {
    std::cout << "Benchmark: " << name << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Iterations: " << result.iterations << std::endl;
    std::cout << "  Min:    " << result.minMs << " ms" << std::endl;
    std::cout << "  Max:    " << result.maxMs << " ms" << std::endl;
    std::cout << "  Avg:    " << result.avgMs << " ms" << std::endl;
    std::cout << "  Median: " << result.medianMs << " ms" << std::endl;
    std::cout << "  StdDev: " << result.stddevMs << " ms" << std::endl;
}

// ============================================================================
// Convenience Functions
// ============================================================================

int64_t GetTimestampMs() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

int64_t GetTimestampUs() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

void SleepMs(int64_t ms) {
    if (ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }
}

void SleepUs(int64_t us) {
    if (us > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(us));
    }
}

} // namespace Qi::Vision::Platform
