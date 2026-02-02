#pragma once

/**
 * @file Timer.h
 * @brief High-resolution timing utilities
 *
 * Provides:
 * - High-resolution time measurement
 * - Scoped timing (RAII)
 * - Simple benchmarking utilities
 *
 * Usage:
 * @code
 * // Simple timing
 * Timer timer;
 * timer.Start();
 * // ... work ...
 * double elapsed = timer.ElapsedMs();
 *
 * // Scoped timing
 * {
 *     ScopedTimer timer("MyFunction");
 *     // ... work ...
 * }  // Prints: "MyFunction: 12.34 ms"
 *
 * // Benchmark
 * double avgTime = Benchmark([]() {
 *     // ... work ...
 * }, 100);  // 100 iterations
 * @endcode
 */

#include <chrono>
#include <cstdint>
#include <string>
#include <functional>
#include <QiVision/Core/Export.h>

namespace Qi::Vision::Platform {

/**
 * @brief High-resolution timer
 *
 * Uses std::chrono::high_resolution_clock for best precision.
 */
class QIVISION_API Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::duration<double>;

    /**
     * @brief Construct and optionally start timer
     * @param autoStart If true, timer starts immediately
     */
    explicit Timer(bool autoStart = false);

    /**
     * @brief Start or restart the timer
     */
    void Start();

    /**
     * @brief Stop the timer
     */
    void Stop();

    /**
     * @brief Reset timer to zero
     */
    void Reset();

    /**
     * @brief Check if timer is running
     */
    bool IsRunning() const { return running_; }

    // =========================================================================
    // Elapsed Time Getters
    // =========================================================================

    /**
     * @brief Get elapsed time in seconds
     */
    double ElapsedSeconds() const;

    /**
     * @brief Get elapsed time in milliseconds
     */
    double ElapsedMs() const;

    /**
     * @brief Get elapsed time in microseconds
     */
    double ElapsedUs() const;

    /**
     * @brief Get elapsed time in nanoseconds
     */
    int64_t ElapsedNs() const;

    /**
     * @brief Get elapsed time as duration
     */
    Duration Elapsed() const;

    // =========================================================================
    // Lap Timing
    // =========================================================================

    /**
     * @brief Get elapsed time and restart timer
     * @return Elapsed time in milliseconds before restart
     */
    double Lap();

private:
    TimePoint startTime_;
    TimePoint stopTime_;
    Duration accumulated_{0};
    bool running_ = false;
};

/**
 * @brief RAII timer that prints elapsed time on destruction
 *
 * Useful for quick profiling of code blocks.
 */
class QIVISION_API ScopedTimer {
public:
    /**
     * @brief Construct and start timing
     * @param name Name to print (identifies this timing)
     * @param printOnDestruct If true, prints elapsed time when destroyed
     */
    explicit ScopedTimer(const std::string& name, bool printOnDestruct = true);

    /**
     * @brief Destructor - prints elapsed time if enabled
     */
    ~ScopedTimer();

    // Non-copyable
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

    /**
     * @brief Get elapsed time in milliseconds
     */
    double ElapsedMs() const { return timer_.ElapsedMs(); }

    /**
     * @brief Print current elapsed time without stopping
     * @param label Optional label for this checkpoint
     */
    void Checkpoint(const std::string& label = "");

    /**
     * @brief Disable printing on destruction
     */
    void Cancel() { printOnDestruct_ = false; }

private:
    std::string name_;
    Timer timer_;
    bool printOnDestruct_;
};

// ============================================================================
// Benchmarking Utilities
// ============================================================================

/**
 * @brief Run a function multiple times and return average time
 * @param func Function to benchmark
 * @param iterations Number of iterations
 * @param warmup Number of warmup iterations (not counted)
 * @return Average time per iteration in milliseconds
 */
QIVISION_API double Benchmark(const std::function<void()>& func,
                 size_t iterations = 100,
                 size_t warmup = 10);

/**
 * @brief Benchmark result with statistics
 */
struct QIVISION_API BenchmarkResult {
    double minMs;       ///< Minimum time
    double maxMs;       ///< Maximum time
    double avgMs;       ///< Average time
    double medianMs;    ///< Median time
    double stddevMs;    ///< Standard deviation
    size_t iterations;  ///< Number of iterations
};

/**
 * @brief Run a detailed benchmark with statistics
 * @param func Function to benchmark
 * @param iterations Number of iterations
 * @param warmup Number of warmup iterations
 * @return Detailed benchmark results
 */
QIVISION_API BenchmarkResult BenchmarkDetailed(const std::function<void()>& func,
                                   size_t iterations = 100,
                                   size_t warmup = 10);

/**
 * @brief Print benchmark result to stdout
 * @param name Name of the benchmark
 * @param result Benchmark result
 */
QIVISION_API void PrintBenchmarkResult(const std::string& name, const BenchmarkResult& result);

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * @brief Get current timestamp in milliseconds since epoch
 */
QIVISION_API int64_t GetTimestampMs();

/**
 * @brief Get current timestamp in microseconds since epoch
 */
QIVISION_API int64_t GetTimestampUs();

/**
 * @brief Sleep for specified milliseconds
 */
QIVISION_API void SleepMs(int64_t ms);

/**
 * @brief Sleep for specified microseconds
 */
QIVISION_API void SleepUs(int64_t us);

} // namespace Qi::Vision::Platform
