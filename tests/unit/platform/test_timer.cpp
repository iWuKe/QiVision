/**
 * @file test_timer.cpp
 * @brief Unit tests for Platform/Timer.h
 */

#include <QiVision/Platform/Timer.h>
#include <gtest/gtest.h>

#include <thread>
#include <cmath>

using namespace Qi::Vision::Platform;

class TimerTest : public ::testing::Test {
protected:
    // Allow some tolerance for timing (1ms)
    static constexpr double TIME_TOLERANCE_MS = 5.0;
};

// ============================================================================
// Timer Basic Tests
// ============================================================================

TEST_F(TimerTest, DefaultConstructor) {
    Timer timer;
    EXPECT_FALSE(timer.IsRunning());
    EXPECT_NEAR(timer.ElapsedMs(), 0.0, TIME_TOLERANCE_MS);
}

TEST_F(TimerTest, AutoStartConstructor) {
    Timer timer(true);
    EXPECT_TRUE(timer.IsRunning());

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_GT(timer.ElapsedMs(), 5.0);
}

TEST_F(TimerTest, StartStop) {
    Timer timer;

    timer.Start();
    EXPECT_TRUE(timer.IsRunning());

    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    timer.Stop();
    EXPECT_FALSE(timer.IsRunning());

    double elapsed = timer.ElapsedMs();
    EXPECT_GE(elapsed, 15.0);  // Should be at least 15ms

    // Elapsed should not change after stop
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_NEAR(timer.ElapsedMs(), elapsed, 0.01);
}

TEST_F(TimerTest, Reset) {
    Timer timer(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    timer.Reset();
    EXPECT_FALSE(timer.IsRunning());
    EXPECT_NEAR(timer.ElapsedMs(), 0.0, 0.01);
}

TEST_F(TimerTest, AccumulatedTime) {
    Timer timer;

    // First run
    timer.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    timer.Stop();
    double first = timer.ElapsedMs();

    // Second run (should accumulate)
    timer.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    timer.Stop();
    double total = timer.ElapsedMs();

    EXPECT_GT(total, first);
    EXPECT_GE(total, 15.0);  // At least 15ms total
}

TEST_F(TimerTest, Lap) {
    Timer timer;
    timer.Start();

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    double lap1 = timer.Lap();
    EXPECT_GE(lap1, 5.0);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    double lap2 = timer.Lap();
    EXPECT_GE(lap2, 5.0);

    // After lap, timer should be reset but running
    EXPECT_TRUE(timer.IsRunning());
}

// ============================================================================
// Time Unit Tests
// ============================================================================

TEST_F(TimerTest, ElapsedSeconds) {
    Timer timer(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    timer.Stop();

    double seconds = timer.ElapsedSeconds();
    EXPECT_GE(seconds, 0.04);
    EXPECT_LE(seconds, 0.15);
}

TEST_F(TimerTest, ElapsedUs) {
    Timer timer(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    timer.Stop();

    double us = timer.ElapsedUs();
    EXPECT_GE(us, 5000.0);  // At least 5ms = 5000us
}

TEST_F(TimerTest, ElapsedNs) {
    Timer timer(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    timer.Stop();

    int64_t ns = timer.ElapsedNs();
    EXPECT_GE(ns, 5000000);  // At least 5ms = 5000000ns
}

// ============================================================================
// ScopedTimer Tests
// ============================================================================

TEST_F(TimerTest, ScopedTimerElapsed) {
    ScopedTimer timer("Test", false);  // Don't print

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    EXPECT_GE(timer.ElapsedMs(), 5.0);
}

TEST_F(TimerTest, ScopedTimerCancel) {
    // Just verify cancel doesn't crash
    ScopedTimer timer("Test", true);
    timer.Cancel();
    // Destructor should not print
}

// ============================================================================
// Benchmark Tests
// ============================================================================

TEST_F(TimerTest, BenchmarkBasic) {
    int counter = 0;
    double avgMs = Benchmark([&counter]() {
        ++counter;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }, 10, 2);

    EXPECT_EQ(counter, 12);  // 2 warmup + 10 iterations
    EXPECT_GE(avgMs, 0.05);   // At least 50us average
}

TEST_F(TimerTest, BenchmarkDetailedStats) {
    auto result = BenchmarkDetailed([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }, 10, 2);

    EXPECT_EQ(result.iterations, 10u);
    EXPECT_GE(result.minMs, 0.5);
    EXPECT_GE(result.avgMs, result.minMs);
    EXPECT_LE(result.avgMs, result.maxMs);
    EXPECT_GE(result.stddevMs, 0.0);
}

TEST_F(TimerTest, BenchmarkMedian) {
    auto result = BenchmarkDetailed([]() {
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }, 11, 2);  // Odd number for exact median

    // Median should be between min and max
    EXPECT_GE(result.medianMs, result.minMs);
    EXPECT_LE(result.medianMs, result.maxMs);
}

// ============================================================================
// Convenience Function Tests
// ============================================================================

TEST_F(TimerTest, GetTimestampMs) {
    int64_t t1 = GetTimestampMs();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    int64_t t2 = GetTimestampMs();

    EXPECT_GT(t2, t1);
    EXPECT_GE(t2 - t1, 5);  // At least 5ms difference
}

TEST_F(TimerTest, GetTimestampUs) {
    int64_t t1 = GetTimestampUs();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    int64_t t2 = GetTimestampUs();

    EXPECT_GT(t2, t1);
    EXPECT_GE(t2 - t1, 5000);  // At least 5000us = 5ms
}

TEST_F(TimerTest, SleepMs) {
    Timer timer(true);
    SleepMs(20);
    timer.Stop();

    EXPECT_GE(timer.ElapsedMs(), 15.0);
}

TEST_F(TimerTest, SleepUs) {
    Timer timer(true);
    SleepUs(20000);  // 20ms
    timer.Stop();

    EXPECT_GE(timer.ElapsedMs(), 15.0);
}

TEST_F(TimerTest, SleepZeroOrNegative) {
    // Should not hang or crash
    Timer timer(true);
    SleepMs(0);
    SleepMs(-10);
    SleepUs(0);
    SleepUs(-100);
    timer.Stop();

    EXPECT_LT(timer.ElapsedMs(), 10.0);  // Should be very fast
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(TimerTest, MultipleStartCalls) {
    Timer timer;
    timer.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Second start should be ignored
    timer.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    timer.Stop();

    // Should have about 20ms, not 10ms (if second start reset)
    EXPECT_GE(timer.ElapsedMs(), 15.0);
}

TEST_F(TimerTest, MultipleStopCalls) {
    Timer timer(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    timer.Stop();
    double first = timer.ElapsedMs();

    // Second stop should be ignored
    timer.Stop();
    EXPECT_NEAR(timer.ElapsedMs(), first, 0.01);
}

TEST_F(TimerTest, ElapsedWhileRunning) {
    Timer timer(true);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    double t1 = timer.ElapsedMs();

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    double t2 = timer.ElapsedMs();

    EXPECT_GT(t2, t1);  // Should increase while running
}

TEST_F(TimerTest, HighResolutionTiming) {
    // Test that we can measure sub-millisecond times
    Timer timer(true);

    // Busy wait for ~100us
    auto start = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - start).count() < 100) {
        // Busy wait
    }

    timer.Stop();

    // Should be measurable (at least 50us)
    EXPECT_GE(timer.ElapsedUs(), 50.0);
    EXPECT_LT(timer.ElapsedUs(), 1000.0);  // Less than 1ms
}
