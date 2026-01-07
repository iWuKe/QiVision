/**
 * @file test_thread.cpp
 * @brief Unit tests for Platform/Thread.h
 */

#include <QiVision/Platform/Thread.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <numeric>
#include <vector>
#include <set>
#include <cmath>

using namespace Qi::Vision::Platform;

class ThreadTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure thread pool is initialized
        ThreadPool::Instance();
    }
};

// ============================================================================
// System Information Tests
// ============================================================================

TEST_F(ThreadTest, GetNumCoresReturnsPositive) {
    size_t cores = GetNumCores();
    EXPECT_GE(cores, 1u);
}

TEST_F(ThreadTest, GetRecommendedThreadCountReturnsPositive) {
    size_t threads = GetRecommendedThreadCount();
    EXPECT_GE(threads, 1u);
    EXPECT_LE(threads, GetNumCores());
}

// ============================================================================
// Thread Pool Basic Tests
// ============================================================================

TEST_F(ThreadTest, ThreadPoolInstanceNotNull) {
    auto& pool = ThreadPool::Instance();
    EXPECT_GT(pool.Size(), 0u);
}

TEST_F(ThreadTest, ThreadPoolIsRunning) {
    auto& pool = ThreadPool::Instance();
    EXPECT_TRUE(pool.IsRunning());
}

TEST_F(ThreadTest, ThreadPoolSubmitReturnsResult) {
    auto& pool = ThreadPool::Instance();

    auto future = pool.Submit([]() {
        return 42;
    });

    EXPECT_EQ(future.get(), 42);
}

TEST_F(ThreadTest, ThreadPoolSubmitWithArgs) {
    auto& pool = ThreadPool::Instance();

    auto future = pool.Submit([](int a, int b) {
        return a + b;
    }, 10, 20);

    EXPECT_EQ(future.get(), 30);
}

TEST_F(ThreadTest, ThreadPoolMultipleTasks) {
    auto& pool = ThreadPool::Instance();
    const int numTasks = 100;

    std::vector<std::future<int>> futures;
    futures.reserve(numTasks);

    for (int i = 0; i < numTasks; ++i) {
        futures.push_back(pool.Submit([i]() {
            return i * i;
        }));
    }

    for (int i = 0; i < numTasks; ++i) {
        EXPECT_EQ(futures[i].get(), i * i);
    }
}

TEST_F(ThreadTest, ThreadPoolExecute) {
    auto& pool = ThreadPool::Instance();
    std::atomic<int> counter{0};

    for (int i = 0; i < 100; ++i) {
        pool.Execute([&counter]() {
            ++counter;
        });
    }

    pool.WaitAll();
    EXPECT_EQ(counter.load(), 100);
}

TEST_F(ThreadTest, ThreadPoolWaitAll) {
    auto& pool = ThreadPool::Instance();
    std::atomic<int> counter{0};

    for (int i = 0; i < 50; ++i) {
        pool.Execute([&counter]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            ++counter;
        });
    }

    pool.WaitAll();
    EXPECT_EQ(counter.load(), 50);
}

// ============================================================================
// ParallelFor Tests
// ============================================================================

TEST_F(ThreadTest, ParallelForEmptyRange) {
    std::atomic<int> counter{0};

    ParallelFor(0, 0, [&counter](size_t) {
        ++counter;
    });

    EXPECT_EQ(counter.load(), 0);
}

TEST_F(ThreadTest, ParallelForSingleElement) {
    std::atomic<int> counter{0};

    ParallelFor(0, 1, [&counter](size_t) {
        ++counter;
    });

    EXPECT_EQ(counter.load(), 1);
}

TEST_F(ThreadTest, ParallelForAllIndicesProcessed) {
    const size_t n = 1000;
    std::vector<std::atomic<int>> flags(n);

    for (size_t i = 0; i < n; ++i) {
        flags[i] = 0;
    }

    ParallelFor(0, n, [&flags](size_t i) {
        flags[i]++;
    });

    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(flags[i].load(), 1) << "Index " << i << " not processed exactly once";
    }
}

TEST_F(ThreadTest, ParallelForWithOffset) {
    const size_t begin = 100;
    const size_t end = 200;
    std::vector<std::atomic<int>> flags(end);

    for (size_t i = 0; i < end; ++i) {
        flags[i] = 0;
    }

    ParallelFor(begin, end, [&flags](size_t i) {
        flags[i]++;
    });

    // Before begin should be untouched
    for (size_t i = 0; i < begin; ++i) {
        EXPECT_EQ(flags[i].load(), 0);
    }

    // Within range should be processed
    for (size_t i = begin; i < end; ++i) {
        EXPECT_EQ(flags[i].load(), 1);
    }
}

TEST_F(ThreadTest, ParallelForWithGrainSize) {
    const size_t n = 1000;
    std::atomic<int> sum{0};

    ParallelFor(0, n, [&sum](size_t i) {
        sum += static_cast<int>(i);
    }, 100);  // Grain size 100

    int expected = static_cast<int>((n - 1) * n / 2);
    EXPECT_EQ(sum.load(), expected);
}

TEST_F(ThreadTest, ParallelForComputation) {
    const size_t n = 10000;
    std::vector<double> input(n);
    std::vector<double> output(n);

    for (size_t i = 0; i < n; ++i) {
        input[i] = static_cast<double>(i);
    }

    ParallelFor(0, n, [&input, &output](size_t i) {
        output[i] = std::sqrt(input[i]) * 2.0;
    });

    for (size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(output[i], std::sqrt(static_cast<double>(i)) * 2.0);
    }
}

// ============================================================================
// ParallelForRange Tests
// ============================================================================

TEST_F(ThreadTest, ParallelForRangeEmptyRange) {
    std::atomic<int> counter{0};

    ParallelForRange(0, 0, [&counter](size_t, size_t) {
        ++counter;
    });

    EXPECT_EQ(counter.load(), 0);
}

TEST_F(ThreadTest, ParallelForRangeCoversAll) {
    const size_t n = 1000;
    std::vector<std::atomic<int>> flags(n);

    for (size_t i = 0; i < n; ++i) {
        flags[i] = 0;
    }

    ParallelForRange(0, n, [&flags](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            flags[i]++;
        }
    });

    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(flags[i].load(), 1) << "Index " << i << " not processed exactly once";
    }
}

TEST_F(ThreadTest, ParallelForRangeWithChunks) {
    const size_t n = 1000;
    std::atomic<size_t> chunkCount{0};

    ParallelForRange(0, n, [&chunkCount](size_t, size_t) {
        ++chunkCount;
    }, 4);  // 4 chunks

    EXPECT_EQ(chunkCount.load(), 4u);
}

// ============================================================================
// ParallelFor2D Tests
// ============================================================================

TEST_F(ThreadTest, ParallelFor2DEmptyImage) {
    std::atomic<int> counter{0};

    ParallelFor2D(0, 100, [&counter](size_t, size_t) {
        ++counter;
    });

    EXPECT_EQ(counter.load(), 0);
}

TEST_F(ThreadTest, ParallelFor2DAllPixelsProcessed) {
    const size_t rows = 100;
    const size_t cols = 100;
    std::vector<std::atomic<int>> flags(rows * cols);

    for (size_t i = 0; i < rows * cols; ++i) {
        flags[i] = 0;
    }

    ParallelFor2D(rows, cols, [&flags, cols](size_t row, size_t col) {
        flags[row * cols + col]++;
    });

    for (size_t i = 0; i < rows * cols; ++i) {
        EXPECT_EQ(flags[i].load(), 1);
    }
}

TEST_F(ThreadTest, ParallelFor2DRangeAllPixelsProcessed) {
    const size_t rows = 100;
    const size_t cols = 100;
    std::vector<std::atomic<int>> flags(rows * cols);

    for (size_t i = 0; i < rows * cols; ++i) {
        flags[i] = 0;
    }

    ParallelFor2DRange(rows, cols, [&flags](size_t rowStart, size_t rowEnd, size_t cols) {
        for (size_t row = rowStart; row < rowEnd; ++row) {
            for (size_t col = 0; col < cols; ++col) {
                flags[row * cols + col]++;
            }
        }
    });

    for (size_t i = 0; i < rows * cols; ++i) {
        EXPECT_EQ(flags[i].load(), 1);
    }
}

// ============================================================================
// Utility Function Tests
// ============================================================================

TEST_F(ThreadTest, ShouldParallelizeSmallWork) {
    // Small work should not parallelize
    EXPECT_FALSE(ShouldParallelize(10, 1000));
    EXPECT_FALSE(ShouldParallelize(1000, 1000));
}

TEST_F(ThreadTest, ShouldParallelizeLargeWork) {
    // Large work should parallelize (if multi-core)
    if (GetNumCores() > 1) {
        EXPECT_TRUE(ShouldParallelize(10000, 1000));
        EXPECT_TRUE(ShouldParallelize(100000, 1000));
    }
}

TEST_F(ThreadTest, CalculateGrainSizePositive) {
    size_t grainSize = CalculateGrainSize(10000);
    EXPECT_GT(grainSize, 0u);
}

TEST_F(ThreadTest, CalculateGrainSizeRespectsMinimum) {
    size_t grainSize = CalculateGrainSize(100, 50);
    EXPECT_GE(grainSize, 50u);
}

// ============================================================================
// Correctness Tests
// ============================================================================

TEST_F(ThreadTest, ParallelSumCorrectness) {
    const size_t n = 100000;
    std::vector<int> data(n);

    for (size_t i = 0; i < n; ++i) {
        data[i] = static_cast<int>(i % 100);
    }

    // Sequential sum
    int seqSum = std::accumulate(data.begin(), data.end(), 0);

    // Parallel sum using atomic
    std::atomic<int> parSum{0};
    ParallelFor(0, n, [&data, &parSum](size_t i) {
        parSum += data[i];
    });

    EXPECT_EQ(parSum.load(), seqSum);
}

TEST_F(ThreadTest, ParallelVectorTransform) {
    const size_t n = 50000;
    std::vector<double> input(n);
    std::vector<double> output(n);

    for (size_t i = 0; i < n; ++i) {
        input[i] = static_cast<double>(i);
    }

    ParallelFor(0, n, [&input, &output](size_t i) {
        output[i] = input[i] * 2.0 + 1.0;
    });

    for (size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(output[i], input[i] * 2.0 + 1.0);
    }
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(ThreadTest, ConcurrentPoolAccess) {
    auto& pool = ThreadPool::Instance();
    const int numThreads = 4;
    const int tasksPerThread = 100;

    std::atomic<int> counter{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&pool, &counter, tasksPerThread]() {
            for (int i = 0; i < tasksPerThread; ++i) {
                pool.Execute([&counter]() {
                    ++counter;
                });
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    pool.WaitAll();
    EXPECT_EQ(counter.load(), numThreads * tasksPerThread);
}

TEST_F(ThreadTest, NestedParallelFor) {
    // Nested parallel for should work (inner runs sequentially on worker)
    const size_t n = 10;
    std::atomic<int> counter{0};

    ParallelFor(0, n, [&counter, n](size_t) {
        // Inner loop runs sequentially
        for (size_t j = 0; j < n; ++j) {
            ++counter;
        }
    });

    EXPECT_EQ(counter.load(), static_cast<int>(n * n));
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(ThreadTest, ParallelForReversedRange) {
    std::atomic<int> counter{0};

    // begin > end should do nothing
    ParallelFor(100, 0, [&counter](size_t) {
        ++counter;
    });

    EXPECT_EQ(counter.load(), 0);
}

TEST_F(ThreadTest, VeryLargeRange) {
    const size_t n = 1000000;
    std::atomic<size_t> sum{0};

    ParallelFor(0, n, [&sum](size_t i) {
        sum += i;
    });

    size_t expected = (n - 1) * n / 2;
    EXPECT_EQ(sum.load(), expected);
}

TEST_F(ThreadTest, SmallRangeSequential) {
    // Very small ranges should run sequentially
    const size_t n = 5;
    std::vector<int> order;
    std::mutex mtx;

    ParallelFor(0, n, [&order, &mtx](size_t i) {
        std::lock_guard<std::mutex> lock(mtx);
        order.push_back(static_cast<int>(i));
    });

    EXPECT_EQ(order.size(), n);

    // All indices should be present
    std::set<int> indices(order.begin(), order.end());
    EXPECT_EQ(indices.size(), n);
}
