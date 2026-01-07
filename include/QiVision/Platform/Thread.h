#pragma once

/**
 * @file Thread.h
 * @brief Thread pool and parallel execution utilities
 *
 * Provides:
 * - Global thread pool for task execution
 * - ParallelFor for data-parallel loops
 * - Automatic work partitioning
 *
 * Usage:
 * @code
 * // Simple parallel for
 * ParallelFor(0, 1000, [&](size_t i) {
 *     process(data[i]);
 * });
 *
 * // With grain size control
 * ParallelFor(0, height, [&](size_t y) {
 *     processRow(y);
 * }, 4);  // Process 4 rows per task
 *
 * // Range-based (for blocking)
 * ParallelForRange(0, height, [&](size_t start, size_t end) {
 *     for (size_t y = start; y < end; ++y) {
 *         processRow(y);
 *     }
 * });
 * @endcode
 */

#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

namespace Qi::Vision::Platform {

// ============================================================================
// System Information
// ============================================================================

/**
 * @brief Get number of hardware threads (logical cores)
 * @return Number of threads, minimum 1
 */
size_t GetNumCores();

/**
 * @brief Get recommended number of worker threads
 * @return GetNumCores() - 1, minimum 1
 *
 * Leaves one core for the main thread.
 */
size_t GetRecommendedThreadCount();

// ============================================================================
// Thread Pool
// ============================================================================

/**
 * @brief Simple thread pool for parallel task execution
 *
 * Singleton pattern - use Instance() to access.
 * Automatically sized based on hardware.
 */
class ThreadPool {
public:
    /**
     * @brief Get global thread pool instance
     */
    static ThreadPool& Instance();

    /**
     * @brief Destructor - waits for all tasks to complete
     */
    ~ThreadPool();

    // Non-copyable
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /**
     * @brief Get number of worker threads
     */
    size_t Size() const { return workers_.size(); }

    /**
     * @brief Check if pool is running
     */
    bool IsRunning() const { return !stop_; }

    /**
     * @brief Submit a task and get a future for the result
     * @param f Function to execute
     * @param args Arguments to pass
     * @return Future for the result
     */
    template<typename F, typename... Args>
    auto Submit(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    /**
     * @brief Submit a task without waiting for result
     * @param f Function to execute
     */
    template<typename F>
    void Execute(F&& f);

    /**
     * @brief Wait for all pending tasks to complete
     */
    void WaitAll();

    /**
     * @brief Get number of pending tasks
     */
    size_t PendingTasks() const;

private:
    explicit ThreadPool(size_t numThreads);

    void WorkerThread();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::condition_variable completionCondition_;
    std::atomic<bool> stop_{false};
    std::atomic<size_t> activeTasks_{0};
};

// ============================================================================
// Parallel For
// ============================================================================

/**
 * @brief Execute a function for each index in range [begin, end)
 * @param begin Start index (inclusive)
 * @param end End index (exclusive)
 * @param func Function to call with each index
 * @param grainSize Minimum iterations per task (0 = auto)
 *
 * The function is called as func(index) for each index.
 * Work is distributed across the thread pool.
 */
template<typename Func>
void ParallelFor(size_t begin, size_t end, Func&& func, size_t grainSize = 0);

/**
 * @brief Execute a function for ranges of indices
 * @param begin Start index (inclusive)
 * @param end End index (exclusive)
 * @param func Function to call with (rangeStart, rangeEnd)
 * @param numChunks Number of chunks (0 = auto based on thread count)
 *
 * More efficient when per-iteration overhead is low.
 * The function handles its own loop: func(start, end)
 */
template<typename Func>
void ParallelForRange(size_t begin, size_t end, Func&& func, size_t numChunks = 0);

/**
 * @brief Execute a 2D parallel for (for image processing)
 * @param rows Number of rows
 * @param cols Number of columns
 * @param func Function to call with (row, col)
 * @param rowGrain Minimum rows per task (0 = auto)
 *
 * Parallelizes over rows, processes columns sequentially.
 * This is cache-friendly for row-major image data.
 */
template<typename Func>
void ParallelFor2D(size_t rows, size_t cols, Func&& func, size_t rowGrain = 0);

/**
 * @brief Execute a 2D parallel for with row ranges
 * @param rows Number of rows
 * @param cols Number of columns
 * @param func Function to call with (rowStart, rowEnd, cols)
 *
 * More efficient version where func handles the row loop.
 */
template<typename Func>
void ParallelFor2DRange(size_t rows, size_t cols, Func&& func);

// ============================================================================
// Utility
// ============================================================================

/**
 * @brief Check if parallel execution is worthwhile
 * @param workSize Total amount of work
 * @param minWorkPerThread Minimum work per thread to justify parallelism
 * @return true if parallel execution is recommended
 */
inline bool ShouldParallelize(size_t workSize, size_t minWorkPerThread = 1000) {
    return workSize >= minWorkPerThread * 2 && GetNumCores() > 1;
}

/**
 * @brief Calculate optimal grain size for ParallelFor
 * @param totalWork Total iterations
 * @param minGrain Minimum grain size
 * @return Recommended grain size
 */
size_t CalculateGrainSize(size_t totalWork, size_t minGrain = 1);

// ============================================================================
// Template Implementations
// ============================================================================

template<typename F, typename... Args>
auto ThreadPool::Submit(F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type>
{
    using ReturnType = typename std::invoke_result<F, Args...>::type;

    auto task = std::make_shared<std::packaged_task<ReturnType()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<ReturnType> result = task->get_future();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_) {
            throw std::runtime_error("ThreadPool is stopped");
        }
        tasks_.emplace([task]() { (*task)(); });
    }

    condition_.notify_one();
    return result;
}

template<typename F>
void ThreadPool::Execute(F&& f) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_) {
            return;
        }
        tasks_.emplace(std::forward<F>(f));
    }
    condition_.notify_one();
}

template<typename Func>
void ParallelFor(size_t begin, size_t end, Func&& func, size_t grainSize) {
    if (begin >= end) return;

    size_t count = end - begin;

    // For small work or single core, run sequentially
    if (count == 1 || !ShouldParallelize(count, 100)) {
        for (size_t i = begin; i < end; ++i) {
            func(i);
        }
        return;
    }

    // Calculate grain size if not specified
    if (grainSize == 0) {
        grainSize = CalculateGrainSize(count);
    }

    // Calculate number of tasks
    size_t numTasks = (count + grainSize - 1) / grainSize;
    auto& pool = ThreadPool::Instance();

    std::vector<std::future<void>> futures;
    futures.reserve(numTasks);

    for (size_t task = 0; task < numTasks; ++task) {
        size_t taskBegin = begin + task * grainSize;
        size_t taskEnd = std::min(taskBegin + grainSize, end);

        futures.push_back(pool.Submit([&func, taskBegin, taskEnd]() {
            for (size_t i = taskBegin; i < taskEnd; ++i) {
                func(i);
            }
        }));
    }

    // Wait for all tasks
    for (auto& f : futures) {
        f.get();
    }
}

template<typename Func>
void ParallelForRange(size_t begin, size_t end, Func&& func, size_t numChunks) {
    if (begin >= end) return;

    size_t count = end - begin;

    // For small work, run sequentially
    if (!ShouldParallelize(count, 100)) {
        func(begin, end);
        return;
    }

    // Calculate number of chunks
    if (numChunks == 0) {
        numChunks = std::min(count, GetNumCores() * 2);
    }
    numChunks = std::min(numChunks, count);

    auto& pool = ThreadPool::Instance();
    std::vector<std::future<void>> futures;
    futures.reserve(numChunks);

    size_t chunkSize = count / numChunks;
    size_t remainder = count % numChunks;

    size_t current = begin;
    for (size_t chunk = 0; chunk < numChunks; ++chunk) {
        size_t thisChunkSize = chunkSize + (chunk < remainder ? 1 : 0);
        size_t chunkEnd = current + thisChunkSize;

        futures.push_back(pool.Submit([&func, current, chunkEnd]() {
            func(current, chunkEnd);
        }));

        current = chunkEnd;
    }

    for (auto& f : futures) {
        f.get();
    }
}

template<typename Func>
void ParallelFor2D(size_t rows, size_t cols, Func&& func, size_t rowGrain) {
    if (rows == 0 || cols == 0) return;

    // Parallelize over rows
    ParallelFor(0, rows, [&func, cols](size_t row) {
        for (size_t col = 0; col < cols; ++col) {
            func(row, col);
        }
    }, rowGrain);
}

template<typename Func>
void ParallelFor2DRange(size_t rows, size_t cols, Func&& func) {
    if (rows == 0 || cols == 0) return;

    ParallelForRange(0, rows, [&func, cols](size_t rowStart, size_t rowEnd) {
        func(rowStart, rowEnd, cols);
    });
}

} // namespace Qi::Vision::Platform
