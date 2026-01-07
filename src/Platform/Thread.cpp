/**
 * @file Thread.cpp
 * @brief Thread pool and parallel execution implementation
 */

#include <QiVision/Platform/Thread.h>

#include <algorithm>

namespace Qi::Vision::Platform {

// ============================================================================
// System Information
// ============================================================================

size_t GetNumCores() {
    unsigned int cores = std::thread::hardware_concurrency();
    return cores > 0 ? static_cast<size_t>(cores) : 1;
}

size_t GetRecommendedThreadCount() {
    size_t cores = GetNumCores();
    return cores > 1 ? cores - 1 : 1;
}

// ============================================================================
// Thread Pool Implementation
// ============================================================================

ThreadPool& ThreadPool::Instance() {
    static ThreadPool instance(GetRecommendedThreadCount());
    return instance;
}

ThreadPool::ThreadPool(size_t numThreads) {
    // Ensure at least 1 thread
    numThreads = std::max<size_t>(numThreads, 1);

    workers_.reserve(numThreads);
    for (size_t i = 0; i < numThreads; ++i) {
        workers_.emplace_back(&ThreadPool::WorkerThread, this);
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }

    condition_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::WorkerThread() {
    while (true) {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(mutex_);

            condition_.wait(lock, [this] {
                return stop_ || !tasks_.empty();
            });

            if (stop_ && tasks_.empty()) {
                return;
            }

            if (!tasks_.empty()) {
                task = std::move(tasks_.front());
                tasks_.pop();
                ++activeTasks_;
            }
        }

        if (task) {
            task();

            {
                std::lock_guard<std::mutex> lock(mutex_);
                --activeTasks_;
            }
            completionCondition_.notify_all();
        }
    }
}

void ThreadPool::WaitAll() {
    std::unique_lock<std::mutex> lock(mutex_);
    completionCondition_.wait(lock, [this] {
        return tasks_.empty() && activeTasks_ == 0;
    });
}

size_t ThreadPool::PendingTasks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tasks_.size() + activeTasks_;
}

// ============================================================================
// Utility Functions
// ============================================================================

size_t CalculateGrainSize(size_t totalWork, size_t minGrain) {
    size_t numThreads = GetNumCores();
    if (numThreads <= 1) {
        return totalWork;  // No parallelism
    }

    // Aim for 4-8 tasks per thread for good load balancing
    size_t targetTasks = numThreads * 4;

    size_t grainSize = (totalWork + targetTasks - 1) / targetTasks;
    return std::max(grainSize, minGrain);
}

} // namespace Qi::Vision::Platform
