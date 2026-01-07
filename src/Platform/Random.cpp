/**
 * @file Random.cpp
 * @brief Random number generation implementation
 */

#include <QiVision/Platform/Random.h>

#include <chrono>
#include <thread>

namespace Qi::Vision::Platform {

Random& Random::Instance() {
    // Thread-local instance for thread safety
    thread_local Random instance;
    return instance;
}

Random::Random() {
    // Initialize with a combination of time and thread ID for uniqueness
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

    // Combine with thread ID for multi-threaded uniqueness
    std::hash<std::thread::id> hasher;
    uint64_t threadHash = hasher(std::this_thread::get_id());

    seed_ = static_cast<uint64_t>(nanos) ^ threadHash;
    gen_.seed(seed_);
}

void Random::SetSeed(uint64_t seed) {
    seed_ = seed;
    gen_.seed(seed);

    // Reset distributions to ensure clean state
    unitDist_.reset();
    normalDist_.reset();
}

// =========================================================================
// Integer Generation
// =========================================================================

uint32_t Random::Uint32() {
    return static_cast<uint32_t>(gen_());
}

uint64_t Random::Uint64() {
    return gen_();
}

int32_t Random::Int(int32_t min, int32_t max) {
    if (min > max) {
        std::swap(min, max);
    }
    std::uniform_int_distribution<int32_t> dist(min, max);
    return dist(gen_);
}

size_t Random::Index(size_t max) {
    if (max == 0) {
        return 0;
    }
    std::uniform_int_distribution<size_t> dist(0, max - 1);
    return dist(gen_);
}

// =========================================================================
// Floating Point Generation
// =========================================================================

float Random::Float() {
    return static_cast<float>(unitDist_(gen_));
}

double Random::Double() {
    return unitDist_(gen_);
}

float Random::Float(float min, float max) {
    if (min > max) {
        std::swap(min, max);
    }
    std::uniform_real_distribution<float> dist(min, max);
    return dist(gen_);
}

double Random::Double(double min, double max) {
    if (min > max) {
        std::swap(min, max);
    }
    std::uniform_real_distribution<double> dist(min, max);
    return dist(gen_);
}

double Random::Gaussian() {
    return normalDist_(gen_);
}

double Random::Gaussian(double mean, double stddev) {
    return mean + stddev * normalDist_(gen_);
}

// =========================================================================
// Boolean Generation
// =========================================================================

bool Random::Bool() {
    return (gen_() & 1) != 0;
}

bool Random::Bool(double probabilityTrue) {
    if (probabilityTrue <= 0.0) return false;
    if (probabilityTrue >= 1.0) return true;
    return unitDist_(gen_) < probabilityTrue;
}

// =========================================================================
// Sampling Functions
// =========================================================================

std::vector<size_t> Random::SampleIndices(size_t n, size_t k) {
    if (k >= n) {
        // Return all indices
        std::vector<size_t> result(n);
        for (size_t i = 0; i < n; ++i) {
            result[i] = i;
        }
        return result;
    }

    std::vector<size_t> result;
    result.reserve(k);

    if (k <= n / 2) {
        // Use selection sampling (good when k << n)
        // Floyd's algorithm for sampling without replacement
        std::vector<bool> selected(n, false);

        while (result.size() < k) {
            size_t idx = Index(n);
            if (!selected[idx]) {
                selected[idx] = true;
                result.push_back(idx);
            }
        }
    } else {
        // Use shuffle-based sampling (better when k is close to n)
        std::vector<size_t> indices(n);
        for (size_t i = 0; i < n; ++i) {
            indices[i] = i;
        }

        // Partial Fisher-Yates shuffle
        for (size_t i = 0; i < k; ++i) {
            size_t j = i + Index(n - i);
            std::swap(indices[i], indices[j]);
        }

        result.assign(indices.begin(), indices.begin() + k);
    }

    return result;
}

} // namespace Qi::Vision::Platform
